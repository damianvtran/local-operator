"""A non-UTF-8 byte on stdin must not kill the app, and a legacy report must still work.

Scope, and why it is a file of its own. The crash this covers is not a mode
negotiation bug — it is the input thread dying on a byte the terminal is entitled
to send — so it is fixed and tested where the bytes are decoded
(``local_operator.tui.input_decode``), not in the pixel-scale gate's files. The
two mechanisms are related but independent, and each has to hold on its own: the
mode resets in ``terminal_modes`` decide what we ASK the terminal for, this
decides what happens to a byte already on its way to us.

``THE PTY TESTS ARE THE ONES THAT MATTER``
(``test_the_app_boots_and_runs_on_a_pty`` and
``test_a_legacy_x10_report_does_not_kill_the_app``). Everything above them
asserts our own contract; those two boot the REAL ``run_tui`` entry point on a
real pty — production boot, real driver, real parser, real ``file_logging`` —
inject the exact bytes from the operator's log, and assert on the app's own
lifetime and log file. The second one is the regression test for the production
failure and fails on the pre-fix tree by exiting the child (measured: the panic
traceback lands in ``local-operator.log`` with
``UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 4``).

The two reports injected are the operator's two, byte for byte:

    ESC [ M <button+32> <x+32> <y+32>, with x=96 (0x80) and x=215 (0xf7)

Every subprocess here gets an ISOLATED HOME and ``LOCAL_OPERATOR_CONFIG_DIR``
and its environment is built from scratch rather than inherited, like the
sibling pixel-mouse pty tests: no ``CMUX_*`` can reach a child (an inherited
``CMUX_WORKSPACE_ID`` lets a booted TUI rename the operator's live cmux
workspaces) and no ``LOP_*`` can make a cell adopt its parent's provider. The
``run_tui`` children DO start a session, so nothing here is a "cheap" arm: they
run against their own config root and a ``FakeSession``, and no live session of
the operator's is reachable from them.
"""

from __future__ import annotations

import json
import os
import re
import select
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from textual import events
from textual._xterm_parser import XTermParser

from local_operator.tui.input_decode import (
    X10_MOUSE_PREFIX,
    NonFatalStdinDecoder,
    decoder_factory,
    install_nonfatal_stdin_decode,
    nonfatal_stdin_decode_installed,
    uninstall_nonfatal_stdin_decode,
    x10_mouse_to_sgr,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The operator's two reports, exactly as their terminal wrote them: the byte in
#: the log is the x coordinate (position 4 = index of x in ESC [ M b x y).
OPERATOR_REPORTS = b"\x1b[M\x20\x80\x4c\x1b[M\x20\xf7\x4c"

#: What those two reports mean, as the text Textual's parser consumes.
OPERATOR_REPORTS_AS_SGR = "\x1b[<0;96;44M\x1b[<0;215;44M"

#: The report frame is six bytes, so a read boundary divides it at one of five
#: interior offsets. Offset 1 is the ESC byte alone, which is the one split the
#: decoder answers by handing the ESC over rather than holding it (see
#: ``input_decode``): holding it would delay the ESC KEY, and one lost report is
#: the cheaper failure.
_REPORT_BYTES = 6


def _splits_at_a_lone_escape(split: int) -> bool:
    """True when ``split`` falls immediately after a report's own ESC byte."""
    return split % _REPORT_BYTES == 1


# -- the translation ----------------------------------------------------------


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        # The operator's first: left button, x=96, y=44.
        (b"\x1b[M\x20\x80\x4c", "\x1b[<0;96;44M"),
        # The operator's second: left button, x=215, y=44.
        (b"\x1b[M\x20\xf7\x4c", "\x1b[<0;215;44M"),
        # Right button (34) at the origin: 1-based cells stay 1-based here and
        # the parser subtracts the one, so this is cell (0, 0).
        (b"\x1b[M\x22\x21\x21", "\x1b[<2;1;1M"),
        # Wheel up (64) and down (65) — the high button codes are the ones an
        # off-by-32 in either direction would silently reinterpret.
        (b"\x1b[M\x60\x30\x30", "\x1b[<64;16;16M"),
        (b"\x1b[M\x61\x30\x30", "\x1b[<65;16;16M"),
    ],
)
def test_a_legacy_report_becomes_the_sgr_form_the_parser_consumes(
    report: bytes, expected: str
) -> None:
    """The X10 values are ``value + 32``; SGR carries the value, and cells are 1-based.

    Asserted as literal text rather than through a helper: this mapping is the
    wire contract the fix rests on, and a test that recomputed it from the same
    expression the implementation uses would agree with any bug.
    """
    assert x10_mouse_to_sgr(report) == expected


def test_a_byte_below_32_is_not_a_report_and_is_dropped() -> None:
    """No encoder can emit a value below 32 after adding 32 to it.

    So such a report is un-interpretable, and the contract for that is "dropped,
    never raised" — a None here is what keeps the caller's decode from inventing
    a coordinate.
    """
    assert x10_mouse_to_sgr(b"\x1b[M\x1f\x30\x30") is None
    assert x10_mouse_to_sgr(b"\x1b[M\x20\x00\x30") is None


def test_the_operator_reports_reach_the_real_parser_as_mouse_events() -> None:
    """The mouse keeps WORKING, asserted on the real parser and not on our own text.

    The control arm is the reason this is one test and not two: the same bytes
    decoded with ``errors="replace"`` — the obvious fix, and the one the module
    docstring rejects — produce NO mouse event at all. A green translation arm
    on its own could mean "our text happens to look right"; the pair says what a
    lossy decode costs, which is the pointer.
    """
    decoded = NonFatalStdinDecoder().decode(OPERATOR_REPORTS)
    assert decoded == OPERATOR_REPORTS_AS_SGR

    parser = XTermParser(False)
    parsed = [m for m in parser.feed(decoded) if isinstance(m, events.MouseEvent)]
    assert [type(m).__name__ for m in parsed] == ["MouseDown", "MouseDown"]
    assert [(m.x, m.y, m.button) for m in parsed] == [(95, 43, 1), (214, 43, 1)]

    lossy = OPERATOR_REPORTS.decode("utf-8", errors="replace")
    assert "\ufffd" in lossy
    assert [m for m in XTermParser(False).feed(lossy) if isinstance(m, events.MouseEvent)] == []


# -- the decoder --------------------------------------------------------------


def test_a_report_split_across_two_reads_translates_exactly_once() -> None:
    """A read boundary can fall anywhere, including inside the six bytes.

    Two split classes, because they are answered differently on purpose:

    - **Inside the introducer but past the first byte** (``ESC [`` or ``ESC [ M``)
      is a partial prefix with nothing ambiguous about it, so it is HELD and the
      report still translates — this is what makes the stateful scan worth its
      state.
    - **At the very first byte** is the documented loss: a lone trailing ``ESC``
      is NOT held, because holding it would delay the ESC KEY by however long
      the user takes to press another key (Textual resolves a lone ESC on its own
      timeout, and a byte we never hand it cannot time out). One report is lost
      when the kernel splits a six-byte write exactly there; the app does not
      notice, and ``decode`` raises nothing. Asserted here so that a future
      "optimisation" that starts holding the ESC has to face this reasoning.
    """
    for split in range(2, len(OPERATOR_REPORTS)):
        if _splits_at_a_lone_escape(split):
            # A read boundary immediately after a report's own ESC byte is the
            # documented loss pinned below, not a hold case.
            continue
        decoder = NonFatalStdinDecoder()
        first = decoder.decode(OPERATOR_REPORTS[:split])
        second = decoder.decode(OPERATOR_REPORTS[split:], final=True)
        assert first + second == OPERATOR_REPORTS_AS_SGR, f"split at {split}: {first + second!r}"

    split_at_the_escape = NonFatalStdinDecoder()
    escaped = split_at_the_escape.decode(OPERATOR_REPORTS[:1])
    rest = split_at_the_escape.decode(OPERATOR_REPORTS[1:], final=True)
    assert escaped == "\x1b"
    # The report is gone — replaced characters, and the parser's legacy arm
    # matches it and finds no handler — but nothing raised and nothing was
    # invented.
    assert OPERATOR_REPORTS_AS_SGR not in escaped + rest
    assert "\ufffd" in rest


def test_a_fragment_never_completes_into_a_half_character() -> None:
    """Every prefix of the report decodes, in order, to at most the full event.

    The property that matters is monotonic: decoding growing prefixes of the
    stream must never emit MORE than the complete stream does, and must never
    emit a replacement character — a fragment held and later decoded as text
    would show up as one. Prefixes that are answered by holding (``ESC[`` and
    ``ESC[M``) emit nothing at all, which is included here rather than special-
    cased.
    """
    for split in range(2, len(OPERATOR_REPORTS)):
        decoder = NonFatalStdinDecoder()
        partial = decoder.decode(OPERATOR_REPORTS[:split])
        assert "\ufffd" not in partial, f"split at {split}: {partial!r}"
        assert OPERATOR_REPORTS_AS_SGR.startswith(partial), f"split at {split}: {partial!r}"


def test_a_truncated_report_at_eof_is_dropped() -> None:
    """EOF can leave a report half-written; there is nothing to complete it.

    The held bytes are raw coordinates and are not text, so they are dropped
    rather than decoded into replacement characters the parser would turn into
    key events.
    """
    decoder = NonFatalStdinDecoder()
    assert decoder.decode(b"\x1b[M\x20") == ""
    assert decoder.decode(b"", final=True) == ""


@pytest.mark.parametrize("byte", range(256))
def test_any_byte_at_all_decodes_without_raising(byte: int) -> None:
    """The load-bearing property, swept over the whole byte space.

    The strict decoder the driver uses by default is the control arm: it raises
    on the high bytes, which is the crash. That control is asserted rather than
    assumed, because a test that only proved our decoder returns a string would
    pass against a decoder that never got the byte either.
    """
    raw = bytes([byte])
    if byte >= 0x80:
        # The control arm: the strict decoder the driver uses by default raises
        # on exactly this byte, which is the crash. Asserted rather than assumed
        # because a test that only proved our decoder returns a string would
        # pass against one that never got the byte either.
        with pytest.raises(UnicodeDecodeError):
            raw.decode("utf-8")
    assert isinstance(NonFatalStdinDecoder().decode(raw, final=True), str)


def test_the_whole_byte_span_decodes_in_one_stream_without_raising() -> None:
    """The sweep above is per byte; this is all 256 of them in one stream.

    A decoder that handled single bytes but broke on a run of them (a state bug
    in the hold, say) would pass the per-byte sweep and fail here.
    """
    decoder = NonFatalStdinDecoder()
    out = decoder.decode(bytes(range(256)))
    assert isinstance(out, str)
    assert "\ufffd" in out  # the un-interpretable bytes are replaced, not raised


def test_the_decoder_hands_text_through_untouched() -> None:
    """A ``str`` is not the driver's contract, but it is not an error either.

    Nobody else's decoder would refuse one, and raising here would be a fatal
    path of exactly the kind this module exists to remove.
    """
    assert NonFatalStdinDecoder().decode("plain text") == "plain text"


def test_a_multi_byte_character_split_across_reads_still_decodes() -> None:
    """The incremental half: the held decoder owns a half-finished character.

    "é" as one character, not as two replacement characters, when the read
    boundary falls inside its two bytes — the reason the ordinary path in this
    decoder is an incremental decoder and not ``bytes.decode``.
    """
    decoder = NonFatalStdinDecoder()
    assert decoder.decode(b"caf\xc3") == "caf"
    assert decoder.decode(b"\xa9") == "é"


# -- the install --------------------------------------------------------------


def test_the_install_points_the_driver_at_the_non_fatal_factory() -> None:
    """The patch is on the name the driver's input thread looks up.

    Read through the module attribute rather than through a local import: a
    decoder built any other way is not the one the driver will use, and this is
    the assertion that the mechanism and the mechanism's target agree.
    """
    import codecs

    import textual.drivers.linux_driver as driver

    assert driver.getincrementaldecoder is codecs.getincrementaldecoder
    assert install_nonfatal_stdin_decode() is True
    assert nonfatal_stdin_decode_installed() is True
    assert driver.getincrementaldecoder is decoder_factory

    decoder = driver.getincrementaldecoder("utf-8")()
    assert isinstance(decoder, NonFatalStdinDecoder)
    assert isinstance(decoder.decode(b"\x1b[M\x20\x80\x4c"), str)

    # Idempotent, and the restore has to be the ORIGINAL object: the module's
    # binding was `from codecs import getincrementaldecoder`, so anything else
    # leaves the driver with a patched name after the test's teardown.
    assert install_nonfatal_stdin_decode() is False
    assert uninstall_nonfatal_stdin_decode() is True
    assert driver.getincrementaldecoder is codecs.getincrementaldecoder
    assert uninstall_nonfatal_stdin_decode() is False


def test_a_missing_driver_module_is_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows has no ``linux_driver``, and the install must not care.

    It imports ``termios``/``tty``, so on Windows the import raises — and
    ``run_tui`` calls this unconditionally on every platform, which would turn
    the fix for a boot failure into a boot failure. The whole surface is asserted
    rather than just the install: a caller that asks ``installed`` or uninstalls
    on that platform must get an answer, not an ImportError.

    ``None`` in ``sys.modules`` is how the stdlib itself simulates an import
    that cannot complete, so this is the real code path and not a stub of it.
    """
    monkeypatch.setitem(sys.modules, "textual.drivers.linux_driver", None)

    assert install_nonfatal_stdin_decode() is False
    assert nonfatal_stdin_decode_installed() is False
    assert uninstall_nonfatal_stdin_decode() is False


def test_only_utf8_is_replaced() -> None:
    """Every other encoding keeps exactly the decoder it asked for."""
    import codecs

    assert decoder_factory("utf-8") is NonFatalStdinDecoder
    assert decoder_factory("UTF8") is NonFatalStdinDecoder
    assert decoder_factory("latin-1") is codecs.getincrementaldecoder("latin-1")


def test_the_driver_still_binds_the_decoder_at_module_level() -> None:
    """Tripwire: a Textual bump that moves the decoder fails here, not in production.

    ``input_decode`` reaches the reader through one specific thing: the
    module-level ``getincrementaldecoder`` name that ``run_input_thread`` binds
    from ``codecs`` at import time (``linux_driver.py:10``) and calls to build
    its decoder (``:430``). A release that inlined ``codecs.getincrementaldecoder``
    or moved the decode elsewhere would keep this suite green while the crash
    returned, so the binding is asserted against the installed source — the same
    shape as the pixel-mouse tripwires, which name their version and fail rather
    than pass silently.

    The ``panic`` assertion is the other half of the claim this file rests on:
    an exception on that thread is FATAL, not logged and swallowed, which is why
    a decode error is an app death. If a future Textual stops panicking on input
    errors this test goes red and the fix's justification (not its mechanism)
    needs re-reading.
    """
    import textual.drivers.linux_driver as driver_module

    source = Path(driver_module.__file__).read_text(encoding="utf-8")
    assert "from codecs import getincrementaldecoder" in source
    assert re.search(r'utf8_decoder\s*=\s*getincrementaldecoder\("utf-8"\)\(\)', source), (
        "the driver no longer builds its stdin decoder from the module-level "
        "getincrementaldecoder name; input_decode's install is now inert"
    )
    assert "self._app.panic" in source


# -- the wiring ---------------------------------------------------------------


_WIRING_CHILD = """
import asyncio
import json
import sys
import types

sys.path.insert(0, {repo!r})

import local_operator.tui as tui
from local_operator.tui import input_decode

calls = []
_real_install = tui.install_nonfatal_stdin_decode


def _install(*args, **kwargs):
    calls.append("decode")
    return _real_install(*args, **kwargs)


tui.install_nonfatal_stdin_decode = _install


class _NoApp(types.ModuleType):
    def __getattr__(self, name):
        raise ImportError("stubbed by the wiring test: " + name)


# The boot stops at the lazy app import, i.e. immediately after the terminal
# patches: anything the recording list is missing was never called, and anything
# that happened by then happened before a driver could exist.
sys.modules["local_operator.tui.app"] = _NoApp("local_operator.tui.app")

try:
    asyncio.run(tui.run_tui(lambda: None))
except ImportError:
    pass

print(
    json.dumps(
        {{
            "calls": calls,
            "installed": input_decode.nonfatal_stdin_decode_installed(),
        }}
    )
)
"""


def _run_wiring_child(tmp_path: Path, smooth_scroll: str) -> dict[str, Any]:
    """Run the real ``run_tui`` with ``TEXTUAL_SMOOTH_SCROLL`` as given.

    ``smooth_scroll`` is passed as an argv value so "absent" and "empty" stay
    distinguishable; the child's environment is built for it and carries no
    ``TEXTUAL_SMOOTH_SCROLL`` of its own.
    """
    import subprocess

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / ".local-operator"),
    }
    out = subprocess.run(
        [sys.executable, "-c", _WIRING_CHILD.format(repo=str(_REPO_ROOT)), smooth_scroll],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.parametrize("smooth_scroll", ["", "1"])
def test_run_tui_installs_the_decode_patch_unconditionally(
    tmp_path: Path, smooth_scroll: str
) -> None:
    """The wiring AND the conditionality, which no pty test can see.

    Two claims, and the second is the one a refactor is most likely to break:

    - ``run_tui`` calls the install at all, before the app is constructed. The
      pty children cannot show this: they boot the real entry point, but a tree
      with the call deleted would still exercise the decoder through nothing —
      they would simply lose the fix, which is the silent failure mode this test
      exists to make loud.
    - It does so in EVERY configuration, including ``TEXTUAL_SMOOTH_SCROLL=1``
      where every other terminal patch in ``run_tui`` stands down. That value
      means "keep Textual's pixel-mouse negotiation", and it must not be able to
      re-arm a fatal decode: the byte on the wire is not the user's to opt out
      of when the decoder is what dies.
    """
    result = _run_wiring_child(tmp_path, smooth_scroll)

    assert result["calls"] == ["decode"], result
    assert result["installed"] is True, result


# -- the app, on a pty --------------------------------------------------------


#: The production boot, driven by its real entry point: ``run_tui`` is where the
#: mode reset, the guard, the parser gate and the decoder install all live, so a
#: child that constructed ``OperatorApp`` itself would test the fix with the fix
#: left out. The session is the suite's own fake, and the factory shape is the
#: one the product passes.
_BOOT_CHILD = """
import asyncio
import sys

sys.path.insert(0, {repo!r})

from local_operator.tui import run_tui
from tests.unit.tui.test_app_pilot import FakeSession, _factory

asyncio.run(run_tui(lambda: _factory(FakeSession())))
"""


class _BootArm:
    """``run_tui`` on its own pty, driven and observed by the parent.

    ``fcntl``/``termios`` are imported in ``__init__`` rather than at module
    scope: they do not exist on Windows, so a module-level import would error
    collection before this file's ``skipif`` could skip it. Repo precedent:
    ``test_pixel_mouse_gate.py:_PtyChild``.
    """

    def __init__(self, child_source: str, tmp_path: Path) -> None:
        import fcntl
        import struct
        import termios

        self._fcntl = fcntl
        self._struct = struct
        self._termios = termios
        self._reaped = False
        self.cfg = tmp_path / ".local-operator"
        self.cfg.mkdir(parents=True, exist_ok=True)

        master, slave = os.openpty()
        #: 45 rows x 160 columns, like the sibling pty arms: enough for the
        #: app's layout, and wide enough that cell x=96 is on the screen.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 45, 160, 0, 0))

        # Built from scratch, not inherited: see the module docstring on CMUX_*
        # and LOP_*. HOME is redirected as well as the config dir, because the
        # cache and skills roots are derived from HOME independently.
        env = {
            "TERM": "xterm-256color",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "LOCAL_OPERATOR_CONFIG_DIR": str(self.cfg),
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
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self._master], [], [], 0.05)
            if not readable:
                continue
            try:
                data = os.read(self._master, 65536)
            except OSError:
                return
            if not data:
                return
            self._chunks.append(data)

    def wait_for_paint(self, minimum: int, timeout: float = 60.0) -> None:
        """Wait until the app has painted at least ``minimum`` bytes.

        A boot takes seconds, and how many depends on the host, so the wait is
        on the OUTPUT rather than on the clock: injecting before the driver's
        input thread exists would make a green arm mean "the harness was early",
        which is the failure mode this whole file is about. The sibling pty
        harnesses wait on the clock; this one cannot, because the injection has
        to be a fact about a booted app.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(b"".join(self._chunks)) >= minimum:
                return
            self.drain(0.1)
        raise AssertionError(
            f"the app never painted {minimum} bytes; captured {len(b''.join(self._chunks))} "
            f"bytes and the child is {'gone' if self.exited() else 'still running'}"
        )

    def send(self, data: bytes) -> None:
        os.write(self._master, data)

    def alive(self, seconds: float = 0.0) -> bool:
        """False once the child has exited; waits ``seconds`` for it to do so.

        Reaping is remembered rather than re-attempted: ``waitpid`` on a child
        that has already been collected raises ``ChildProcessError``, and this
        is asked several times through a test.
        """
        deadline = time.monotonic() + seconds
        while True:
            if not self._reaped:
                try:
                    done, _ = os.waitpid(self._pid, os.WNOHANG)
                except ChildProcessError:
                    done = self._pid
                if done:
                    self._reaped = True
                    return False
            if time.monotonic() >= deadline:
                return not self._reaped
            self.drain(0.1)

    def exited(self) -> bool:
        return not self.alive()

    def output(self) -> bytes:
        self.drain(0.3)
        return b"".join(self._chunks)

    def log_text(self) -> str:
        """The app's own rotating log, which is where a panic is recorded."""
        logs = self.cfg / "logs"
        if not logs.is_dir():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace") for path in sorted(logs.iterdir())
        )

    def close(self) -> None:
        try:
            os.kill(self._pid, 9)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(self._pid, 0)
        except ChildProcessError:
            pass
        try:
            os.close(self._master)
        except OSError:
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="pty semantics are POSIX-only")
def test_the_app_boots_and_runs_on_a_pty(tmp_path: Path) -> None:
    """The boot-and-run assertion, on its own arm.

    "At the very least that the TUI can boot and run" is a separate claim from
    "it survives a bad byte", and it is the one a change to the boot path breaks
    first. Liveness here is a REPAINT caused by input — a keystroke the app
    echoes — rather than the process merely existing, because a frozen app holds
    a pid too. Nothing is injected, so this arm is also the control for the
    crash test below: if it is red, that test's result means nothing.
    """
    arm = _BootArm(_BOOT_CHILD.format(repo=str(_REPO_ROOT)), tmp_path)
    try:
        arm.wait_for_paint(20_000)
        before = len(arm.output())
        arm.send(b"/")
        arm.drain(1.0)
        assert arm.alive(10.0), "the app exited while idle after boot"
        assert len(arm.output()) > before, "the app never repainted for a keystroke"
        assert "UnicodeDecodeError" not in arm.log_text()
    finally:
        arm.close()


@pytest.mark.skipif(sys.platform == "win32", reason="pty semantics are POSIX-only")
def test_a_legacy_x10_report_does_not_kill_the_app(tmp_path: Path) -> None:
    """The production failure, and the regression test for it.

    On the pre-fix tree this arm exits: the strict decoder raises on the 0x80
    byte, ``LinuxDriver._run_input_thread`` answers with ``self._app.panic(...)``
    and the traceback lands in the app's own log —
    ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 4``,
    which is the operator's log line, byte for byte. Both assertions are kept
    because they fail for different reasons and both are the bug: the process
    dying (the app is gone) and the decode error (it died of THAT, not of
    something else a pty arm might trip over).

    The keystroke after the injection is the "and it is still working" half: a
    surviving-but-wedged app would pass the liveness check alone.
    """
    arm = _BootArm(_BOOT_CHILD.format(repo=str(_REPO_ROOT)), tmp_path)
    try:
        arm.wait_for_paint(20_000)
        arm.send(OPERATOR_REPORTS)
        arm.drain(1.0)

        assert arm.alive(10.0), (
            "the app exited after a legacy X10 mouse report; " f"log: {arm.log_text()[-800:]!r}"
        )
        assert "UnicodeDecodeError" not in arm.log_text(), (
            "the decode error reached the log: this is the reported crash "
            f"surviving as a logged failure rather than a clean run; log: {arm.log_text()[-800:]!r}"
        )

        before = len(arm.output())
        arm.send(b"/")
        arm.drain(1.0)
        assert arm.alive(10.0), "the app exited while echoing a keystroke after the report"
        assert len(arm.output()) > before, "the app stopped repainting after the report"
    finally:
        arm.close()


def test_the_prefix_the_decoder_scans_for_is_the_parsers_own() -> None:
    """Cross-check the introducer, because both sides of the seam rely on it.

    Textual's mouse regex accepts the legacy form as ``M`` plus three characters
    (``_re_mouse_event``, ``_xterm_parser.py:29``), so the six bytes this module
    treats as one report are exactly the six the parser would consume — which is
    what makes translating in place (rather than hand-building an event) the
    right shape. Asserted against the compiled pattern rather than against text
    in the file: a Textual bump that changed the frame would otherwise leave
    this module emitting a sequence the parser splits differently, and only the
    pty arm might notice.
    """
    from textual import _xterm_parser as parser_module

    assert X10_MOUSE_PREFIX == b"\x1b[M"
    legacy = parser_module._re_mouse_event.match("\x1b[M\x20\x80\x4c")
    assert legacy is not None, "the parser no longer accepts the legacy X10 frame"
    assert legacy.group(1) == "M\x20\x80\x4c"
    assert parser_module._re_mouse_event.match("\x1b[<0;96;44M") is not None
