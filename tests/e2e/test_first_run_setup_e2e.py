"""First-run setup state over the REAL CLI — the wiring the unit shape faked.

The setup-state unit tests drive a factory that RAISES
``HostingNotConfiguredError`` — the only in-process shape that reached the
state — while the shipped viewer factory swallowed that error and built a cold
viewer. Everything gated on the setup state was therefore unreachable on a real
fresh install: the splash's ``/login`` cue, her R28 view and the R26 routing
(review round 1, U1/Q1 — the reviewer's note: "a raising test factory cannot
catch this class"). This stage boots the real binary over a pty in an isolated
root and reads the painted frames, so the wiring itself is what is under test.

The pty machinery follows ``tests/unit/tui/test_pixel_mouse_gate.py``'s
``_PtyChild``: fork, ``setsid``, make the slave the controlling tty, exec the
real module. The child is reaped by exact pid in every path.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import signal
import struct
import sys
import termios
import time
from pathlib import Path

from tests.e2e.harness import NO_NOTIFY_ENV

#: The painted frame, with the escapes a pty writes. OSC first (window title),
#: then CSI styling: neither carries text, and stripping them makes the screen
#: a plain substring surface.
_OSC = re.compile(r"\x1b\][^\x07]*\x07")
_CSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _decode(chunks: list[bytes]) -> str:
    raw = b"".join(chunks).decode("utf-8", "replace")
    return _CSI.sub("", _OSC.sub("", raw))


#: Deadline for one painted fact, everywhere in this file. MEASURED on this
#: host during the round-2 remediation: the no-provider boot's first paint
#: takes 42-48 s under fleet load, on BOTH the delta head and its pre-delta
#: sibling — the old 60 s ceiling sat inside that noise and failed the file
#: while the product was correct (the same load-sensitivity class as Q2).
#: These are ceilings, not sleeps: ``wait_for`` returns the moment the fact
#: paints, so a generous window costs a healthy run nothing.
_PAINT_DEADLINE = 120.0


class _TuiChild:
    """The real CLI on its own pty: drain, read, send, reap by pid."""

    def __init__(self, root: Path) -> None:
        home = root / "home"
        home.mkdir()
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("CMUX_") and not k.startswith("LOP_")
        }
        env.pop("NO_COLOR", None)
        env["HOME"] = str(home)
        env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / ".local-operator")
        env["TERM"] = "xterm-256color"
        # The same knobs every other e2e host pins, for the same reasons: a
        # driven frame must not repaint a title or raise a notification on the
        # machine's real desktop, and the shimmer timer would turn "is the
        # loop painting" into a question about an animation clock.
        env["LOCAL_OPERATOR_NO_SHIMMER"] = "1"
        env["LOCAL_OPERATOR_NO_TERMINAL_TITLE"] = "1"
        env.update(NO_NOTIFY_ENV)
        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
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
                os.execve(sys.executable, [sys.executable, "-m", "local_operator.cli"], env)
            except BaseException:
                os._exit(127)
        os.close(slave)
        self.master = master
        self.pid = pid
        self.chunks: list[bytes] = []

    def drain(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self.master], [], [], 0.1)
            if not readable:
                continue
            try:
                data = os.read(self.master, 65536)
            except OSError:
                return
            if not data:
                return
            self.chunks.append(data)

    def screen(self) -> str:
        return _decode(self.chunks)

    def wait_for(self, needle: str, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if needle in self.screen():
                return True
            self.drain(0.5)
        return needle in self.screen()

    def send(self, data: str) -> None:
        os.write(self.master, data.encode())

    def reap(self) -> None:
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(40):
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.25)
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_fresh_cli_boot_enters_setup_and_she_answers_at_the_cue(tmp_path: Path) -> None:
    """R28 on the shipped binary: the setup state, her view, and one cue.

    Before the U1 fix this boot painted NO cue anywhere (the factory swallowed
    ``HostingNotConfiguredError`` into a cold viewer), ``/aida`` opened a
    normal conversation with no view, and a typed message was refused with the
    launcher's shared "Settings > Providers" wording — desktop vocabulary on a
    terminal, naming a surface the TUI does not have.
    """
    child = _TuiChild(tmp_path)
    try:
        # 1. The setup state is REACHABLE: the splash carries the cue, and the
        #    band says `setup` rather than a model that does not exist.
        assert child.wait_for("no provider configured", _PAINT_DEADLINE), child.screen()[-2500:]
        assert "setup" in child.screen()

        # 2. `/aida` opens HER view at the same cue.
        child.send("/aida\r")
        assert child.wait_for("chief of staff", _PAINT_DEADLINE), child.screen()[-2500:]

        # 3. A typed message is refused with the shared cue — the message
        #    version says it was not sent — and never with the desktop copy.
        child.send("hello there\r")
        assert child.wait_for("your message was not sent", _PAINT_DEADLINE), child.screen()[-2500:]
        assert "can't reply yet" in child.screen()
        assert "Settings > Providers" not in child.screen()
        # D1: the no-provider block must not promise the first-run greeting on
        # installs whose predicate will never fire.
        assert "introduce herself" not in child.screen()
    finally:
        child.reap()


def test_the_first_login_routes_to_her_and_arms_the_greeting(tmp_path: Path) -> None:
    """R26 on the shipped binary: the setup exit arms ``aida-greeting``.

    The login is the real paste-provider flow (a deepseek key is stored, not
    validated), so this also pins that the rebuild after `/login` runs on the
    shipped path — the seam the U1 defect skipped. The greeting is asserted on
    the wake index and the ledger, which are the two durable facts the fire
    and the once-only guard hang off; the delivery itself is covered by the
    engine tests and QA's live walk.
    """
    child = _TuiChild(tmp_path)
    root = tmp_path / ".local-operator"
    try:
        assert child.wait_for("no provider configured", _PAINT_DEADLINE), child.screen()[-2500:]
        child.send("/login deepseek\r")
        assert child.wait_for("Paste your DeepSeek API key", _PAINT_DEADLINE), child.screen()[
            -2500:
        ]
        child.send("sk-test-not-a-real-key\r")

        state = root / "aida" / "state.json"
        ledger = root / "aida" / "onboarding.json"
        deadline = time.monotonic() + _PAINT_DEADLINE
        armed = False
        stamped = False
        while time.monotonic() < deadline:
            child.drain(0.5)
            if not state.exists():
                continue
            session_id = json.loads(state.read_text()).get("session_id")
            entry = root / "wakes" / f"{session_id}.json"
            if entry.exists() and "aida-greeting" in entry.read_text():
                armed = True
            # The stamp trails the row by ~2 ms measured, but the writer is the
            # app's own arm sequence, not this test's, so BOTH facts are polled
            # inside the deadline (QA round 2, Q2: a single-shot read here
            # slipped inside that window under fleet load while the product was
            # correct — row armed, stamp not yet written).
            if ledger.exists():
                try:
                    if json.loads(ledger.read_text()).get("greeted_at") is not None:
                        stamped = True
                except ValueError:
                    pass  # a torn read mid-write; the next lap re-reads
            if armed and stamped:
                break
        assert armed, child.screen()[-2500:]
        assert stamped, child.screen()[-2500:]
    finally:
        child.reap()
