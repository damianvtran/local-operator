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

from local_operator.providers.login_catalog import RECOMMENDED_LOGIN_COMMAND
from local_operator.providers.registry import get_provider_definition
from local_operator.tui.app import AIDA_NO_PROVIDER_CUE
from tests.e2e.harness import NO_NOTIFY_ENV

#: The setup cue is asserted THROUGH the shipped constant, never as a literal
#: (CI round 1 of this file: the needles here still spelled the copy this PR
#: had already replaced — `no provider configured` — so both tests spent their
#: whole deadline matching a sentence the product no longer paints, and went
#: red while the frames were correct). A literal that duplicates shipped copy
#: is a second copy of it: the next copy change silently strands the test again,
#: which is exactly what happened once. Everything asserted as TEXT below comes
#: from the module that paints it.
_SETUP_CUE = AIDA_NO_PROVIDER_CUE
_LOGIN_COMMAND = RECOMMENDED_LOGIN_COMMAND

#: ``KeyPromptBlock`` titles itself ``f"Paste your {provider_label} API key"``
#: (``tui/widgets/key_prompt.py``), and ``_login_callbacks`` passes the
#: definition's own ``name`` as that label. Derived rather than spelled, so a
#: relabelled provider moves the needle with it.
_DEEPSEEK = get_provider_definition("deepseek")
# Narrowed rather than chained (the same shape the callbacks fix uses): the
# lookup is Optional, and a provider that vanished from the registry should say
# so at collection instead of raising an AttributeError out of a needle.
assert _DEEPSEEK is not None, "the deepseek definition is the login leg's provider"
_DEEPSEEK_LABEL = str(_DEEPSEEK.name)
_PASTE_KEY_NEEDLE = f"Paste your {_DEEPSEEK_LABEL} API key"

#: Fragments of composed sentences that have no exported constant: the refusal
#: (`tui/app.py`, ``… can't reply yet — your message was not sent: {cue}``) and
#: her view's opening (`… — your chief of staff. She can't reply yet: {cue}`).
#: The CUE half of both is the constant above; these are the only literals left
#: and each names the sentence it comes from.
_REFUSAL_NEEDLE = "your message was not sent"
_AIDA_VIEW_NEEDLE = "chief of staff"

#: Desktop vocabulary that must NEVER appear on this surface (it names a pane
#: the TUI does not have) — a real string in ``session/runtime/launch.py``, so
#: its absence here is a discrimination rather than a word nobody uses.
_DESKTOP_COPY = "Settings > Providers"

#: The painted frame, with the escapes a pty writes. OSC first (window title),
#: then CSI styling: neither carries text, and stripping them makes the screen
#: a plain substring surface.
_OSC = re.compile(r"\x1b\][^\x07]*\x07")
_CSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _decode(chunks: list[bytes]) -> str:
    raw = b"".join(chunks).decode("utf-8", "replace")
    return _CSI.sub("", _OSC.sub("", raw))


#: How long a gone child must STAY gone before the login poll calls it a
#: failure. Covers the rebuild handover: a process replaced mid-flow leaves a
#: gap, and reading that gap as a crash would turn a working run red.
_CHILD_GONE_GRACE = 5.0

#: Grace between ``SIGTERM`` and ``SIGKILL`` in ``_TuiChild.reap``. Measured:
#: the child ignores the polite signal for at least 10 s, so this only bounds
#: how long a doomed process is left running (see the method's docstring).
_REAP_GRACE = 1.0

#: Deadline for one painted fact, everywhere in this file — a WEDGE DETECTOR,
#: not a paint budget. ``wait_for`` returns the moment the fact paints, so the
#: number here never costs a healthy run anything; its only job is to fail
#: loudly instead of hanging if the app stops painting.
#:
#: MEASURED on this host (2026-10-08, fleet load 11-14, three runs, isolated
#: HOME): the no-provider boot's FIRST paint — the cue on the splash — took
#: 3.67 / 3.67 / 3.69 s from spawn, and every later fact in this file painted
#: in 0.26-0.32 s. The whole first screen was up in 4.55 s. A fresh install
#: paints in a few seconds; nothing here needs tens of them.
#:
#: The figure this replaced said 42-48 s "under fleet load" and raised the
#: ceiling from 60 s to 120 s on that basis. It cannot have been a paint time:
#: it was recorded while the needles below still spelled copy this PR had
#: already removed (``no provider configured``), so the wait it timed never
#: matched and every run burned whatever ceiling it was given — which is also
#: why the file went red on CI after passing locally. The ceiling stays
#: generous because a loaded CI shard is slower than this host and a wedge is
#: what it guards; the needles, not the ceiling, are what had to change.
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
        #: Wait status from ``alive()``'s ``WNOHANG`` reap, or ``None`` while the
        #: child is running (or when it was never ours to wait on).
        self.exit_status: int | None = None

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

    def alive(self) -> bool:
        """Whether the CLI is still running — by pid, and reaping it if it is not.

        A dead child is the failure mode this file actually met once (see
        ``_PAINT_DEADLINE``): the poll spun its whole budget against a process
        that was gone, so a crash read as a 132 s timeout and the assertion
        printed a cleared screen with nothing to go on. ``WNOHANG`` is what
        distinguishes "gone" from "a zombie nothing has reaped yet".
        """
        try:
            reaped, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return False
        if reaped == self.pid:
            self.exit_status = status
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def exit_note(self) -> str:
        """What is known about a child that is no longer running."""
        if self.exit_status is None:
            return "the pid is gone (it was never ours to wait on)"
        if os.WIFSIGNALED(self.exit_status):
            return f"killed by signal {os.WTERMSIG(self.exit_status)}"
        return f"exited with status {os.WEXITSTATUS(self.exit_status)}"

    def reap(self) -> None:
        """End the child by EXACT pid: ``SIGTERM`` for a grace, then ``SIGKILL``.

        The grace is short because the polite signal is measurably ignored
        here: measured on this host (2026-10-08), the child survives the full
        40 x 0.25 s of the old 10 s window and died only at the ``SIGKILL``
        after it — so every test in the file paid 10.1 s of teardown for a
        signal the process was never going to honour, and the two-test file
        cost ~20 s more than its own work did. Nothing is lost by killing
        sooner: the durability this file depends on (the wake index entry, the
        onboarding ledger) is asserted BEFORE the teardown runs, and the pid is
        still the exact one this instance forked.
        """
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + _REAP_GRACE
        while time.monotonic() < deadline:
            try:
                os.kill(self.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        # A killed pid can still be a zombie until the parent reaps it, so
        # ``kill(pid, 0)`` alone would read as alive here; the wait below is
        # for the signal to be DELIVERED, which is what keeps the next test's
        # pty from inheriting a live reader.
        for _ in range(20):
            try:
                reaped, _status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return
            if reaped == self.pid:
                return
            time.sleep(0.05)


def test_a_fresh_cli_boot_enters_setup_and_she_answers_at_the_cue(tmp_path: Path) -> None:
    """R28 on the shipped binary: the setup state, her view, and one cue.

    Before the U1 fix this boot painted NO cue anywhere (the factory swallowed
    ``HostingNotConfiguredError`` into a cold viewer), ``/aida`` opened a
    normal conversation with no view, and a typed message was refused with the
    launcher's shared "Settings > Providers" wording — desktop vocabulary on a
    terminal, naming a surface the TUI does not have. The cue asserted here is
    ``AIDA_NO_PROVIDER_CUE`` (``Connect an AI account first: type
    /login radient``); the sentence it replaced is not in the product at all.
    """
    child = _TuiChild(tmp_path)
    try:
        # 1. The setup state is REACHABLE: the splash carries the cue, and the
        #    band says `setup` rather than a model that does not exist.
        assert child.wait_for(_SETUP_CUE, _PAINT_DEADLINE), child.screen()[-2500:]
        assert _LOGIN_COMMAND in child.screen()
        assert "setup" in child.screen()

        # 2. `/aida` opens HER view at the same cue.
        child.send("/aida\r")
        assert child.wait_for(_AIDA_VIEW_NEEDLE, _PAINT_DEADLINE), child.screen()[-2500:]

        # 3. A typed message is refused with the shared cue — the message
        #    version says it was not sent — and never with the desktop copy.
        child.send("hello there\r")
        assert child.wait_for(_REFUSAL_NEEDLE, _PAINT_DEADLINE), child.screen()[-2500:]
        assert "can't reply yet" in child.screen()
        assert _DESKTOP_COPY not in child.screen()
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
        assert child.wait_for(_SETUP_CUE, _PAINT_DEADLINE), child.screen()[-2500:]
        child.send("/login deepseek\r")
        assert child.wait_for(_PASTE_KEY_NEEDLE, _PAINT_DEADLINE), child.screen()[-2500:]
        child.send("sk-test-not-a-real-key\r")

        state = root / "aida" / "state.json"
        ledger = root / "aida" / "onboarding.json"
        deadline = time.monotonic() + _PAINT_DEADLINE
        armed = False
        stamped = False
        gone_since: float | None = None
        while time.monotonic() < deadline:
            child.drain(0.5)
            if state.exists():
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
            # A GONE CHILD IS CHECKED AFTER the facts, never before: everything
            # the arm sequence writes is durable before the process ends, so a
            # clean exit between the write and this lap must still pass. What
            # this catches is the other shape — the login child dying with the
            # row unwritten — which used to burn the whole deadline and report a
            # cleared screen (measured once in ~28 runs on this host). The grace
            # is for a HANDOVER: a rebuild that replaces the process must be
            # given its moment before "gone" is read as "failed".
            if not child.alive():
                gone_since = gone_since or time.monotonic()
                if time.monotonic() - gone_since >= _CHILD_GONE_GRACE:
                    raise AssertionError(
                        f"the login child is gone ({child.exit_note()}) and the wake index "
                        f"still has no aida-greeting row\n{child.screen()[-2000:]}"
                    )
        assert armed, child.screen()[-2500:]
        assert stamped, child.screen()[-2500:]
    finally:
        child.reap()
