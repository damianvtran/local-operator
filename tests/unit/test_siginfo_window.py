"""The sensor's install/restore windows and burst delivery: a signal must never wedge the process.

WHY THIS FILE EXISTS (review round 3 on #3215, R-1/R-2, QA Q-series). ``siginfo``'s
sensor is a ``SA_SIGINFO`` action whose last act is ``_restore`` + ``_redeliver``. If it
ever runs with no saved action to restore it re-raises the signal at itself forever: a
process spinning at ~91% CPU that ignores even a sanctioned stop until SIGKILL. So the
invariant is pinned here: **a sensor is never live without its saved action** (``arm``
publishes before it installs, ``disarm`` restores before it forgets).

THE SENSOR IS ONE-SHOT PER PROCESS and nothing re-arms it. An earlier revision of the
gate re-armed it after every refusal; QA round 4 measured that at 1-3 ms bursts over
12,507 signals: 12 receipts naming the runtime's OWN pid, 3 hangs (after which a second
sanctioned SIGTERM also failed) and 5 SIGSEGVs inside the ctypes callback. The burst
cell below pins the one-shot contract against exactly that rig.

EVERY CELL RUNS ITS SUBJECT IN A CHILD PROCESS with a hard timeout, because the failure
being tested is "never returns"; a hang must be a red cell, not a stuck suite. The
children are this test's own throwaways (synthetic, no session store, no live fleet),
killed by exact pid in ``finally``. Darwin only: the sensor is armed nowhere else.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="siginfo arms its sensor on Darwin only"
)

_ENV_STRIP = ("CMUX_", "LOP_")


def _child_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith(_ENV_STRIP)}


def _run(script: str, *args: str, timeout: float = 25.0) -> tuple[str, str | None]:
    """Run ``script`` in a child. ``(stdout, None)`` on exit, ``(stdout, "hung")`` on timeout."""
    child = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_child_env(),
        start_new_session=True,
    )
    try:
        out, _ = child.communicate(timeout=timeout)
        return out, None
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        out, _ = child.communicate()
        return out, "hung"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


# A signal delivered at the exact instant the sensor becomes the live action: the hook
# wraps libc's sigaction and self-signals right after the INSTALL syscall returns.
_HOOK = """
import os, signal, sys
from local_operator import siginfo

ran = []
signal.signal(signal.SIGUSR2, lambda *_: ran.append("python handler ran"))
real = siginfo._libc()

class Hooked:
    def __init__(self):
        self.fire = False
    def sigaction(self, sig, new, old):
        rc = real.sigaction(sig, new, old)
        if self.fire and new is not None:
            self.fire = False
            os.kill(os.getpid(), sig)   # lands the instant the new action is live
        return rc
    def __getattr__(self, name):
        return getattr(real, name)

hook = Hooked()
siginfo._LIBC, siginfo._LIBC_TRIED = hook, True
SIG = int(signal.SIGUSR2)
"""


def test_a_signal_landing_right_after_the_sensor_goes_live_still_returns() -> None:
    """Deliver a signal in the exact gap after ``sigaction(sensor)`` returns."""
    body = _HOOK + """
hook.fire = True
# ``arm`` MAY answer False here: the signal ran the sensor, which restored the action,
# and arm's own verify step then (correctly) refuses to call that a live sensor. What
# is pinned is that the process RETURNS and the product's handler ran.
siginfo.arm(SIG)
import time
time.sleep(0.2)
print("returned", ran, siginfo.take(SIG))
"""
    out, verdict = _run(body)
    assert verdict is None, f"the process wedged on a signal in the arm window:\n{out}"
    assert "returned" in out and "python handler ran" in out, out


def test_a_signal_landing_just_before_disarm_restores_still_returns() -> None:
    """The mirror window: ``disarm`` must restore BEFORE it forgets the saved action.

    The hook fires the signal immediately BEFORE the restoring ``sigaction`` runs, while
    the sensor is still the live action. Forgetting first (the old order) leaves the
    sensor with nothing to restore and it re-raises at the process forever.
    """
    body = _HOOK + """
class Pre(Hooked):
    def sigaction(self, sig, new, old):
        if self.fire and new is not None:
            self.fire = False
            os.kill(os.getpid(), sig)   # the sensor is STILL live: it runs here
        return real.sigaction(sig, new, old)

pre = Pre()
siginfo._LIBC, siginfo._LIBC_TRIED = pre, True
assert siginfo.arm(SIG)
pre.fire = True
siginfo.disarm(SIG)
import time
time.sleep(0.2)
print("returned", ran)
"""
    out, verdict = _run(body)
    assert verdict is None, f"the process wedged on a signal in the disarm window:\n{out}"
    assert "returned" in out and "python handler ran" in out, out


def test_the_harness_detects_a_wedge_when_the_saved_state_is_dropped() -> None:
    """NEGATIVE CONTROL: with ``_SAVED`` dropped the sensor re-raises forever, and this
    harness reports it — so the cells above cannot pass because the harness is blind."""
    body = _HOOK + """
assert siginfo.arm(SIG)
siginfo._SAVED.pop(SIG)          # the un-published state the old order exposed
os.kill(os.getpid(), SIG)
print("returned")
"""
    out, verdict = _run(body, timeout=6.0)
    assert verdict == "hung", f"the control did not wedge, so the harness proves nothing:\n{out}"


_BURST = """
import asyncio, json, os, signal, sys
from local_operator import siginfo
from local_operator.session.runtime import signal_receipt

flag, out, parent = sys.argv[1], sys.argv[2], int(sys.argv[3])
seen = {"named": 0, "none": 0, "self": 0, "wrong": 0, "arrivals": 0, "sanctioned": 0}

async def main():
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def on_term():
        seen["arrivals"] += 1
        # The router's own first act: claim the capture. NOTHING re-arms the sensor.
        sender = signal_receipt.sender_facts(int(signal.SIGTERM))
        if sender.get("state") == "known":
            seen["named"] += 1
            if sender["pid"] == os.getpid():
                seen["self"] += 1
            elif sender["pid"] != parent:
                seen["wrong"] += 1
        else:
            seen["none"] += 1
        if os.path.exists(flag):            # a sanctioned stop: proceed, exactly once
            seen["sanctioned"] += 1
            done.set()

    loop.add_signal_handler(signal.SIGTERM, on_term)
    assert siginfo.arm(int(signal.SIGTERM))
    print("ready", flush=True)
    await done.wait()

asyncio.run(main())
json.dump(seen, open(out, "w"))
"""


def test_a_burst_of_refused_signals_neither_hangs_nor_crashes_nor_self_attributes(
    tmp_path: Path,
) -> None:
    """>=1000 SIGTERMs at 1-3 ms spacing, then a sanctioned stop, against the one-shot sensor.

    Contract: no hang (the child exits 0 on the sanctioned stop), no crash (exit 0, not
    -SIGSEGV), no self-attribution (a NAMED receipt never carries the child's own pid),
    and the sender is named at most ONCE — on the first delivery — with every later
    arrival ``no-siginfo``. The numbers are printed so a run is reviewable.
    """
    flag, out = tmp_path / "sanctioned", tmp_path / "seen.json"
    import random
    import time

    child = subprocess.Popen(
        [sys.executable, "-c", _BURST, str(flag), str(out), str(os.getpid())],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_child_env(),
        start_new_session=True,
    )
    sent = 0
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        rng = random.Random(3215)
        while sent < 1000:
            os.kill(child.pid, signal.SIGTERM)
            sent += 1
            time.sleep(rng.uniform(0.001, 0.003))
        flag.write_text("stage")  # the ladder's marker, in miniature
        # ONE sanctioned SIGTERM, then wait. Re-sending on a timer raced the child's
        # exit: a late signal landing after ``loop.close()`` restored SIG_DFL killed it
        # (-15) under heavy CPU load. The burst's last signals are still being drained
        # when the flag appears, so the sanctioned stop is any arrival that sees the
        # flag; if the burst was fully drained already, this one signal is that arrival.
        time.sleep(0.2)
        if child.poll() is None:
            os.kill(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            pass
        code = child.poll()
        assert code is not None, "the sanctioned stop did not end the child (hung or deaf)"
        assert code == 0, f"the child crashed (exit {code}; -11 is SIGSEGV)"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    seen = json.loads(out.read_text())
    print(f"burst numbers: sent={sent} exit={code} {seen}")
    assert sent >= 1000
    assert seen["sanctioned"] == 1, seen
    assert seen["self"] == 0, f"a receipt named the runtime's own pid: {seen}"
    assert seen["wrong"] == 0, f"a receipt named a pid that never sent: {seen}"
    assert seen["named"] <= 1, f"the one-shot sensor named more than the first signal: {seen}"
    assert seen["named"] + seen["none"] == seen["arrivals"]
    assert seen["arrivals"] >= 100, f"too few deliveries reached the handler: {seen}"
