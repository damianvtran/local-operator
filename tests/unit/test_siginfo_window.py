"""The sensor's re-install window: a signal landing INSIDE it must never wedge the process.

WHY THIS FILE EXISTS (review round 3 on #3215, R-1/R-2, QA Q-series). ``siginfo``'s
sensor is a ``SA_SIGINFO`` action whose last act is ``_restore`` + ``_redeliver``. If it
ever runs with no saved action to restore it re-raises the signal at itself forever: a
process spinning at ~91% CPU that ignores even a sanctioned stop until SIGKILL. The
sanction gate re-arms the sensor in a RUNNING process after every refusal, which made
that window reachable in production (0.2 ms arrivals hung 4 of 8 runs in review), so
the invariant is pinned here: **a sensor is never live without its saved action**.

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


@pytest.mark.parametrize("path", ["arm", "rearm"])
def test_a_signal_landing_right_after_the_sensor_goes_live_still_returns(path: str) -> None:
    """Deliver a signal in the exact gap after ``sigaction(sensor)`` returns."""
    body = (
        _HOOK
        + (
            """
hook.fire = True
# ``arm`` MAY answer False here: the signal ran the sensor, which restored the action,
# and arm's own verify step then (correctly) refuses to call that a live sensor. What
# is pinned is that the process RETURNS and the product's handler ran.
siginfo.arm(SIG)
"""
            if path == "arm"
            else """
assert siginfo.arm(SIG)
import time
os.kill(os.getpid(), SIG); time.sleep(0.2)        # spend the sensor
assert siginfo.take(SIG) is not None
hook.fire = True
assert siginfo.rearm(SIG)
"""
        )
        + """
import time
time.sleep(0.2)
print("returned", ran, siginfo.take(SIG))
"""
    )
    out, verdict = _run(body)
    assert verdict is None, f"the process wedged on a signal in the {path} window:\n{out}"
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
import asyncio, json, os, random, signal, sys, time
from local_operator import siginfo
from local_operator.session.runtime import signal_receipt

flag, out = sys.argv[1], sys.argv[2]
seen = {"named": 0, "none": 0, "self": 0, "arrivals": 0, "sanctioned": 0}

async def main():
    loop = asyncio.get_running_loop()
    done = asyncio.Event()

    def on_term():
        seen["arrivals"] += 1
        sender = signal_receipt.sender_facts(int(signal.SIGTERM))   # claims the capture
        if sender.get("state") == "known":
            seen["named"] += 1
            if sender["pid"] == os.getpid():
                seen["self"] += 1
        else:
            seen["none"] += 1
        if os.path.exists(flag):            # a sanctioned stop: proceed, exactly once
            seen["sanctioned"] += 1
            done.set()
            return
        signal_receipt.rearm_sender_capture(int(signal.SIGTERM))   # a REFUSAL's re-arm

    loop.add_signal_handler(signal.SIGTERM, on_term)
    assert siginfo.arm(int(signal.SIGTERM))
    print("ready", flush=True)
    await done.wait()

asyncio.run(main())
json.dump(seen, open(out, "w"))
"""


def test_a_burst_of_refused_signals_neither_hangs_nor_self_attributes(tmp_path: Path) -> None:
    """>=200 SIGTERMs at 0.2-3 ms spacing against a runtime that re-arms on each refusal,
    then one sanctioned stop: the child must exit, no receipt may name the child's own
    pid, and the sanctioned stop must proceed."""
    flag, out = tmp_path / "sanctioned", tmp_path / "seen.json"
    import random
    import time

    child = subprocess.Popen(
        [sys.executable, "-c", _BURST, str(flag), str(out)],
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
        while sent < 300:
            os.kill(child.pid, signal.SIGTERM)
            sent += 1
            time.sleep(rng.uniform(0.0002, 0.003))
        flag.write_text("stage")  # the ladder's marker, in miniature
        deadline = time.monotonic() + 20.0
        while child.poll() is None and time.monotonic() < deadline:
            os.kill(child.pid, signal.SIGTERM)  # a coalesced signal can be lost: keep asking
            time.sleep(0.05)
        assert child.poll() == 0, "the sanctioned stop did not end the child (wedged or lost)"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    seen = json.loads(out.read_text())
    print(f"burst numbers: sent={sent} {seen}")
    assert seen["sanctioned"] == 1, seen
    assert seen["self"] == 0, f"a receipt named the runtime's own pid: {seen}"
    assert seen["named"] + seen["none"] == seen["arrivals"]
    assert seen["arrivals"] >= 200, f"too few deliveries reached the handler: {seen}"
