"""``local_operator.siginfo``: the one-shot SA_SIGINFO sender sensor.

TWO KINDS OF TEST HERE, and the split is deliberate.

The LAYOUT and DEGRADATION cells run anywhere: they pin the word offsets the
module reads out of a ``siginfo_t``, and they pin that every unsupported path
answers "no facts" rather than a guess. The REAL-DELIVERY cells spawn a child
process, have it arm the sensor exactly as the runtime and the exec worker do
(``signal.signal`` first, ``arm`` second), and then really signal it — because
"si_pid is the signaller" is a claim about a kernel interface, and a fabricated
``siginfo_t`` would test this module against its own assumptions. Those cells are
Darwin-only for the reason the module states: it is the only layout this change
could execute against a real kernel.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import local_operator
from local_operator import siginfo

#: The child script. ``{root}`` is the tree to import the product from (the
#: parent's own, so a worktree run tests the worktree rather than an installed
#: copy), ``{directory}`` the conversation directory the receipt is written to,
#: and ``{signal_name}`` the signal under test.
#:
#: IT PRINTS TWO LINES AND EXITS, which is what makes the parent's job bounded:
#: ``READY <pid> <armed>`` once the handler and the sensor are in place, then
#: ``RECORDED <receipt json>`` from inside the signal handler, after the receipt
#: has been written to disk. A child that never gets its signal times out on its
#: own (exit 3) instead of parking a pytest worker forever.
_CHILD = r"""
import json, os, signal, sys, time
from pathlib import Path

sys.path.insert(0, {root!r})
from local_operator import siginfo
from local_operator.session.runtime import signal_receipt

DIRECTORY = Path({directory!r})
SIGNAL = getattr(signal, {signal_name!r})

def handler(number, frame):
    receipt = signal_receipt.record(
        DIRECTORY,
        kind="runtime",
        session_id="siginfo-child",
        pid=os.getpid(),
        started_at=1760000000.0,
        name=SIGNAL.name,
        number=int(SIGNAL),
        in_flight=False,
        action="stop",
    )
    print("RECORDED " + json.dumps(receipt), flush=True)
    os._exit(0)

signal.signal(SIGNAL, handler)
ARMED = bool(siginfo.arm(SIGNAL))
if ARMED and SIGNAL is signal.SIGALRM:
    signal.setitimer(signal.ITIMER_REAL, 0.2)
print("READY %d %s" % (os.getpid(), ARMED), flush=True)
deadline = time.time() + 10
while time.time() < deadline:
    time.sleep(0.02)
os._exit(3)
"""

_DARWIN_ONLY = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the sensor is armed on Darwin only — see siginfo's platform-support note",
)


def _read_line(proc: subprocess.Popen[str], timeout: float = 30.0) -> str:
    """One line from ``proc``, or an assertion if it does not arrive in time.

    Bounded on purpose: a blocking ``readline`` on a child that never printed is
    how a test suite hangs instead of failing, so the read happens on a daemon
    thread that this call stops waiting for.
    """
    stream = proc.stdout
    assert stream is not None, "the child was spawned without a stdout pipe"
    box: list[str] = []
    reader = threading.Thread(target=lambda: box.append(stream.readline()), daemon=True)
    reader.start()
    reader.join(timeout)
    assert box, "the child printed nothing within the bound"
    return box[0].strip()


def _spawn(tmp_path: Path, *, signal_name: str = "SIGTERM") -> subprocess.Popen[str]:
    root = str(Path(local_operator.__file__).resolve().parent.parent)
    script = _CHILD.format(root=root, directory=str(tmp_path), signal_name=signal_name)
    return subprocess.Popen(  # noqa: S603 — the arguments are this file's own literals
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    )


def _receipt_from(line: str) -> dict[str, Any]:
    assert line.startswith("RECORDED "), f"expected the receipt line, got {line!r}"
    return json.loads(line[len("RECORDED ") :])


def test_the_sender_word_offsets_are_pinned_per_platform() -> None:
    """``si_pid``/``si_uid`` sit where each platform's headers put them — no assumptions.

    Darwin's ``struct __siginfo`` declares the pair immediately after
    ``si_signo``/``si_errno``/``si_code`` (words 3 and 4). Linux's ``asm-generic``
    layout — what x86_64 and aarch64 use — inserts ``int __pad0`` to align the
    ``_sifields`` union, so the same pair is words 4 and 5. A single shared offset
    would be wrong on one of them, and wrong in the direction that prints a
    plausible uid for somebody else's signal.
    """
    assert siginfo._LAYOUTS["darwin"] == {"signo": 0, "errno": 1, "code": 2, "pid": 3, "uid": 4}
    assert siginfo._LAYOUTS["linux"] == {"signo": 0, "errno": 1, "code": 2, "pid": 4, "uid": 5}
    assert "linux" not in siginfo._ARMED_PLATFORMS, (
        "Linux is written down but not armed: its layout was not executed on the host "
        "this change was developed on, and a wrong pid is worse than 'unavailable'"
    )


def test_the_darwin_sigaction_struct_matches_the_header() -> None:
    """The struct installed over the product's handler is ``sys/signal.h``'s, field for field.

    A mistake here is not cosmetic: it installs an action with a corrupted mask or
    flags, so the sizes and offsets are asserted rather than trusted.
    """
    import ctypes

    struct = siginfo._SigActionDarwin
    word = ctypes.sizeof(ctypes.c_void_p)
    assert ctypes.sizeof(struct) == word + 8
    assert struct.handler.offset == 0
    assert struct.mask.offset == word
    assert struct.flags.offset == word + 4


def test_a_platform_this_module_cannot_verify_arms_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Cannot capture" is a supported answer, and it must not raise on any platform.

    Windows is the interesting case (no ``sigaction``, no ``SA_SIGINFO``), and it
    is also the one whose whole runtime boot must not notice this module exists.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    assert siginfo.supported() is False
    assert siginfo.arm(signal.SIGTERM) is False
    assert siginfo.armed(signal.SIGTERM) is False
    assert siginfo.take(signal.SIGTERM) is None


def test_an_unarmed_signal_captures_nothing() -> None:
    """With nothing armed, every reader answers "no capture" — never a placeholder pid."""
    assert siginfo.take(signal.SIGUSR2) is None
    assert siginfo.armed(signal.SIGUSR2) is False
    assert siginfo.disarm(signal.SIGUSR2) is False


def _header_siginfo(signo: int, pid: int, uid: int, code: int = 0) -> Any:
    """A ``siginfo_t`` laid out by the PLATFORM'S HEADER, independent of ``_LAYOUTS``.

    The fixture must not be built from the table under test: that would make the
    decode agree with itself whatever the offsets. These are the two real headers,
    spelled as ``ctypes`` structures:

    * Darwin (``sys/signal.h``): ``si_signo, si_errno, si_code, si_pid, si_uid``.
    * Linux, 64-bit (glibc ``bits/types/siginfo_t.h``): the same three ints, then
      ``int __pad0`` (explicit padding, ``__WORDSIZE == 64``) so the ``_sifields``
      union is 8-aligned, then ``_kill.si_pid, _kill.si_uid`` — bytes 16 and 20.

    The first version of this test used a Darwin-shaped buffer on every platform
    and was red on the Linux shard (decode read pid 501 / uid 0 from a buffer with
    pid in word 3): the decoder's Linux offsets were right and the fixture was not.
    """
    import ctypes

    fields: list[tuple[str, Any]] = [
        ("si_signo", ctypes.c_int32),
        ("si_errno", ctypes.c_int32),
        ("si_code", ctypes.c_int32),
    ]
    if sys.platform == "linux":
        fields.append(("__pad0", ctypes.c_int32))
    fields += [
        ("si_pid", ctypes.c_int32),
        ("si_uid", ctypes.c_int32),
        ("rest", ctypes.c_int32 * 26),
    ]
    cls = type("SigInfo", (ctypes.Structure,), {"_fields_": fields})
    info = cls()
    info.si_signo, info.si_code, info.si_pid, info.si_uid = signo, code, pid, uid
    return info


@pytest.mark.skipif(
    sys.platform not in ("darwin", "linux"), reason="only these two layouts are written down"
)
def test_a_siginfo_that_disagrees_with_its_own_delivery_is_refused() -> None:
    """``si_signo`` is the structural check: a buffer that fails it is not a ``siginfo_t``.

    This is the guard that makes a wrong offset degrade to "no facts" instead of a
    confident wrong pid: word 0 is the one field every layout agrees on, so if it
    does not name the signal being delivered, nothing else read from that pointer
    is reported. The fixture is the host platform's REAL header layout (see
    :func:`_header_siginfo`), so the byte offsets in ``_LAYOUTS`` are checked
    against the header and not against themselves.
    """
    import ctypes

    info = _header_siginfo(int(signal.SIGUSR1), pid=1234, uid=501, code=7)
    pointer = ctypes.addressof(info)
    ok = siginfo._read(pointer, int(signal.SIGUSR1))
    assert ok == {"signo": int(signal.SIGUSR1), "pid": 1234, "uid": 501, "code": 7}, ok
    assert siginfo._read(pointer, int(signal.SIGUSR2)) is None
    assert siginfo._read(None, int(signal.SIGUSR1)) is None


def test_the_linux_decode_reads_a_linux_shaped_buffer_on_any_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_read`` with the Linux table decodes a glibc-shaped buffer, whatever the host.

    The offsets-only test above proves the numbers; this one runs the DECODER with
    them, so a Linux CI shard and this macOS host exercise the same code on the
    same bytes. ``sys.platform`` is patched only for the lookup inside ``_read``.
    """
    import ctypes

    class LinuxSigInfo64(ctypes.Structure):
        _fields_ = [
            ("si_signo", ctypes.c_int32),
            ("si_errno", ctypes.c_int32),
            ("si_code", ctypes.c_int32),
            ("__pad0", ctypes.c_int32),
            ("si_pid", ctypes.c_int32),
            ("si_uid", ctypes.c_int32),
            ("rest", ctypes.c_int32 * 26),
        ]

    info = LinuxSigInfo64(si_signo=int(signal.SIGTERM), si_code=0, si_pid=4242, si_uid=1000)
    monkeypatch.setattr(sys, "platform", "linux")
    got = siginfo._read(ctypes.addressof(info), int(signal.SIGTERM))
    assert got == {"signo": int(signal.SIGTERM), "pid": 4242, "uid": 1000, "code": 0}, got
    # And the Darwin-shaped read of the SAME bytes is the wrong answer this guards against.
    monkeypatch.setattr(sys, "platform", "darwin")
    wrong = siginfo._read(ctypes.addressof(info), int(signal.SIGTERM))
    assert wrong is not None and wrong["pid"] != 4242


def test_the_linux_sender_offsets_are_the_64_bit_header_offsets() -> None:
    """Linux ``si_pid``/``si_uid`` sit at bytes 16 and 20 (words 4 and 5), on any host.

    Derived from glibc's ``siginfo_t`` (three ints, ``int __pad0`` under
    ``__WORDSIZE == 64``, then the ``_sifields`` union whose ``_kill`` member
    starts ``si_pid, si_uid``). Checked here with a structure built from that
    declaration so it runs on every platform. 32-bit Linux has no ``__pad0`` and is
    NOT covered by ``_LAYOUTS``; Linux is unarmed, so nothing reads this table in
    production, and the table is what a future arming change must prove on a real
    kernel first.
    """
    import ctypes

    class LinuxSigInfo64(ctypes.Structure):
        _fields_ = [
            ("si_signo", ctypes.c_int32),
            ("si_errno", ctypes.c_int32),
            ("si_code", ctypes.c_int32),
            ("__pad0", ctypes.c_int32),
            ("si_pid", ctypes.c_int32),
            ("si_uid", ctypes.c_int32),
        ]

    layout = siginfo._LAYOUTS["linux"]
    assert layout["pid"] * 4 == LinuxSigInfo64.si_pid.offset == 16
    assert layout["uid"] * 4 == LinuxSigInfo64.si_uid.offset == 20


@_DARWIN_ONLY
def test_a_real_sigterm_names_the_process_that_sent_it(tmp_path: Path) -> None:
    """THE HEADLINE CELL: a real SIGTERM, a known signaller, and a receipt that names it.

    The child arms the sensor the way both production callers do (Python-level
    handler first — ``arm`` wraps whatever action is there), then this test sends
    it SIGTERM. Three things must be true at once, or the change has not landed:
    the child's own handler still ran (the sensor handed the signal back rather
    than swallowing it), the receipt names THIS process as the sender, and the
    durable artifact on disk says the same thing the returned dict does.
    """
    proc = _spawn(tmp_path)
    try:
        ready = _read_line(proc)
        assert ready.startswith("READY "), ready
        _, pid_text, armed_text = ready.split()
        assert int(pid_text) == proc.pid, (pid_text, proc.pid)
        assert armed_text == "True", "the sensor did not arm on this host"

        os.kill(proc.pid, signal.SIGTERM)
        receipt = _receipt_from(_read_line(proc))
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:  # pragma: no cover — only on a failed assertion above
            proc.kill()
            proc.wait(timeout=15)

    assert receipt is not None
    entry = receipt["signals"][-1]
    assert entry["name"] == "SIGTERM" and entry["sanction"] == "none"
    sender = entry["sender"]
    assert sender["state"] == "known", sender
    assert sender["pid"] == os.getpid(), (sender, os.getpid())
    assert sender["uid"] == os.getuid(), sender
    assert isinstance(sender["code"], int), sender

    on_disk = json.loads((tmp_path / "runtime-signal.json").read_text())
    assert on_disk["signals"][-1]["sender"] == sender


@_DARWIN_ONLY
def test_without_the_capture_the_receipt_keeps_its_old_shape(tmp_path: Path) -> None:
    """An unarmed process must write EXACTLY the receipt it wrote before this existed.

    The unavailable shape is a persisted contract: readers, renderers and the
    pinned sentences downstream all key off it, so "the sensor is not there" has to
    be indistinguishable from "the sensor had not been written yet".
    """
    root = str(Path(local_operator.__file__).resolve().parent.parent)
    script = _CHILD.format(root=root, directory=str(tmp_path), signal_name="SIGTERM")
    # Same child, minus the arming call: the process a build without this module
    # would run.
    script = script.replace("ARMED = bool(siginfo.arm(SIGNAL))", "ARMED = False")
    proc = subprocess.Popen(  # noqa: S603 — this file's own literals
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_line(proc)
        assert ready.split()[-1] == "False", ready
        os.kill(proc.pid, signal.SIGTERM)
        receipt = _receipt_from(_read_line(proc))
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=15)

    assert receipt["signals"][-1]["sender"] == {
        "state": "unavailable",
        "reason": "no-siginfo",
        "could_be": "same-uid-or-root",
    }


@_DARWIN_ONLY
def test_a_kernel_delivered_signal_reports_no_sending_process(tmp_path: Path) -> None:
    """``si_pid == 0`` is the KERNEL talking, and the receipt says so instead of "pid 0".

    A timer is a real kernel-originated delivery (measured on this host: a
    ``setitimer`` SIGALRM arrives with ``si_pid 0, si_uid 0, si_code 0``), which is
    why this cell uses one rather than hand-building a ``siginfo_t``: the claim is
    about what the kernel fills in, so the kernel has to fill it in.
    """
    proc = _spawn(tmp_path, signal_name="SIGALRM")
    try:
        ready = _read_line(proc)
        assert ready.split()[-1] == "True", ready
        receipt = _receipt_from(_read_line(proc))
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=15)

    sender = receipt["signals"][-1]["sender"]
    assert receipt["signals"][-1]["name"] == "SIGALRM"
    assert sender == {"state": "unavailable", "reason": "kernel-originated"}, sender


@_DARWIN_ONLY
def test_the_sensor_is_spent_after_one_delivery(tmp_path: Path) -> None:
    """ONE CAPTURE, ONE CLAIM: a repeat delivery must not re-use the first sender.

    Two independent facts are asserted here, and the second is the one that would
    otherwise ship a lie.

    The SENSOR IS SPENT: it restores the saved action before re-raising, so a
    second delivery runs the product's handler with no ctypes frame — that is what
    bounds the added risk to one signal per process.

    And because of that, a repeat has NO capture of its own. The capture that
    exists describes only the first delivery, so "the second receipt names the
    first sender" would be a confidently wrong attribution — the exact outcome
    worse than ``unavailable``. A repeat sweep whose two signals came from
    different pids would look like one sender. Hence :func:`siginfo.take`
    consumes, and the second receipt says ``no-siginfo`` about itself, which is
    true.
    """
    root = str(Path(local_operator.__file__).resolve().parent.parent)
    script = _CHILD.format(root=root, directory=str(tmp_path), signal_name="SIGTERM")
    script = script.replace(
        '    print("RECORDED " + json.dumps(receipt), flush=True)\n    os._exit(0)',
        '    print("RECORDED " + json.dumps(receipt), flush=True)\n'
        "    if os.environ.get('CHILD_SECOND') and receipt['count'] < 2:\n"
        "        return\n"
        "    os._exit(0)",
    )
    env = dict(os.environ, CHILD_SECOND="1")
    proc = subprocess.Popen(  # noqa: S603 — this file's own literals
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert _read_line(proc).split()[-1] == "True"
        os.kill(proc.pid, signal.SIGTERM)
        first = _receipt_from(_read_line(proc))
        time.sleep(0.3)
        os.kill(proc.pid, signal.SIGTERM)
        second = _receipt_from(_read_line(proc))
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:  # pragma: no cover
            proc.kill()
            proc.wait(timeout=15)

    assert first["signals"][-1]["sender"]["pid"] == os.getpid()
    assert second["count"] == 2
    assert second["signals"][-1]["sender"] == {
        "state": "unavailable",
        "reason": "no-siginfo",
        "could_be": "same-uid-or-root",
    }, "a repeat delivery owns no capture, so it names nobody"


#: The RUNTIME's install sequence, in the order ``session/runtime/process.py``
#: uses it: the loop takes the handler through the shared platform helper, and
#: only then is the sensor armed over it. Mirrored here rather than imported from
#: ``process.amain`` (which needs a whole session, provider and socket) because
#: the part under test is exactly those two calls and their order — a real
#: asyncio loop, the real helper, a real SIGTERM, and the receipt written by the
#: loop callback that ``_on_signal`` would be.
_RUNTIME_CHILD = r"""
import asyncio, json, os, signal, sys
from pathlib import Path

sys.path.insert(0, {root!r})
from local_operator import siginfo
from local_operator.procstate import install_loop_signal_handlers
from local_operator.session.runtime import signal_receipt

DIRECTORY = Path({directory!r})

def on_signal():
    receipt = signal_receipt.record(
        DIRECTORY,
        kind="runtime",
        session_id="siginfo-runtime",
        pid=os.getpid(),
        started_at=1760000000.0,
        name="SIGTERM",
        number=int(signal.SIGTERM),
        in_flight=False,
        action="stop",
    )
    print("RECORDED " + json.dumps(receipt), flush=True)
    os._exit(0)

async def main():
    loop = asyncio.get_running_loop()
    install_loop_signal_handlers(loop, {{signal.SIGTERM: on_signal}})
    siginfo.arm(int(signal.SIGTERM))
    print("READY %d %s" % (os.getpid(), siginfo.armed(int(signal.SIGTERM))), flush=True)
    await asyncio.Event().wait()

asyncio.run(main())
"""


@_DARWIN_ONLY
def test_the_runtimes_own_install_order_names_the_sender(tmp_path: Path) -> None:
    """The runtime's SEQUENCE (loop handler, then ``arm``), on a real asyncio loop.

    WHY THE ORDER NEEDS ITS OWN CELL: ``loop.add_signal_handler`` reaches
    ``sigaction`` itself, so arming before it would have the loop's own install
    replace the sensor and ``process.amain`` would write ``unavailable`` forever.
    The unit cells above prove the sensor works over ``signal.signal``; this one
    proves it over the *asyncio* path the runtime actually takes, through the same
    ``procstate.install_loop_signal_handlers`` call, with a real SIGTERM from a
    known pid.
    """
    root = str(Path(local_operator.__file__).resolve().parent.parent)
    script = _RUNTIME_CHILD.format(root=root, directory=str(tmp_path))
    proc = subprocess.Popen(  # noqa: S603 — this file's own literals
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_line(proc)
        assert ready.startswith("READY "), ready
        assert ready.split()[-1] == "True", "the sensor did not arm on this host"

        os.kill(proc.pid, signal.SIGTERM)
        receipt = _receipt_from(_read_line(proc))
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:  # pragma: no cover — only on a failed assertion above
            proc.kill()
            proc.wait(timeout=15)

    sender = receipt["signals"][-1]["sender"]
    assert sender["state"] == "known", sender
    assert sender["pid"] == os.getpid(), (sender, os.getpid())
    on_disk = json.loads((tmp_path / "runtime-signal.json").read_text())
    assert on_disk["signals"][-1]["sender"] == sender


@_DARWIN_ONLY
def test_a_signal_the_process_ignores_or_defaults_is_not_armed() -> None:
    """No Python-level action means no receipt, so there is nothing to capture for.

    Two things this prevents. Over ``SIG_IGN`` the sensor would briefly turn a
    deliberate ignore into an action — a behaviour change an instrument must not
    make. Over ``SIG_DFL`` it would install a ctypes handler in a process that
    dies at the signal without writing anything, so the only thing it could add is
    risk. Both answer ``False`` and leave the disposition untouched, which the
    read-back asserts.
    """
    # A signal this process has NO handler for: the kernel's default action is not
    # something a receipt can be written under.
    assert siginfo.arm(signal.SIGWINCH) is False
    assert siginfo.armed(signal.SIGWINCH) is False

    # And the deliberate-ignore case, explicitly.
    previous = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        assert siginfo.arm(signal.SIGUSR1) is False
        assert siginfo.armed(signal.SIGUSR1) is False
        assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN, "the ignore was disturbed"
    finally:
        signal.signal(signal.SIGUSR1, previous)

    # SIG_DFL, spelled out, because the default for SIGUSR1 is terminate.
    previous = signal.signal(signal.SIGUSR1, signal.SIG_DFL)
    try:
        assert siginfo.arm(signal.SIGUSR1) is False
        assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGUSR1, previous)
