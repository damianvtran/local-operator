"""WHO sent this termination signal — the one fact a signal's TARGET has never
been able to record.

WHY THIS MODULE EXISTS. On 2026-10-30 at 21:57:52 -0400 twelve background exec
workers, at least five session runtimes and two ``lop serve`` daemons all took a
SIGTERM inside 1.33 s (``~/.local-operator/logs/exec-jobs.jsonl``, jobs
ecf65e7945f9, 4ccb31a83f15, c3a13a730695 and nine more). Every one of those
arrival rows says the same thing:

    "sender": {"state": "unavailable", "reason": "no-siginfo",
               "could_be": "same-uid-or-root"}

and there is no ``logs/stop-sweeps.jsonl``, so the stop ladder never ran and
nobody staged a marker. The fleet was killed by somebody, and the product could
not name them — which makes every recurrence unfalsifiable: the same receipt
would be written whether the sender was the desktop app being force-quit, a
terminal, or a stray ``killall``.

WHY IT WAS NOT NAMEABLE. The handler CPython installs for a Python-level signal
handler is a plain ``sa_handler`` (``void (*)(int)``, no ``SA_SIGINFO``), so the
kernel's ``siginfo_t`` — which carries the sender's pid and uid — is discarded
in the C layer before any Python code runs. ``asyncio``'s
``loop.add_signal_handler`` sits on top of exactly that (``signal.signal`` plus a
wakeup fd), so the runtime's own SIGTERM path inherits the gap. The portable
alternatives are absent where it matters: on this host (macOS 27.0.1, CPython
3.12.13) ``signal.sigwaitinfo`` and ``signal.sigtimedwait`` do not exist, and
``sigwait`` returns only the signal number. ``SA_SIGINFO`` is the only
interface that carries a sender, so this module installs one — over the
handler that already owns the signal, without taking the signal away from it.

WHAT IT DOES, PRECISELY. :func:`arm` saves the action currently installed
(whatever CPython, asyncio or the caller put there) and installs a
``SA_SIGINFO`` sensor in its place. When the signal arrives, the sensor:

1. reads ``si_pid``, ``si_uid`` and ``si_code`` out of the ``siginfo_t`` the
   kernel handed it, and only if ``si_signo`` agrees with the signal being
   delivered (a wrong offset must degrade to "no facts", never to a wrong
   pid);
2. restores the action it saved — one syscall, no Python logic;
3. re-raises the signal to its own process, so the platform's OWN delivery runs
   exactly as it would have: CPython's C handler writes its wakeup-fd byte and
   trips the pending signal, ``asyncio`` wakes, and the Python-level handler the
   product installed runs on the loop thread, on the same call stack it always
   did. Measured on this host with a real child process as the signaller
   (``tests/unit/test_siginfo.py``): the loop callback ran and the captured pid
   was the child's.

WHY THAT DANCE RATHER THAN THE WORK IN THE HANDLER. A Python callback entered
from a C signal frame is not async-signal-safe: CPython deliberately defers
Python-level handlers to a pending call for that reason. So the sensor does as
little as it can — read six ints, one ``sigaction``, one ``kill`` — and hands
the signal straight back. SAID PLAINLY: the sensor IS still a Python frame (a
``ctypes`` callback takes the GIL on whichever thread the kernel chose), so the
residual risk is a GIL-internals deadlock if the signal lands inside the
interpreter's own GIL hand-off. That was not reproduced (an 8-thread
GIL-churning probe completed the sensor 40/40 in review), and the mitigation is
the bound, not a proof: it runs ONCE per process, on the first delivery.

AFTER THAT FIRST DELIVERY the sensor is spent (it restored the saved action
before re-raising), so every later delivery takes the ordinary path with no
ctypes frame. The FIRST delivery is not behaviourally identical to this module
being absent, and the exact limit matters:

* UNDER AN ASYNCIO LOOP HANDLER (the runtime and the exec worker, the only
  callers) it is equivalent in the way that counts: CPython's C handler writes
  its wakeup-fd byte on the re-raised signal, the loop wakes from
  ``select``/``kqueue`` and runs the callback.
* UNDER A BARE ``signal.signal`` HANDLER it is NOT guaranteed equivalent. While
  the sensor runs the signal is masked on the interrupted thread, so the
  re-raise is delivered to another thread; CPython trips its pending flag there
  but does not interrupt a main thread parked in a blocking wait. Measured in
  review: with a CPU-busy thread and the main thread in ``Event.wait(5)`` the
  handler did not run within 5 s when armed, and ran when plain. The same
  review measured that ``SA_RESTART`` (set by :func:`arm`) keeps a main thread
  blocked in ``read``/``recv``/``waitpid`` from taking ``EINTR``, delaying the
  handler until the call returns. Neither affects an asyncio loop, and both are
  recorded as deferred rather than papered over.

ONE CAPTURE, ONE CLAIM (:func:`take`). Because the sensor fires once, a repeat
delivery has no capture of its own — and the capture that DOES exist describes
only the delivery that made it. Handing it out repeatedly would let a later
receipt name the first sender for its own signal, which is worse than
"unavailable": a confidently wrong attribution. So the facts are consumed by the
first reader and every later receipt says ``no-siginfo`` about itself, which is
true of that receipt.

WHAT IT CAN STILL NOT SAY, and must not pretend to:

* **SIGKILL, a crash, a power loss**: no handler runs, so nothing here can be
  called. Those stay on the acting party's stop marker or on no evidence at all.
* **A kernel-originated signal** carries ``si_pid == 0`` (measured: a
  ``setitimer`` SIGALRM delivers ``si_pid 0, si_uid 0, si_code 0``). A pid of 0
  means "the kernel", NOT "process zero", and :func:`take` reports it that
  way; the receipt renders such a delivery as unattributed rather than printing
  "pid 0".
* **Which thread** was chosen by the kernel, on a delivery that is already over
  by the time it is read. Not recorded, not claimed.

PLATFORM SUPPORT IS SPLIT ON PURPOSE. ``siginfo_t``'s sender fields sit after
three ints on both Darwin and Linux, but the offsets differ: Darwin has no
padding (``si_pid`` at word 3), while Linux's ``asm-generic`` layout — used by
x86_64 and aarch64 — puts a pad word there (``si_pid`` at word 4). Both layouts
are written down in :data:`_LAYOUTS` and pinned by unit tests, but only
``darwin`` is ARMED, because that is the only layout this change could execute
against a real kernel: a wrong offset would print a confidently false pid into an
incident report, which is worse than today's "unavailable". Armed stays a
frozenset for that reason — adding Linux is a small, tested change, not a
rewrite.

stdlib plus ``ctypes`` only, no new dependencies: this is imported from the
runtime's boot path and from the exec worker, and neither may pay for anything
heavier. EVERY PUBLIC FUNCTION HERE IS TOTAL — it returns a value or a
``None``/``False``, and never raises into a boot path or a signal path.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import signal
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: ``sigaction`` flags (``sys/signal.h``). ``SA_RESETHAND``/``SA_NODEFER`` are
#: cleared from whatever the saved action had: the first would have the kernel
#: reset the disposition to ``SIG_DFL`` behind our back (losing the graceful
#: handler the product installed), the second would let a second signal re-enter
#: the sensor while it is still running.
SA_RESTART = 0x0002
SA_RESETHAND = 0x0004
SA_NODEFER = 0x0010
SA_SIGINFO = 0x0040

#: ``signal.SIG_DFL``/``signal.SIG_IGN`` as ``sigaction`` carries them, kept
#: PRIVATE: a public ``siginfo.SIG_DFL`` would invite callers to reach for this
#: module where the stdlib constant is the right one. These two are the actions no
#: receipt can be written under (see :func:`arm`).
_SIG_DFL = 0
_SIG_IGN = 1

#: Word indices into the ``siginfo_t`` a ``SA_SIGINFO`` handler is given.
#:
#: ``{"signo": 0, "errno": 1, "code": 2}`` is shared. The sender pair is not:
#: Darwin's ``struct __siginfo`` (``sys/signal.h``) declares ``si_pid``/``si_uid``
#: immediately after ``si_code`` (words 3 and 4), while Linux's ``asm-generic``
#: layout — what x86_64 and aarch64 use — inserts ``int __pad0`` to 16-byte align
#: the ``_sifields`` union, so the same pair is words 4 and 5. Reading the wrong
#: pair yields a plausible-looking wrong uid, which is why the platform is a
#: lookup rather than an assumption.
_LAYOUTS: dict[str, dict[str, int]] = {
    # measured on this host: a child's ``kill(2)`` gave si_pid == child pid and
    # si_uid == 501, i.e. the documented offsets are the ones the kernel fills.
    "darwin": {"signo": 0, "errno": 1, "code": 2, "pid": 3, "uid": 4},
    # from the asm-generic layout; NOT executed by this change (see the module
    # docstring on why only Darwin is armed).
    "linux": {"signo": 0, "errno": 1, "code": 2, "pid": 4, "uid": 5},
}

#: The platforms this module will actually arm. See the module docstring: an
#: unexecuted layout must degrade to "unavailable", not to a guessed pid.
_ARMED_PLATFORMS = frozenset({"darwin"})


class _SigActionDarwin(ctypes.Structure):
    """Darwin's ``struct sigaction``: handler union, ``sigset_t`` (a uint32), flags.

    Layout from ``sys/signal.h``; 16 bytes. Getting this wrong would install a
    handler with a corrupted mask/flags, so the fields are spelled out rather
    than approximated with an opaque buffer.
    """

    _fields_ = [
        ("handler", ctypes.c_void_p),
        ("mask", ctypes.c_uint32),
        ("flags", ctypes.c_int),
    ]


#: Per-platform ``sigaction`` structures. A platform missing here cannot be
#: armed — :func:`arm` returns ``False`` and the receipt says "unavailable".
_STRUCTS: dict[str, type[ctypes.Structure]] = {"darwin": _SigActionDarwin}

#: The signature every ``SA_SIGINFO`` handler is called with: signal number,
#: pointer to the ``siginfo_t``, pointer to the ``ucontext_t`` (unused here —
#: nothing about a termination signal needs the interrupted machine state).
_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

#: Serialises arm/disarm, which run once at boot but may race a unit test.
_LOCK = threading.RLock()

#: Captured facts per signal number, written by the sensor and read by
#: :func:`take`. A plain dict assignment is atomic under the GIL, which is
#: what makes a lock unnecessary on the signal path itself.
_FACTS: dict[int, dict[str, int]] = {}

#: The action saved before arming, per signal. Kept after the sensor fires
#: because that is what the sensor restores the disposition FROM.
_SAVED: dict[int, Any] = {}

#: One ctypes callback per armed signal, held here for the lifetime of the
#: process: a garbage-collected callback would leave the kernel calling freed
#: memory on the next delivery.
_CALLBACKS: dict[int, Any] = {}

#: Re-raises a sensor has sent at ITS OWN pid and that have not yet been seen arrive,
#: per signal. A sensor ends by restoring the product's action and re-raising; if a
#: sensor was RE-INSTALLED before that re-raise is delivered (the sanction gate re-arms
#: after every refusal), the re-raise lands on the new sensor and would be captured as
#: an ordinary delivery whose sender is this process or pid 0 — a wrong pid in an
#: incident report, which this module rates worse than "unavailable" (QA round 3
#: measured it 1-2 times per 40-60 bursts). The sensor that finds a pending count
#: consumes it and captures nothing. ``take`` clears the count: a delivery that
#: reached the product's handler has no re-raise left to skip, so a quiet re-armed
#: sensor can never swallow the NEXT real signal's sender.
#:
#: WHAT THIS CANNOT DO, stated because the mask is the reason it is a counter and not a
#: flag: a signal is blocked while its own handler runs, so a same-thread re-raise is
#: delivered only AFTER the sensor returned — an "in flight" marker would already be
#: gone. A real second signal arriving inside the same instant as a re-armed sensor
#: with a pending count is consumed instead of captured: a missing fact, never a wrong one.
_SELF_RERAISES: dict[int, int] = {}

#: ``libc``, resolved once. ``None`` means "no ctypes/libc on this host", which
#: every caller treats as "cannot capture".
_LIBC: Any | None = None
_LIBC_TRIED = False


def _libc() -> Any | None:
    """``libc``, with ``sigaction``'s return type declared, or ``None``. Never raises.

    ``find_library("c")`` is the portable spelling; ``CDLL(None)`` (the running
    program's own symbols, which on Darwin re-export libSystem) is the fallback
    for a host where that lookup fails. Resolved ONCE: a boot path may arm
    several signals, and a failing lookup must not be retried on a signal path.

    ONLY ``restype`` IS DECLARED, deliberately. ``sigaction``'s arguments are
    pointers to a PLATFORM-SPECIFIC ``struct sigaction``, so declaring
    ``argtypes`` for one layout makes another platform's call fail at the ctypes
    boundary (``expected LP_c_void_p instance``) — an instrument that reports
    "cannot capture" on every platform but the one its author tested. Leaving the
    pointer arguments undeclared passes ``byref`` of the layout this platform
    chose, which is what :data:`_STRUCTS` exists for.
    """
    global _LIBC, _LIBC_TRIED
    if _LIBC_TRIED:
        return _LIBC
    _LIBC_TRIED = True
    try:
        name = ctypes.util.find_library("c")
        lib = ctypes.CDLL(name, use_errno=True) if name else ctypes.CDLL(None, use_errno=True)
        lib.sigaction.restype = ctypes.c_int
        _LIBC = lib
    except Exception:  # noqa: BLE001 — no libc means "no capture", never a boot failure
        logger.debug("siginfo: libc/sigaction unavailable; sender capture disabled", exc_info=True)
        _LIBC = None
    return _LIBC


def _struct_type() -> type[ctypes.Structure] | None:
    return _STRUCTS.get(sys.platform)


def supported() -> bool:
    """Whether this platform can capture a sender at all. Never raises."""
    return sys.platform in _ARMED_PLATFORMS and _struct_type() is not None and _libc() is not None


def armed(sig: int) -> bool:
    """Whether a sensor is currently installed for ``sig``. Never raises."""
    try:
        return int(sig) in _CALLBACKS
    except Exception:  # noqa: BLE001
        return False


def _read(info: int | None, expected_signo: int) -> dict[str, int] | None:
    """The sender facts in one ``siginfo_t``, or ``None`` when they are not readable.

    STRICT ON THE ONE FIELD THAT VALIDATES THE POINTER: ``si_signo`` is word 0 on
    every layout here, so a ``siginfo_t`` whose ``si_signo`` disagrees with the
    signal being delivered is not a ``siginfo_t`` — the capture is refused rather
    than reported. The rest is defensive: a ``ValueError``/``OSError`` out of
    ctypes is "no facts", never an exception on a signal path.
    """
    layout = _LAYOUTS.get(sys.platform)
    if layout is None or not info:
        return None
    try:
        words = ctypes.cast(info, ctypes.POINTER(ctypes.c_int32))
        # Bound explicitly: a truncated buffer would otherwise read whatever
        # follows it, and the sender pair is exactly the read that must not be
        # invented.
        values = {key: int(words[index]) for key, index in layout.items()}
    except Exception:  # noqa: BLE001 — a signal path never propagates
        return None
    if values["signo"] != int(expected_signo):
        return None
    return {
        "signo": values["signo"],
        "pid": values["pid"],
        "uid": values["uid"],
        "code": values["code"],
    }


def take(sig: int) -> dict[str, int] | None:
    """The sender facts for the FIRST delivery of ``sig`` — CLAIMED ONCE, then gone.

    ``pid`` is the sending PROCESS (``si_pid``) and ``uid`` its real uid
    (``si_uid``); ``code`` is ``si_code`` verbatim, recorded rather than
    interpreted because its meaning is origin-dependent (``kill(2)`` measured
    ``0`` on Darwin, where the userspace headers spell ``SI_USER`` as 0x10001,
    and a kernel timer measured ``0`` too — so the code alone cannot tell a
    sender from the kernel; the pid can, and 0 there means "the kernel").

    THE CALLER MUST READ ``pid == 0`` AS "NO SENDER", not as process zero.

    WHY IT TAKES RATHER THAN READS, and this is the whole correctness argument
    for the second and later signal a process receives. The sensor fires once
    (it restores the platform's action before re-raising), so a repeat delivery
    is handled with no capture of its own — and if this function kept answering
    with the FIRST delivery's facts, every later receipt would name the first
    sender for its own signal. That is a confidently wrong attribution, which is
    the one outcome worse than "unavailable": a repeat sweep would look like it
    came from the same pid even when it did not. So the capture belongs to the
    delivery that made it, exactly one reader may claim it, and the receipt
    written for a later signal says ``no-siginfo`` — a true statement about that
    receipt's own evidence.

    ``dict.pop`` is atomic under the GIL, so the claim needs no lock even though
    the sensor writes the same mapping from a signal frame.
    """
    try:
        facts = _FACTS.pop(int(sig), None)
        _SELF_RERAISES.pop(int(sig), None)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(facts, dict):
        return None
    return dict(facts)


def _restore(sig: int) -> None:
    """Put the saved action back for ``sig``. One syscall; never raises."""
    saved = _SAVED.get(sig)
    lib = _libc()
    if saved is None or lib is None:
        return
    try:
        if lib.sigaction(sig, ctypes.byref(saved), None) != 0:
            logger.debug("siginfo: restoring sigaction for %s failed", sig)
    except Exception:  # noqa: BLE001
        logger.debug("siginfo: restoring sigaction for %s raised", sig, exc_info=True)


def _redeliver(sig: int) -> None:
    """Send ``sig`` to this process again, so the RESTORED action handles it.

    This is the second half of the handoff: after :func:`_restore`, a process
    delivery runs the platform's own handler exactly as if this module had never
    intercepted it — CPython writes its wakeup-fd byte, ``asyncio`` wakes, and
    the loop callback runs on the thread it always ran on.

    Two spellings because losing the signal would leave the process ignoring its
    shutdown: ``os.kill`` to the pid (what every sender in this incident family
    used) and ``signal.raise_signal`` as the fallback. Both are legal in a
    signal context; a failure here is logged nowhere (logging from a signal
    frame is not safe) and simply leaves the next delivery to the sensor, which
    is still installed precisely so a retry is possible.
    """
    try:
        os.kill(os.getpid(), sig)
        return
    except Exception:  # noqa: BLE001 — try the other spelling
        pass
    try:
        signal.raise_signal(sig)
    except Exception:  # noqa: BLE001 — see the docstring: the sensor stays armed
        pass


def _handler(expected_signo: int) -> Any:
    """Build the ``SA_SIGINFO`` sensor for one signal number.

    The body is the entire cost this module adds to a signal delivery, so it is
    deliberately tiny and has no branch that can raise out of it: capture, then
    ALWAYS restore and re-raise (``finally``), because a sensor that swallowed a
    SIGTERM would turn an instrument into a process that ignores its shutdown.
    """

    def sensor(signo: int, info: int, _ucontext: int) -> None:
        number = int(signo)
        try:
            pending = _SELF_RERAISES.get(number, 0)
            if pending > 0:
                # A predecessor sensor's own re-raise, not a delivery from anybody:
                # consume it, capture nothing (see ``_SELF_RERAISES``). Restore and
                # redeliver below still run, so the product's handler gets the signal.
                _SELF_RERAISES[number] = pending - 1
            else:
                facts = _read(info, expected_signo)
                if facts is not None and number not in _FACTS:
                    _FACTS[number] = facts
                _SELF_RERAISES[number] = 1
        except Exception:  # noqa: BLE001 — never propagate out of a signal frame
            pass
        finally:
            _restore(number)
            _redeliver(number)

    return _CALLBACK(sensor)


def arm(sig: int) -> bool:
    """Install the one-shot sender sensor for ``sig``. Idempotent; never raises.

    CALL THIS AFTER THE HANDLER THAT SHOULD ACTUALLY HANDLE THE SIGNAL IS IN
    PLACE. ``signal.signal`` and ``loop.add_signal_handler`` both call
    ``sigaction`` themselves, so arming first would have the product's own
    install silently replace the sensor, and :func:`take` would keep answering
    forever. The saved action is therefore the product's handler, which is
    exactly what :func:`_restore` puts back before re-raising. Two consequences
    of that contract, both deliberate:

    * a ``SIG_DFL``/``SIG_IGN`` action is REFUSED, so "arm after the handler" is
      enforced by the code and not only by this paragraph: a process with no
      Python-level action has nothing that could ever read a capture, and arming
      over an ignore would convert a deliberate ignore into an action;
    * a caller that later REPLACES the action (``loop.remove_signal_handler``,
      a bare ``signal.signal``) takes the sensor with it. Nothing re-arms, and
      no receipt is harmed: the receipt for a signal that process no longer
      handles has nothing to say about the sender, and says ``no-siginfo``.

    Returns ``True`` when the sensor is now in place (armed here or already),
    ``False`` where it cannot be — a caller reads ``False`` as "this receipt
    keeps its old wording", never as an error.
    """
    try:
        number = int(sig)
    except Exception:  # noqa: BLE001
        return False
    with _LOCK:
        if number in _CALLBACKS:
            return True
        if not supported():
            logger.debug(
                "siginfo: sender capture not supported on %s; receipts stay 'unavailable'",
                sys.platform,
            )
            return False
        lib = _libc()
        struct = _struct_type()
        if lib is None or struct is None:
            return False
        try:
            saved = struct()
            if lib.sigaction(number, None, ctypes.byref(saved)) != 0:
                logger.debug("siginfo: could not read the current action for %s", number)
                return False
            current = int(saved.handler or 0)
            if current in (_SIG_DFL, _SIG_IGN):
                # NOTHING WOULD EVER READ THE CAPTURE. Every receipt path in this
                # product installs a Python-level handler before arming (that is
                # what makes the facts readable at all), so a default or ignored
                # action means the signal ends this process — or is deliberately
                # ignored — without anybody writing a receipt. Arming there would
                # put a ctypes handler in a process for no fact, and over SIG_IGN
                # it would briefly convert a deliberate ignore into an action,
                # which is a behaviour change this instrument has no business
                # making.
                logger.debug(
                    "siginfo: %s has no Python-level action to read the capture; not arming",
                    number,
                )
                return False
            callback = _handler(number)
            flags = (int(saved.flags) | SA_SIGINFO | SA_RESTART) & ~(SA_RESETHAND | SA_NODEFER)
            sensor = struct(
                handler=ctypes.cast(callback, ctypes.c_void_p).value,
                mask=int(saved.mask),
                flags=flags,
            )
            # PUBLISH BEFORE INSTALLING, NEVER AFTER. The sensor's last act is
            # ``_restore`` + ``_redeliver``; with no saved action to restore it
            # re-raises at itself forever (a process spinning at ~91% CPU that
            # ignores even a sanctioned stop until SIGKILL — measured with 0.2 ms
            # arrivals). A signal can land the instant ``sigaction`` returns, so
            # the saved action has to exist BEFORE the sensor can run. The failure
            # paths below withdraw it again, while the sensor is not live.
            _SAVED[number] = saved
            _CALLBACKS[number] = callback
            if lib.sigaction(number, ctypes.byref(sensor), None) != 0:
                _SAVED.pop(number, None)
                _CALLBACKS.pop(number, None)
                logger.debug("siginfo: could not install the sensor for %s", number)
                return False
            verified = struct()
            if lib.sigaction(number, None, ctypes.byref(verified)) != 0 or not (
                int(verified.flags) & SA_SIGINFO
            ):
                # The kernel did not take an SA_SIGINFO action: put back whatever
                # was there rather than leave a handler whose arguments we cannot
                # trust (a plain sa_handler called with three arguments would read
                # the siginfo pointer as a ``frame`` object).
                lib.sigaction(number, ctypes.byref(saved), None)
                _SAVED.pop(number, None)
                _CALLBACKS.pop(number, None)
                return False
            _FACTS.pop(number, None)
            _SELF_RERAISES.pop(number, None)
            logger.debug("siginfo: sender sensor armed for signal %s", number)
            return True
        except Exception:  # noqa: BLE001 — arming must never fail a boot
            logger.debug("siginfo: arming the sender sensor failed", exc_info=True)
            return False


def disarm(sig: int) -> bool:
    """Remove the sensor for ``sig`` and forget what it captured. Never raises.

    For tests and for a caller that wants the platform's action back verbatim
    before this module is reloaded; production paths arm once and leave it. A
    sensor that has already fired is a no-op here (the disposition is the saved
    one again), which is why the forget step is unconditional.
    """
    try:
        number = int(sig)
    except Exception:  # noqa: BLE001
        return False
    with _LOCK:
        # RESTORE FIRST, FORGET AFTER — the mirror of :func:`arm`'s publish-before-
        # install. While the sensor is the live action its saved action must stay
        # published: forgetting first opens a window in which a delivery runs the
        # sensor with nothing to restore and re-raises at itself forever.
        saved = _SAVED.get(number)
        lib = _libc()
        restored = False
        if saved is not None and lib is not None:
            try:
                restored = lib.sigaction(number, ctypes.byref(saved), None) == 0
            except Exception:  # noqa: BLE001
                logger.debug("siginfo: disarming raised", exc_info=True)
        _CALLBACKS.pop(number, None)
        _SAVED.pop(number, None)
        _FACTS.pop(number, None)
        _SELF_RERAISES.pop(number, None)
        return restored


def rearm(sig: int) -> bool:
    """Put the sensor back after it fired, WITHOUT EVER UN-PUBLISHING ITS STATE. Never raises.

    The sensor is one-shot: it restores the platform's action on its first delivery
    and :func:`arm` on an already-armed signal is a no-op. That cost nothing while the
    first terminating signal ended the process; under the victim-side sanction gate a
    REFUSED signal leaves the process running, so the sensor has to be reinstated for
    the signal that matters (the sanctioned stop). ``disarm`` + ``arm`` was the first
    spelling and it was WRONG: it un-published the saved action, and a delivery landing
    in that window ran the sensor with nothing to restore, which re-raises at the
    process forever (measured: 0.2 ms arrivals hung 4 of 8 runs). Here ``_SAVED`` and
    ``_CALLBACKS`` stay published for the whole operation, so a sensor can never be live
    without its saved action.

    What is saved is whatever Python-level action is installed NOW (the product's own
    handler, which the spent sensor restored — or a replacement, if somebody installed
    one since): that is exactly what the sensor's next delivery must hand back.

    Returns ``True`` when the sensor is live afterwards (already live counts), ``False``
    when it cannot be — a signal this module never armed, a ``SIG_DFL``/``SIG_IGN``
    action, or a failed syscall. A ``False`` leaves the platform's own action in place,
    so the only cost is that the NEXT receipt says ``no-siginfo``.

    A signal landing in the gap between reading the current action and installing the
    sensor takes the ordinary path (the product's handler) and is recorded
    ``no-siginfo`` — a true statement about that receipt's own evidence, and the ONLY
    outcome that window can have.
    """
    try:
        number = int(sig)
    except Exception:  # noqa: BLE001
        return False
    with _LOCK:
        callback = _CALLBACKS.get(number)
        lib = _libc()
        struct = _struct_type()
        if callback is None or lib is None or struct is None or number not in _SAVED:
            return False
        try:
            current = struct()
            if lib.sigaction(number, None, ctypes.byref(current)) != 0:
                return False
            sensor_address = ctypes.cast(callback, ctypes.c_void_p).value
            if int(current.handler or 0) == sensor_address:
                return True  # the sensor never fired (or is already back): nothing to do
            if int(current.handler or 0) in (_SIG_DFL, _SIG_IGN):
                return False  # nobody would read a capture: see ``arm``
            sensor = struct(
                handler=sensor_address,
                mask=int(current.mask),
                flags=(int(current.flags) | SA_SIGINFO | SA_RESTART) & ~(SA_RESETHAND | SA_NODEFER),
            )
            # Publish the (possibly new) saved action BEFORE installing: the sensor is
            # not live here, so this write is unobservable until the install below.
            _SAVED[number] = current
            _FACTS.pop(number, None)
            return lib.sigaction(number, ctypes.byref(sensor), None) == 0
        except Exception:  # noqa: BLE001
            logger.debug("siginfo: re-arming the sensor raised", exc_info=True)
            return False
