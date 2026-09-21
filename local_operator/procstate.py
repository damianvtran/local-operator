"""The one answer to "is this pid still a process?" — including zombies.

WHY THIS IS ITS OWN MODULE
--------------------------
Three modules ask this question about a pid that some *other* process owns, and
each asks it to decide whether something a dead owner left behind may be taken
over:

- :mod:`local_operator.session_lease` — may this transcript's sole-writer claim
  be acquired, or is its holder still working?
- :mod:`local_operator.session.runtime.registry` — may this discovery record be
  reaped, or is its runtime still live?
- :mod:`local_operator.resume` — should an attach be offered, or refused
  because the session is already open in another process?

There are TWO questions here, and one of them was missing until a reused pid
made a session unopenable (2026-09-21): "is this pid a live process" is not the
same as "is this pid still the process that wrote this record". A pid is not an
identity — the kernel hands the number to the next process that wants one as
soon as the owner is reaped — so a claim whose pid was recycled reads live
forever. That case is a :func:`birth_token` matter, and the token is read from
the SAME platform probe as the zombie answer, so neither can be asked without
the other being available.

They must agree, and the cost of disagreement is not a stale row: discovery
reporting "gone" while the lease reports "held" is a session **no interface can
open and no mechanism can recover**. That is not hypothetical — see
:func:`is_zombie` for the incident that produced this module.

All three callers are on the resume/attach/startup path and each documents that
it must stay stdlib-only and import-light (``registry`` sits on the CLI startup
path; ``session_lease`` and ``resume`` must be consultable without the engine
or the mobile stack). A leaf module with no local imports is therefore the only
home that satisfies all of them at once, which is exactly why the probe was
written twice before and got the zombie case right in only one of the two.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

#: ``os.O_BINARY`` where the platform has it (Windows), ``0`` everywhere else.
#:
#: **Every ``os.open`` + ``os.write`` pair in this package must OR this into its
#: flags**, and the reason is not style. On Windows ``os.open`` opens in the
#: CRT's TEXT mode unless ``O_BINARY`` is given, and text mode TRANSLATES: a
#: ``0x0A`` byte in the buffer handed to ``os.write`` is written as
#: ``0x0D 0x0A``. Every caller here writes BYTES -- a master key, a sealed
#: secret, a JSONL frame whose length a reader checks -- so the translation
#: corrupts exactly the payloads that have to be byte-exact.
#:
#: It is INTERMITTENT, which is what kept it hidden: a 32-byte master key is
#: only corrupted when the random key happens to CONTAIN ``0x0A`` (about one run
#: in eight), so the Windows probe passed on most runs and then read back
#: ``master.key is 33 bytes; a master key is 32`` on another. Off Windows this
#: is ``0``, so ``flags | O_BINARY`` is bit-identical to ``flags`` and POSIX
#: behaviour cannot change.
#:
#: It lives HERE, next to the rest of this package's platform facts, and not in
#: :mod:`local_operator.paths` where it was first written: ``paths`` is on the
#: runner core's FORBIDDEN_PREFIXES (tests/unit/evaluation/runner/
#: test_isolation.py), so importing it from the evidence store broke the
#: runner's isolation -- the suite caught that on the first CI run after the
#: fix. This module is a stdlib-only leaf, so every layer may reach it.
O_BINARY = getattr(os, "O_BINARY", 0)


# ---------------------------------------------------------------------------
# Platform process primitives
# ---------------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS (cross-platform work, 2026-09-18). Four questions
# below have a DIFFERENT ANSWER ON WINDOWS, and each was previously answered at
# its own call site — twice in two cases, with the copies disagreeing. The
# failure mode is not a wrong answer, it is a destructive one: ``os.kill(pid,
# 0)`` on Windows is ``TerminateProcess``, so a liveness PROBE killed the
# process it was asked about, silently, on a path that runs on every ``lop``
# invocation. One implementation cited by every caller is the only shape that
# keeps this right, which is why these live in the leaf module the rest of the
# package already treats as the single answer to "what is this pid?".
#
# Nothing here imports the rest of the package: ``registry`` sits on the CLI
# startup path and ``session_lease``/``resume`` must stay consultable without
# the engine (see the module docstring).

#: How this platform is spelled in a sentence a USER reads, as opposed to the
#: identifier the runtime uses.
#:
#: ``sys.platform`` is ``win32`` there -- a CPython identifier that appears
#: nowhere else an operator can see -- so a refusal that interpolates it tells
#: them the feature "cannot run on win32", and an upgrade summary announces
#: "(this host is win32)". The mapping was spelled in two modules and MISSING
#: from the third message this branch added, which is the drift a single home
#: exists to stop; ``secrets/peer.py`` and ``update.py`` both read THIS.
#:
#: A module constant rather than a function, because it is patched by tests and
#: ``os.name`` cannot be: ``pathlib`` reads it at call time, so patching it to
#: ``nt`` makes the next ``Path(...)`` a ``WindowsPath`` and the host that is
#: running the test cannot construct one.
PLATFORM_LABEL = "Windows" if os.name == "nt" else sys.platform


#: The platform, read ONCE and through :func:`is_windows` rather than inline.
#: ``sys.platform`` rather than ``os.name`` because a test that flips the
#: platform must patch ONE name: patching ``os.name`` process-wide makes
#: ``pathlib`` refuse to construct a path at all on the host running the test
#: (``cannot instantiate 'WindowsPath' on your system``), which turns a
#: platform test into a broken one.
_PLATFORM = sys.platform


def is_windows() -> bool:
    """Whether this process is on Windows, spelled once for every branch.

    The single home for the question: the process primitives below, the bash
    tool's interpreter resolution and refusal, and anything else that must know
    read THIS, so a platform branch cannot drift from another that disagrees.
    """
    return _PLATFORM == "win32"


#: ``PROCESS_QUERY_LIMITED_INFORMATION``. The least right ``OpenProcess`` needs
#: to ASK about a process, which is deliberate: the pid may be another user's,
#: and a probe must never carry a right to touch what it is asking about.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

#: Windows' ``ERROR_INVALID_PARAMETER`` — ``OpenProcess``'s answer for a pid
#: that is not a process. It is the ONLY error that means "gone": access
#: denied (a pid owned by another account or a higher integrity level) means
#: the pid may well be live, and reporting it dead would let a caller act on a
#: process that is still working.
_WINDOWS_ERROR_INVALID_PARAMETER = 87


def pid_liveness(pid: int) -> bool | None:
    """Whether ``pid`` is a live process: True, False, or None when unprovable.

    THE WINDOWS BRANCH IS NOT AN OPTIMISATION, IT IS A CORRECTNESS FIX.
    ``os.kill(pid, 0)`` is a liveness probe on POSIX and a KILL on Windows:
    CPython's ``os_kill_impl`` takes the ``GenerateConsoleCtrlEvent`` path only
    for ``CTRL_C_EVENT``/``CTRL_BREAK_EVENT`` and otherwise calls
    ``TerminateProcess(handle, sig)`` — and the documented reading of that is
    "any other value for sig will cause the process to be unconditionally
    killed... and the exit code will be set to sig". Signal 0 is any other
    value. A probe that reports ``True`` after killing its subject is the worst
    possible shape: silent to the caller and destructive to the target. So
    win32 asks the kernel the liveness question directly — ``OpenProcess`` —
    exactly as :mod:`local_operator.session_lease` already did.

    ``None`` means "could not prove either answer" (an unusable kernel32, or an
    ``OpenProcess`` failure other than ERROR_INVALID_PARAMETER). It is
    three-valued rather than a bool because the callers disagree about what to
    do with doubt and both are right for their question: :func:`pid_alive`
    fails CLOSED (a live process called dead is unrecoverable), while
    ``session_lease`` reports the refusal honestly instead of guessing.
    """
    if pid <= 0:
        return False
    if is_windows():  # pragma: no cover - exercised on Windows hosts
        return _windows_liveness(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another account's process: it exists, and it cannot be a corpse of
        # ours. Alive is also the safe direction for every caller.
        return True
    except OSError:
        # ESRCH and EPERM are the two answers this probe can give, and both are
        # caught above; anything else means the kernel did not answer the
        # question at all. `None` says exactly that rather than inventing a
        # verdict, and `pid_alive` reads it as the fail-closed "alive".
        # Unreachable on POSIX for a valid int pid, so no macOS/Linux caller
        # changes behaviour here.
        return None
    return True


def _windows_liveness(pid: int) -> bool | None:  # pragma: no cover - Windows only
    """``OpenProcess`` as the liveness primitive, in ``pid_liveness``'s tri-state.

    Split out so the ERROR_INVALID_PARAMETER test and the ``CloseHandle``
    discipline live in one place. Deliberately imports ``ctypes`` lazily: the
    module is imported on the CLI startup path on every platform, and a
    POSIX host must not pay for a Windows-only binding.
    """
    try:
        import ctypes

        # `getattr` for the same reason tools/eval.py uses it: the Windows
        # bindings do not exist on a POSIX host, so the module must still
        # import (and pyright must still understand it) everywhere.
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        last_error = getattr(ctypes, "get_last_error")
        return False if last_error() == _WINDOWS_ERROR_INVALID_PARAMETER else None
    except Exception:
        # No kernel32 binding, or a call that raised: this is doubt, never
        # evidence of death.
        return None


def pid_alive(pid: int) -> bool:
    """Windows-safe "is this pid a running process?" — the shared probe.

    Existence only: a zombie reads as alive here, because the zombie question
    is POSIX-specific and needs its own probe (:func:`is_zombie`). Callers that
    must not treat a corpse as a live owner ask that one AFTER this says alive.

    Fails CLOSED: any doubt answers True. The two errors are not symmetric —
    calling a live owner dead lets a second writer take a transcript a working
    runtime is still appending to, while calling a dead one live only costs the
    recovery of its claim.
    """
    return pid_liveness(pid) is not False


def supports_loop_signals() -> bool:
    """Whether ``asyncio``'s ``loop.add_signal_handler`` works on this platform.

    Reported rather than probed, because the probe IS the failure:
    ``asyncio.BaseEventLoop.add_signal_handler`` is a stub that raises
    ``NotImplementedError``, and only the UNIX selector loop overrides it. The
    Windows default policy installs the Proactor loop
    (``windows_events._WindowsProactorEventLoopPolicy``), which takes the stub —
    so a runtime that installs handlers unconditionally dies before it can
    serve anything, and this codebase never calls
    ``asyncio.set_event_loop_policy``. Callers branch on THIS and fall back to
    ``signal.signal`` (which Windows supports for ``SIGINT``/``SIGBREAK``
    only), rather than discovering the stub at runtime.
    """
    return not is_windows()


def install_loop_signal_handlers(loop: Any, handlers: Mapping[int, Callable[[], None]]) -> bool:
    """Ask ``loop`` to run ``handlers``; degrade to ``signal.signal`` where it will not.

    ``loop.add_signal_handler`` IS UNIX-ONLY (``asyncio.unix_events`` overrides
    it; ``BaseEventLoop``'s is a stub that raises ``NotImplementedError``), and
    the Windows default policy installs the Proactor loop, which takes the stub.
    Every server boot path here that installed handlers unconditionally
    therefore died with a traceback before it could serve anything — on the one
    platform where a foreground command is the only way to run the daemon.

    The fallback is ``signal.signal``, which Windows does support for
    ``SIGINT``/``SIGBREAK``. Its handler runs on the main thread OUTSIDE the
    running loop, so it must hand the callback over with
    ``call_soon_threadsafe`` instead of touching loop state from a signal
    context; ``signal.signal`` also refuses anywhere but the main thread
    (``ValueError``), which is a condition to survive rather than to propagate.

    Returns True when the loop took at least one handler itself. Never raises:
    every caller is booting a server, and a signal it could not install is not a
    reason to fail to bind. A caller with an OPTIONAL signal (``SIGUSR1`` has no
    Windows equivalent) filters it out of ``handlers`` with ``hasattr(signal,
    ...)`` first — an absent constant is a name error here, and a deliberate
    omission at the call site.
    """
    import signal as signal_module

    installed = False
    for sig, callback in handlers.items():
        try:
            loop.add_signal_handler(sig, callback)
            installed = True
            continue
        except (NotImplementedError, RuntimeError, ValueError, AttributeError):
            pass
        try:
            signal_module.signal(sig, lambda *_args, _cb=callback: loop.call_soon_threadsafe(_cb))
        except (OSError, RuntimeError, ValueError):
            # Unhandleable on this platform (or not the main thread). The
            # caller's socket/stop path is still live, so carry on.
            continue
        installed = True
    return installed


def hard_kill_signal() -> int | None:
    """``SIGKILL`` where it exists, ``None`` on Windows.

    ``signal.SIGKILL`` is documented "Availability: Unix", so naming it in a
    stop ladder raises ``AttributeError`` on Windows *before* any kill is
    attempted — rung 3 fails on the platform where the ladder is needed most.
    ``None`` is the honest answer, and the caller's job is then
    :func:`terminate_process_tree`, which does what Windows actually offers.
    """
    import signal

    return getattr(signal, "SIGKILL", None)


class DetachedPopenKwargs(TypedDict, total=False):
    """The ``Popen`` kwargs that detach a child. See :func:`detached_popen_kwargs`.

    A TypedDict rather than ``dict[str, Any]`` so the KEYS are known to the
    analyzer at every spread site: an opaque map made every ``Popen(...)`` call
    that passed it unverifiable, and the type parameter pyright then inferred
    (``str``, from a spread it could not see through) disagreed with the
    declared ``Popen[bytes]`` of the runtime spawn.
    """

    start_new_session: bool
    creationflags: int


def detached_popen_kwargs() -> DetachedPopenKwargs:
    """``Popen`` kwargs that REALLY detach a child, per platform.

    On POSIX this is ``start_new_session=True`` (``setsid``): the child leaves
    this process group, so a Ctrl-C in this terminal does not reach it and it
    outlives us.

    WINDOWS SILENTLY IGNORES ``start_new_session`` — ``Popen`` documents it
    "(POSIX only)" and the Windows ``_execute_child`` parameter is literally
    named ``unused_start_new_session``. No error is raised, so the flag looks
    honoured while the child keeps the parent's console: ``CTRL_C_EVENT`` in
    that console and a console close both reach it, which is precisely what
    the callers pass the flag to prevent. The platform's equivalent is
    ``DETACHED_PROCESS`` (no console inherited at all) plus
    ``CREATE_NEW_PROCESS_GROUP`` (a group of its own, so a targeted
    ``CTRL_BREAK_EVENT`` can still reach only it). ``CREATE_NO_WINDOW`` is
    deliberately NOT used here: the documented behaviour is that it is ignored
    when combined with ``DETACHED_PROCESS``, and it says nothing about console
    inheritance, which is the property this function is about.

    Returned as kwargs rather than a boolean so a caller merges them into its
    own ``Popen``/``create_subprocess_*`` call and cannot half-apply them.
    """
    if is_windows():  # pragma: no cover - exercised on Windows hosts
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)) | int(
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        )
        return {"creationflags": creationflags}
    return {"start_new_session": True}


def terminate_process_tree(pid: int, *, force: bool = False) -> bool:
    """Stop ``pid`` and its descendants; True when a stop was DELIVERED.

    POSIX signals the process GROUP when the pid leads one — a shell tool child
    is spawned ``start_new_session``, so its pgid IS its pid — and signals the
    pid alone when it does not, which is the case that matters: `killpg` there
    would target the group the CALLER belongs to. ``force`` selects ``SIGKILL``
    over ``SIGTERM``.

    WINDOWS HAS NO SIGNAL TO DELIVER AND THE ``force`` FLAG CANNOT BE HONOURED:
    a detached child has no console to receive ``CTRL_BREAK_EVENT``, and
    ``TerminateProcess`` — what ``taskkill`` reaches for — is the only stop the
    kernel offers. Both modes therefore terminate; ``force`` only decides
    whether ``taskkill`` is given ``/F``, and the graceful rung on this platform
    is the control socket's ``stop`` op, which the stop ladder tries first.
    ``taskkill /T`` rather than ``TerminateProcess`` on the one pid because the
    descendants are the point.

    Never raises: a caller is on a stop path where an exception would abort the
    ladder that is trying to make the process go away. False means "nothing was
    there to stop" (the pid is already gone), not "the stop failed quietly".
    """
    if pid <= 0:
        return False
    if is_windows():  # pragma: no cover - exercised on Windows hosts
        return _taskkill_tree(pid, force=force)
    import signal

    sig = signal.SIGKILL if force else signal.SIGTERM
    # Declared with its Optional BEFORE the probe, so the fallback below is an
    # assignment rather than a second declaration of the same name.
    pgid: int | None
    try:
        pgid = os.getpgid(pid)
    except OSError:
        # The pid is gone, or is not one we may ask about. Signal the PID and
        # let it report the same thing rather than inventing an answer.
        #
        # NOT ``pgid = pid`` (agent review round 2, recorded item): that makes
        # the comparison below true and takes the ``killpg`` branch, which is
        # the OPPOSITE of this comment -- and `killpg` on a pid we could not
        # prove leads a group may signal the group THIS CALLER belongs to,
        # killing the caller and its siblings. The comment stated the intent;
        # the code did something else. None here means "signal the pid alone".
        pgid = None
    try:
        if pgid is not None and pgid == pid:
            # The pid LEADS its own group: a shell command spawned
            # `start_new_session` is exactly this, and killing only the leader
            # would leave the children it spawned behind. `killpg` is what the
            # callers did before this helper existed.
            os.killpg(pgid, sig)
        else:
            # NOT a group leader. `killpg(getpgid(pid))` here would signal the
            # group this caller itself belongs to — killing the caller and its
            # siblings — which is how a stop helper becomes the incident. Signal
            # the pid, which is the tree this process actually owns.
            os.kill(pid, sig)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _taskkill_tree(pid: int, *, force: bool) -> bool:  # pragma: no cover - Windows only
    """``taskkill`` the pid's tree; see :func:`terminate_process_tree`.

    ``SystemRoot``-anchored rather than PATH-resolved, matching
    ``tools.eval``'s last-resort killer: a hijacked PATH must not be able to
    redirect a kill. ``taskkill`` exits 128 for "no such process", which is the
    already-gone answer rather than a failure.
    """
    taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
    argv = [taskkill, "/PID", str(pid), "/T"]
    if force:
        argv.append("/F")
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _is_zombie_state(state: str) -> bool:
    """Whether a ``ps``/``/proc`` state field names an exited process.

    ONE HOME for the spelling, because two probes read it now (the single-pid
    :func:`is_zombie` and the batched :func:`zombie_states`): ``ps`` prints a
    zombie as ``Z`` and may append flag characters (``Z+``), so the test is a
    prefix rather than an equality.
    """
    return state.strip().upper().startswith("Z")


# ---------------------------------------------------------------------------
# Birth tokens: "is this pid still the process that wrote this record?"
# ---------------------------------------------------------------------------
#
# WHY A PID IS NOT AN IDENTITY (the incident this section exists for).
# A session runtime died and left its sole-writer claim behind naming pid 1969;
# the kernel then gave pid 1969 to an unrelated live process. Every probe in this
# package asked "is there a process with this pid", which the stranger answered
# yes to, so the dead owner's claim read live for as long as the stranger happened
# to hold the number: engage waited out its 30 s deadline without spawning (the
# user was told "the runtime is reconnecting"), the lease refused the session from
# every interface, and the attach guard reported it as owned. The round-3 zombie
# fix cannot reach this case — its reasoning is "the pid is not reused while the
# corpse lingers", which an unreaped zombie satisfies and a reaped one does not,
# one instant after it is reaped.
#
# WHAT THE TOKEN IS: the process's START TIME, which the platform will answer
# for any live pid and which cannot be made true again once that process is
# gone.
#
#   - macOS / BSD: ``ps -o lstart=`` — whole seconds (see ``_PS_RENDER_ENV`` for
#     why the child's environment is fixed, and ``ProcessSample.is_birth`` for
#     what
#     the resolution does and does not catch);
#   - Linux: ``/proc/<pid>/stat`` field 22, the start time in clock ticks since
#     boot — 10 ms at the usual 100 Hz, and fork-free;
#   - Windows: none, deliberately. There the process identity is a HANDLE rather
#     than a process-table entry, and a pid can be recycled the instant its
#     handle closes, so every caller keeps today's pid-liveness behaviour there.
#
# The token is an OPAQUE STRING paired with a SCHEME, and the two are compared
# together: the scheme names what was measured (``ps-lstart-c-v1``,
# ``proc-starttime-v1``) and the token is that measurement, verbatim. A token
# written with a scheme this build does not produce — another platform, or a
# build that measured something else — is DOUBT rather than a mismatch. That
# direction matters: see ``same_birth`` and ``ProcessSample.is_birth``.

#: `ps -o lstart=` (macOS/BSD): the process start time, as `ps` renders it.
#:
#: Scheme-tagged in the CLAIM rather than in the token bytes: the tag is what a
#: reader compares first, so a token written by a build that measured something
#: else is UNREADABLE rather than different, and unreadable means "keep the
#: holder" (:func:`same_birth`). The version suffix is the point — if the `ps`
#: invocation ever changes (padding, a different `ps`, an extra field), the tag
#: must move, or every live owner in the fleet reads as mismatched at once.
BIRTH_SCHEME_PS_LSTART = "ps-lstart-c-v1"

#: `/proc/<pid>/stat` field 22 (Linux): start time in clock ticks since boot.
BIRTH_SCHEME_PROC_STARTTIME = "proc-starttime-v1"

#: The `ps` keywords that answer BOTH per-process questions in ONE invocation.
#:
#: One probe rather than two because every caller of the identity question is
#: also asking the zombie question, and the zombie probe's cost is what the
#: engage loop's cadence is set by (2.4-4.6 ms per fork against the 23-30 µs
#: budget of one dense poll iteration). `lstart` rides the invocation that was
#: already being spent instead of adding a second one beside it.
#:
#: It is also a CORRECTNESS property, not only a cost one: a second fork for the
#: identity would sample a DIFFERENT instant, so the pid could be recycled
#: between the liveness answer and the token answer and the successor would
#: answer the identity question. One line, one sample, both answers.
_PS_SAMPLE_KEYWORDS = "pid=,state=,lstart="

#: The environment every `ps` child is given. CORRECTNESS, not tidiness:
#: `lstart` is rendered in the process table's LOCAL time zone and locale, so
#: two callers with different `LC_ALL`/`TZ` would sample DIFFERENT tokens for the
#: SAME live process. A false mismatch is the one dangerous outcome of this whole
#: mechanism — it says "the writer is gone" and lets a second runtime take a
#: transcript a working process is still appending to — so the rendering is fixed
#: in the ONE place the probe lives, where no caller can forget it.
#:
#: Measured on this host (2026-09-21): `TZ=UTC /bin/ps -o lstart= -p <pid>` prints
#: `13:52:44` where `TZ=Asia/Tokyo` prints `22:52:44` for the same process — and
#: `LC_ALL=C` is what keeps the month name parseable at all.
_PS_PINNED_ENV = {"LC_ALL": "C", "TZ": "UTC"}


@dataclass(frozen=True)
class ProcessSample:
    """One probe's answers about a pid: is it a corpse, and is it still the writer?

    ONE sample, TWO answers, and that is a correctness property as much as a
    cost one: the two questions are asked about the same holder by the same
    callers, and a probe that answered them from separate instants could report
    a pid that was a corpse when it was asked about liveness and a stranger by
    the time it was asked about identity.
    """

    #: Exited but unreaped. See :func:`is_zombie`.
    zombie: bool

    #: The measurement itself, opaque to every reader but the platform that
    #: produced it, paired with :func:`birth_scheme` (WHICH measurement this is).
    #: ``None`` when the platform (or this probe, on this run) produced none —
    #: DOUBT, never death: see :func:`same_birth`.
    birth: str | None

    def is_birth(self, scheme: str | None, token: str | None) -> bool | None:
        """Whether THIS sample's process is the one that produced ``token``.

        The whole comparison, and its only home: :func:`same_birth` is the
        pid-based accessor for callers that hold no sample, and the callers that
        do hold one — the acquisition and reap paths, which need the zombie
        answer from the same instant — ask the sample directly. A second
        spelling of this comparison beside it is exactly the drift
        :func:`same_birth` documents.

        See :func:`same_birth` for the three-valued contract.
        """
        if not scheme or not token:
            return None
        if scheme != birth_scheme():
            # A scheme this build does not produce: written on another platform,
            # or by a build that measured something else. Unreadable, not a
            # mismatch — the difference is a live writer's claim.
            return None
        if self.birth is None:
            return None
        return self.birth == token


def birth_scheme() -> str | None:
    """What this platform's birth token measures, or ``None`` where it cannot.

    Windows is the ``None`` case, deliberately: liveness there is a HANDLE
    question (``OpenProcess``), a terminated process is immediately reusable,
    and there is no process-table field to sample. Every caller therefore writes
    no birth fields, reads them as unreadable, and falls back to exactly today's
    pid-liveness behaviour — see :func:`same_birth`.
    """
    if is_windows():
        return None
    return BIRTH_SCHEME_PROC_STARTTIME if os.path.isdir("/proc") else BIRTH_SCHEME_PS_LSTART


def _ps_samples(pids: Sequence[int]) -> dict[int, ProcessSample]:
    """ONE ``ps`` invocation answering both questions for a whole pid SET.

    ``ps`` takes a pid LIST, which is what makes the batch cost one fork rather
    than one per pid — the difference between a probe that scales with the record
    population and one that does not. A pid ``ps`` does not report (it exited
    between the caller's signal-0 check and this call) is simply ABSENT from the
    result: existence is the caller's question and it has already asked it, so
    inventing a verdict here would answer a different one.

    The start time is taken as `ps` RENDERS it (``lstart``, under the pinned
    ``_PS_PINNED_ENV``) and is never parsed into a date. That is deliberate: the
    token only ever has to equal another rendering of the same process, and a
    parse is where a locale, a format surprise or a clock step could turn two
    renderings of one process into two different values — the mismatch direction
    that costs a live writer its claim. Left as text, those surprises can only
    make two tokens differ, which is the safe side.
    """
    if not pids:
        return {}
    try:
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [
                "/bin/ps",
                "-o",
                _PS_SAMPLE_KEYWORDS,
                "-p",
                ",".join(str(pid) for pid in pids),
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
            env={**os.environ, **_PS_PINNED_ENV},
        )
    except Exception:  # noqa: BLE001 — an unprobeable set is treated as alive
        return {}
    samples: dict[int, ProcessSample] = {}
    # ``pid=`` suppresses the header; ``state`` is padded to its column, so the
    # line splits into pid, state, and the five fields of ``lstart``.
    for line in result.stdout.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        rendered = " ".join(fields[2].split())
        samples[int(fields[0])] = ProcessSample(
            zombie=_is_zombie_state(fields[1]),
            birth=rendered or None,
        )
    return samples


def _proc_samples(pids: Sequence[int]) -> dict[int, ProcessSample]:
    """The Linux answer: state and start time from ``/proc/<pid>/stat``.

    One read per pid and no fork. Field 22 (``starttime``, in clock ticks since
    boot) is the counterpart of ``ps``'s ``lstart``, and it is kept as the RAW
    INTEGER: the token is compared against another sample of the same process, so
    ticks are sufficient, exact, and immune to NTP steps and to any TZ or DST
    rule — strictly better than a wall-clock rendering, which is why they are
    never converted.
    """
    samples: dict[int, ProcessSample] = {}
    for pid in pids:
        try:
            data = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            continue
        # The command name is parenthesised and may itself contain spaces and
        # parentheses, so the remaining fields are read from the LAST ')'.
        tail = data.rpartition(")")[2].split()
        # tail[0] is field 3 (state); ``starttime`` (field 22) is tail[19].
        if len(tail) < 20:
            continue
        starttime = tail[19]
        samples[pid] = ProcessSample(
            zombie=_is_zombie_state(tail[0]),
            birth=starttime if starttime.isdigit() else None,
        )
    return samples


def process_samples(pids: Iterable[int]) -> dict[int, ProcessSample]:
    """Both facts for a pid set, from the platform's ONE probe.

    Windows answers ``{}`` rather than probing (see :func:`birth_scheme`), which
    is the same answer the current implementation gives and the reason its
    callers read absence as "not a zombie".
    """
    wanted = sorted({int(pid) for pid in pids if int(pid) > 0})
    if not wanted or is_windows():
        return {}
    if os.path.isdir("/proc"):
        return _proc_samples(wanted)
    return _ps_samples(wanted)


def process_sample(pid: int) -> ProcessSample | None:
    """The same sample, for one pid; ``None`` when this host cannot answer.

    ``None`` means a Windows host, a ``ps`` that failed, or a pid with no
    ``/proc`` entry left. Every caller treats it as doubt and leaves a holder's
    claim exactly where it is (:func:`same_birth`); the alternative, calling an
    unproven pid dead, is the one direction that forks a live transcript.
    """
    if pid <= 0:
        return None
    return process_samples([pid]).get(pid)


def birth_token(pid: int) -> str | None:
    """The birth token of the process now holding ``pid``, or None.

    Raw, and meaningful only together with :func:`birth_scheme` — the pair is
    what a claim records and what :func:`same_birth` compares. The single
    accessor for the platform's answer, so a caller that wants only identity does
    not reach for a probe of its own: two spellings of "when did this process
    start" are two answers, and this module exists because the probes it replaced
    disagreed.
    """
    sample = process_sample(pid)
    return None if sample is None else sample.birth


#: ``pid -> token`` for THIS process. The token is fixed at process creation and
#: cannot change while the process lives, so one probe answers every later
#: question — but it is keyed by pid and re-sampled if that ever differs, so a
#: ``fork``ed child (which inherits this module's globals) cannot answer with its
#: parent's identity. Getting that wrong would write a claim naming a writer that
#: never held it.
_SELF_BIRTH: tuple[int, str] | None = None


def self_birth_token() -> str | None:
    """This process's own birth token, or ``None`` where the platform has none.

    A FAILED SAMPLE IS NOT CACHED: ``None`` means "the platform has no token"
    (Windows) or "this probe failed", and caching the second would poison every
    later call in that process — one transient ``ps`` failure and every claim the
    runtime writes from then on would carry no identity at all. Only a token is
    remembered, and only under the pid it was read for.
    """
    global _SELF_BIRTH
    pid = os.getpid()
    if _SELF_BIRTH is not None and _SELF_BIRTH[0] == pid:
        return _SELF_BIRTH[1]
    token = birth_token(pid)
    if token is not None:
        _SELF_BIRTH = (pid, token)
    return token


def same_birth(scheme: str | None, token: str | None, pid: int) -> bool | None:
    """Whether the process at ``pid`` is the one that produced ``token``.

    THREE-VALUED, and the third value is the load-bearing one:

    - ``True`` — a process holds this pid and it is the one that wrote the
      record. A caller must treat such a holder as LIVE.
    - ``False`` — a process holds this pid and it is NOT the writer: the writer
      is gone and the number was reused. The record is stale, and the
      generation-fenced recovery path may take it.
    - ``None`` — the question could not be asked: no recorded birth at all (an
      older build's record), no sample (``ps`` failed, ``/proc`` unreadable), or
      a scheme this build does not produce (written elsewhere, or by a build
      that measured something else). DOUBT IS NOT DEATH, exactly as it is not
      for :func:`pid_liveness`: calling a live writer dead is what puts two
      runtimes on one transcript, while calling a dead one live leaves its
      record where it was.

    **What the macOS token's one-second resolution does and does not catch.**
    ``lstart`` is rendered to whole seconds, so two processes born inside the SAME
    second are indistinguishable to it — and a same-second spawn is ORDINARY, not
    exotic: two processes started back to back share a token, measured on this host
    (Linux's ticks are 10 ms, and a pid reused there requires the counter to wrap
    through ``pid_max``). What that means for a claim, stated as the frequency it
    is rather than as a bound: a false ``True`` needs the current occupant of the
    pid to have been born in the same second as the WRITER, and since the writer
    had to die and be reaped before the pid could be handed on, it means the writer
    started and was reaped inside one second — a process whose whole life fits in
    one tick. A false ``True`` leaves the claim un-takeable until the stranger
    exits — the incident's symptom, which the same interfaces report honestly — and
    never a second writer. Measured on
    this host: a canary pid killed immediately was not handed to any of the next
    400 short-lived processes in 19 s, so on macOS the reuse itself is the slow
    part (the counter wraps through ``pid_max``).
    """
    if not scheme or not token:
        return None
    sample = process_sample(pid)
    if sample is None:
        return None
    return sample.is_birth(scheme, token)


def zombie_states(pids: Iterable[int]) -> dict[int, bool]:
    """The zombie verdict for a whole pid SET, in ONE probe.

    WHY THIS EXISTS (review round 1, MINOR 3 / QA Q1). ``registry.classify``'s
    derived policy spends the ``ps`` fork on every record whose heartbeat has
    gone quiet, and the desktop feed re-runs the whole scan on a one-second
    clock — so a population of quiet-but-alive records (the wedged sessions this
    feature most cares about) cost one fork PER RECORD PER SECOND: measured at
    88 forks on each probe with 200 records, a probe of 1.7 s, and the feed's
    10 Hz doorbell collapsing to 0.5 Hz. The question is per pid, but the probe
    need not be: ask it once for the set.

    MEMOISATION WAS THE OTHER CANDIDATE AND WAS REJECTED, deliberately: any
    cache of the "not a zombie" answer delays the alive -> zombie transition by
    its staleness bound, which is exactly the transition the round-3 U10 fix
    exists to catch promptly (a ``kill -9``'d runtime must stop reading as live
    as soon as its beat is quiet). A batched probe keeps every answer as fresh
    as the unbatched one — it changes how many forks the answers cost, never how
    old they are — so the freshness guarantee is preserved rather than traded.

    Pids that cannot be probed are absent from the result; the caller decides
    what absence means (``registry.scan`` reads it as "not a zombie", the same
    fail-closed answer as a failed single probe).

    Windows answers ``{}`` rather than probing: a terminated process is
    immediately reusable there, and liveness is a *handle* question
    (``OpenProcess``), not a process-table one — the answer the POSIX probe
    would reach through a doomed ``/bin/ps`` fork, without the fork.

    The probe is :func:`process_samples`, the SAME one that answers the birth-token
    question. That is deliberate: the two questions are asked about the same
    holder by the same callers, and one probe means a caller can never be told
    that a pid is a corpse with one answer and a stranger with another. On macOS
    this also means the identity tag rides the fork this batch was already
    spending, rather than a second fork beside it.
    """
    wanted = sorted({int(pid) for pid in pids if int(pid) > 0})
    return {pid: sample.zombie for pid, sample in process_samples(wanted).items()}


def is_zombie(pid: int) -> bool:
    """Whether this pid is an exited-but-unreaped process.

    **A ZOMBIE IS NOT A LIVE PROCESS, AND ``kill(pid, 0)`` CANNOT TELL YOU
    THAT.** Signal 0 succeeds against a process that has exited but has not
    been reaped yet, so every probe built on it reports such a pid as alive
    for as long as its parent fails to reap it — and for a runtime spawned by
    a long-lived TUI, that parent may never reap it at all. The pid is not
    reused while it lingers, so the wrong answer is stable, not a race — **and
    that is the limit of what this probe can promise.** The number is free the
    instant the corpse is reaped, so "this pid is a live process" stops being a
    statement about the WRITER one instant later. That second question — "is
    this pid still the process that wrote this record?" — is
    :func:`ProcessSample.is_birth`, and a caller that asks only this one is asking
    half of what it needs. Everything that arbitrates a sole-writer claim asks
    both.

    Where the wrong answer costs a row: `registry` fixed this for discovery
    records in round 3 (U10) — a SIGKILLed runtime reported ``live`` with
    ``0B`` RSS in `lop sessions`. Where it costs the SESSION: the lease probe
    kept calling such an owner live, so nothing could take its claim over —
    ``acquire_session_lease`` refused every attempt and
    ``reap_proven_dead_session_claim`` declined to remove it, because both
    require the holder to be *proven dead*. An operator's session whose
    runtime was killed while its TUI parent lived on was therefore
    un-attachable from every interface (TUI ``/resume``, ``lop exec
    --resume``, phone attach) with the message "already open in pid N", where
    N was a corpse.

    Fails CLOSED (returns False, i.e. "treat as alive") on any doubt: calling
    a live process dead would let a second writer take a transcript a working
    runtime is still appending to, and a forked trajectory is far worse than
    the unrecovered claim this exists to remove.

    Deliberately NOT ``psutil``: this module is stdlib-only by contract and
    ``/proc`` does not exist on macOS, so on POSIX the fallback is a ``ps``
    fork — measured at 2.4-4.6 ms across runs on an M-series box, tracking
    host load, against ~1 µs for signal-0. Callers therefore spend it only where the answer changes
    what they do. Concretely, these are the places that may pay it:

    - ``session_lease._pid_state`` — every acquisition and reaper decision, and
      the legacy ``.session.pid``-only branch, all of which require a holder to
      be *proven* dead before they move its claim.
    - ``resume.live_runtime_pid`` — after signal 0 has already said "exists",
      because at its user-facing call sites (the TUI's ``/resume``,
      ``lop exec --resume``, the phone's attach) that answer decides whether
      someone is refused the session they asked for. Its ``check_zombie=False``
      mode exists for one caller, the engage loop's dense discovery pass, where
      the same answer can only cost a wait.
    - ``registry.pid_alive(check_zombie=True)`` — ``registry.scan`` spends it on
      a record whose heartbeat has already gone quiet, so a healthy session's
      row stays fork-free. That scan asks through :func:`zombie_states`, i.e.
      ONE fork for the whole quiet set rather than one per quiet record: the
      answer is identical and the caller is the one that has the population in
      hand.

    Two places deliberately do NOT, and both are latency trades rather than
    safety ones: ``registry.pid_alive``'s default, which keeps ``scan``
    fork-free for healthy records, and the engage loop's two probes inside its
    dense 10 ms grid (``find_runtime_record``'s owner lookup and
    ``_lease_holder``), where a fork costs more than the wait it would shorten.
    Neither can take a claim — only ``session_lease`` does that, and it always
    asks.

    The probe is :func:`zombie_states`, so the single-pid answer and the batch a
    whole scan asks for cannot drift apart: this is that function asked about
    one pid.
    """
    return zombie_states([pid]).get(pid, False)
