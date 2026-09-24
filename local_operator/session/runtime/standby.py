"""A pre-imported runtime interpreter this CONSOLE forked, adopted by its next cold engage.

WHY THIS EXISTS — THE MEASUREMENT
=================================
A cold engage (a brand-new conversation, or switching to one whose runtime is
not running) spawns ``python -m local_operator.session.runtime.process`` and
waits for it to publish. Under the operator's real load (~17-25 live runtimes,
load average 100-140 on 14 cores, measured 2026-09-24) that wait was 3-4 s wall
(``scripts/bench_cold_engage.py``: child spawn -> publish median 2.4-3.0 s).
Attributed with a CPU-clock cProfile of a fresh child:

    interpreter + the runtime's import graph     ~1,100 ms CPU   (~70%)
    session construction (create_session)          ~440 ms CPU
    serve + publish                                  ~15 ms CPU

and on that host 100 ms of CPU costs 1.0-1.7 s of WALL (a pure CPU spin,
measured the same afternoon), because the scheduler is shared with ~1,000
processes. So the only lever is removing CPU from the critical path, and import
pruning cannot do it: session construction needs all but a handful of the ~450
modules it loads. A standby pays that import, and the next cold engage hands its
spawn to the already-warm process instead of forking a fresh interpreter.

WHY THE CHANNEL IS PRIVATE — THE ONE PROPERTY EVERYTHING ELSE FOLLOWS FROM
==========================================================================
**This console forks the standby, and the only channel to it is the socketpair
end that fork inherited.** There is no path, port, lock file or directory any
other process can find, bind or connect to, because there is none at all.

That is not a simplification of the earlier design; it is the correction of a
security defect in it (agent review round 1, R1-1). That revision kept one
standby per MACHINE and reached it over ``<root>/run/standby/standby.sock``.
Same-uid processes cannot be isolated from each other — a model-authored
``bash`` call runs as exactly this uid — so anything writable by this uid can be
bound FIRST by any of them, and the console would then have handed that process
the operator capability (issue #1310's escalation: the capability is what lets
a runtime's own gate be moved from ``ask`` to ``auto``). A peer-pid or
``LOCAL_PEERPID`` check does not close it either: the impostor's pid is just as
real, and ``secrets/peer.py`` records that same-uid peers can only be identified
by LINEAGE. Lineage is what an inherited descriptor IS: the value can reach a
process only if this process's own ``fork``/``exec`` put the descriptor in it,
so the capability cannot reach anything this console did not start. There is no
ownership, mode, path or handshake that could have the same strength.

THE CONSEQUENCE, STATED PLAINLY: one standby per WARMING CONSOLE, not one per
machine. On this host the desktop app's ``lop serve`` daemon is the singleton
that warms (so the desktop surface keeps one spare per machine, exactly as
intended), while each interactive TUI that opens new conversations warms its
own. The earlier machine-wide sharing is not recoverable by any means that keeps
the capability out of a stranger's hands — see the trade-off section of the PR
body — and the mitigations here are: warming is opt-in per process and only
enabled at the TUI and ``serve`` launch points; it is triggered lazily AFTER an
engage (never at boot, never on the critical path); every standby exits on idle
(:data:`IDLE_REAP_S`), when its console goes away (EOF on the inherited
descriptor, detected immediately), and when its root disappears.

WHAT IT IS NOT — THE OPERATOR'S CONSTRAINTS, AND HOW EACH IS HELD
================================================================
(a) ONE PER CONSOLE, and never more: the process holds at most one standby, and
    a spawn is refused while one is alive. A console with no standby spawns cold
    exactly as before — the standby is an optimisation, never a dependency.

(b) NEVER A STALE BUILD, NEVER A STALE CONFIG. Adoption is refused — the standby
    exits — when any of these moved since it finished warming:

    * the interpreter the caller would spawn (``launch._spawn_interpreter``,
      i.e. the CURRENT generation after a ``lop-update``) is not in the same venv
      the standby runs;
    * ``update.installed_build`` (version + ``.lop-source`` ref) differs;
    * any loaded ``local_operator`` module file's mtime moved (a same-path
      rebuild, an editable checkout edited under it — measured at 0.2 ms for the
      ~140 files, so it is checked on every request rather than sampled);
    * ``config.yml``'s stat key (``config.config_file_key``) differs from the one
      recorded at warm, for the operator's explicit concern that a spare warmed
      under one configuration must not construct a session under another.

    A requester whose root or warm-sensitive environment differs is DECLINED (it
    keeps waiting for a matching one) rather than retired, because a standby that
    exited on every mismatch would be killed by whichever host engaged next.

(c) INVISIBLE AS A SESSION until it is adopted. The standby publishes nothing a
    session reader lists: no record in ``run/mobile`` (so not in ``lop
    sessions``, the desktop feed or the mobile list), no boot record, no lease,
    no transcript, no analytics row — and it never runs ``create_session``, so it
    never takes the store-maintenance or analytics passes, which start from
    session construction. It does not arm the stall watchdog
    (``stall_watchdog.arm`` has exactly one call site, the runtime ``__main__``
    guard, and a standby reaches it only through :func:`_become_runtime`). Its
    argv is ``-m local_operator.session.runtime.standby``, which the residency
    census (``reclaim.parse_process_row``) matches as a whole word against
    ``RUNTIME_MODULE`` and therefore does NOT count as a runtime.

    AT adoption it must become countable and attributable, because it is now a
    runtime the sweep's orphan protection has to see:

    * the module word and the ``[standby]``/``id=--------`` label are rewritten
      IN PLACE in the process's own argv memory (``_rename_argv``) — same
      lengths, and macOS/Linux ``ps`` read argv straight from that memory
      (verified with ``ps -ww``). Where the rewrite is impossible, the adoption
      is refused rather than leaving an uncountable runtime behind;
    * a BOOT RECORD is written BEFORE construction starts. That is what fixes
      R1-3: the adopted process is older than its session (its ``etime`` counts
      from the warm) and its ``ps -E`` environment block is this console's, so
      ``reclaim``'s young rung and ``session_id_of`` had nothing to read. The
      boot record is the artifact those readers already prefer, and writing it
      one step earlier makes both true for the whole construction window.

(d) COST, measured: max RSS 47-158 MB (median ~130 MB) per standby, sitting at 0%
    CPU in a blocking read. It is warmed on a daemon thread after an engage, so it
    costs an engage nothing; it exits on adoption, on idle
    (:data:`IDLE_REAP_S`), and when its console goes away.

    HOW LONG THE WARM TAKES, CORRECTED (QA round 1, QW1). The first revision kept
    the warm in Darwin's background band for its whole life and claimed "about a
    minute". That claim was false on the host this exists for: measured at load
    180-227, standbys spent 16 minutes of wall on 0.25-0.29 s of CPU each and had
    not finished, so the feature failed open to a cold spawn exactly when it
    mattered. :func:`_warm` now measures its own progress and abandons the band if
    it is starved (:data:`WARM_BAND_MAX_S` caps the polite phase at 45 s), which
    bounds the warm at ~15-45 s from there at that load — see the PR body for the
    measurement at load 150+. Two cold engages inside that window still both take
    the cold path.

    A CONSEQUENCE FOR THE CONSTRAINTS ABOVE: this buys the win per CONSOLE, not
    per machine. The desktop app's ``lop serve`` daemon is a singleton and keeps
    one spare per machine as intended, but each interactive TUI that opens new
    conversations warms its own — the price of the capability never reaching a
    process this console did not start.

RIG NOTE (agent review round 1, R1-6, and it is now structural rather than a
policy): a rig never leaves a standby behind, because a standby's life is tied
to the process that warmed it — when the rig ends, its descriptor closes and the
standby sees EOF and exits. A rig that keeps its ROOT while its warming process
stays alive can still hold one for up to :data:`IDLE_REAP_S`; the root-gone check
covers the case where the root is deleted first.

WHAT IT DOES NOT FIX: session construction itself (~440 ms CPU before this PR's
config memo, ~180 ms after) still runs after adoption, because it depends on the
session id, cwd and model the engage names. See the PR body for the bound that
leaves on a host at load 100+.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The module a standby runs as. SAME LENGTH as ``types.RUNTIME_MODULE`` on
#: purpose: adoption rewrites this word in place (see ``_rename_argv``), and an
#: in-place rewrite cannot grow the string. ``test_standby`` pins the lengths.
STANDBY_MODULE = "local_operator.session.runtime.standby"

#: The argv flag carrying the INHERITED descriptor number the standby listens on.
#: A descriptor number is not a secret: the descriptor itself is what nobody else
#: has, which is the whole property this design rests on. Spelled once, like
#: ``approval.OPERATOR_FD_FLAG``, because the spawner and the child are different
#: processes and a drift between them is a silent "no standby".
STANDBY_FD_FLAG = "--standby-fd"

#: How long a standby waits for an adoption before exiting on its own. Long
#: enough that an operator who opens new conversations every few minutes always
#: finds one warm; short enough that a console gives the memory back within the
#: quarter hour. A standby is no longer shared between consoles, so this is now
#: purely the memory bound rather than an availability policy.
IDLE_REAP_S = 900.0

#: How long the engage side waits for a standby's answer before giving up and
#: spawning cold. A healthy standby answers in single-digit ms (it is parked in a
#: read); the bound exists for one that is wedged, so a wedged standby costs at
#: most this on top of today's cold spawn.
ADOPT_TIMEOUT_S = 2.0

#: Environment switch that turns the whole mechanism off — no warm and no
#: adoption — for an operator who wants the cold spawn exactly, and for the test
#: suite, which sets it process-wide in ``tests/conftest.py``: warming is opt-in
#: per process (:func:`enable_warming`, called only at the TUI and ``serve`` CLI
#: launch points), but the suite drives exactly those launch points through
#: ``cli.main()``, and a standby is a detached process by design, so without the
#: switch a run leaves live interpreters behind (measured, and why it is there).
DISABLE_ENV = "LOP_RUNTIME_STANDBY_DISABLED"

#: The keys of the spawn contract a standby is allowed to receive. The same
#: names ``launch._spawn_runtime`` writes into a cold child's environment, and
#: nothing else: a standby must end up in exactly the state a cold child starts
#: in.
CONTRACT_KEYS = (
    "LOP_MOBILE_CHILD_CWD",
    "LOP_MOBILE_CHILD_RESUME",
    "LOP_MOBILE_CHILD_PROVIDER",
    "LOP_MOBILE_CHILD_MODEL",
    "LOP_MOBILE_CHILD_EFFORT",
    "LOP_MODEL_SELECTION_OVERRIDE",
    "LOP_RUNTIME_DEFER_MATERIALISE",
)

#: The modules warmed beyond the runtime's own top-level imports and
#: ``session_factory._WARM_IMPORTS``: the smallest set whose transitive closure
#: covers every module a first session construction imports lazily (a greedy
#: cover over the 199 modules measured, 76 of them ours). A module missing here
#: is not a correctness problem — it is imported on first use exactly as before —
#: only a slower adoption.
_WARM_EXTRA: tuple[str, ...] = (
    "local_operator.classification",
    "local_operator.tools.registry",
    "local_operator.ui_console",
    "local_operator.tui.notify",
    "local_operator.secrets.access",
    "local_operator.agents",
    "local_operator.guides",
    "local_operator.session.attention",
    "local_operator.update",
    "local_operator.wakes.store",
    "local_operator.browser_bridge.resources",
    "local_operator.config_watch",
    "local_operator.context_files",
    "local_operator.fork",
    "local_operator.mcp.resources",
    "local_operator.model.prices",
    "local_operator.secrets.session",
    "local_operator.session.cleanup",
    "local_operator.session.runtime.journal",
    "local_operator.variables",
)

#: Environment names a WARMED interpreter may already have acted on, so a
#: requester whose values differ must get a cold child instead. Measured, not
#: guessed: an ``os.environ`` read hook over the whole warm recorded exactly
#: ``HOME``, ``LANG``/``LANGUAGE``/``LC_*``, ``ARCHFLAGS``, ``OTEL_*``,
#: ``PYDANTIC_*``, ``PYTHON*``/``_PYTHON*`` — and nothing else; every product
#: variable (``LOCAL_OPERATOR_*``, ``LOP_*``) is read at CALL time, after
#: adoption has installed the requester's environment. ``TMPDIR``/``TZ`` are
#: added because ``tempfile`` and ``time`` cache them on first use, and
#: ``LOP_BUILD_PREFIX`` because the warm's build stamp is read through it. Every
#: other name (``PATH``, API keys, a terminal's variables, the desktop token) is
#: simply replaced by the requester's value before the runtime starts, which is
#: exactly what a cold child would have inherited.
_WARM_SENSITIVE_NAMES = frozenset(
    {"HOME", "LANG", "LANGUAGE", "ARCHFLAGS", "TMPDIR", "TZ", "LOP_BUILD_PREFIX"}
)
_WARM_SENSITIVE_PREFIXES = ("LC_", "OTEL_", "PYDANTIC_", "PYTHON", "_PYTHON")


def _venv_of(interpreter: str) -> str:
    """The environment an interpreter path runs in, comparable across spellings.

    NOT ``realpath(interpreter)``: a standby is exec'd through the branded
    hardlink beside the interpreter (``procname.spawn_identity``), so its
    ``sys.executable`` is ``<venv>/bin/Local Operator`` while the requester names
    ``<venv>/bin/python3`` — and a venv's ``python`` is itself a symlink to the
    base interpreter, which every venv on the machine shares. The venv directory
    is the identity that decides which build is imported.
    """
    return os.path.realpath(Path(interpreter).parent.parent)


def disabled() -> bool:
    """Off by switch, and off where an inherited descriptor cannot be trusted.

    The private channel is a POSIX socketpair; Windows has no ``pass_fds``, and
    its inheritance rules would hand the console's descriptor to any child it
    starts. Windows therefore keeps the cold spawn.
    """
    if os.name != "posix":
        return True
    return os.environ.get(DISABLE_ENV, "") not in ("", "0", "false", "no", "off")


def _warm_sensitive(env: "dict[str, str] | os._Environ[str]") -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if (key in _WARM_SENSITIVE_NAMES or key.startswith(_WARM_SENSITIVE_PREFIXES))
        and key not in CONTRACT_KEYS
        and key != "LOP_RUNTIME_ADOPT_SESSION"
    }


# ---------------------------------------------------------------------------
# The console side: one standby, one private descriptor, no path anywhere
# ---------------------------------------------------------------------------

#: Whether THIS process warms standbys. Off by default: only a long-lived
#: interface host turns it on (:func:`enable_warming`), so a unit test, a script,
#: a ``lop exec`` or a benchmark that engages a runtime never leaves an idle
#: interpreter behind it. A list, not a bare global, so the flag is mutated
#: rather than rebound.
_WARMING: list[bool] = [False]

#: The live standby, or ``None``. At most one per process, and never replaced
#: while it is alive.
_WARM: list["_Standby | None"] = [None]

#: Serialises the spawn/consume transition between the warming thread and the
#: engage path.
_LOCK = threading.Lock()


class _Standby:
    """A forked, warming interpreter and the private descriptor to it.

    ``sock`` is this process's end of the socketpair whose other end the child
    inherited at ``exec``. Nothing else on the machine holds it, which is why
    possession of it is the whole authentication story (see the module
    docstring): the capability is handed over this descriptor and nowhere else.
    """

    def __init__(self, proc: "subprocess.Popen[bytes]", sock: socket.socket, root: Path) -> None:
        self.proc = proc
        self.sock = sock
        self.root = root
        #: Set once the child says it is warm. Kept as a cached answer so the
        #: engage path never blocks on a read that has not arrived yet.
        self.ready = False
        #: Set when an adoption SUCCEEDS, i.e. this standby became a runtime. The
        #: monitor thread reads it to tell "consumed" from "died and must be
        #: replaced".
        self.consumed = False

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        """Drop the descriptor. Never raises; called on every exit path."""
        try:
            self.sock.close()
        except OSError:
            pass


def enable_warming(root: "Path | None" = None) -> None:
    """Make this process a warmer, and warm one standby now, off the caller's thread.

    Called by the TUI and the ``serve`` daemon at their launch points, so the
    FIRST new conversation or cold switch already finds a standby. Never raises:
    a console that cannot warm simply spawns cold.
    """
    if disabled() or os.environ.get("LOP_MOBILE_CHILD_RESUME"):
        return
    try:
        from local_operator.paths import config_dir
        from local_operator.session.runtime.launch import _spawn_interpreter

        target = root if root is not None else config_dir()
        # A real directory path or nothing: a standby outlives its warmer, so a
        # root that is a test double (or a relative path resolved against some
        # later cwd) would leave a process bound to a directory nobody owns.
        if not isinstance(target, Path) or not target.is_absolute():
            return
        # The standby resolves its own root from the environment it inherits, and
        # a console has exactly one. An engage for a different root is declined
        # by ``try_adopt`` rather than served from a spare warmed for another
        # store.
        if Path(target) != config_dir():
            return
        interpreter = _spawn_interpreter()
    except Exception:  # noqa: BLE001 — a missing warm is a slower first engage
        logger.debug("could not resolve a standby target", exc_info=True)
        return
    _WARMING[0] = True
    warm_in_background(Path(target), interpreter)


def warm_in_background(root: Path, interpreter: str) -> None:
    """:func:`ensure_warm` on a daemon thread: never on an engage's critical path."""
    if not _WARMING[0]:
        return
    threading.Thread(
        target=ensure_warm, args=(root, interpreter), name="lop-standby-warm", daemon=True
    ).start()


#: How long a standby which exits without being adopted must have lived before a
#: replacement is warmed for it. The loop breaker: a root that has been deleted
#: (``root-gone``) makes a fresh standby exit immediately, and re-warming it
#: forever would be a fork storm.
REWARM_MIN_LIFE_S = 5.0

#: Delay before a replacement spawn, so an exit that repeats cannot spin.
REWARM_DELAY_S = 1.0

#: The least time between two monitor-driven re-warms. What it bounds: a host whose
#: tree is being edited (an editable checkout retires a standby as ``tree-moved``)
#: would otherwise re-warm after every retirement, each one a fork plus a warm.
#: An engage's own ``warm_in_background`` is not subject to it, because that one
#: follows a spawn the operator asked for.
REWARM_MIN_INTERVAL_S = 60.0

#: When the last monitor-driven re-warm happened. A list so it is mutated, not
#: rebound, matching ``_WARMING``'s shape.
_LAST_REWARM: list[float] = [0.0]


def _monitor(warm: _Standby) -> None:
    """Wait on a standby and, if it left WITHOUT being adopted, warm another.

    WHY THIS EXISTS (QA round 1, QW2). A standby retires itself whenever an input
    it must be current on moved — measured: a TUI rewrote ``config.yml`` 5.9 s
    after boot, which retired the standby as ``config-moved``. Without a monitor
    the replacement waits for the next engage, so the FIRST new conversation
    after a boot is exactly the one that goes cold: the warm would be spent on
    the wrong session. Here the console notices the exit when it happens and
    starts the replacement immediately, so the window is a warm rather than a
    whole cycle.

    A CONSUMED standby exits the same way and is not replaced here — the engage
    path's own ``warm_in_background`` after the spawn does that, with the
    interpreter the new session actually used.
    """
    started = time.monotonic()
    try:
        warm.proc.wait()
    except Exception:  # noqa: BLE001 - a monitor must never take a thread down
        return
    if warm.consumed:
        return
    with _LOCK:
        if _WARM[0] is not warm:
            return
        _WARM[0] = None
    if not _WARMING[0] or disabled():
        return
    if not warm.root.is_dir():
        # ``root-gone``: a replacement would exit for the same reason, so the
        # loop breaker here is the cause rather than a timer.
        return
    if time.monotonic() - started < REWARM_MIN_LIFE_S:
        logger.debug("standby for %s exited at once; not re-warming", warm.root)
        return
    now = time.monotonic()
    if now - _LAST_REWARM[0] < REWARM_MIN_INTERVAL_S:
        return
    _LAST_REWARM[0] = now
    time.sleep(REWARM_DELAY_S)
    try:
        from local_operator.session.runtime.launch import _spawn_interpreter

        ensure_warm(warm.root, _spawn_interpreter())
    except Exception:  # noqa: BLE001 - a missing replacement is a slower engage
        logger.debug("could not replace the standby for %s", warm.root, exc_info=True)


def _start_monitor(warm: _Standby) -> None:
    threading.Thread(target=_monitor, args=(warm,), name="lop-standby-monitor", daemon=True).start()


def ensure_warm(root: Path, interpreter: str) -> None:
    """Start a standby for ``root`` unless one is already alive. Never raises.

    Called from ``launch.engage_runtime`` after each spawn decision, so the NEXT
    cold engage finds one — never before the user's own engage, and never from a
    runtime child (a runtime carries the spawn contract in its environment, and a
    runtime that warmed spares would make every session a warmer).
    """
    if not _WARMING[0] or disabled() or os.environ.get("LOP_MOBILE_CHILD_RESUME"):
        return
    try:
        from local_operator.paths import config_dir

        if Path(root) != config_dir():
            return
        with _LOCK:
            current = _WARM[0]
            if current is not None and current.alive():
                return
            if current is not None:
                current.close()
            _WARM[0] = _spawn_standby(Path(root), interpreter)
            fresh = _WARM[0]
        # OUTSIDE the lock: the monitor immediately blocks in ``proc.wait()`` and
        # only takes the lock if the child leaves without being adopted.
        _start_monitor(fresh)
    except Exception:  # noqa: BLE001 — a missing warm is a slower next engage, never a failure
        logger.debug("could not warm a standby for %s", root, exc_info=True)


def _spawn_standby(root: Path, interpreter: str) -> "_Standby":
    """Fork the warming interpreter, handing it ONE end of a private socketpair.

    ``pass_fds`` is what makes the other end unreachable by anything else: it is
    the only descriptor that survives the child's ``exec`` (``close_fds`` closes
    every other one), and the console keeps its own end with ``O_CLOEXEC`` set so
    no later child of this process — a tool subprocess, an ``exec --background``
    worker — inherits it either.
    """
    from local_operator import procname
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.procstate import detached_popen_kwargs

    console_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    child_fd = child_end.fileno()
    env = dict(os.environ)
    for key in CONTRACT_KEYS + ("LOP_RUNTIME_ADOPT_SESSION",):
        env.pop(key, None)
    # Pinned, because the child resolves its root from the environment and a
    # warmer's own environment is the only one it should serve. Safe to pin: the
    # config dir is read at call time, not by the warm, and adoption replaces
    # this whole environment with the requester's anyway (``_apply_environment``).
    env[CONFIG_DIR_ENV] = str(root)
    # The label is the ANON session label with ``[standby]`` in place of
    # ``[session]`` and a placeholder id: same lengths as the real label, so the
    # adoption rename is in place.
    label = procname.LABEL_SESSION_ANON.replace("[session]", "[standby]")
    if interpreter != sys.executable:
        argv0, executable = procname.spawn_identity_for_interpreter(
            label, interpreter, id="--------"
        )
    else:
        argv0, executable = procname.spawn_identity(label, id="--------")
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [argv0, SAFE_PATH_FLAG, "-m", STANDBY_MODULE, STANDBY_FD_FLAG, str(child_fd)],
            executable=executable,
            env=env,
            # Detached stdio: the standby has no session to speak for yet, and its
            # own console is a full-screen app whose terminal a stray write would
            # paint over. On adoption ``_redirect_output`` points both at the
            # session's capture, which is where a runtime's own traceback must
            # land (see R1-2 in the module docstring of ``_become_runtime``).
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(child_fd,),
            **detached_popen_kwargs(),
        )
    except BaseException:
        # A failed spawn must not leak either end of the socketpair.
        child_end.close()
        console_end.close()
        raise
    child_end.close()
    return _Standby(proc, console_end, root)


#: What a standby may say before it is asked anything, as ONE byte (the console
#: reads it with a zero timeout, so the frame must arrive whole and mean something
#: on its own). ``READY`` is the normal case; ``FAILED`` says the warm raised, and
#: the console logs it and spawns cold.
_READY = b"R"
_FAILED = b"E"


class AdoptedRuntime:
    """The ``Popen``-shaped handle the engage loop expects, backed by a real child.

    It is a REAL ``Popen`` this process forked (through the standby), so ``pid``
    is the console's own child, and the console's own record of "I started the
    runtime behind this record" is telling the truth about it — and ``poll()``
    reports the child's actual status and reaps it, instead of the previous
    design's zombie probe guessing at a pid that was never ours.
    """

    #: ``launch`` and the engage loop read these to treat an adopted candidate
    #: exactly like a forked one.
    lop_adopted_standby = True

    def __init__(self, proc: "subprocess.Popen[bytes]", capture: Path | None) -> None:
        self._proc = proc
        self.lop_capture_path = capture

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> "int | None":
        return self._proc.poll()

    @property
    def returncode(self) -> "int | None":
        return self._proc.returncode


def adoption_possible() -> bool:
    """Whether a warmed standby for this process is ready to take a spawn.

    A PURE probe for the caller that must decide BEFORE it builds anything: the
    engage path mints a capability handoff for whichever route it takes, and a
    console with no standby (the common case — ``lop exec``, a test, the phone
    daemon) should not mint one it will not use. Never blocks: :func:`_ready`
    reads a byte that was already written when the warm finished.

    It can retire a standby whose warm failed, which is the only side effect and
    is the same one :func:`try_adopt` would have caused a moment later.
    """
    if disabled():
        return False
    with _LOCK:
        warm = _WARM[0]
    if warm is None:
        return False
    return _ready(warm)


def try_adopt(
    root: Path,
    interpreter: str,
    env: dict[str, str],
    capture: Path,
    cap_fd: "int | None",
) -> "AdoptedRuntime | None":
    """Hand a cold child's whole spawn to this console's standby, or ``None``.

    ``env`` is EXACTLY the environment ``launch._spawn_runtime`` would have given
    the cold child, and ``cap_fd`` is the child end of the operator-capability
    handoff, passed by ``SCM_RIGHTS`` so the value travels on a descriptor exactly
    as it does into a cold child (``harness/approval.py``): the adopted runtime
    reads it with the same ``--operator-fd`` reader, and the capability never
    touches argv, the environment or a file.

    WHO MAY ADOPT, and why there is no check here to look for: only this process
    holds the descriptor this request is written to (see the module docstring).
    The value reaches a process only through this process's own fork, so there is
    no peer to authenticate — and nothing to steal the capability with.

    Never raises: a standby that is not ready, one that declined, a torn reply,
    or a wedged one that misses :data:`ADOPT_TIMEOUT_S` all answer ``None``, and
    the caller does exactly what it did before this module existed.
    """
    if disabled():
        return None
    with _LOCK:
        warm = _WARM[0]
        if warm is None:
            return None
        if not warm.alive():
            warm.close()
            _WARM[0] = None
            return None
    if Path(root) != warm.root:
        return None
    if not _ready(warm):
        return None
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = ""
    request = {
        "op": "adopt",
        "root": str(root),
        "interpreter": str(interpreter),
        "env": dict(env),
        "cwd": cwd,
        "capture": str(capture),
        "has_cap_fd": cap_fd is not None,
    }
    adopted: "AdoptedRuntime | None" = None
    try:
        warm.sock.settimeout(ADOPT_TIMEOUT_S)
        # ONE MESSAGE: the frame and the capability descriptor together, so the
        # child's first read gets the length prefix and the fd in the same
        # ``recvmsg`` (see :func:`_recv_request`). A separate marker byte would be
        # indistinguishable from the start of that prefix.
        body = json.dumps(request).encode("utf-8")
        payload = len(body).to_bytes(4, "big") + body
        if cap_fd is not None:
            socket.send_fds(warm.sock, [payload], [cap_fd])
        else:
            socket.send_fds(warm.sock, [payload], [])
        reply = _recv(warm.sock)
        if isinstance(reply, dict) and reply.get("ok"):
            adopted = AdoptedRuntime(warm.proc, capture)
    except (OSError, ValueError):
        logger.debug("standby adoption failed; spawning cold", exc_info=True)
    finally:
        # The standby served its one session (or is no longer usable): drop the
        # descriptor and stop tracking it, so the next engage warms a replacement
        # with a FRESH handoff rather than reusing this one.
        with _LOCK:
            if _WARM[0] is warm:
                warm.consumed = True
                _WARM[0] = None
        warm.close()
        if adopted is None:
            _retire(warm)
    return adopted


def _ready(warm: _Standby) -> bool:
    """Whether the child has said it is warm. Cached, and never blocks past now.

    A zero timeout, because this runs on the engage path: a standby whose warm is
    still in flight must cost the engage nothing at all. The byte was written
    when the warm finished, so a warm standby's answer is already in the socket
    buffer and this read returns immediately.
    """
    if warm.ready:
        return True
    try:
        warm.sock.settimeout(0)
        answer = warm.sock.recv(1)
    except (BlockingIOError, TimeoutError, OSError):
        return False
    if answer == _READY:
        warm.ready = True
        return True
    if answer == _FAILED:
        logger.warning("the runtime standby could not warm; new sessions spawn cold")
    elif answer:
        logger.debug("unexpected standby handshake %r", answer)
    with _LOCK:
        if _WARM[0] is warm:
            _WARM[0] = None
    warm.close()
    _retire(warm)
    return False


def _retire(warm: _Standby) -> None:
    """End a standby that cannot serve, by EXACT pid of a child this process owns.

    Never by name: this fleet runs ~25 agents whose own children carry similar
    argv, and an unscoped kill has already taken out another session's process
    tree once. SIGTERM first because that is what a standby expects; SIGKILL only
    if it ignores one, and only for the pid this process forked.
    """
    try:
        if warm.proc.poll() is None:
            warm.proc.terminate()
            try:
                warm.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                warm.proc.kill()
    except Exception:  # noqa: BLE001 - a retire is best-effort BY CONTRACT
        # Not just OSError, and not fatal: this runs on the engage path (a
        # refused or timed-out adoption) and on the console's way out. A standby
        # that cannot be signalled is still harmless - it holds one end of a
        # socketpair nothing else has, and it exits on its own when this console
        # closes the other end, when its root goes away, or on idle.
        logger.debug("could not end standby %s", getattr(warm, "pid", "?"), exc_info=True)


def reset_for_tests() -> None:
    """End the tracked standby and clear the flags. Tests only — production never calls it."""
    with _LOCK:
        warm = _WARM[0]
        _WARM[0] = None
        _WARMING[0] = False
        _LAST_REWARM[0] = 0.0
    if warm is not None:
        # ``consumed`` so the monitor does not start a replacement for a standby
        # this call ended on purpose.
        warm.consumed = True
        warm.close()
        _retire(warm)


# ---------------------------------------------------------------------------
# Wire: one length-prefixed JSON object each way, over the private descriptor
# ---------------------------------------------------------------------------


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    sock.sendall(len(body).to_bytes(4, "big") + body)


def _recv(sock: socket.socket) -> Any:
    header = _read_exact(sock, 4)
    size = int.from_bytes(header, "big")
    if size > 1 << 20:
        raise ValueError("standby frame too large")
    return json.loads(_read_exact(sock, size).decode("utf-8"))


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ValueError("standby connection closed mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


# ---------------------------------------------------------------------------
# The standby side
# ---------------------------------------------------------------------------


def _loaded_tree_stamp() -> dict[str, int]:
    """``{file: mtime_ns}`` for every loaded ``local_operator`` module file."""
    stamp: dict[str, int] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("local_operator"):
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            stamp[path] = os.stat(path).st_mtime_ns
        except OSError:
            stamp[path] = -1
    return stamp


def _tree_moved(recorded: dict[str, int]) -> bool:
    """Whether any file the warm IMPORTED has changed since it was imported.

    The recorded files only, re-stat'ed: a module first imported AFTER the warm
    (the guard's own lazy imports, a timer's) is not a change to what was warmed,
    and comparing whole stamps read that as a moved tree — measured: the first
    idle check retired every standby with ``tree-moved`` on an untouched checkout.
    """
    for path, mtime in recorded.items():
        try:
            now = os.stat(path).st_mtime_ns
        except OSError:
            now = -1
        if now != mtime:
            return True
    return False


class _Warmth:
    """What the standby recorded when it finished warming: the adoption guards.

    Two different questions, answered separately because they want different
    reactions:

    * :meth:`stale` — has anything this interpreter's imports depend on moved
      since the warm? If so the standby is worthless to EVERY requester, so it
      EXITS (the console warms a replacement against what is current).
    * the per-request checks in :func:`_await_request` — is this requester one a
      cold child of THIS interpreter would serve identically? If not, the standby
      DECLINES and keeps waiting, because exiting on a mismatch would end a spare
      whose console merely engaged from another directory.
    """

    def __init__(self, root: Path) -> None:
        from local_operator import update
        from local_operator.config import CONFIG_FILE_NAME, config_file_key

        self.root = root
        self.config_file = root / CONFIG_FILE_NAME
        self.config_key = config_file_key(self.config_file)
        self.build = update.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None)
        self.tree = _loaded_tree_stamp()
        self.venv = os.path.realpath(sys.prefix)

    def stale(self) -> str:
        """Why no requester may have this standby any more, or ``""``."""
        from local_operator import update
        from local_operator.config import config_file_key

        if not self.root.is_dir():
            # A deleted root (a finished test rig, a removed sandbox): nothing
            # will ever adopt here again, and the process must not outlive it.
            return "root-gone"
        # THE GENERATION. ``lop-update`` installs a new generation beside this
        # one and flips ``current``; this interpreter's own files do not change,
        # so neither the tree stamp nor ``installed_build`` of its own prefix
        # would notice. What a cold spawn would run now is the question.
        current = update.current_interpreter()
        if current is not None and _venv_of(str(current)) != self.venv:
            return "generation-moved"
        if update.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None) != self.build:
            return "build-moved"
        if _tree_moved(self.tree):
            return "tree-moved"
        if config_file_key(self.config_file) != self.config_key:
            return "config-moved"
        return ""


def _rename_argv(old: bytes, new: bytes) -> bool:
    """Rewrite one argv word IN PLACE, in the memory ``ps`` reads it from.

    WHY. The residency census identifies a runtime by the ``-m RUNTIME_MODULE``
    word in its argv (``reclaim.parse_process_row``). An adopted standby IS a
    runtime and must be counted — the sweep's protection of attached runtimes and
    its orphan reclaim both depend on the census — but it was exec'd as
    ``-m ...standby``. macOS's ``ps`` reads the argument area of the process
    (``KERN_PROCARGS2``) and Linux reads ``/proc/<pid>/cmdline``, both backed by
    the same memory ``_NSGetArgv``/``argv`` point at, so an equal-length
    overwrite is visible to every reader (verified with ``ps -ww`` on this host).
    ``sys.orig_argv`` keeps the original, so nothing in-process changes.

    Returns False when the platform offers no way to find argv; the caller then
    refuses adoption rather than leaving an uncountable runtime.
    """
    if len(old) != len(new):
        return False
    try:
        if sys.platform == "darwin":
            libc = ctypes.CDLL(None)
            libc._NSGetArgc.restype = ctypes.POINTER(ctypes.c_int)
            libc._NSGetArgv.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            argc = libc._NSGetArgc().contents.value
            argv = libc._NSGetArgv().contents
            addresses = [argv[i] for i in range(argc)]
        elif sys.platform.startswith("linux"):
            return _rename_argv_linux(old, new)
        else:
            return False
        renamed = False
        for address in addresses:
            if not address:
                continue
            word = ctypes.string_at(address)
            at = word.find(old)
            if at >= 0:
                ctypes.memmove(address + at, new, len(new))
                renamed = True
        return renamed
    except Exception:  # noqa: BLE001 — no argv access means "cannot rename"
        logger.debug("argv rename unavailable", exc_info=True)
        return False


def _rename_argv_linux(old: bytes, new: bytes) -> bool:
    """Linux: argv is the contiguous region ``/proc/self/stat`` fields 48-49 name."""
    with open("/proc/self/stat", "rb") as handle:
        fields = handle.read().rsplit(b")", 1)[1].split()
    start, end = int(fields[45]), int(fields[46])
    with open("/proc/self/mem", "r+b", buffering=0) as mem:
        mem.seek(start)
        area = mem.read(end - start)
        at = area.find(old)
        if at < 0:
            return False
        mem.seek(start + at)
        mem.write(new)
    return True


def _standby_fd_from_argv(argv: "list[str]") -> "int | None":
    """The inherited descriptor number, or ``None`` when this is not a standby.

    A NUMBER, not a value: the descriptor is the secret, and a number that names
    nothing in another process's table is useless there.
    """
    for index, item in enumerate(argv):
        if item == STANDBY_FD_FLAG and index + 1 < len(argv):
            candidate = argv[index + 1]
            break
        if item.startswith(f"{STANDBY_FD_FLAG}="):
            candidate = item.partition("=")[2]
            break
    else:
        return None
    return int(candidate) if candidate.lstrip("-").isdigit() else None


def _recv_request(sock: socket.socket) -> "tuple[dict[str, Any], list[int]]":
    """One request frame AND the descriptors that came with it.

    THE DESCRIPTOR RIDES THE SAME MESSAGE AS THE FRAME'S FIRST BYTES, not a second
    message with a marker byte: a byte sent alongside the ancillary data is
    indistinguishable from the start of the frame's length prefix, so the reader
    saw a torn frame (found by this change's own round-1 tests, which is what the
    tests are for). The kernel attaches an ``SCM_RIGHTS`` set to the first byte of
    the message it was sent with, so reading the 4-byte header with ``recv_fds``
    gets both, and the body follows with a plain read.

    Raises ``ValueError`` on EOF — the console having gone away, which the caller
    reads as this process's exit signal.
    """
    header = b""
    fds: list[int] = []
    while len(header) < 4:
        chunk, received, _flags, _addr = socket.recv_fds(
            sock, 4 - len(header), max(1, 4 - len(fds))
        )
        if not chunk:
            for fd in fds:
                os.close(fd)
            raise ValueError("standby channel closed")
        header += chunk
        fds.extend(received)
    size = int.from_bytes(header, "big")
    if size > 1 << 20:
        raise ValueError("standby frame too large")
    body = b""
    while len(body) < size:
        chunk = sock.recv(size - len(body))
        if not chunk:
            raise ValueError("standby channel closed")
        body += chunk
    return json.loads(body.decode("utf-8")), fds


def _close_all(fds: "list[int]") -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _await_request(sock: socket.socket) -> "dict[str, Any] | None":
    """Warm, then wait for an adoption this standby may take.

    Returns the request to serve, or ``None`` when the standby should leave
    quietly (idle, its console gone, or its build/root/config moved). Raises only
    for a genuine programming error: every condition a normal machine produces is
    handled here, because a standby that dies noisily would be a worse citizen
    than one that simply is not there.
    """
    from local_operator.paths import config_dir

    root = config_dir()
    # Lowest scheduling class while warming: the warm is speculative work and must
    # never take CPU from a session that is doing real work on this host.
    _background_priority(True)
    try:
        _warm()
    except BaseException:  # noqa: BLE001 — a failed warm means "no standby"
        logger.debug("standby warm failed", exc_info=True)
        _background_priority(False)
        _send_byte(sock, _FAILED)
        return None
    _background_priority(False)
    # The guard snapshot is taken at NORMAL priority: it is a handful of stats and
    # two small file reads, and in the background band a host at load 100+ starved
    # it for over a minute after the imports had already finished.
    warmth = _Warmth(root)
    warm_env = _warm_sensitive(os.environ)
    _send_byte(sock, _READY)
    deadline = time.monotonic() + IDLE_REAP_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        sock.settimeout(min(remaining, 30.0))
        try:
            request, fds = _recv_request(sock)
        except (TimeoutError, socket.timeout):
            # A stale warm is worth nothing: checked between waits too, so a
            # standby does not sit on a superseded build for the whole idle
            # window after a ``lop-update``.
            if warmth.stale():
                return None
            continue
        except (OSError, ValueError):
            # EOF here is the console having gone away — the descriptor it held
            # was the only one, so its close is this process's exit signal.
            return None
        if not isinstance(request, dict) or request.get("op") != "adopt":
            _close_all(fds)
            continue
        reason, retire = _refusal(warmth, warm_env, request)
        if reason:
            _close_all(fds)
            # Sent even on the way out: the console logs the reason, and an
            # unexplained EOF is what a retirement used to look like.
            _refuse(sock, reason)
            if retire:
                return None
            continue
        if request.get("has_cap_fd") and not fds:
            _refuse(sock, "capability-descriptor-missing")
            continue
        cap_fd = fds[0] if fds else -1
        _close_all(fds[1:])
        # LAST, and after every check that can decline: this is the one step that
        # cannot be undone, and it is taken only for a request this standby is
        # actually going to serve.
        if not _rename_argv(STANDBY_MODULE.encode(), _runtime_module().encode()):
            # No way to make this process countable by the census: leaving it as a
            # runtime would hide it from the sweep, which is worse than no spare.
            _refuse(sock, "argv-rename-unavailable")
            return None
        return {**request, "cap_fd": cap_fd}


def _refusal(
    warmth: _Warmth, warm_env: dict[str, str], request: dict[str, Any]
) -> "tuple[str, bool]":
    """Why this standby must not take this request, or ``("", False)``.

    Returns ``(reason, retire)``. ``retire`` means the standby is worthless to
    EVERYONE (an input it warmed against moved) so it should exit; a reason with
    ``retire`` False is a decline that leaves it waiting, because it is still the
    right spare for the host that warmed it — a TUI and the desktop daemon
    legitimately differ in environment, and a spare that exited on every mismatch
    would be killed by whichever of the two engaged next.

    The reason is SENT to the console either way, so a retirement appears in that
    host's log as ``config-moved`` rather than as an unexplained EOF.
    """
    stale = warmth.stale()
    if stale:
        return stale, True
    if str(request.get("root") or "") != str(warmth.root):
        return "other-root", False
    if _venv_of(str(request.get("interpreter") or "")) != warmth.venv:
        return "other-venv", False
    env = request.get("env")
    env = env if isinstance(env, dict) else {}
    if _warm_sensitive(env) != warm_env:
        # The requester's interpreter settings, locale, home or build prefix
        # differ from the ones this process imported under: only a fresh
        # interpreter is equivalent.
        return "other-environment", False
    # The one step that cannot be undone is taken by the CALLER, as the last
    # precondition before it commits: this function only answers questions.
    return "", False


def _refuse(sock: socket.socket, reason: str) -> None:
    try:
        _send(sock, {"ok": False, "reason": reason})
    except OSError:
        pass


def _send_byte(sock: socket.socket, byte: bytes) -> None:
    """One byte, in one write: the handshake the console reads with a zero timeout."""
    try:
        sock.sendall(byte)
    except OSError:
        pass


def _commit_adoption(sock: socket.socket, request: dict[str, Any]) -> int:
    """Become the runtime's process: argv, name, environment, boot record, reply.

    Called OUTSIDE the caller's error handling, deliberately (agent review round
    1, R1-2). Everything up to the previous line is a standby that may fail
    quietly; from here this process is the session's runtime, and a failure must
    be as loud as it is in a cold child — non-zero exit and a traceback on stderr,
    which by then is the session's capture file that ``engage_runtime`` reads for
    its actionable startup reason.

    Returns the descriptor number carrying the capability (``-1`` for none).
    """
    env = request.get("env")
    env = env if isinstance(env, dict) else {}
    # The module word was renamed by ``_await_request`` as the last precondition;
    # these two complete the row a reader of ``ps`` sees.
    _rename_argv(b"[standby]", b"[session]")
    _rename_argv(b"id=--------", _label_id(env).encode())
    cap_fd = int(request.get("cap_fd", -1))
    _apply_environment(env)
    _change_directory(str(request.get("cwd") or ""))
    _redirect_output(str(request.get("capture") or ""))
    # BEFORE the session is constructed, so the sweep can see this runtime's
    # session and its true age for the whole construction window (R1-3). The
    # runtime rewrites this same record at its own boot boundary.
    _write_adoption_boot_record(env)
    _send(sock, {"ok": True, "pid": os.getpid()})
    # The private descriptor has served its purpose, and the runtime must not hold
    # one nothing will ever write to again: close it before any tool subprocess
    # could be started.
    try:
        sock.close()
    except OSError:
        pass
    return cap_fd


def _write_adoption_boot_record(env: dict[str, Any]) -> None:
    """Publish this process's boot record now, before it constructs anything.

    ``run/host``'s record is the artifact ``reclaim`` and the roster already
    prefer for attribution, so writing it one step earlier is what makes an
    adopted runtime visible to them exactly as a forked one is. Best-effort, for
    the reason ``journal.write_boot_record``'s own contract gives: a runtime whose
    boot record cannot be written must still run its turns.
    """
    session_id = str(env.get("LOP_MOBILE_CHILD_RESUME") or "")
    if not session_id:
        return
    try:
        from local_operator import update
        from local_operator.session.runtime.journal import write_boot_record

        build = update.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None)
        write_boot_record(session_id, build, cwd=str(env.get("LOP_MOBILE_CHILD_CWD") or ""))
    except Exception:  # noqa: BLE001 — instrumentation, never a boot gate
        logger.debug("adopted runtime could not write its boot record", exc_info=True)


def _runtime_module() -> str:
    from local_operator.session.runtime.types import RUNTIME_MODULE

    return RUNTIME_MODULE


WARM_BAND_MIN_WALL_S = 20.0
WARM_BAND_MAX_S = 45.0
WARM_BAND_MIN_CPU_RATIO = 0.02


def _background_priority(on: bool) -> None:
    """Darwin's background band while warming; normal again before serving.

    PRIO_DARWIN_BG throttles CPU and I/O. Reset before the channel is answered, so
    an adopted standby constructs the session at the same priority a cold child
    would. A platform without it keeps its normal priority, which only means the
    warm competes as a cold spawn would have.
    """
    which = getattr(os, "PRIO_DARWIN_PROCESS", None)
    band = getattr(os, "PRIO_DARWIN_BG", None)
    if which is None or band is None:
        return
    try:
        os.setpriority(which, 0, band if on else 0)
    except OSError:
        logger.debug("could not change the standby's scheduling band", exc_info=True)


def _warm() -> None:
    """Import what a runtime child and a first session construction import.

    Exactly the work a cold child pays before it can publish, minus anything that
    depends on the session: nothing here opens the store, reads a transcript,
    takes a lease or starts a background pass (verified with an audit hook over
    the whole warm: zero opens or listings under the config root). The tokenizer
    rides along for the reason ``warm_session_imports`` gives.

    THE BAND IS ABANDONED IF IT STARVES THE WARM (QA round 1, QW1). The caller
    starts this process in Darwin's background band, which is the polite choice (a
    speculative spare must not compete with the sessions already serving). But
    the band is not a bound: measured at load 180-227, eight standbys spent 16
    MINUTES of wall on 0.25-0.29 s of CPU each, all of them runnable — so
    "available about a minute after boot" was false on exactly the host this
    exists for, and the feature quietly failed open to a cold spawn. The same
    warm at normal priority takes 15-42 s there.

    So progress is measured, not assumed: before each import, if the warm has
    been running for at least :data:`WARM_BAND_MIN_WALL_S` and has either spent
    under :data:`WARM_BAND_MIN_CPU_RATIO` of that wall on CPU or run past
    :data:`WARM_BAND_MAX_S`, the band is dropped FOR GOOD and the rest of the warm
    runs at normal priority. That gives a ceiling instead of an unbounded wait
    while keeping the polite behaviour on a host with room to spare.
    """
    import importlib

    started_wall = time.monotonic()
    started_cpu = time.process_time()

    def _leave_the_band_if_starved() -> None:
        elapsed = time.monotonic() - started_wall
        if elapsed < WARM_BAND_MIN_WALL_S:
            return
        spent = time.process_time() - started_cpu
        if elapsed >= WARM_BAND_MAX_S or spent / elapsed < WARM_BAND_MIN_CPU_RATIO:
            _background_priority(False)

    # NOT ``local_operator.session.runtime.process`` itself: adoption runs that
    # module as ``__main__`` (see :func:`_become_runtime`), exactly as ``python
    # -m`` does in a cold child, and ``runpy`` warns when the module it is about
    # to run is already imported. Its own top-level imports are warmed by name.
    _leave_the_band_if_starved()
    import local_operator.session.runtime.server  # noqa: F401
    import local_operator.session.runtime.serving  # noqa: F401
    import local_operator.session.runtime.stall_watchdog  # noqa: F401
    from local_operator.session_factory import _WARM_IMPORTS

    # ``_WARM_IMPORTS`` directly rather than ``warm_session_imports()``: that
    # helper also starts the BYTECODE warm, a subprocess that belongs to a
    # long-lived host (``bytecode.warm_bytecode_cache_in_background``'s own
    # docstring says a runtime child must not start it).
    for name in _WARM_IMPORTS + _WARM_EXTRA:
        _leave_the_band_if_starved()
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — a warm-up must never be the failure
            logger.debug("standby warm skipped %s", name, exc_info=True)
    try:
        from local_operator.compaction.tokens import warm_tokenizer

        _leave_the_band_if_starved()
        warm_tokenizer()
    except Exception:  # noqa: BLE001
        logger.debug("standby tokenizer warm skipped", exc_info=True)


def _label_id(env: dict[str, Any]) -> str:
    """``id=<first 8 of the session>``, padded to the placeholder's width.

    The same 8 characters ``launch._spawn_runtime`` puts in a cold child's label,
    so ``ps`` and Activity Monitor show an adopted runtime exactly as they show a
    cold one. Padded rather than truncated because the rename is in place and must
    keep the length.
    """
    ident = str(env.get("LOP_MOBILE_CHILD_RESUME") or "")[:8]
    return ("id=" + ident).ljust(len("id=--------"), "-")


def _apply_environment(env: dict[str, Any]) -> None:
    """Replace this process's environment with the one a cold child would get.

    WHOLE, not merged: a cold child inherits exactly ``env`` and nothing else, and
    an adopted standby must be indistinguishable from it — a key the warmer had
    and the requester does not would otherwise survive into the session (the
    ``LOP_*`` leakage AGENTS.md "Isolating a run" documents). Warm-sensitive names
    already matched (see ``_WARM_SENSITIVE_NAMES``), so nothing replaced here was
    read by an import that already ran.

    THE ``ps -E`` VIEW IS NOT REPAIRED HERE, and cannot be: ``os.environ`` and
    ``putenv`` do not rewrite the argument area ``ps -Eww`` reads. That is why the
    boot record is written at adoption (R1-3): reclaim and the roster read the
    record for attribution instead of the process's environment.
    """
    wanted = {str(k): str(v) for k, v in env.items() if isinstance(v, str)}
    for key in list(os.environ):
        if key not in wanted:
            del os.environ[key]
    for key, value in wanted.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def _change_directory(cwd: str) -> None:
    """The requester's working directory, which a cold child inherits.

    ``launch._spawn_runtime`` passes no ``cwd=``, so the cold child starts where
    the engaging host stands; the session's OWN directory rides the contract
    (``LOP_MOBILE_CHILD_CWD``) and is what the session uses. Best-effort: a
    vanished directory leaves the standby where it was, which only matters to code
    that reads ``os.getcwd()`` before the session applies its own.
    """
    if not cwd:
        return
    try:
        os.chdir(cwd)
    except OSError:
        logger.debug("could not enter the requester's directory", exc_info=True)


def _redirect_output(capture: str) -> None:
    """Point stdout/stderr at the caller's capture file, as a cold spawn does.

    ``engage_runtime`` reads that file to report WHY a candidate died (the
    actionable startup reasons), so an adopted standby must write its construction
    failures to the same place a cold child would — and so must a runtime that
    crashes hours later.
    """
    if not capture:
        return
    try:
        # ``O_CREAT`` as well as append, at the mode the spawn path's ``mkstemp``
        # makes: the capture is this process's log, and a runtime whose traceback
        # went nowhere because the file was missing would be the R1-2 defect in a
        # second shape.
        fd = os.open(capture, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    except OSError:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


def _become_runtime(operator_fd: int) -> int:
    """Run the runtime module as ``__main__``: from here on, a cold child.

    Called OUTSIDE the standby's error handling, on purpose: this is the point
    after which a failure belongs to the session rather than to a speculative
    spare, so the exception propagates and the process exits NON-ZERO with its
    traceback in the capture file — which is what lets the console report
    "connect a provider" instead of a generic error after three retries (agent
    review round 1, R1-2). The earlier revision caught it here and returned 0 with
    an empty capture, so every adopted failure read as a success.

    ``runpy.run_module(..., run_name="__main__")`` is the machinery ``python -m``
    itself uses, so the runtime's ``__main__`` guard runs unchanged — brand, ARM
    THE STALL BOUND, ``main()`` — and ``stall_watchdog.arm`` keeps its one call
    site (pinned by ``test_the_only_arm_site_is_the_runtime_entry_point``). A
    standby never arms it: the warm is not a session.

    The operator capability arrives exactly as it does for a cold child — as the
    NUMBER of a descriptor in ``sys.argv`` (``--operator-fd <n>``), read and
    closed by the capability reader in ``harness/approval.py``, from
    ``process.main``. The descriptor is the one ``SCM_RIGHTS`` delivered over this
    console's private channel, i.e. the child end of the console's own handoff.

    ``SystemExit`` from the guard's ``sys.exit(main())`` propagates to this
    process's own exit, as it would in a cold child.
    """
    import runpy

    from local_operator.harness.approval import OPERATOR_FD_FLAG

    sys.argv = [sys.argv[0]] + (
        [OPERATOR_FD_FLAG, str(operator_fd)] if operator_fd is not None and operator_fd >= 0 else []
    )
    runpy.run_module(_runtime_module(), run_name="__main__", alter_sys=True)
    return 0


def main() -> int:
    """The standby entry point: wait, then become a runtime or leave quietly."""
    from local_operator import procname

    procname.brand_this_process()
    fd = _standby_fd_from_argv(sys.argv[1:])
    if fd is None:
        logger.debug("standby started without an inherited descriptor; nothing to do")
        return 0
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)
    try:
        request = _await_request(sock)
    except Exception:  # noqa: BLE001 — before adoption a failure is just "no standby"
        logger.debug("standby exited on an error while waiting", exc_info=True)
        return 0
    if request is None:
        return 0
    # PAST THIS LINE A FAILURE IS THE SESSION'S, not the spare's: no handler here,
    # so the exception reaches stderr (the capture) and the exit status is non-zero.
    operator_fd = _commit_adoption(sock, request)
    return _become_runtime(operator_fd)


if __name__ == "__main__":
    sys.exit(main())
