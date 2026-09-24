"""One pre-imported runtime interpreter per config root, adopted by the next cold engage.

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
processes. Wall time is therefore ~10x CPU on every path, and the only way to
move a number the user feels is to remove CPU from the critical path, not to
reorder it. Import pruning cannot do that: session construction needs all but
a handful of the ~450 modules it loads (the greedy cover in the PR evidence).

So the imports are paid BEFORE anyone asks: one standby interpreter per config
root imports the whole runtime graph and waits. The next cold engage for that
root hands it the session's spawn contract over a unix socket instead of
forking a new interpreter; the standby becomes an ordinary runtime (same module,
same ``amain``, same lease arbitration) and a fresh standby is warmed behind it.

WHAT A STANDBY IS NOT — THE OPERATOR'S CONSTRAINTS, AND HOW EACH IS HELD
=======================================================================
(a) ONE PER CONFIG ROOT, MACHINE-WIDE, not one per host: ~20 TUIs run at once
    here, and a spare per TUI would be ~20 idle interpreters. A ``flock`` on
    ``run/standby/lock`` inside the root is held for the standby's whole life by
    the standby ITSELF — so the lock is released by the kernel when it exits or
    is killed, and a second warmer finds it held and does nothing. A host that
    finds no ready standby spawns cold exactly as before: the standby is an
    optimisation, never a dependency.

(b) NEVER A STALE BUILD, NEVER A STALE CONFIG. Adoption is refused — the
    standby exits and the caller spawns cold — when any of these moved since the
    standby finished warming:

    * the interpreter the caller would spawn (``launch._spawn_interpreter``,
      i.e. the CURRENT generation after a ``lop-update``) is not the one the
      standby runs;
    * ``update.installed_build`` (version + ``.lop-source`` ref) differs;
    * any loaded ``local_operator`` module file's mtime moved (a same-path
      rebuild, an editable checkout edited under it — measured at 0.2 ms for
      the ~140 files, so it is checked on every adoption rather than sampled);
    * ``config.yml``'s stat key (``config.config_file_key``) differs from the
      one recorded at warm, for the operator's explicit concern that a spare
      warmed under one configuration must not construct a session under another.

    The standby also exits on its own after :data:`IDLE_REAP_S` so a machine
    that stops opening conversations does not keep an interpreter forever.

(c) INVISIBLE AS A SESSION. The standby publishes nothing a session reader
    lists until it is adopted: no record in ``run/mobile`` (so not in ``lop
    sessions``, the desktop feed or the mobile list), no boot record in
    ``run/host``, no lease, no transcript, no analytics row, and it never runs
    ``create_session`` — so it never takes the store-maintenance or analytics
    passes, which are started from session construction. It does not arm the
    stall watchdog (``stall_watchdog.arm`` has exactly one call site, the
    runtime ``__main__`` guard, and a standby does not reach it until adopted).
    Its argv is ``-m local_operator.session.runtime.standby``, which the
    residency census (``reclaim.parse_process_row``) matches as a whole word
    against ``RUNTIME_MODULE`` and therefore does NOT count as a runtime.

    ON ADOPTION it must become countable, because it is then a runtime like any
    other and the sweep's orphan protection has to see it. The module word is
    rewritten IN PLACE in the process's own argv memory (``_rename_argv``):
    ``...runtime.standby`` and ``...runtime.process`` are the same length, and
    macOS/Linux ``ps`` read argv straight from that memory — verified on this
    host, ``ps -ww`` shows the rewritten word. The label's ``[standby]`` becomes
    ``[session]`` the same way. Where the rewrite is not possible the adoption is
    refused rather than leaving an uncountable runtime behind.

COST, STATED: one interpreter per config root that has an interactive host, at
~135-150 MB max RSS measured (the same as a runtime child right after
construction, less the session), idle at 0% CPU in a blocking ``accept``.

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
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The module a standby runs as. SAME LENGTH as ``types.RUNTIME_MODULE`` on
#: purpose: adoption rewrites this word in place (see ``_rename_argv``), and an
#: in-place rewrite cannot grow the string. ``test_standby`` pins the lengths.
STANDBY_MODULE = "local_operator.session.runtime.standby"

#: The directory, inside a config root's ``run`` tree, that holds the lock and
#: the socket's rendezvous file. Not ``run/mobile`` or ``run/host``: every
#: session reader scans those, and a standby must be in none of them.
STANDBY_DIRNAME = "run/standby"

#: How long a standby waits for an adoption before exiting on its own. Long
#: enough that an operator who opens conversations every few minutes always
#: finds one warm; short enough that a machine that stops opening them gives
#: the memory back within the quarter hour.
IDLE_REAP_S = 900.0

#: How long the engage side waits for a standby's answer before giving up and
#: spawning cold. A healthy standby answers in single-digit ms (it is parked in
#: ``accept``); the bound exists for one that is wedged, so a wedged standby can
#: cost at most this on top of today's cold spawn.
ADOPT_TIMEOUT_S = 2.0

#: Environment switch that turns the whole mechanism off — no warm and no
#: adoption — for an operator who wants the cold spawn exactly. The test suite
#: does not need it: warming is opt-in per process (:func:`enable_warming`, called
#: only by the TUI and desktop-daemon boot paths), so a test that engages a
#: runtime never leaves a warmed interpreter behind unless it asked for one.
DISABLE_ENV = "LOP_RUNTIME_STANDBY_DISABLED"

#: The keys of the spawn contract a standby is allowed to receive. The same
#: names ``launch._spawn_runtime`` writes into a cold child's environment, and
#: nothing else: a standby must end up in exactly the state a cold child starts
#: in, so the adoption request is applied as those keys and only those.
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
#: cover over the 199 modules measured, 76 of them ours). Warming them is what
#: makes the post-adoption construction pay ~180 ms of CPU instead of ~400.
#: A module missing here is not a correctness problem — it is imported on
#: first use exactly as before — only a slower adoption.
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

#: sockaddr_un.sun_path is 104 bytes on macOS and bind() fails past 103 — the
#: same bound ``secrets.protocol.MAX_SOCKET_PATH`` measured.
_MAX_SOCKET_PATH = 103


#: Environment names a WARMED interpreter may already have acted on, so a
#: requester whose values differ must get a cold child instead. Measured, not
#: guessed: an ``os.environ`` read hook over the whole warm recorded exactly
#: ``HOME``, ``LANG``/``LANGUAGE``/``LC_*``, ``ARCHFLAGS``, ``OTEL_*``,
#: ``PYDANTIC_DISABLE_PLUGINS``, ``PYTHON*``/``_PYTHON*`` — and nothing else;
#: every product variable (``LOCAL_OPERATOR_*``, ``LOP_*``) is read at CALL time,
#: after adoption has installed the requester's environment. ``TMPDIR``/``TZ``
#: are added because ``tempfile`` and ``time`` cache them on first use, and
#: ``LOP_BUILD_PREFIX`` because the warm's build stamp is read through it.
#: Every other name (``PATH``, API keys, a terminal's variables, the desktop
#: token) is simply replaced by the requester's value before the runtime starts,
#: which is exactly what a cold child would have inherited. Keeping product
#: prefixes OUT of this set is what lets one standby serve both a TUI and the
#: desktop daemon, whose environments differ in exactly those names.
_WARM_SENSITIVE_NAMES = frozenset(
    {"HOME", "LANG", "LANGUAGE", "ARCHFLAGS", "TMPDIR", "TZ", "LOP_BUILD_PREFIX"}
)
_WARM_SENSITIVE_PREFIXES = ("LC_", "OTEL_", "PYDANTIC_", "PYTHON", "_PYTHON")


def _venv_of(interpreter: str) -> str:
    """The environment an interpreter path runs in, comparable across spellings.

    NOT ``realpath(interpreter)``: a standby is exec'd through the branded
    hardlink beside the interpreter (``procname.spawn_identity``), so its
    ``sys.executable`` is ``<venv>/bin/Local Operator`` while the requester
    names ``<venv>/bin/python3`` — and a venv's ``python`` is itself a symlink
    to the base interpreter, which every venv on the machine shares. The venv
    directory is the identity that decides which build is imported.
    """
    return os.path.realpath(Path(interpreter).parent.parent)


def disabled() -> bool:
    """Off by switch, and off where the adoption channel does not exist.

    The operator capability rides to the standby as a passed descriptor
    (``socket.send_fds``, POSIX ``SCM_RIGHTS``), so a platform without it — and
    Windows, whose handoff is an inheritable handle — keeps the cold spawn.
    """
    if os.name != "posix" or not hasattr(socket, "send_fds"):
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


def standby_dir(root: Path, *, create: bool = True) -> Path:
    path = root / STANDBY_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    return path


def socket_path(root: Path, *, create: bool = True) -> Path:
    """The rendezvous socket for ``root``'s standby, short enough to bind.

    Inside the root's 0700 ``run/standby`` when it fits; otherwise under a
    private per-uid directory in ``TMPDIR`` named by a digest of the root, the
    same fallback shape the secret broker uses, so a deep test root still works
    and two roots never share a socket. ``create=False`` is the READER's form:
    the engage path only asks whether a socket is there, and must not leave a
    directory in a store (or in ``TMPDIR``) on every spawn by asking.
    """
    natural = standby_dir(root, create=create) / "standby.sock"
    if len(str(natural)) <= _MAX_SOCKET_PATH:
        return natural
    import hashlib

    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    uid = os.getuid() if hasattr(os, "getuid") else 0
    directory = Path(tempfile.gettempdir()) / f"lop-standby-{uid}-{digest}"
    if create:
        from local_operator.secrets.protocol import ensure_runtime_dir

        directory = ensure_runtime_dir(directory)
    return directory / "standby.sock"


# ---------------------------------------------------------------------------
# The engage side: adopt one if it is there, warm one behind it either way
# ---------------------------------------------------------------------------


class AdoptedRuntime:
    """What ``launch._spawn_runtime`` returns when a standby took the session.

    Duck-types the two things ``engage_runtime`` reads off a candidate — ``pid``
    and ``poll()``/``returncode``, which it uses to tell a candidate still
    CONSTRUCTING from one that died — plus ``lop_capture_path``. The standby is
    not our child (it was started detached by whichever host warmed it), so
    liveness is signal-0 plus the zombie probe rather than ``waitpid``.
    """

    #: The engage loop polls a constructing candidate every 10 ms, and the zombie
    #: probe is a ``ps`` fork (3.9 ms idle, far more at load 100). Signal-0 on
    #: every poll, the fork at most this often: a standby's parent is the host
    #: that warmed it, which may not reap it promptly, so a died-unreaped standby
    #: is still noticed within a quarter second rather than at the 30 s deadline.
    ZOMBIE_PROBE_S = 0.25

    def __init__(self, pid: int, capture: Path | None) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.lop_capture_path = capture
        self.lop_adopted_standby = True
        self._probed_at = 0.0

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        from local_operator.session.runtime.registry import pid_alive

        now = time.monotonic()
        probe = now - self._probed_at >= self.ZOMBIE_PROBE_S
        if probe:
            self._probed_at = now
        if not pid_alive(self.pid, check_zombie=probe):
            # Not our child: its real status is unknowable here. 1 is the
            # honest "it ended without publishing", which is all the engage
            # loop asks of a returncode.
            self.returncode = 1
        return self.returncode


def try_adopt(
    root: Path,
    interpreter: str,
    env: dict[str, str],
    capture: Path,
    operator_fd: int | None,
) -> AdoptedRuntime | None:
    """Hand a cold child's whole spawn to ``root``'s standby, or ``None`` to spawn cold.

    ``env`` is EXACTLY the environment ``launch._spawn_runtime`` would have given
    the cold child, and ``operator_fd`` the child's end of the operator-capability
    handoff, passed by ``SCM_RIGHTS`` so the value travels on a descriptor exactly
    as it does into a cold child (``harness/approval.py``): the adopted runtime
    reads it with the same ``--operator-fd`` reader, and the capability never
    touches argv, the environment or a file.

    WHO MAY ADOPT. The socket is 0600 in a 0700 directory, so only this uid can
    connect, and anything running as this uid can already start a runtime of its
    own for any session with a capability it minted itself (``python -m
    local_operator.session.runtime.process``). Adoption therefore grants nothing
    a cold spawn does not; it only skips the imports.

    Never raises: every failure — no standby, a refusal, a timeout, a torn
    reply — is ``None``, and the caller does exactly what it did before this
    module existed.
    """
    if disabled():
        return None
    try:
        path = socket_path(root, create=False)
        if not path.exists():
            return None
    except Exception:  # noqa: BLE001 — no rendezvous, no standby
        return None
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = ""
    request = {
        "op": "adopt",
        "interpreter": str(interpreter),
        "env": dict(env),
        "cwd": cwd,
        "capture": str(capture),
        "has_operator_fd": operator_fd is not None,
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(ADOPT_TIMEOUT_S)
            sock.connect(str(path))
            # The descriptor rides on its own one-byte message, ahead of the
            # frame, so the frame itself can be written with a plain sendall.
            socket.send_fds(sock, [b"F"], [operator_fd] if operator_fd is not None else [])
            _send(sock, request)
            reply = _recv(sock)
    except (OSError, ValueError):
        logger.debug("standby adoption unavailable for %s", root, exc_info=True)
        return None
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        logger.info(
            "standby declined adoption (%s); spawning cold",
            (reply or {}).get("reason", "no reply") if isinstance(reply, dict) else "bad reply",
        )
        return None
    pid = reply.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    return AdoptedRuntime(pid, capture)


def ensure_warm(root: Path, interpreter: str) -> None:
    """Start a standby for ``root`` unless one is warming or waiting already.

    Called from ``launch.engage_runtime`` on user-driven engages, so the NEXT
    cold engage finds one. Cheap when one exists: a single non-blocking
    ``flock`` probe on the lock the live standby holds. Never raises.

    NEVER FROM A RUNTIME: a runtime child carries the spawn contract in its
    environment, and a runtime that warmed spares would make every session a
    warmer. Only interface hosts (the TUI, the desktop daemon) warm.
    """
    if not _WARMING[0] or disabled() or os.environ.get("LOP_MOBILE_CHILD_RESUME"):
        return
    try:
        lock = standby_dir(root) / "lock"
        if _lock_held(lock):
            return
        _spawn_standby(root, interpreter)
    except Exception:  # noqa: BLE001 — a missing warm is a slower next engage, never a failure
        logger.debug("could not warm a standby for %s", root, exc_info=True)


#: Whether THIS process warms standbys. Off by default: only a long-lived
#: interface host turns it on (:func:`enable_warming`), so a unit test, a script,
#: a ``lop exec`` or a benchmark that engages a runtime never leaves an idle
#: interpreter behind it. A list, not a bare global, so the flag is mutated
#: rather than rebound.
_WARMING: list[bool] = [False]


def enable_warming(root: "Path | None" = None) -> None:
    """Make this process a warmer, and warm one standby now, off the caller's thread.

    Called by the TUI and the desktop daemon at boot, so the FIRST new
    conversation or cold switch after a launch already finds a standby. The
    probe is one ``flock``; the spawn, when needed, is a fork — both on a daemon
    thread so neither host's event loop waits on them. Never raises.
    """
    if disabled() or os.environ.get("LOP_MOBILE_CHILD_RESUME"):
        return
    _WARMING[0] = True
    try:
        from local_operator.paths import config_dir
        from local_operator.session.runtime.launch import _spawn_interpreter

        target = root if root is not None else config_dir()
        interpreter = _spawn_interpreter()
    except Exception:  # noqa: BLE001 — a missing warm is a slower first engage
        logger.debug("could not resolve a standby target", exc_info=True)
        return
    warm_in_background(target, interpreter)


def warm_in_background(root: Path, interpreter: str) -> None:
    """:func:`ensure_warm` on a daemon thread: never on an engage's critical path."""
    if not _WARMING[0]:
        return
    import threading

    threading.Thread(
        target=ensure_warm, args=(root, interpreter), name="lop-standby-warm", daemon=True
    ).start()


def _lock_held(lock: Path) -> bool:
    import fcntl

    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _spawn_standby(root: Path, interpreter: str) -> None:
    """Start one detached standby. It takes the lock itself; a racer loses there.

    The environment is this host's, stripped of the spawn contract so a standby
    never inherits a session identity from whoever warmed it, and pinned to the
    root it serves. The label is the ANON session label with ``[standby]`` in
    place of ``[session]`` (same length, so the adoption rename is in place).
    """
    from local_operator import procname
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.procstate import detached_popen_kwargs

    env = dict(os.environ)
    for key in CONTRACT_KEYS + ("LOP_RUNTIME_ADOPT_SESSION",):
        env.pop(key, None)
    # Pinned, because the engage names its root explicitly and a warmer's own
    # environment may resolve a different one. Safe to pin: the config dir is
    # read at call time, not by the warm, and adoption replaces this whole
    # environment with the requester's anyway (``_apply_environment``).
    env[CONFIG_DIR_ENV] = str(root)
    label = procname.LABEL_SESSION_ANON.replace("[session]", "[standby]")
    if interpreter != sys.executable:
        argv0, executable = procname.spawn_identity_for_interpreter(
            label, interpreter, id="--------"
        )
    else:
        argv0, executable = procname.spawn_identity(label, id="--------")
    subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [argv0, SAFE_PATH_FLAG, "-m", STANDBY_MODULE],
        executable=executable,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **detached_popen_kwargs(),
    )


# ---------------------------------------------------------------------------
# Wire: one length-prefixed JSON object each way
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
    (the guard's own lazy imports, a timer's) is not a change to what was
    warmed, and comparing whole stamps read that as a moved tree — measured: the
    first idle check retired every standby with ``tree-moved`` on an untouched
    checkout.
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
      EXITS (the next engage warms a replacement against what is current).
    * the per-request checks in :func:`_serve` — is this requester one a cold
      child of THIS interpreter would serve identically? If not, the standby
      DECLINES and keeps waiting, because it is still right for the hosts it
      was warmed for (a TUI and the desktop daemon can differ in environment,
      and a spare that exited on every mismatch would be killed by whichever of
      the two engaged next).
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
    runtime and must be counted — the sweep's protection of attached runtimes
    and its orphan reclaim both depend on the census — but it was exec'd as
    ``-m ...standby``. macOS's ``ps`` reads the argument area of the process
    (``KERN_PROCARGS2``) and Linux reads ``/proc/<pid>/cmdline``, both backed by
    the same memory ``_NSGetArgv``/``argv`` point at, so an equal-length
    overwrite is visible to every reader (verified with ``ps -ww`` on this
    host). ``sys.orig_argv`` keeps the original, so nothing in-process changes.

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


def _serve(root: Path) -> int:
    """Warm, then wait for one adoption or the idle bound. Returns an exit status."""
    import fcntl

    lock_fd = os.open(str(standby_dir(root) / "lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0  # another standby holds this root: the racer that lost exits 0
    # Held for the life of the process, released by the kernel on any exit.
    # Lowest scheduling class while warming: the warm is speculative work and
    # must never take CPU from a session that is doing real work on this host.
    _background_priority(True)
    _warm()
    warmth = _Warmth(root)
    _background_priority(False)
    path = socket_path(root)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    os.chmod(path, 0o600)
    listener.listen(4)
    warm_env = _warm_sensitive(os.environ)
    deadline = time.monotonic() + IDLE_REAP_S
    request: Any = None
    operator_fd: int | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 0
            listener.settimeout(min(remaining, 30.0))
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                # A stale warm is worth nothing: check the guards between waits
                # too, so a standby does not sit on a superseded build for the
                # whole idle window after a ``lop-update``.
                if warmth.stale():
                    return 0
                continue
            with conn:
                conn.settimeout(ADOPT_TIMEOUT_S)
                try:
                    _marker, fds, _flags, _addr = socket.recv_fds(conn, 1, 1)
                    request = _recv(conn)
                except (OSError, ValueError):
                    continue
                if not isinstance(request, dict) or request.get("op") != "adopt":
                    for fd in fds:
                        os.close(fd)
                    continue
                env = request.get("env")
                env = env if isinstance(env, dict) else {}
                stale = warmth.stale()
                reason = stale
                if not reason and _venv_of(str(request.get("interpreter") or "")) != warmth.venv:
                    reason = "interpreter-differs"
                if not reason and _warm_sensitive(env) != warm_env:
                    # The requester's interpreter settings, locale, home or
                    # product switches differ from the ones this process
                    # imported under: only a fresh interpreter is equivalent.
                    reason = "environment-differs"
                if not reason and bool(request.get("has_operator_fd")) != bool(fds):
                    reason = "operator-fd-missing"
                if not reason and not _rename_argv(
                    STANDBY_MODULE.encode(), _runtime_module().encode()
                ):
                    # Nothing renamed yet (the module word is the first rename
                    # and it failed), so the standby is still a clean standby.
                    stale = reason = "argv-rename-unavailable"
                if reason:
                    for fd in fds:
                        os.close(fd)
                    try:
                        _send(conn, {"ok": False, "reason": reason})
                    except OSError:
                        pass
                    if stale:
                        return 0
                    continue
                _rename_argv(b"[standby]", b"[session]")
                _rename_argv(b"id=--------", _label_id(env).encode())
                operator_fd = fds[0] if fds else None
                _apply_environment(env)
                _change_directory(str(request.get("cwd") or ""))
                _redirect_output(str(request.get("capture") or ""))
                _send(conn, {"ok": True, "pid": os.getpid()})
            break
    finally:
        listener.close()
        try:
            path.unlink()
        except OSError:
            pass
        # The lock goes with the listener: the next warmer may start its
        # replacement the moment this one stops being a standby.
        os.close(lock_fd)
    return _become_runtime(operator_fd)


def _runtime_module() -> str:
    from local_operator.session.runtime.types import RUNTIME_MODULE

    return RUNTIME_MODULE


def _background_priority(on: bool) -> None:
    """Darwin's background band while warming; normal again before serving.

    PRIO_DARWIN_BG throttles CPU and I/O. Reset before the socket opens, so an
    adopted standby constructs the session at the same priority a cold child
    would. A platform without it keeps its normal priority, which only means
    the warm competes as a cold spawn would have.
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

    Exactly the work a cold child pays before it can publish, minus anything
    that depends on the session: nothing here opens the store, reads a
    transcript, takes a lease or starts a background pass (verified with an
    audit hook over the whole warm: zero opens or listings under the config
    root). The tokenizer rides along for the reason ``warm_session_imports``
    gives.
    """
    import importlib

    # NOT ``local_operator.session.runtime.process`` itself: adoption runs that
    # module as ``__main__`` (see :func:`_become_runtime`), exactly as ``python
    # -m`` does in a cold child, and ``runpy`` warns when the module it is about
    # to run is already imported. Its own top-level imports are warmed by name.
    import local_operator.session.runtime.server  # noqa: F401
    import local_operator.session.runtime.serving  # noqa: F401
    import local_operator.session.runtime.stall_watchdog  # noqa: F401
    from local_operator.session_factory import _WARM_IMPORTS

    # ``_WARM_IMPORTS`` directly rather than ``warm_session_imports()``: that
    # helper also starts the BYTECODE warm, a subprocess that belongs to a
    # long-lived host (``bytecode.warm_bytecode_cache_in_background``'s own
    # docstring says a runtime child must not start it).
    for name in _WARM_IMPORTS + _WARM_EXTRA:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — a warm-up must never be the failure
            logger.debug("standby warm skipped %s", name, exc_info=True)
    try:
        from local_operator.compaction.tokens import warm_tokenizer

        warm_tokenizer()
    except Exception:  # noqa: BLE001
        logger.debug("standby tokenizer warm skipped", exc_info=True)


def _label_id(env: dict[str, Any]) -> str:
    """``id=<first 8 of the session>``, padded to the placeholder's width.

    The same 8 characters ``launch._spawn_runtime`` puts in a cold child's
    label, so ``ps`` and Activity Monitor show an adopted runtime exactly as
    they show a cold one. Padded rather than truncated because the rename is in
    place and must keep the length.
    """
    ident = str(env.get("LOP_MOBILE_CHILD_RESUME") or "")[:8]
    return ("id=" + ident).ljust(len("id=--------"), "-")


def _apply_environment(env: dict[str, Any]) -> None:
    """Replace this process's environment with the one a cold child would get.

    WHOLE, not merged: a cold child inherits exactly ``env`` and nothing else,
    and an adopted standby must be indistinguishable from it — a key the warmer
    had and the requester does not would otherwise survive into the session
    (the ``LOP_*`` leakage AGENTS.md "Isolating a run" documents).
    Warm-sensitive names already matched (see ``_WARM_SENSITIVE_NAMES``), so
    nothing replaced here was read by an import that already ran.
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
    vanished directory leaves the standby where it was, which only matters to
    code that reads ``os.getcwd()`` before the session applies its own.
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
    actionable startup reasons), so an adopted standby must write its
    construction failures to the same place a cold child would.
    """
    if not capture:
        return
    try:
        fd = os.open(capture, os.O_WRONLY | os.O_APPEND)
    except OSError:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


def _become_runtime(operator_fd: int | None) -> int:
    """Run the runtime module as ``__main__``: from here on, a cold child.

    ``runpy.run_module(..., run_name="__main__")`` is the machinery ``python -m``
    itself uses, so the runtime's ``__main__`` guard runs unchanged — brand,
    ARM THE STALL BOUND, ``main()`` — and ``stall_watchdog.arm`` keeps its one
    call site (pinned by ``test_the_only_arm_site_is_the_runtime_entry_point``).
    A standby never arms it: the warm is not a session, and the bound is
    armed only by this line, after adoption.

    The operator capability arrives exactly as it does for a cold child — as the
    NUMBER of a descriptor in ``sys.argv`` (``--operator-fd <n>``), read and
    closed by ``approval.read_operator_cap_from_argv`` inside ``process.main``.
    The descriptor is the one ``SCM_RIGHTS`` delivered, i.e. the child end of the
    requester's own handoff; its number is not a secret.

    ``SystemExit`` from the guard's ``sys.exit(main())`` propagates to this
    process's own exit, as it would in a cold child.
    """
    import runpy

    from local_operator.harness.approval import OPERATOR_FD_FLAG

    sys.argv = [sys.argv[0]] + (
        [OPERATOR_FD_FLAG, str(operator_fd)] if operator_fd is not None else []
    )
    runpy.run_module(_runtime_module(), run_name="__main__", alter_sys=True)
    return 0


def main() -> int:
    from local_operator import procname
    from local_operator.paths import config_dir

    procname.brand_this_process()
    root = config_dir()
    try:
        return _serve(root)
    except Exception:  # noqa: BLE001 — a standby that fails simply is not there
        logger.debug("standby exited on an error", exc_info=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
