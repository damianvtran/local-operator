"""The session host: the launchd-owned parent of every runtime it starts.

``python -m local_operator.session.runtime.host_service`` — the unit
:mod:`~local_operator.session.runtime.host_install` registers. Design:
``docs/design-surface-independence.md`` §2-§4 and §6.

WHAT IT IS FOR. A runtime forked by a surface (the desktop backend, the TUI, the
phone relay) is that surface's descendant, and on 2026-10-30 twelve workers and at
least five runtimes took SIGTERM within 1.33 s of an app force-quit while the
runtimes the launchd-owned wake supervisor had forked survived the same instant.
This process is that second kind of parent, with one job: fork a runtime when a
surface asks, and be the runtime's direct parent so ``wait()`` still works (the one
thing a parent buys; double-forked daemonisation would lose it).

OWNERSHIP CAP — the host may SPAWN a runtime and may NEVER STOP one. Concretely,
and testable against this file's API surface:

* nothing here calls ``os.kill``, ``killpg``, ``Popen.terminate`` / ``kill`` or
  ``send_signal``, and :func:`serve` has no stop path that touches a child
  (rule R3: a host restart must not become a fleet stop — which is the reason the
  wake supervisor's ``owned_runtimes.sweep`` is NOT reused and must not be added
  here). ``tests/unit/session/runtime/test_session_host.py`` pins this by AST.
* a runtime ends only by its own residency policy or by the stop ladder on a
  person's instruction (``control.stop_session``).

THE OPERATOR CAPABILITY IS RELAYED, NEVER HELD (design §4 option (b)). The request
carries the CHILD END of a socketpair the surface made, by ``SCM_RIGHTS``. This
process hands that descriptor to the runtime it forks (``--operator-fd <n>`` and
``pass_fds``) and closes its own copy; it never reads from it, so the 32-byte
value — which the surface writes into the other end after it has this process's
reply — never exists in this process's memory. :func:`_fork_runtime` is the only
place a received descriptor is used, and :func:`_handle_spawn` closes every
received descriptor on every path (``finally``).

NO ``subprocess`` OF ITS OWN, no sweep, no stop. The module's imports are the
evidence: ``macos_disclaim.spawn_disclaimed`` to fork, ``launch.runtime_command``
for the one definition of a runtime's command line.

SINGLE INSTANCE (design §11 risk 2: two hosts is the first failure mode). An
exclusive ``flock`` on ``<root>/run/session-host/host.lock`` is held for the
process's life; a host that cannot take it within :data:`LOCK_WAIT_S` exits 0 as an
observer-that-never-became. The wait exists because ``launchctl kickstart -k``
SIGKILLs the old host and starts the new one immediately: the lock is released
when the old process is reaped, and a successor that conceded on its first try
would exit 0 — which ``KeepAlive: {SuccessfulExit: false}`` leaves down.

COALESCING (design §3, an optimisation and not load-bearing). Two requests for one
session arriving within :data:`COALESCE_WINDOW_S` of a still-running fork get the
SAME pid back and fork once. The transcript lease, taken by the CHILD
(``session_lease.acquire_session_lease``), stays the authority on who runs:
without coalescing the extra candidate would simply lose the lease and exit; with
it the host does not fork a doomed process. ``forks`` and ``coalesced`` in the
``stats`` op make both visible.

FAILURE IS REPORTED, NOT RETRIED. A refusing request (bad root, bad capture,
unreadable interpreter, fork error) answers ``{"ok": false, "reason": ...}``; the
host never retries on the surface's behalf, because the engage loop owns the
retry budget and a second fork behind its back is the double-spawn this design
exists to prevent.
"""

from __future__ import annotations

import errno
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_operator.session.runtime import session_host

logger = logging.getLogger("local_operator.session.runtime.host_service")

#: See the module docstring ("SINGLE INSTANCE"). Sized past a kickstart's
#: teardown, short enough that a genuinely duplicate host does not hang.
LOCK_WAIT_S = 8.0

#: See the module docstring ("COALESCING"). Long enough to cover a runtime's
#: construction (the engage deadline is 30 s), short enough that a session whose
#: previous runtime is retiring is not pinned to it.
COALESCE_WINDOW_S = 30.0

#: Concurrent connections served. A flood of connects must not become a thread
#: flood in the process every runtime descends from.
MAX_CONNECTIONS = 64

#: Read deadline on a request frame, so a connected-and-silent peer cannot hold a
#: handler thread.
REQUEST_READ_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# The fork — the only place a received descriptor is used
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpawnRequest:
    """A validated spawn ask. Everything the fork needs, nothing the host keeps."""

    session_id: str
    interpreter: str
    env: dict[str, str]
    cwd: str
    capture: str


def _is_runtime_interpreter(candidate: str) -> bool:
    """Whether ``candidate`` is an interpreter this host may exec as a runtime.

    THE SOCKET IS SAME-UID REACHABLE, so a request is untrusted input (design §4:
    an impostor that wins the bind race is not authenticated). Left to name ANY
    executable, a request was an arbitrary exec whose parent is the launchd job and
    whose ancestry holds no requester -- more than the design's "a descriptor for
    its own child", and the very ancestry the 2026-10-30 forensics read (#3222 R-6).
    So the host accepts only the interpreters a legitimate engage can name: its own
    (``sys.executable``) and the current generation's (``_spawn_interpreter``, the
    one ``launch`` sends), compared by realpath. The argv after it is fixed
    (``-m RUNTIME_MODULE``), so this bounds what a request can run to this install.

    RESIDUAL RISK ACCEPTED FOR THIS SLICE: ``env`` is still the requester's, so a
    same-uid peer can steer the interpreter (``PYTHONPATH``, ``PYTHONSTARTUP``) --
    but that peer already runs arbitrary code as this uid and can ``exec`` anything
    itself; what it gains is only the laundering of ancestry. A uid check would add
    nothing: the socket is 0600 in a 0700 directory, so every peer that can connect
    already IS this uid. A surface running a source checkout's venv is refused on
    this route until it is migrated (slice 3) -- an honest failure, never a local fork.
    """
    from local_operator.session.runtime.launch import _spawn_interpreter

    if not os.path.isabs(candidate) or not os.access(candidate, os.X_OK):
        return False
    try:
        wanted = os.path.realpath(candidate)
        return wanted in {
            os.path.realpath(sys.executable),
            os.path.realpath(_spawn_interpreter()),
        }
    except OSError:
        return False


class Refusal(Exception):
    """A request the host will not act on; ``str(exc)`` is the reason sent back."""


#: A seam so the host's behaviour (coalescing, reporting, the lifecycle) is
#: testable with a stand-in child, and so the production fork stays a single
#: function. ``(request, cap_fd) -> Popen-like`` with ``pid`` / ``wait(timeout)``.
Spawner = Callable[[SpawnRequest, "int | None"], Any]


def _fork_runtime(request: SpawnRequest, cap_fd: int | None) -> Any:
    """Fork ONE runtime candidate, detached and disclaimed, as THIS process's child.

    The same spawn ``launch._spawn_runtime_local`` makes, with the same argv
    (``launch.runtime_command``), the same stdio contract (stdout and stderr to the
    surface's capture file so ``engage_runtime`` can report the child's last words
    if it dies before taking the lease) and the same detachment
    (``spawn_disclaimed``: own session and process group, out of the responsibility
    chain) — only the parent differs, which is the whole point.

    ``cap_fd`` is passed through ``pass_fds`` and its NUMBER goes in argv
    (``--operator-fd``), exactly as ``OperatorCapHandoff`` does for a local fork.
    The caller closes its own copy afterwards; this function neither reads nor
    writes it.
    """
    from local_operator.harness.approval import OPERATOR_FD_FLAG
    from local_operator.macos_disclaim import spawn_disclaimed
    from local_operator.session.runtime.launch import runtime_command

    operator_argv = [OPERATOR_FD_FLAG, str(cap_fd)] if cap_fd is not None else []
    argv, executable = runtime_command(request.session_id, request.interpreter, operator_argv)
    # The capture file is the SURFACE's (created 0600 by its ``mkstemp``); the host
    # only appends to it. O_NOFOLLOW + a regular-file check, and no O_CREAT: a peer
    # naming ``/etc/...`` or a symlink must get a refusal, not a write.
    capture_fd = os.open(request.capture, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    handle = os.fdopen(capture_fd, "wb")
    try:
        cwd = request.cwd if request.cwd and os.path.isdir(request.cwd) else None
        return spawn_disclaimed(
            argv,
            executable=executable,
            env=request.env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            pass_fds=(cap_fd,) if cap_fd is not None else (),
            close_fds=True,
        )
    finally:
        handle.close()


# ---------------------------------------------------------------------------
# The host
# ---------------------------------------------------------------------------


@dataclass
class _Child:
    """One runtime this host forked, kept only to report its exit and to coalesce."""

    pid: int
    started: float
    process: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    returncode: int | None = None
    done: bool = False
    subscribers: list[socket.socket] = field(default_factory=list)


class SessionHost:
    """The request handler. Pure of sockets' lifecycle so it is testable with a stand-in fork."""

    def __init__(
        self,
        root: Path,
        *,
        spawner: Spawner = _fork_runtime,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root)
        self._spawner = spawner
        self._clock = clock
        self._fork_lock = threading.Lock()
        self._children: dict[str, _Child] = {}
        self.forks = 0
        self.coalesced = 0

    # -- framing -----------------------------------------------------------

    def serve_connection(self, conn: socket.socket) -> None:
        """Answer one connection. Never raises; owns ``conn`` unless a child adopts it."""
        from local_operator.session.runtime.standby import _recv_request, _send

        fds: list[int] = []
        keep_open = False
        try:
            conn.settimeout(REQUEST_READ_TIMEOUT_S)
            request, fds = _recv_request(conn)
            op = request.get("op") if isinstance(request, dict) else None
            if request.get("v") != session_host.PROTOCOL_VERSION:
                _send(conn, {"ok": False, "reason": "unsupported protocol version"})
            elif op == "ping":
                _send(conn, {"ok": True, "pid": os.getpid(), "root": str(self.root)})
            elif op == "stats":
                _send(conn, {"ok": True, **self.stats()})
            elif op == "spawn":
                keep_open = self._handle_spawn(conn, request, fds)
            else:
                _send(conn, {"ok": False, "reason": f"unknown op {op!r}"})
        except (OSError, ValueError, TypeError):
            logger.debug("session host: connection failed", exc_info=True)
        finally:
            # EVERY descriptor that arrived with this request is closed here, on
            # every path: the host holds the capability's child end only for the
            # length of the fork, and never past the request (design §4: "no
            # capability outlives the request in flight").
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if not keep_open:
                try:
                    conn.close()
                except OSError:
                    pass

    def stats(self) -> dict[str, Any]:
        with self._fork_lock:
            return {
                "forks": self.forks,
                "coalesced": self.coalesced,
                "pid": os.getpid(),
                "children": sorted(c.pid for c in self._children.values() if not c.done),
            }

    # -- spawn -------------------------------------------------------------

    def _validate(self, request: dict[str, Any], fds: list[int]) -> SpawnRequest:
        try:
            requested_root = Path(str(request.get("root", ""))).resolve()
        except (OSError, ValueError):
            raise Refusal("the request names an unreadable store") from None
        if requested_root != self.root.resolve():
            # A surface on another store must not use this host: the socket path
            # is per-root so this only fires for a hand-built request, and a
            # runtime forked for the wrong store would write the wrong transcript.
            raise Refusal("this host serves a different store")
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id or "/" in session_id:
            raise Refusal("the request names no session")
        env = request.get("env")
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            raise Refusal("the request carries no environment")
        interpreter = request.get("interpreter")
        if not isinstance(interpreter, str) or not _is_runtime_interpreter(interpreter):
            raise Refusal("the interpreter is not this install's")
        capture = request.get("capture")
        if not isinstance(capture, str) or not capture:
            raise Refusal("the request names no capture file")
        try:
            info = os.lstat(capture)
        except OSError:
            raise Refusal("the capture file does not exist") from None
        import stat

        # ``hasattr`` IS the platform question (the xplat probe reads it as the
        # guard): the host is launchd-only, but this module is imported -- and its
        # AST pins run -- on Linux and Windows CI.
        if (
            not stat.S_ISREG(info.st_mode)
            or not hasattr(os, "getuid")
            or info.st_uid != os.getuid()
        ):
            raise Refusal("the capture file is not this account's regular file")
        wants_cap = bool(request.get("has_cap_fd"))
        if wants_cap and len(fds) != 1:
            raise Refusal("the request promised one capability descriptor")
        if not wants_cap and fds:
            raise Refusal("the request carried an unexpected descriptor")
        cwd = request.get("cwd")
        return SpawnRequest(
            session_id=session_id,
            interpreter=interpreter,
            env=dict(env),
            cwd=cwd if isinstance(cwd, str) else "",
            capture=capture,
        )

    def _handle_spawn(self, conn: socket.socket, request: dict[str, Any], fds: list[int]) -> bool:
        """Fork (or coalesce), reply, then adopt ``conn`` for the exit report.

        Returns True when ``conn`` now belongs to the child's reaper.
        """
        from local_operator.session.runtime.standby import _send

        try:
            spawn = self._validate(request, fds)
        except Refusal as refusal:
            _send(conn, {"ok": False, "reason": str(refusal)})
            return False
        cap_fd = fds[0] if fds else None
        with self._fork_lock:
            existing = self._children.get(spawn.session_id)
            if (
                existing is not None
                and not existing.done
                and self._clock() - existing.started < COALESCE_WINDOW_S
            ):
                self.coalesced += 1
                child, coalesced = existing, True
            else:
                try:
                    process = self._spawner(spawn, cap_fd)
                except Exception as exc:  # noqa: BLE001 — answered, never retried
                    logger.warning("session host: fork failed for a session: %s", exc)
                    _send(conn, {"ok": False, "reason": f"fork failed: {exc}"})
                    return False
                child = _Child(pid=int(process.pid), started=self._clock(), process=process)
                self._children[spawn.session_id] = child
                self.forks += 1
                coalesced = False
                threading.Thread(
                    target=self._reap,
                    args=(spawn.session_id, child),
                    name=f"reap-{child.pid}",
                    daemon=True,
                ).start()
        _send(conn, {"ok": True, "pid": child.pid, "coalesced": coalesced})
        with child.lock:
            if not child.done:
                child.subscribers.append(conn)
                return True
            returncode = child.returncode
        _send(conn, {"event": "exit", "returncode": returncode})
        return False

    def _reap(self, session_id: str, child: _Child) -> None:
        """Wait on ONE child (by pid, via its own handle) and report its exit.

        ``wait`` is what a direct parent buys, and it is the reason this is a
        thread per child rather than ``SIGCHLD`` or ``waitpid(-1)`` (see
        ``launch._harvest_on_exit``: they are process-wide and can steal another
        waiter's status). The report window is bounded
        (:data:`session_host.DEATH_REPORT_WINDOW_S`); past it the subscribers are
        released and the thread keeps waiting only to reap, so a long-lived
        runtime costs one parked thread and no socket.
        """
        from local_operator.session.runtime.standby import _send

        try:
            returncode = child.process.wait(timeout=session_host.DEATH_REPORT_WINDOW_S)
        except subprocess.TimeoutExpired:
            with child.lock:
                released, child.subscribers = child.subscribers, []
            for conn in released:
                try:
                    conn.close()
                except OSError:
                    pass
            returncode = child.process.wait()
            with child.lock:
                child.returncode, child.done = returncode, True
            self._forget(session_id, child)
            return
        with child.lock:
            child.returncode, child.done = returncode, True
            subscribers, child.subscribers = child.subscribers, []
        for conn in subscribers:
            try:
                _send(conn, {"event": "exit", "returncode": returncode})
            except (OSError, ValueError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        self._forget(session_id, child)

    def _forget(self, session_id: str, child: _Child) -> None:
        with self._fork_lock:
            if self._children.get(session_id) is child:
                del self._children[session_id]


# ---------------------------------------------------------------------------
# The process: election, socket, accept loop
# ---------------------------------------------------------------------------


def _take_lock(root: Path, *, wait_s: float) -> int | None:
    """The single-instance election; the fd is held (never closed) for the process's life."""
    from local_operator.secrets.protocol import ensure_runtime_dir
    from local_operator.wakes.lock import _try_lock

    ensure_runtime_dir(session_host.host_dir(root))
    fd = os.open(session_host.lock_path(root), os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + wait_s
    while True:
        if _try_lock(fd):
            return fd
        if time.monotonic() >= deadline:
            os.close(fd)
            return None
        time.sleep(0.1)


def _bind(root: Path) -> socket.socket:
    """Bind the host socket, reaping a corpse left by a host that died."""
    path = session_host.socket_path(root)
    session_host.prepare_socket_dir(root)
    if path.exists():
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(path))
        except OSError:
            path.unlink(missing_ok=True)
        else:
            raise OSError(errno.EADDRINUSE, f"another session host answers on {path}")
        finally:
            probe.close()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        os.chmod(path, 0o600)
        server.listen(MAX_CONNECTIONS)
    except BaseException:
        server.close()
        raise
    return server


def serve(root: Path, stop: threading.Event, *, host: SessionHost | None = None) -> int:
    """Run until ``stop`` is set. Returns the exit code (0 on every deliberate end).

    THE EXIT PATH SIGNALS NOTHING (rule R3). Closing the listener and unlinking the
    socket are the whole teardown: the children are not touched, are not in this
    process's group (they are ``setsid``), and are reparented to launchd when this
    process goes. The successor host finds them through the store like every other
    surface, and serves new requests without any knowledge of them.
    """
    lock_fd = _take_lock(root, wait_s=LOCK_WAIT_S)
    if lock_fd is None:
        # A clean exit: the other host is serving, and ``SuccessfulExit: false``
        # means launchd does not respawn a loser into a loop.
        logger.info("session host: another host holds the election for %s; exiting", root)
        return 0
    server = _bind(root)
    path = session_host.socket_path(root)
    try:
        inode = path.stat().st_ino
    except OSError:
        inode = None
    handler = host or SessionHost(root)
    slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    logger.info("session host: serving %s on %s (pid %d)", root, path, os.getpid())
    server.settimeout(0.5)
    try:
        while not stop.is_set():
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                if stop.is_set():
                    break
                raise
            if not slots.acquire(blocking=False):
                conn.close()
                continue

            def run(conn: socket.socket = conn) -> None:
                try:
                    handler.serve_connection(conn)
                finally:
                    slots.release()

            threading.Thread(target=run, name="session-host-conn", daemon=True).start()
    finally:
        server.close()
        # By inode, not path: a successor that already rebound the path must not
        # have ITS socket deleted by a draining predecessor (the broker's QA Q4).
        try:
            if inode is not None and path.stat().st_ino == inode:
                path.unlink()
        except OSError:
            pass
        os.close(lock_fd)
    return 0


def main(argv: list[str] | None = None) -> int:
    """``python -m local_operator.session.runtime.host_service`` — the LaunchAgent entry."""
    from local_operator.logger import configure_cli_logging
    from local_operator.paths import config_dir

    configure_cli_logging()
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    # SIGTERM / SIGINT end THIS process only. Nothing else is signalled, by design.
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    # A closed terminal must not end the host (it has none, but a hand run has).
    # A no-op HANDLER and not SIG_IGN: an ignored disposition is inherited across
    # exec, so every runtime would be born ignoring SIGHUP, whereas a handled
    # signal resets to default in the child. ``process._install_sighup_ignore``
    # documents deliberately not planting an inheritable ignore (#3222 N-1).
    signal.signal(signal.SIGHUP, lambda _signum, _frame: None)
    return serve(config_dir(), stop)


if __name__ == "__main__":
    sys.exit(main())
