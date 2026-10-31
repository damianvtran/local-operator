"""The session host's protocol and its client: how a surface asks for a runtime.

WHY THIS EXISTS (design-surface-independence §1-§4). On 2026-10-30 twelve exec
workers and at least five session runtimes took SIGTERM within 1.33 s when the
operator force-quit a frozen desktop app. Every victim descended from the app;
runtimes forked by the launchd-owned wake supervisor survived the same instant.
The fix this module serves is that NO SURFACE FORKS A RUNTIME: a launchd-owned
process (:mod:`local_operator.session.runtime.host_service`) is the direct parent
of every runtime it starts, and a surface only ASKS it. A runtime whose ancestry
contains no surface cannot be taken by a surface's death.

THIS FILE IS THE SURFACE'S HALF, and it is deliberately small and import-light
(stdlib plus the framing helpers ``standby`` already owns): it is loaded on the
engage path of the TUI, the desktop backend and the phone relay.

* :func:`spawn_authority` — who forks. ``AUTHORITY_LOCAL`` is today's behaviour
  and is the default in this slice; the host route is opt-in until the surfaces
  are migrated (slice 3). It is read per call so a test or an operator switch
  needs no restart.
* :func:`request_spawn` — the cold-start ladder and the request. Its docstring is
  the contract for rule R4: **there is no rung that forks locally.**
* :class:`HostedProcess` — what the surface holds instead of a ``Popen``. It has
  ``pid``, ``poll()`` and ``wait()`` — the death facts the engage loop needs
  (``launch._spawn_runtime``: the process's liveness is the only sound death
  signal in the window before the lease is taken) — and **no ``kill``,
  ``terminate`` or ``send_signal``**. A surface holding a handle that can signal
  a runtime is the reachability this design removes.

TRANSPORT. ``<root>/run/session-host/host.sock``, mode 0600, in a 0700
directory; framed JSON in both directions, with the operator capability's
descriptor riding ``SCM_RIGHTS`` on the request's first message. The framing is
``standby``'s (``_send`` / ``_recv`` / ``_recv_request``) rather than a second
copy: the request is the same one ``standby.try_adopt`` already sends, only the
owner of the listening end changed. The request cannot ride the wake index, the
``run/serve`` HTTP plane or ``launchctl``: the capability must travel as a
descriptor and ``harness/approval.py`` forbids it touching a file, the
environment or argv.

THE OPERATOR CAPABILITY IS RELAYED, NEVER HELD (§4 option (b)). The surface mints
the 32-byte value and keeps it; the host receives only the CHILD END of a
socketpair, hands it to the runtime it forks and closes its own copy, and never
reads from it. The surface writes the value into the parent end after the reply
names the runtime's pid (``launch._spawn_runtime_hosted``). The host therefore
sees a descriptor and never a value — and an impostor that wins the bind race on
the socket receives a descriptor for ITS OWN child, which it could already obtain
by calling ``engage_runtime`` itself. Requester identity is not authenticated;
provenance is preserved.

NO IMPLICIT FALLBACK ANYWHERE IN THIS FILE. Every failure — platform without a
supervisor, an installer refusal, a host that never answers, a refusing reply —
raises :class:`SessionHostUnavailable`, which ``launch`` turns into a
``RuntimeStartupError`` carrying a keyed sentence. Nothing here imports
``subprocess`` to start a runtime, and nothing imports ``macos_disclaim``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The switch for who forks. Opt-in in this slice: ``1`` routes every
#: ``launch._spawn_runtime`` through the host. A process that is the HOST never
#: reads it (the host forks through its own function, not through ``launch``).
SPAWN_AUTHORITY_ENV = "LOP_SESSION_HOST"

AUTHORITY_LOCAL = "local"
AUTHORITY_HOST = "host"

#: Wire version. A host and a surface on different builds are the NORMAL state of
#: a mixed-generation fleet; a version the host does not speak is refused in words
#: rather than misread.
PROTOCOL_VERSION = 1

#: Where the socket lives, under the config root. A directory of its own: ``run/host``
#: holds boot records and ``run/mobile`` the discovery records (design §2).
HOST_DIRNAME = "run/session-host"
SOCKET_NAME = "host.sock"
LOCK_NAME = "host.lock"

#: ``sockaddr_un.sun_path`` is 104 bytes on macOS and ``bind`` fails past 103
#: (measured; ``secrets.protocol.MAX_SOCKET_PATH`` records the same number). The
#: config root is operator-controlled and tests use deep ``tmp_path`` trees, so
#: the length fallback below is a real path, not a theoretical one.
_MAX_SOCKET_PATH = 103

#: Bounded connect, inside the engage's own deadline (design §2: "connect with a
#: bounded timeout inside the engage's own deadline"). A host that is bootstrapped
#: by this very call needs ~1-2 s to import and bind; 10 s leaves a third of the
#: default 30 s engage deadline for the fork and construction behind it.
CONNECT_DEADLINE_S = 10.0

#: One request/reply round trip. The host's reply is the fork's return, which is
#: milliseconds; this only bounds a wedged host.
REPLY_TIMEOUT_S = 15.0

#: How long the host keeps a connection open to report a child's EXIT after
#: replying with its pid. The report exists for the construction window — a
#: candidate that dies before it takes the lease must read as "died, with the
#: child's own reason" rather than "timed out" — so it is bounded: past it the
#: runtime is established, the host closes the connection, and the surface's
#: proxy falls back to a pid liveness probe. Bounding it is also what keeps a
#: surface from parking a socket and a thread per runtime for the runtime's life.
DEATH_REPORT_WINDOW_S = 120.0


class SessionHostUnavailable(RuntimeError):
    """The session host could not be reached or declined; nothing was forked.

    ``code`` names which keyed sentence applies (``unsupported`` / ``unavailable``
    / ``refused``) so the caller renders copy in the active locale instead of
    this module carrying English; ``detail`` is for the log only.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"session host {code}: {detail}" if detail else f"session host {code}")
        self.code = code
        self.detail = detail


def unavailable_sentence(code: str) -> str:
    """The keyed, localised sentence for a :class:`SessionHostUnavailable` ``code``.

    Rendered here so ``launch`` stays free of copy. The codes are a closed set
    (``unsupported`` / ``unavailable`` / ``refused``); an unknown one reads as
    ``unavailable`` — the safe wording, because it promises nothing about the
    cause and says nothing was started.
    """
    from local_operator.i18n.keys import wire_errors_session_host as keys
    from local_operator.i18n.text import tr

    text = tr("wire.errors.session_host")
    if code == "unsupported":
        return text(keys.unsupported())
    if code == "refused":
        return text(keys.refused())
    return text(keys.unavailable())


def spawn_authority() -> str:
    """Who forks a runtime in THIS process: ``AUTHORITY_HOST`` or ``AUTHORITY_LOCAL``."""
    value = os.environ.get(SPAWN_AUTHORITY_ENV, "")
    return AUTHORITY_HOST if value not in ("", "0", "false", "no", "off") else AUTHORITY_LOCAL


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def host_dir(root: Path) -> Path:
    """The directory the lock lives in — a PATH only, creates nothing."""
    return Path(root) / HOST_DIRNAME


def lock_path(root: Path) -> Path:
    return host_dir(root) / LOCK_NAME


def socket_path(root: Path) -> Path:
    """Where the host listens for ``root``, with the sun_path length fallback.

    The primary location is under the root, inside a 0700 directory, so the socket
    inherits a directory that already excludes other accounts. Past the kernel's
    path limit it moves to a short, uid-qualified directory named from a hash of
    the root (two stores must never share a socket: a surface would reach a host
    that serves a different store). Both the host and every surface compute the
    same answer from the same root, so no file has to carry the choice. Creates
    nothing; :func:`prepare_socket_dir` is the writer's spelling.
    """
    primary = host_dir(root) / SOCKET_NAME
    if len(str(primary)) <= _MAX_SOCKET_PATH:
        return primary
    digest = hashlib.sha256(str(Path(root)).encode("utf-8")).hexdigest()[:12]
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"lop-session-host-{uid}-{digest}" / SOCKET_NAME


def prepare_socket_dir(root: Path) -> Path:
    """Create (0700, ownership-verified) the directory the socket lives in.

    ``ensure_runtime_dir`` refuses a directory another uid owns: on a shared
    ``/tmp`` an attacker could pre-create the fallback directory and wait for the
    host to put a socket in it.
    """
    from local_operator.secrets.protocol import ensure_runtime_dir

    directory = socket_path(root).parent
    ensure_runtime_dir(directory)
    return directory


# ---------------------------------------------------------------------------
# The request, from the surface
# ---------------------------------------------------------------------------


class HostedProcess:
    """The surface's handle on a runtime the host forked: death facts, no signals.

    ``poll()`` is ``None`` while the candidate runs and its exit status once the
    host has reported it. The host reports for :data:`DEATH_REPORT_WINDOW_S`; past
    that (or if the host itself dies, which does NOT kill the runtime — rule R3)
    the proxy answers from a pid liveness probe, with ``-1`` meaning "exited,
    status unavailable" — the same reading ``DisclaimedProcess`` gives a child it
    cannot reap.

    DELIBERATELY NO ``kill`` / ``terminate`` / ``send_signal``. The one thing a
    surface must not be able to do through a spawn handle is end the runtime.
    """

    def __init__(self, pid: int, *, coalesced: bool, connection: socket.socket | None) -> None:
        self.pid = pid
        self.returncode: int | None = None
        #: True when the host answered with a runtime ANOTHER request had already
        #: caused: this engage forked nothing, so its capability descriptor reached
        #: no runtime (``launch._spawn_runtime_hosted`` therefore does not remember it).
        self.lop_coalesced = coalesced
        self._connection = connection
        self._lock = threading.Lock()
        self._detached = connection is None
        self._done = threading.Event()
        if connection is not None:
            threading.Thread(target=self._read_exit, name=f"host-exit-{pid}", daemon=True).start()

    def _read_exit(self) -> None:
        from local_operator.session.runtime.standby import _recv

        connection = self._connection
        assert connection is not None
        try:
            connection.settimeout(DEATH_REPORT_WINDOW_S + REPLY_TIMEOUT_S)
            frame = _recv(connection)
            if isinstance(frame, dict) and frame.get("event") == "exit":
                code = frame.get("returncode")
                with self._lock:
                    self.returncode = int(code) if isinstance(code, int) else -1
        except (OSError, ValueError):
            # The host went away or the window closed: fall back to the probe.
            logger.debug("session host: exit report unavailable for %s", self.pid, exc_info=True)
        finally:
            with self._lock:
                self._detached = True
            try:
                connection.close()
            except OSError:
                pass
            self._done.set()

    def poll(self) -> int | None:
        with self._lock:
            if self.returncode is not None:
                return self.returncode
            detached = self._detached
        if not detached:
            return None
        from local_operator import procstate

        if procstate.pid_alive(self.pid) and not procstate.is_zombie(self.pid):
            return None
        with self._lock:
            if self.returncode is None:
                self.returncode = -1
            return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                import subprocess

                raise subprocess.TimeoutExpired(
                    ["session-host-runtime", str(self.pid)], timeout or 0.0
                )
            self._done.wait(0.05)


def _connect(path: Path, timeout_s: float) -> socket.socket:
    """One connection attempt; raises ``OSError`` on failure."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        sock.connect(str(path))
    except BaseException:
        sock.close()
        raise
    return sock


def ping(root: Path, *, timeout_s: float = 2.0) -> bool:
    """Whether a host is answering for ``root`` — a round trip, not a file check."""
    from local_operator.session.runtime.standby import _recv, _send

    try:
        sock = _connect(socket_path(root), timeout_s)
    except OSError:
        return False
    try:
        sock.settimeout(timeout_s)
        _send(sock, {"op": "ping", "v": PROTOCOL_VERSION})
        reply = _recv(sock)
        return isinstance(reply, dict) and bool(reply.get("ok"))
    except (OSError, ValueError):
        return False
    finally:
        sock.close()


def ensure_session_host(root: Path) -> None:
    """Make a host run for ``root``, or raise :class:`SessionHostUnavailable`.

    THE COLD-START LADDER'S FIRST RUNG (design §2): answer from a live host when
    there is one (no ``launchctl`` call on the hot path), otherwise ask the
    supervisor to bootstrap (absent) or kickstart (loaded but stopped). It never
    forks a runtime and never starts the host as a child of the caller — the host
    must be launchd's, or it is the app's child and the bug is back.

    Split out as a module-level function because it is the SEAM a test of the
    ladder (and CI's Gate A, which runs the host as an ordinary subprocess)
    replaces.
    """
    if ping(root, timeout_s=1.0):
        return
    from local_operator.session.runtime import host_install

    outcome = host_install.ensure_host_installed(Path(root))
    if not outcome.ok:
        raise SessionHostUnavailable(outcome.code, outcome.reason)


def _connect_with_deadline(root: Path, deadline_s: float) -> socket.socket:
    path = socket_path(root)
    deadline = time.monotonic() + deadline_s
    delay = 0.05
    last: OSError | None = None
    while True:
        try:
            return _connect(path, min(2.0, max(0.1, deadline - time.monotonic())))
        except OSError as exc:
            last = exc
        if time.monotonic() >= deadline:
            raise SessionHostUnavailable("unavailable", f"no answer on {path}: {last}")
        time.sleep(delay)
        delay = min(delay * 2, 0.5)


def request_spawn(
    root: Path,
    *,
    session_id: str,
    cwd: str,
    env: dict[str, str],
    interpreter: str,
    capture: Path,
    cap_fd: int | None,
    connect_deadline_s: float = CONNECT_DEADLINE_S,
) -> HostedProcess:
    """Ask the host to fork a runtime for ``session_id``; returns its death-fact handle.

    THE COLD-START LADDER, AND ITS ABSENT FINAL RUNG (rule R4). In order:
    ``ensure_session_host`` (bootstrap/kickstart through the supervisor) → connect
    with a bounded timeout → send. If any step fails this RAISES
    :class:`SessionHostUnavailable`. There is deliberately no ``except`` that
    forks locally, and no import in this module that could: a "host is down, fork
    it ourselves" rung would fire exactly when the host is unavailable — which is
    when a surface is most likely about to die — and would re-create the incident
    silently. ``request_spawn`` also never retries a REFUSED request or sends a
    second request for one engage (design §3): the engage loop owns the retry
    budget.

    ``cap_fd`` is the child end of the operator-capability socketpair; it is sent
    by ``SCM_RIGHTS`` and remains the caller's to close.
    """
    from local_operator.session.runtime.standby import _recv

    ensure_session_host(root)
    sock = _connect_with_deadline(root, connect_deadline_s)
    try:
        sock.settimeout(REPLY_TIMEOUT_S)
        request: dict[str, Any] = {
            "op": "spawn",
            "v": PROTOCOL_VERSION,
            "root": str(root),
            "session_id": session_id,
            "interpreter": str(interpreter),
            "env": dict(env),
            "cwd": _process_cwd(),
            "capture": str(capture),
            "has_cap_fd": cap_fd is not None,
        }
        body = _encode(request)
        if cap_fd is not None:
            # ONE MESSAGE: the frame's first bytes and the descriptor together, so
            # the host's first read gets the length prefix and the fd in the same
            # ``recvmsg`` (see ``standby._recv_request``).
            sent = socket.send_fds(sock, [body], [cap_fd])
            # A stream ``sendmsg`` may take only part of a large frame (the
            # environment alone can run to tens of KB). The descriptor rode the
            # first bytes, so the remainder is plain data -- ``standby.try_adopt``
            # does not need this because it is bounded by one warm env; here a
            # short send would be a torn frame the host reads as a closed channel.
            if sent < len(body):
                sock.sendall(body[sent:])
        else:
            sock.sendall(body)
        reply = _recv(sock)
    except (OSError, ValueError) as exc:
        sock.close()
        raise SessionHostUnavailable("unavailable", f"request failed: {exc}") from exc
    except BaseException:
        sock.close()
        raise
    if not isinstance(reply, dict) or not reply.get("ok"):
        sock.close()
        reason = str(reply.get("reason", "")) if isinstance(reply, dict) else "torn reply"
        raise SessionHostUnavailable("refused", reason)
    pid = reply.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        sock.close()
        raise SessionHostUnavailable("refused", "reply carried no pid")
    return HostedProcess(pid, coalesced=bool(reply.get("coalesced")), connection=sock)


def _process_cwd() -> str:
    try:
        return os.getcwd()
    except OSError:
        return ""


def _encode(payload: dict[str, Any]) -> bytes:
    import json

    body = json.dumps(payload).encode("utf-8")
    return len(body).to_bytes(4, "big") + body
