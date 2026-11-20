"""Refuse loopback TCP connections whose peer process belongs to another OS user.

WHY THIS EXISTS. ``lop serve`` binds loopback only (``cli.py``
``_serve_bind_is_loopback_only``, no override), but "loopback" means "any
process on this machine", including processes of OTHER local accounts. A
standalone daemon authenticates no native caller, and ``/v1/static/*`` serves
any readable file unauthenticated to a loopback-accepted connection even in the
desktop-managed posture (``server/utils/static_roots.py``: ``<img>``/``<video>``
cannot carry the bearer). So on a shared Mac or Linux box another user could
read the operator's files through the daemon. This module closes that: the
kernel already knows who owns the CLIENT end of every loopback connection, and
we ask it once per connection, before a single request byte is parsed.

DESIGN (follow-up D, architect design of 2026-11-19, slice 1 = macOS + Linux):

* **Where.** At uvicorn's connection layer, via an HTTP protocol subclass passed
  as ``uvicorn.Config(http=...)``. ``connection_made`` pauses reading, the
  lookup runs on a small dedicated executor (never the event loop: the desktop
  and TUI share it), and the connection is resumed or answered with a raw
  ``403``/``503`` + ``connection: close``. ASGI middleware was rejected: it sees
  ``scope["client"]`` per request, not the connection, which would force a cache
  keyed on ``(ip, port)`` -- unsafe, because a same-user client that closes with
  RST skips TIME_WAIT and a foreign process can rebind that port and inherit
  the verdict. The verdict lives on the connection object and dies with it.
* **Match the full 4-tuple.** The client socket we look for has
  local = our ``peername`` and foreign = our ``sockname``, addresses included.
  Ports alone would admit a foreign process on ``127.0.0.1:X`` while a same-user
  client used ``127.0.0.2:X``. IPv4-mapped IPv6 (``::ffff:a.b.c.d``) is
  normalised to IPv4 on both sides.
* **macOS: libproc.** ``proc_listpids(PROC_UID_ONLY, getuid())`` then each pid's
  socket fds (what ``lsof`` does). A non-root caller CANNOT inspect another
  uid's process (EPERM), so "found in one of my processes" means same user and
  "not found" means another user, root, or already gone -- all refused. The
  ``netstat`` sysctl (``net.inet.tcp.pcblist_n``) is unusable here: measured to
  return only the calling process's sockets. A small LRU of pids that recently
  owned admitted connections is scanned first (the desktop's network helper
  then costs ~0.1 ms instead of a full scan).
* **Linux: ``/proc/net/tcp`` + ``tcp6``.** World-readable, lists every socket
  in the netns regardless of ``hidepid``, and carries the owning uid directly.
  Linux can see foreign owners, so uid 0 (root: ``sudo curl``, root relays) is
  admitted -- root can already read everything this user owns (manager decision
  1, 2026-11-19). The files are opened once at startup and re-read from offset
  0, so a foreign user exhausting fds cannot break the lookup.

POLICY (``LOCAL_OPERATOR_PEER_UID_CHECK`` = ``auto`` (default) | ``on`` | ``off``,
read once per process). The operator's constraint is ZERO user-visible friction:

* ``auto`` enforces on native macOS/Linux and SKIPS, with exactly one WARNING,
  in containers and WSL (relays make uids meaningless there), on unsupported
  platforms (Windows until slice 2), when the mechanism is missing, or when the
  startup self-probe fails. A host where the lookup is meaningless must never
  lock the desktop out.
* ``on`` ignores the container/WSL heuristics (QA in a container). It cannot
  enforce where the mechanism is absent or its self-probe fails; those still
  skip with the warning rather than refuse everything.
* ``off`` disables it, with one INFO line.
* After a passing self-probe, a lookup ERROR or timeout is retried once
  immediately and then refuses THAT connection with 503 (manager decision 2):
  a foreign user can induce errors (fd/CPU exhaustion), so fail-open per
  connection would hand them the gate. Never auto-disabled at runtime; a restart
  re-decides.
* Only connections ACCEPTED on a loopback address are checked. Anything else
  (an unsupported bare-uvicorn embed on a LAN address, a Unix socket) passes
  through untouched to the rooted-mode gate in ``static_roots``.

The policy is resolved in a background thread when ``lop serve`` starts (so the
self-probe never delays startup); connections that arrive before it settles
wait for it, paused. The ``--reload`` child resolves it lazily on its first
connection. Applies to standalone and desktop-managed daemons alike.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import ctypes
import ipaddress
import logging
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger("local_operator.server.peer_check")

ENV_VAR = "LOCAL_OPERATOR_PEER_UID_CHECK"

# Per-attempt wait for one lookup, INCLUDING queueing on the 2-worker executor.
# Measured (design §2, macOS 27.2, ~590-850 same-uid pids): hit 1-30 ms, full
# miss ~50 ms; estimated ~150 ms on a ~2k-process dev machine. A burst of
# parallel desktop connections queues behind 2 workers, so the budget covers a
# few full scans back to back rather than the single-scan 250 ms the design
# sketched; exceeding it is an ERROR (refused), so it must sit well above any
# legitimate wait.
LOOKUP_TIMEOUT_S = 1.0
# Pending lookups allowed before new connections are refused as ERROR. Bounds
# the memory and thread-pool pressure a connection flood can create.
MAX_PENDING_LOOKUPS = 64
# A not-found peer is rescanned once after this pause before refusing: covers
# an fd being passed between processes or an exec in flight (design §4).
NOT_FOUND_RESCAN_S = 0.02
# Self-probe: the child interpreter gets this long to start and connect (Python
# start-up, not the mechanism), and the lookup itself must answer within
# SELF_PROBE_LOOKUP_BUDGET_S (design §3: 500 ms).
SELF_PROBE_CONNECT_TIMEOUT_S = 10.0
SELF_PROBE_LOOKUP_BUDGET_S = 0.5
# Refusal logging: one line per peer uid (or "unknown") per minute.
REFUSAL_LOG_INTERVAL_S = 60.0


def _raw_response(status: str, body: bytes) -> bytes:
    # Raw bytes, not an ASGI response: the refusal happens before uvicorn has
    # parsed a request, and no application code may run for a refused peer.
    # The body is a developer-facing diagnostic (a client of ours never sees
    # it), so it stays English like a log line.
    head = (
        f"HTTP/1.1 {status}\r\ncontent-type: text/plain\r\n"
        f"content-length: {len(body)}\r\nconnection: close\r\n\r\n"
    )
    return head.encode("ascii") + body


_RESPONSE_403 = _raw_response(
    "403 Forbidden", b"refused: the connecting process is not this daemon's OS user"
)
_RESPONSE_503 = _raw_response(
    "503 Service Unavailable", b"peer user check failed; retry the request"
)

Address = ipaddress.IPv4Address | ipaddress.IPv6Address
Endpoint = tuple[Address, int]


def normalize_endpoint(host: str, port: int) -> Endpoint:
    """Parse a socket address, mapping ``::ffff:a.b.c.d`` to IPv4.

    A scope suffix (``fe80::1%lo0``) is dropped; loopback addresses carry none.
    """
    addr = ipaddress.ip_address(host.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return addr, int(port)


class PeerLookup(Protocol):
    """Answers "which uid owns the client socket of this connection?".

    ``client`` is the peer's endpoint (our ``peername``), ``server`` ours (our
    ``sockname``). Returns the owning uid, or ``None`` when no visible socket
    matches. Raises on a mechanism fault.
    """

    mechanism: str
    #: Whether uid 0 is admitted. Only a mechanism that can positively see a
    #: root owner can set this (Linux); macOS never sees one.
    admits_root: bool

    def owner_uid(self, client: Endpoint, server: Endpoint) -> int | None: ...


# --------------------------------------------------------------------------
# Linux: /proc/net/tcp{,6}
# --------------------------------------------------------------------------

# TCP states (include/net/tcp_states.h). TIME_WAIT rows report uid 0 because no
# socket owns them any more, and LISTEN rows are never a client end, so both are
# skipped: a TIME_WAIT row must never let the root rule admit a stale tuple.
_PROC_SKIP_STATES = frozenset({"06", "0A"})


def parse_proc_net_address(field: str) -> Endpoint:
    """Decode ``/proc/net/tcp`` ``ADDR:PORT`` hex.

    The address is printed as 32-bit words in HOST byte order (one word for
    IPv4, four for IPv6); the port is a plain big-endian number.
    """
    hex_addr, hex_port = field.split(":")
    words = [int(hex_addr[i : i + 8], 16) for i in range(0, len(hex_addr), 8)]
    raw = b"".join(struct.pack("=I", w) for w in words)
    addr: Address
    if len(raw) == 4:
        addr = ipaddress.IPv4Address(raw)
    elif len(raw) == 16:
        addr = ipaddress.IPv6Address(raw)
        if addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
    else:
        raise ValueError(f"unexpected /proc/net/tcp address width: {field!r}")
    return addr, int(hex_port, 16)


def find_uid_in_proc_net(text: str, client: Endpoint, server: Endpoint) -> int | None:
    """Return the uid of the row whose local = ``client`` and remote = ``server``."""
    port_suffix = f":{client[1]:04X}"
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8 or not parts[1].endswith(port_suffix):
            continue  # cheap pre-filter on the client port before decoding
        if parts[3] in _PROC_SKIP_STATES:
            continue
        if (
            parse_proc_net_address(parts[1]) == client
            and parse_proc_net_address(parts[2]) == server
        ):
            return int(parts[7])
    return None


class LinuxProcNetLookup:
    """``/proc/net/tcp`` + ``tcp6`` reader over descriptors opened once."""

    mechanism = "linux-procfs"
    admits_root = True

    def __init__(self, paths: tuple[str, ...] = ("/proc/net/tcp", "/proc/net/tcp6")) -> None:
        self._tables: list[tuple[int, threading.Lock]] = []
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except FileNotFoundError:
                # tcp6 is absent on a host booted with ipv6.disable=1.
                if path.endswith("tcp6"):
                    continue
                raise
            self._tables.append((fd, threading.Lock()))
        if not self._tables:
            raise OSError("no /proc/net/tcp table could be opened")

    @staticmethod
    def _read_all(fd: int) -> str:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks).decode("ascii", "replace")

    def owner_uid(self, client: Endpoint, server: Endpoint) -> int | None:
        # Both tables are searched regardless of family: a v4 client socket is
        # listed in ``tcp`` while a v6-mapped one is listed in ``tcp6``.
        for fd, lock in self._tables:
            with lock:  # lseek+read on a shared fd is not thread-safe
                text = self._read_all(fd)
            uid = find_uid_in_proc_net(text, client, server)
            if uid is not None:
                return uid
        return None


# --------------------------------------------------------------------------
# macOS: libproc
# --------------------------------------------------------------------------

_PROC_UID_ONLY = 4
_PROC_PIDLISTFDS = 1
_PROC_PIDFDSOCKETINFO = 3
_PROX_FDTYPE_SOCKET = 2
# ``struct socket_fdinfo`` (<sys/proc_info.h>), measured on macOS 27.2 arm64:
# 792 bytes; ``psi.soi_family`` @184, ``psi.soi_kind`` @256 (SOCKINFO_TCP = 2),
# then the ``in_sockinfo`` that opens ``tcp_sockinfo``: fport @264, lport @268
# (network-order ports in an int), ``insi_vflag`` @288 (1 = v4, 2 = v6), faddr
# @296 and laddr @312 (16-byte ``in6_addr``, or 12 pad bytes + ``in_addr``).
# These are ABI offsets the kernel could move; the startup self-probe resolves a
# real child connection through them, so a drift skips (with a warning) rather
# than refusing everyone.
_SOCKET_FDINFO_SIZE = 792
_OFF_FAMILY = 184
_OFF_KIND = 256
_OFF_FPORT = 264
_OFF_LPORT = 268
_OFF_VFLAG = 288
_OFF_FADDR = 296
_OFF_LADDR = 312
_SOCKINFO_TCP = 2
_INI_IPV4 = 0x1
_HINT_LRU_SIZE = 32


def _mac_addr(raw: bytes, offset: int, vflag: int) -> Address:
    if vflag & _INI_IPV4:
        return ipaddress.IPv4Address(raw[offset + 12 : offset + 16])
    addr = ipaddress.IPv6Address(raw[offset : offset + 16])
    return addr.ipv4_mapped or addr


class MacLibprocLookup:
    """Scan this uid's own processes' sockets for the client end."""

    mechanism = "macos-libproc"
    admits_root = False

    def __init__(self) -> None:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self._listpids = lib.proc_listpids
        self._listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        self._pidinfo = lib.proc_pidinfo
        self._pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._fdinfo = lib.proc_pidfdinfo
        self._fdinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self._uid = os.getuid()
        # Pids that recently owned an admitted connection, most recent last.
        # Sound against pid reuse: a reused pid that now belongs to another uid
        # is uninspectable (EPERM), so it can never produce a match.
        self._hints: OrderedDict[int, None] = OrderedDict()
        self._hints_lock = threading.Lock()
        self._local = threading.local()  # per-worker ctypes buffers

    def _buffers(self) -> tuple[Any, Any]:
        bufs = getattr(self._local, "bufs", None)
        if bufs is None:
            bufs = (
                (ctypes.c_uint32 * 2 * 1024)(),
                ctypes.create_string_buffer(_SOCKET_FDINFO_SIZE),
            )
            self._local.bufs = bufs
        return bufs

    def _list_pids(self) -> list[int]:
        need = self._listpids(_PROC_UID_ONLY, self._uid, None, 0)
        if need <= 0:
            raise OSError(ctypes.get_errno(), "proc_listpids failed")
        count = need // 4 + 64  # headroom for processes spawned in between
        pids = (ctypes.c_int * count)()
        got = self._listpids(_PROC_UID_ONLY, self._uid, pids, ctypes.sizeof(pids))
        if got <= 0:
            raise OSError(ctypes.get_errno(), "proc_listpids failed")
        return [p for p in pids[: got // 4] if p > 0]

    def _pid_owns(self, pid: int, client: Endpoint, server: Endpoint) -> bool:
        fds, sbuf = self._buffers()
        while True:
            got = self._pidinfo(pid, _PROC_PIDLISTFDS, 0, fds, ctypes.sizeof(fds))
            if got <= 0:
                return False  # gone, a zombie, or not ours (EPERM)
            if got < ctypes.sizeof(fds):
                break
            fds = (ctypes.c_uint32 * 2 * (len(fds) * 2))()  # table full: grow, re-read
            self._local.bufs = (fds, sbuf)
        for k in range(got // 8):
            fd, fdtype = fds[k][0], fds[k][1]
            if fdtype != _PROX_FDTYPE_SOCKET:
                continue
            if (
                self._fdinfo(pid, fd, _PROC_PIDFDSOCKETINFO, sbuf, _SOCKET_FDINFO_SIZE)
                < _SOCKET_FDINFO_SIZE
            ):
                continue
            raw = sbuf.raw
            if struct.unpack_from("<i", raw, _OFF_KIND)[0] != _SOCKINFO_TCP:
                continue
            lport = socket.ntohs(struct.unpack_from("<I", raw, _OFF_LPORT)[0] & 0xFFFF)
            fport = socket.ntohs(struct.unpack_from("<I", raw, _OFF_FPORT)[0] & 0xFFFF)
            if lport != client[1] or fport != server[1]:
                continue
            vflag = raw[_OFF_VFLAG]
            if (
                _mac_addr(raw, _OFF_LADDR, vflag) == client[0]
                and _mac_addr(raw, _OFF_FADDR, vflag) == server[0]
            ):
                return True
        return False

    def owner_uid(self, client: Endpoint, server: Endpoint) -> int | None:
        with self._hints_lock:
            hints = list(reversed(self._hints))
        for pid in hints:
            if self._pid_owns(pid, client, server):
                self._remember(pid)
                return self._uid
        tried = set(hints)
        for pid in self._list_pids():
            if pid in tried:
                continue
            if self._pid_owns(pid, client, server):
                self._remember(pid)
                return self._uid
        return None

    def _remember(self, pid: int) -> None:
        with self._hints_lock:
            self._hints.pop(pid, None)
            self._hints[pid] = None
            while len(self._hints) > _HINT_LRU_SIZE:
                self._hints.popitem(last=False)


# --------------------------------------------------------------------------
# Applicability, self-probe and the resolved gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateStatus:
    """What ``/health`` reports: ``mode`` plus a mechanism or a reason."""

    mode: str  # "enforcing" | "skipped" | "disabled"
    mechanism: str | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, str]:
        out = {"mode": self.mode}
        if self.mechanism:
            out["mechanism"] = self.mechanism
        if self.reason:
            out["reason"] = self.reason
        return out


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(65536)
    except OSError:
        return None


def detect_environment_skip(
    env: Mapping[str, str],
    *,
    exists: Callable[[str], bool] = os.path.exists,
    read_text: Callable[[str], str | None] = _read_text,
) -> str | None:
    """Return why ``auto`` must skip on this host (container/WSL), else ``None``."""
    for marker in ("/.dockerenv", "/run/.containerenv"):
        if exists(marker):
            return f"container: {marker}"
    if env.get("container"):
        return f"container: container={env['container']}"
    cgroup = read_text("/proc/1/cgroup") or ""
    for word in ("docker", "kubepods", "containerd", "lxc", "libpod"):
        if word in cgroup:
            return f"container: /proc/1/cgroup names {word}"
    for var in ("WSL_DISTRO_NAME", "WSL_INTEROP"):
        if env.get(var):
            return f"wsl: {var} is set"
    if "microsoft" in (read_text("/proc/sys/kernel/osrelease") or "").lower():
        return "wsl: kernel osrelease names microsoft"
    return None


def default_lookup_factory(platform: str = sys.platform) -> PeerLookup:
    """Build the native mechanism, or raise ``LookupError`` naming why not."""
    if platform == "darwin":
        try:
            return MacLibprocLookup()
        except (OSError, AttributeError) as exc:
            raise LookupError(f"unsupported: libproc unavailable ({exc})") from exc
    if platform.startswith("linux"):
        try:
            return LinuxProcNetLookup()
        except OSError as exc:
            raise LookupError(f"unsupported: /proc/net/tcp unreadable ({exc})") from exc
    name = "windows" if platform.startswith("win") else platform
    raise LookupError(f"unsupported: {name}")


def self_probe(lookup: PeerLookup) -> str | None:
    """Resolve a real CHILD process's loopback connection; ``None`` means pass.

    A child, not an in-process socket, because what must hold is that sockets
    in OTHER processes of ours are visible (a sandbox could hide them) and that
    the macOS struct offsets still decode a real connection. A negative control
    follows: a tuple nobody owns must NOT resolve, or the mechanism would be
    admitting everything.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    child: subprocess.Popen[bytes] | None = None
    conn: socket.socket | None = None
    try:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.settimeout(SELF_PROBE_CONNECT_TIMEOUT_S)
        port = srv.getsockname()[1]
        # -I -S: isolated, no site import -- the child only needs ``socket``,
        # and skipping site keeps its start-up to tens of milliseconds.
        code = (
            f"import socket,sys;s=socket.create_connection(('127.0.0.1',{port}));sys.stdin.read()"
        )
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        conn, peer = srv.accept()
        client = normalize_endpoint(peer[0], peer[1])
        server = normalize_endpoint(*conn.getsockname()[:2])
        started = time.monotonic()
        uid = lookup.owner_uid(client, server)
        elapsed = time.monotonic() - started
        if uid != os.getuid():
            return f"self-probe: child connection resolved to uid {uid}, expected {os.getuid()}"
        if elapsed > SELF_PROBE_LOOKUP_BUDGET_S:
            budget_ms = SELF_PROBE_LOOKUP_BUDGET_S * 1000
            return f"self-probe: lookup took {elapsed * 1000:.0f} ms (budget {budget_ms:.0f} ms)"
        # Port 0 is never a real client port, so this tuple has no owner.
        if lookup.owner_uid((client[0], 0), server) is not None:
            return "self-probe: an unowned tuple resolved to an owner"
        return None
    except Exception as exc:  # noqa: BLE001 -- any probe fault means "do not enforce"
        return f"self-probe: {type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()
        srv.close()
        if child is not None:
            try:
                if child.stdin is not None:
                    child.stdin.close()
                child.wait(timeout=5)
            except Exception:  # noqa: BLE001
                child.kill()
                child.wait()


@dataclass(frozen=True)
class Decision:
    admit: bool
    response: bytes = b""


class PeerGate:
    """The resolved policy for this process: a status and, when enforcing, a lookup."""

    def __init__(self, status: GateStatus, lookup: PeerLookup | None = None) -> None:
        self.status = status
        self.lookup = lookup
        self._uid = os.getuid() if hasattr(os, "getuid") else -1
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._last_logged: dict[str, float] = {}
        self._last_error = ""

    @property
    def enforcing(self) -> bool:
        return self.status.mode == "enforcing" and self.lookup is not None

    def _submit(self, client: Endpoint, server: Endpoint) -> concurrent.futures.Future[int | None]:
        assert self.lookup is not None
        with self._pending_lock:
            if self._pending >= MAX_PENDING_LOOKUPS:
                raise RuntimeError("too many pending peer lookups")
            self._pending += 1
            if self._executor is None:
                # Dedicated, so a connection flood cannot starve the app's own
                # default executor (design "Thread pool pressure").
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="peer-uid-check"
                )
        fut = self._executor.submit(self.lookup.owner_uid, client, server)

        def _done(_f: object) -> None:
            with self._pending_lock:
                self._pending -= 1

        fut.add_done_callback(_done)
        return fut

    async def _attempt(self, client: Endpoint, server: Endpoint) -> tuple[bool, int | None]:
        """One lookup; ``(ok, uid)`` where ``ok`` is False on ERROR/timeout."""
        try:
            fut = self._submit(client, server)
            uid = await asyncio.wait_for(asyncio.wrap_future(fut), LOOKUP_TIMEOUT_S)
            return True, uid
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a fault is ERROR, decided by the caller
            self._last_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            return False, None

    def _refuse_on_error(self) -> Decision:
        self._log_refusal("error", f"lookup error after one retry: {self._last_error}")
        return Decision(False, _RESPONSE_503)

    def _admits(self, uid: int | None) -> bool:
        if uid is None:
            return False
        if uid == self._uid:
            return True
        return uid == 0 and self.lookup is not None and self.lookup.admits_root

    async def check(self, client: Endpoint, server: Endpoint) -> Decision:
        if not self.enforcing:
            return Decision(True)
        ok, uid = await self._attempt(client, server)
        if not ok:
            # ONE immediate retry (manager decision 2), then fail closed.
            ok, uid = await self._attempt(client, server)
            if not ok:
                return self._refuse_on_error()
        if uid is None:
            # Not found: an fd being passed or an exec in flight; rescan once.
            await asyncio.sleep(NOT_FOUND_RESCAN_S)
            ok, uid = await self._attempt(client, server)
            if not ok:
                return self._refuse_on_error()
        if self._admits(uid):
            return Decision(True)
        if uid is None:
            self._log_refusal(
                "unknown", f"client {client[0]}:{client[1]} is not owned by uid {self._uid}"
            )
        else:
            self._log_refusal(str(uid), f"client {client[0]}:{client[1]} is owned by uid {uid}")
        return Decision(False, _RESPONSE_403)

    def _log_refusal(self, key: str, detail: str) -> None:
        now = time.monotonic()
        last = self._last_logged.get(key)
        if last is not None and now - last < REFUSAL_LOG_INTERVAL_S:
            return
        self._last_logged[key] = now
        logger.warning(
            "peer-uid check: refused a loopback connection (%s); "
            "set %s=off if this is a client you trust",
            detail,
            ENV_VAR,
        )


def resolve_gate(
    env: Mapping[str, str] | None = None,
    *,
    lookup_factory: Callable[[], PeerLookup] = default_lookup_factory,
    detect_skip: Callable[[Mapping[str, str]], str | None] = detect_environment_skip,
    probe: Callable[[PeerLookup], str | None] = self_probe,
) -> PeerGate:
    """Decide this process's policy and log the ONE notice about it."""
    env = os.environ if env is None else env
    raw = (env.get(ENV_VAR) or "auto").strip().lower()
    note = ""
    if raw not in ("auto", "on", "off"):
        note = f" (ignored unrecognised {ENV_VAR}={raw!r}; using auto)"
        raw = "auto"
    if raw == "off":
        logger.info("peer-uid check: disabled by %s=off", ENV_VAR)
        return PeerGate(GateStatus("disabled", reason=f"{ENV_VAR}=off"))

    def _skip(reason: str) -> PeerGate:
        logger.warning(
            "peer-uid check: skipped (%s); other OS users on this host are not refused.%s "
            "Set %s=on to enforce.",
            reason,
            note,
            ENV_VAR,
        )
        return PeerGate(GateStatus("skipped", reason=reason))

    if raw == "auto":
        reason = detect_skip(env)
        if reason is not None:
            return _skip(reason)
    try:
        lookup = lookup_factory()
    except LookupError as exc:
        return _skip(str(exc))
    failure = probe(lookup)
    if failure is not None:
        return _skip(failure)
    logger.info(
        "peer-uid check: enforcing (%s); loopback connections from other OS users are refused%s",
        lookup.mechanism,
        note,
    )
    return PeerGate(GateStatus("enforcing", mechanism=lookup.mechanism), lookup)


# The process-wide resolution, started once (``ensure_resolving``) and read by
# every connection. A Future rather than a value so connections that arrive
# while the self-probe runs wait for it instead of racing it.
_resolution: concurrent.futures.Future[PeerGate] | None = None
_resolution_lock = threading.Lock()


def ensure_resolving() -> concurrent.futures.Future[PeerGate]:
    """Start resolving the policy in a background thread (idempotent)."""
    global _resolution
    with _resolution_lock:
        if _resolution is None:
            fut: concurrent.futures.Future[PeerGate] = concurrent.futures.Future()

            def _run() -> None:
                try:
                    fut.set_result(resolve_gate())
                except BaseException as exc:  # noqa: BLE001 -- never lock out on a bug here
                    logger.warning("peer-uid check: skipped (resolution failed: %s)", exc)
                    fut.set_result(
                        PeerGate(GateStatus("skipped", reason=f"resolution failed: {exc}"))
                    )

            threading.Thread(target=_run, name="peer-uid-check-resolve", daemon=True).start()
            _resolution = fut
        return _resolution


def current_status() -> dict[str, str] | None:
    """The resolved status for ``/health``, or ``None`` before/without resolution."""
    fut = _resolution
    if fut is None or not fut.done():
        return None
    return fut.result().status.as_dict()


def _is_loopback_sockname(sockname: object) -> bool:
    if not isinstance(sockname, tuple) or len(sockname) < 2:
        return False  # a Unix socket or unknown: not ours to gate
    try:
        return normalize_endpoint(str(sockname[0]), int(sockname[1]))[0].is_loopback
    except ValueError:
        return False


def make_gated_protocol(
    base: type[asyncio.Protocol],
    gate_source: Callable[[], concurrent.futures.Future[PeerGate]] = ensure_resolving,
) -> type[asyncio.Protocol]:
    """Subclass a uvicorn HTTP protocol so each connection is peer-checked first.

    ``connection_made`` is a semi-private uvicorn API; ``tests/unit/server/
    test_peer_uid_check.py`` boots real servers through this class so an
    uvicorn upgrade that changes it fails loudly.
    """

    class PeerCheckedProtocol(base):  # type: ignore[misc,valid-type]
        _peer_check_task: asyncio.Task[None] | None = None

        def connection_made(self, transport: asyncio.BaseTransport) -> None:
            super().connection_made(transport)  # type: ignore[arg-type]
            sockname = transport.get_extra_info("sockname")
            peername = transport.get_extra_info("peername")
            if not _is_loopback_sockname(sockname) or not isinstance(peername, tuple):
                return
            fut = gate_source()
            if fut.done() and not fut.result().enforcing:
                return  # skipped/disabled: zero per-connection cost
            flow = getattr(self, "flow", None)
            # Through uvicorn's FlowControl when present, so its own reading
            # state stays consistent with the transport's.
            (flow or transport).pause_reading()  # type: ignore[union-attr]
            self._peer_check_task = asyncio.get_running_loop().create_task(
                self._peer_verify(transport, fut, peername, sockname)
            )

        async def _peer_verify(
            self,
            transport: Any,
            fut: concurrent.futures.Future[PeerGate],
            peername: tuple[Any, ...],
            sockname: tuple[Any, ...],
        ) -> None:
            try:
                gate = await asyncio.wrap_future(fut)
                client = normalize_endpoint(str(peername[0]), int(peername[1]))
                server = normalize_endpoint(str(sockname[0]), int(sockname[1]))
                decision = await gate.check(client, server)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- fail closed on a bug after enforcing
                logger.warning("peer-uid check: refused a connection on an internal error: %s", exc)
                decision = Decision(False, _RESPONSE_503)
            if transport.is_closing():
                return
            if decision.admit:
                flow = getattr(self, "flow", None)
                (flow or transport).resume_reading()
                return
            transport.write(decision.response)
            transport.close()

    PeerCheckedProtocol.__name__ = f"PeerChecked{base.__name__}"
    PeerCheckedProtocol.__qualname__ = PeerCheckedProtocol.__name__
    return PeerCheckedProtocol


_DEFAULT_CLASS: type[asyncio.Protocol] | None = None


def gated_protocol_class() -> type[asyncio.Protocol]:
    """uvicorn's ``auto`` HTTP protocol (httptools if installed, else h11), gated."""
    global _DEFAULT_CLASS
    if _DEFAULT_CLASS is None:
        from uvicorn.protocols.http.auto import AutoHTTPProtocol

        _DEFAULT_CLASS = make_gated_protocol(AutoHTTPProtocol)
    return _DEFAULT_CLASS


#: Import string for ``uvicorn.run(http=...)`` (the ``--reload`` path, where the
#: app and protocol are re-imported in a child process).
PROTOCOL_IMPORT_STRING = "local_operator.server.peer_check:PeerCheckedHTTPProtocol"


def __getattr__(name: str) -> Any:
    # Lazy so importing this module (``/health``) never imports uvicorn's
    # protocol stack; uvicorn's ``import_from_string`` resolves it on demand.
    if name == "PeerCheckedHTTPProtocol":
        return gated_protocol_class()
    raise AttributeError(name)
