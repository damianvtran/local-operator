"""Who is on the other end of the socket, and may they be served (design §2.1).

This module is the security boundary of the whole feature. Everything else in
``local_operator.secrets`` decides *what* a caller gets; this decides *whether*
a caller is one of ours at all.

**The asymmetry the design spends its entire budget on.** A same-uid process
CAN read another process's environment (``ps eww``, ``KERN_PROCARGS2`` — design
§2.2, spike 2), so no secret an authorized caller could hold in an environment
variable can authenticate it: a capability token in the environment is a token
the attacker reads. A same-uid process CANNOT read another's memory without
tripping a macOS authorization prompt (``task_for_pid`` rc=5, ``lldb`` blocked
on ``system.privilege.taskport.debug`` — spikes 3 and 5). So the master key
lives in a daemon's RAM, and the only thing a caller presents is *the connection
itself*: the kernel tells the server the peer's pid, and no forgeable payload
crosses the wire.

**What is therefore deliberately absent.** There is no token, no shared secret,
no nonce and no inherited-fd capability. An inherited connection is actively
*anti*-authenticating (spike 6/7): ``LOCAL_PEERPID`` reports the pid of whoever
called ``connect()``, and that identity survives the connector's death, so a
process handed a live socket on fd 3 borrows the identity of a process that no
longer exists. Every client connects for itself; see
:mod:`local_operator.secrets.client`.

**What authorization actually means here, stated without overclaiming.** A peer
is authorized iff it is a registered, still-live lop session, or a live
descendant of one. Becoming a registered session is not free: it requires the
registration ticket from the 0700 secrets directory (design §2.1, review R1) —
before that gate existed, any process could register itself and, because the
ancestry walk yields the peer first, become its own authorizing ancestor.

That proves LINEAGE, not INTENT (design §9.1): a script that runs as a child of
an agent's shell inherits that lineage and is allowed, and a script that simply
runs ``lop`` itself becomes a legitimate session. And what it is worth depends
entirely on the tier — in ``keyfile`` mode the ticket and the master key sit in
the same directory, so an attacker who can take one has the other and this
boundary stops nobody; in ``passphrase`` mode the key is not on disk at all and
the boundary is load-bearing. §8's table states this per tier. It is a real
increase in attacker cost over a plaintext key file, and it is **not** a vault.

**Code-signature verification is not available.** ``csops(CS_OPS_CDHASH)``
returned rc=-1 for a sibling process on this machine (spike 1), so "is the peer
a genuine python running our code" cannot be asked. Anything in review that
proposes tightening this by checking the peer binary should know that was
measured and rejected, not overlooked.

Stdlib only, on purpose: the broker imports this before it has a key, and a
failure to authenticate must never depend on the crypto stack loading.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import socket
import struct
import sys
from dataclasses import dataclass

#: Ceiling on the ancestry walk. Deep enough for any real chain here (session →
#: shell → wrapper → interpreter is 4) and bounded so a pathological or
#: adversarially-constructed parent chain cannot spin the broker.
MAX_ANCESTRY_DEPTH = 24

# --- macOS: getsockopt names, measured in spike 1 -----------------------------
SOL_LOCAL = 0
LOCAL_PEERPID = 0x002
#: The audit token: ``(auid, euid, egid, ruid, rgid, pid, asid, pidversion)``.
#: ``pidversion`` is the monotonic per-process generation that makes the pid
#: pin race-free; it is the reason this is read instead of ``LOCAL_PEERPID``
#: alone. Available for the CONNECTING peer only — the kernel exposes no way to
#: read another arbitrary process's pidversion, which is why ancestors are
#: pinned by start time and unique id instead (see :func:`process_info`).
LOCAL_PEERTOKEN = 0x006

#: ``kinfo_proc`` field offsets on 64-bit Darwin. Determined EMPIRICALLY rather
#: than transcribed from a header (spike 4, re-verified on this machine before
#: this module was written): a 648-byte record whose ``p_pid`` is at 40 and
#: ``e_ppid`` at 560. The start time is at offset **0**, not the 8 a reading of
#: ``extern_proc`` suggests — offset 8 yields ``sec=289903``, a 1970s timestamp,
#: while offset 0 yields the true epoch second. Do not "fix" these to match a
#: header without re-running that probe.
_KINFO_PROC_MIN_SIZE = 564
_OFF_START_SEC = 0
_OFF_P_PID = 40
_OFF_E_PPID = 560

_CTL_KERN, _KERN_PROC, _KERN_PROC_PID = 1, 14, 1
_PROC_PIDUNIQIDENTIFIERINFO = 17

_IS_DARWIN = sys.platform == "darwin"
_IS_LINUX = sys.platform.startswith("linux")


class _UniqIdentifierInfo(ctypes.Structure):
    """``struct proc_uniqidentifierinfo`` — the per-process unique id pair.

    ``p_uniqueid`` is a monotonic 64-bit id that, unlike a pid, is never
    recycled, and ``p_puniqueid`` is the parent's. Together they let the
    ancestry walk verify each hop instead of trusting a bare ppid; see
    :func:`_walk`.
    """

    _fields_ = [
        ("p_uuid", ctypes.c_ubyte * 16),
        ("p_uniqueid", ctypes.c_uint64),
        ("p_puniqueid", ctypes.c_uint64),
        ("p_reserve2", ctypes.c_uint64),
        ("p_reserve3", ctypes.c_uint64),
        ("p_reserve4", ctypes.c_uint64),
    ]


#: Resolved C library, and whether resolution has been attempted. Two names
#: rather than a sentinel object so the cache stays typed as ``CDLL | None``:
#: a sentinel would widen it to ``object`` and every call site would need a
#: cast to use it.
_LIBC_CACHE: ctypes.CDLL | None = None
_LIBC_RESOLVED = False


def _libc() -> ctypes.CDLL | None:
    """The C library, or ``None`` where it cannot be loaded.

    Resolved lazily and tolerantly: a platform where this fails must fail
    CLOSED at the authorization call (no ancestry, no authorization), never
    raise at import and take the whole CLI down with it.
    """
    global _LIBC_CACHE, _LIBC_RESOLVED
    if not _LIBC_RESOLVED:
        _LIBC_RESOLVED = True
        try:
            name = ctypes.util.find_library("c")
            _LIBC_CACHE = ctypes.CDLL(name, use_errno=True) if name else None
        except OSError:  # pragma: no cover - platform-dependent
            _LIBC_CACHE = None
    return _LIBC_CACHE


class PeerAuthenticationUnavailable(Exception):
    """This platform cannot identify a socket peer, so nothing may be served.

    Raised rather than returning "unauthorized" so the caller can say *why* it
    is refusing. The distinction matters to an operator: "you are not a
    descendant of a lop session" is a policy answer, while this is "the broker
    cannot make that determination on this OS and is therefore refusing
    everyone". Both deny; only one is a bug report.
    """


@dataclass(frozen=True)
class ProcessIdentity:
    """A pid pinned to the specific process incarnation that held it.

    A bare pid is not an identity — pids are recycled, and a check that
    re-resolves one later can be answered by a different process than the one
    that was authorized. Every field here is captured ONCE, at connect or
    registration time, and compared as a unit afterwards.

    ``pidversion`` is the strongest of these but is only readable for a direct
    socket peer; ``unique_id`` is readable for any pid on macOS and is never
    recycled; ``start_time`` is the portable fallback and is what Linux uses.
    A comparison uses whichever fields both sides actually have (see
    :meth:`same_process_as`), so no platform silently degrades to comparing
    pids alone.
    """

    pid: int
    #: Process start time. Epoch seconds on macOS; clock ticks since boot on
    #: Linux (``/proc/<pid>/stat`` field 22). Never compared ACROSS platforms,
    #: only against another sample taken the same way on the same host.
    start_time: int
    #: macOS audit-token generation. ``None`` for ancestors and on Linux.
    pidversion: int | None = None
    #: macOS ``p_uniqueid``. ``None`` on Linux.
    unique_id: int | None = None
    #: macOS ``p_puniqueid`` — the parent's ``unique_id``, used to verify a
    #: walk hop actually links to the process we then examine.
    parent_unique_id: int | None = None

    def same_process_as(self, other: "ProcessIdentity") -> bool:
        """Is ``other`` the same process incarnation as this one?

        Compares every discriminator BOTH samples carry. The pid alone is never
        sufficient and is never the only thing compared: at least one
        recycle-proof field (``pidversion``, ``unique_id``) or the start time
        must agree, and any field present on both sides that disagrees is a
        rejection.
        """
        if self.pid != other.pid:
            return False
        if self.unique_id is not None and other.unique_id is not None:
            # Never recycled, so this settles it outright.
            return self.unique_id == other.unique_id
        if self.pidversion is not None and other.pidversion is not None:
            return self.pidversion == other.pidversion
        return self.start_time == other.start_time


def _sysctl_kinfo(pid: int) -> bytes | None:
    """Raw ``kinfo_proc`` for ``pid`` on macOS, or ``None`` if it is gone."""
    libc = _libc()
    if libc is None:
        return None
    mib = (ctypes.c_int * 4)(_CTL_KERN, _KERN_PROC, _KERN_PROC_PID, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 4, None, ctypes.byref(size), None, 0) != 0 or size.value == 0:
        return None
    buffer = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 4, buffer, ctypes.byref(size), None, 0) != 0:
        return None
    raw = buffer.raw[: size.value]
    # A dead pid yields a zero-length record rather than an error; treating a
    # short buffer as "gone" keeps the walk from unpacking garbage offsets.
    if len(raw) < _KINFO_PROC_MIN_SIZE:
        return None
    return raw


def _unique_ids(pid: int) -> tuple[int, int] | None:
    """``(p_uniqueid, p_puniqueid)`` for ``pid`` on macOS, or ``None``."""
    libc = _libc()
    if libc is None or not hasattr(libc, "proc_pidinfo"):
        return None
    info = _UniqIdentifierInfo()
    written = libc.proc_pidinfo(
        ctypes.c_int(pid),
        ctypes.c_int(_PROC_PIDUNIQIDENTIFIERINFO),
        ctypes.c_uint64(0),
        ctypes.byref(info),
        ctypes.c_int(ctypes.sizeof(info)),
    )
    if written != ctypes.sizeof(info):
        return None
    return int(info.p_uniqueid), int(info.p_puniqueid)


def _linux_stat(pid: int) -> tuple[int, int] | None:
    """``(ppid, start_ticks)`` from ``/proc/<pid>/stat``, or ``None`` if gone.

    Parsed from the LAST ``)`` rather than by splitting on whitespace: field 2
    is the executable name in parentheses and may itself contain spaces and
    parentheses, which is the classic way a naive ``/proc`` parser is fooled
    into reading the wrong field — and here the wrong field would be a
    security decision.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read()
    except (OSError, ValueError):
        return None
    end_of_comm = data.rfind(b")")
    if end_of_comm == -1:
        return None
    fields = data[end_of_comm + 2 :].split()
    # After the comm, field indices shift: state is 0, ppid 1, starttime 19.
    if len(fields) < 20:
        return None
    try:
        return int(fields[1]), int(fields[19])
    except ValueError:
        return None


def process_info(pid: int) -> ProcessIdentity | None:
    """Identify a live process by pid, or return ``None`` if it is gone.

    ``None`` is a refusal, not an error: a pid that vanishes mid-walk cannot be
    authorized, because whatever replaces it is a different process and the
    whole point of this module is to never conflate the two.
    """
    if pid <= 0:
        return None
    if _IS_DARWIN:
        raw = _sysctl_kinfo(pid)
        if raw is None:
            return None
        start_time = struct.unpack_from("=q", raw, _OFF_START_SEC)[0]
        # Guard the offsets rather than trusting them: if a future OS moves the
        # layout, a mismatched p_pid means every subsequent field is garbage
        # and the only safe response is to stop authorizing, not to guess.
        if struct.unpack_from("=i", raw, _OFF_P_PID)[0] != pid:
            return None
        unique = _unique_ids(pid)
        return ProcessIdentity(
            pid=pid,
            start_time=int(start_time),
            unique_id=unique[0] if unique else None,
            parent_unique_id=unique[1] if unique else None,
        )
    if _IS_LINUX:
        stat = _linux_stat(pid)
        if stat is None:
            return None
        return ProcessIdentity(pid=pid, start_time=stat[1])
    return None


def parent_pid(pid: int) -> int | None:
    """The parent pid of ``pid``, or ``None`` if it cannot be determined."""
    if _IS_DARWIN:
        raw = _sysctl_kinfo(pid)
        if raw is None:
            return None
        return int(struct.unpack_from("=i", raw, _OFF_E_PPID)[0])
    if _IS_LINUX:
        stat = _linux_stat(pid)
        return None if stat is None else stat[0]
    return None


def peer_identity(connection: socket.socket) -> ProcessIdentity:
    """Identify the process at the other end of ``connection``.

    **This is the pin.** Everything downstream compares against the value
    returned here, captured at connect time, and nothing ever re-resolves a
    bare pid afterwards — re-resolution is precisely how a pid-reuse attack
    wins, because the answer arrives from whatever process holds the number by
    then rather than from the one that connected.

    On macOS the audit token is preferred over ``LOCAL_PEERPID`` because it
    carries ``pidversion`` in the same atomic read as the pid, so the pid and
    its generation cannot disagree. On Linux ``SO_PEERCRED`` yields the pid
    directly and the start time from ``/proc`` completes the pin.

    Raises :class:`PeerAuthenticationUnavailable` on a platform that offers
    neither, so the broker can refuse everyone with an accurate reason rather
    than authorizing on an identity it never established.
    """
    if _IS_DARWIN:
        try:
            raw = connection.getsockopt(SOL_LOCAL, LOCAL_PEERTOKEN, 32)
            token = struct.unpack("=8I", raw[:32])
            pid, pidversion = int(token[5]), int(token[7])
        except OSError as exc:
            raise PeerAuthenticationUnavailable(
                f"could not read the peer's audit token: {exc}"
            ) from exc
        identity = process_info(pid)
        if identity is None:
            raise PeerAuthenticationUnavailable(
                f"peer pid {pid} was gone before it could be identified"
            )
        return ProcessIdentity(
            pid=pid,
            start_time=identity.start_time,
            pidversion=pidversion,
            unique_id=identity.unique_id,
            parent_unique_id=identity.parent_unique_id,
        )

    if _IS_LINUX:
        try:
            # SO_PEERCRED does not exist in the macOS socket module, so this
            # cannot be a bare attribute access: peer.py is imported on both
            # platforms and an AttributeError at import would take the CLI
            # down. 17 is the Linux constant (<asm-generic/socket.h>).
            so_peercred = getattr(socket, "SO_PEERCRED", 17)
            raw = connection.getsockopt(socket.SOL_SOCKET, so_peercred, struct.calcsize("3i"))
            pid = int(struct.unpack("3i", raw)[0])
        except OSError as exc:
            raise PeerAuthenticationUnavailable(
                f"could not read SO_PEERCRED from the peer: {exc}"
            ) from exc
        identity = process_info(pid)
        if identity is None:
            raise PeerAuthenticationUnavailable(
                f"peer pid {pid} was gone before it could be identified"
            )
        return identity

    raise PeerAuthenticationUnavailable(
        f"peer authentication is not implemented on {sys.platform}; the secret broker "
        "refuses every request rather than serving one it cannot attribute"
    )


def _walk(start: int, max_depth: int = MAX_ANCESTRY_DEPTH):
    """Yield ``ProcessIdentity`` for ``start`` and each ancestor, upward.

    Stops at pid 1 INCLUSIVE — pid 1 is yielded, then the walk ends. An earlier
    version of this loop (spike 9) used ``while pid > 1``, which never examines
    pid 1 itself; that is invisible on a normal macOS host where no session is
    pid 1, and wrong inside a container where the session frequently IS pid 1.
    Measured: the same probe denied a legitimate descendant in Docker until the
    bound was corrected.

    A hop is only followed when the child's recorded parent unique id matches
    the parent we are about to examine (macOS, where that field exists). That
    closes a TOCTOU in the walk: if a parent exits mid-walk and its pid is
    recycled, the bare ppid would lead into an unrelated process, and this
    stops rather than continuing into it.
    """
    seen: set[int] = set()
    pid = start
    for _ in range(max_depth):
        if pid <= 0 or pid in seen:
            return
        seen.add(pid)
        identity = process_info(pid)
        if identity is None:
            return
        yield identity
        if pid == 1:
            return
        parent = parent_pid(pid)
        if parent is None or parent <= 0:
            return
        if identity.parent_unique_id is not None:
            parent_identity = process_info(parent)
            if parent_identity is None or parent_identity.unique_id is None:
                return
            if parent_identity.unique_id != identity.parent_unique_id:
                # The pid we were pointed at is no longer the process that
                # actually fathered this one. Refuse to keep walking.
                return
        pid = parent


def authorize(peer: ProcessIdentity, sessions: dict[int, ProcessIdentity]) -> tuple[bool, str]:
    """Is ``peer`` a registered session, or a live descendant of one?

    ``sessions`` maps pid to the identity pinned when that session registered.
    A pid found in it is only accepted when the identity still matches, so a
    session that died and whose pid was reissued to an attacker's process
    authorizes nothing.

    **A peer that is itself registered is authorized as itself, not by its own
    ancestry (review R1).** ``_walk`` yields the peer as the first element of
    its own chain, so treating that hop like any other made every process that
    registered itself its own authorizing ancestor — the second half of the
    unauthenticated-``register`` bypass. Gating ``register`` closed the door;
    this keeps the peer from being its own key. Standing to register is proven
    by the ticket (:func:`~local_operator.secrets.keys.registration_ticket`),
    and only then does a session speak for itself here.

    Returns ``(allowed, reason)``; the reason is recorded in the audit trail
    and shown to the operator, so it is written to be read by a human deciding
    whether a denial was correct.
    """
    if not sessions:
        return False, "no lop session is registered with the broker"
    depth = 0
    for index, identity in enumerate(_walk(peer.pid)):
        registered = sessions.get(identity.pid)
        if registered is None:
            depth += 1
            continue
        # A registered pid whose process is not the one that registered it: the
        # session died and the pid came back around. Never authorized, and the
        # walk stops rather than continuing past an impostor.
        if not registered.same_process_as(identity):
            return False, (
                f"pid {identity.pid} is not the session that registered it "
                "(the registered session has exited and its pid was reused)"
            )
        if index > 0:
            return True, (
                f"descendant of registered session {identity.pid} "
                f"(start {identity.start_time} matches)"
            )
        # The peer IS the session. Verify the connect-time pin, which is what
        # makes `peer_identity`'s "everything downstream compares against the
        # value returned here" contract true (review R7): ``peer`` carries the
        # pidversion the kernel reported for THIS connection, and a re-read of
        # the pid must still be that same incarnation.
        if not registered.same_process_as(peer):
            return False, (
                f"pid {identity.pid} is registered but is not the process on this "
                "connection (the registered session has exited and its pid was reused)"
            )
        return True, f"registered session {identity.pid} acting for itself"
    return False, f"no registered lop session among {depth} ancestor(s)"
