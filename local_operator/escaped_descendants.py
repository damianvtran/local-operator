"""Find, reap and prove gone the descendants that LEFT the owned process group.

WHY THIS EXISTS
---------------
``scripts/run_bounded.py`` owns exactly one process group: the leader is held
unreaped, so the group number is reserved and every member is provably the
command's (see :mod:`local_operator.owned_process_group`). A descendant that
calls ``setsid()`` (Chrome, Electron helpers, a daemonising server) or
``setpgid()`` into another group is outside that reservation. Before this module
the cleanup-required certificate described the owned group only, and the stderr
summary still read ``cleanup=proven-empty`` -- measured 2026-11-19 by lopdev
fd36: an ``rc=124`` run whose Chrome (own pgid AND session) outlived the reap
while the record said ``cleanup=proven-empty``. Reproduced here with a child
that forks, ``setsid()``s and sleeps: rc 124, ``cleanup=proven-empty``, and the
sleeper alive afterwards under ``systemd --user`` (pid 3058, the nearest
subreaper on that host).

HOW THE POPULATION IS MADE COUNTABLE
-------------------------------------
Parentage survives ``setsid``, but NOT the death of the parent: an orphan
re-parents to the nearest child subreaper (or init), and from that moment no
``ppid`` walk from the wrapper can find it. On Linux the wrapper therefore marks
itself ``PR_SET_CHILD_SUBREAPER`` before spawning, so every orphan of the command
-- including a double-forked daemon whose middle process exited at once --
re-parents to the WRAPPER and stays inside its ``ppid`` closure. That closure is
then a complete census of everything the command ever started that is still
alive or unreaped.

The closure also contains processes that are NOT the command's: the wrapper's
own helpers (they share the wrapper's session) and, for an in-process caller such
as a test, children it already had before the spawn. Neither can ever share a
session with the command: the leader starts a NEW session, and a process can
only ever leave its session for another new one (``setsid``), never join an
existing one. So a closure member is the command's exactly when its session is
neither the wrapper's own nor one that existed among the wrapper's descendants
before the spawn (``baseline``).

WHAT IS PROVEN AND WHAT IS NOT
------------------------------
* Proven gone: two consecutive complete ``/proc`` scans (no unreadable row)
  find no live escaped member. Zombies that are our
  direct children are reaped (``waitpid`` on that exact pid) so they cannot
  linger in the next census; a zombie is not live.
* Unproven, named by pid: a member still alive at the deadline, one we were not
  permitted to signal, an unreadable census, or a host where the subreaper could
  not be set (macOS has no ``prctl``; cleanup-required already refuses before
  spawn there, so this is a defensive branch) -- an orphan re-parented elsewhere
  is then invisible, and invisibility is never reported as absence.
* PID reuse: each signal re-reads the row immediately before ``kill`` and sends
  only when ``starttime`` AND parent are unchanged from the scan; a recycled pid
  has a different birth and is skipped, never signalled. Direct-child zombies
  cannot be recycled until we reap them.
* Signals go per pid, never ``killpg``: a pgid number outside our reservation
  can be reused once its members die, and every member of an escaped group is in
  the census anyway.

Owned-group members are deliberately excluded from signalling: that group stays
under :class:`~local_operator.owned_process_group.OwnedGroup` authority. Only its
orphaned ZOMBIES (re-parented to us by the subreaper) are reaped here, because
without the subreaper init would have reaped them and the terminal census would
otherwise see them.

Stdlib only, so the ``python -S`` bootstrap of ``run_bounded`` holds.
"""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import sys
import time
from dataclasses import dataclass, field

#: linux/prctl.h. Not exported by the stdlib on any Python version.
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37

_POLL_SECONDS = 0.02


@dataclass(frozen=True)
class Row:
    pid: int
    state: str
    ppid: int
    pgid: int
    sid: int
    threads: int
    birth: str

    @property
    def dead(self) -> bool:
        # A zombie whose leader thread exited while others still run is NOT
        # dead: its threads can still fork (same rule as owned_process_group's
        # ``_exited``). Treat it as live so it is never reported gone.
        return self.state in ("Z", "X") and self.threads <= 1


class CensusUnknown(Exception):
    """A scan that could not see every row; never a spelling of absence."""


def _read_row(pid: int) -> Row | None:
    """One ``/proc/<pid>/stat`` row; None only when the kernel says it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            data = handle.read().decode("ascii", "replace")
    except OSError as exc:
        # ESRCH: a task exiting between open() and read() (measured on Linux
        # CI, see owned_process_group.strict_linux). Both mean "gone".
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return None
        raise CensusUnknown(f"unreadable /proc/{pid}/stat (errno {exc.errno})") from exc
    # comm is parenthesised and may contain spaces/parentheses: split after the
    # LAST ')'. tail[0]=state(3) ppid(4) pgrp(5) session(6) ... threads(20)=17,
    # starttime(22)=19.
    tail = data.rpartition(")")[2].split()
    if len(tail) < 20 or not tail[17].isdigit() or not tail[19].isdigit():
        raise CensusUnknown(f"short /proc/{pid}/stat row")
    return Row(
        pid=pid,
        state=tail[0],
        ppid=int(tail[1]),
        pgid=int(tail[2]),
        sid=int(tail[3]),
        threads=int(tail[17]),
        birth=tail[19],
    )


def _scan() -> dict[int, Row]:
    """Every row; a pid that exits mid-scan is simply gone, not unknown.

    A vanished descendant is not a disproof: its children re-parent to the
    subreaper and appear in the next scan, and the sweep demands two
    consecutive clean scans.
    """
    rows: dict[int, Row] = {}
    try:
        names = os.listdir("/proc")
    except OSError as exc:
        raise CensusUnknown(f"/proc not listable (errno {exc.errno})") from exc
    for name in names:
        if not name.isdigit():
            continue
        row = _read_row(int(name))
        if row is None:
            continue
        rows[row.pid] = row
    return rows


def _closure(rows: dict[int, Row], root: int) -> dict[int, Row]:
    children: dict[int, list[int]] = {}
    for row in rows.values():
        children.setdefault(row.ppid, []).append(row.pid)
    found: dict[int, Row] = {}
    stack = list(children.get(root, ()))
    while stack:
        pid = stack.pop()
        if pid in found or pid == root:
            continue
        found[pid] = rows[pid]
        stack.extend(children.get(pid, ()))
    return found


def _libc():
    # CDLL(None) is the running process's own symbol table, which on Linux
    # includes libc's prctl. ctypes.util.find_library is avoided on purpose: on
    # Linux it FORKS ldconfig/gcc, which is both slow on a loaded host and a
    # subprocess.Popen call the wrapper's tests forbid.
    return ctypes.CDLL(None, use_errno=True)


def enable_subreaper() -> tuple[bool, bool | None]:
    """Mark this process a child subreaper. Returns (enabled, previous value).

    Linux only. ``previous`` is what :func:`restore_subreaper` puts back, so an
    in-process caller (a test) is not left a subreaper after the run.
    """
    if not sys.platform.startswith("linux"):
        return False, None
    try:
        libc = _libc()
        previous = ctypes.c_int(0)
        if libc.prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(previous), 0, 0, 0) != 0:
            return False, None
        if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            return False, None
        return True, bool(previous.value)
    except (OSError, AttributeError):
        return False, None


def restore_subreaper(previous: bool | None) -> None:
    if previous is None or previous:
        return
    try:
        libc = _libc()
        libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
    except (OSError, AttributeError):
        pass


@dataclass
class Tracker:
    """The command's escaped population, relative to one wrapper process."""

    root: int
    owned_pgid: int
    subreaper: bool
    own_sid: int = field(default_factory=lambda: os.getsid(0))
    #: Sessions present among the root's descendants before the spawn: not ours.
    baseline_sids: frozenset[int] = frozenset()

    @classmethod
    def before_spawn(cls, *, subreaper: bool, root: int | None = None) -> "Tracker":
        root = os.getpid() if root is None else root
        try:
            rows = _scan()
            baseline = frozenset(row.sid for row in _closure(rows, root).values())
        except CensusUnknown:
            # Without a baseline an in-process caller's own foreign-session
            # children would be indistinguishable from escapees. Signal nothing
            # we cannot attribute: an unreadable baseline disables the sweep's
            # authority (sweep() then reports unproven).
            baseline = None  # type: ignore[assignment]
        tracker = cls(root=root, owned_pgid=0, subreaper=subreaper)
        tracker.baseline_sids = baseline if baseline is not None else frozenset()
        tracker._baseline_ok = baseline is not None
        return tracker

    _baseline_ok: bool = True

    def bind(self, owned_pgid: int) -> None:
        self.owned_pgid = owned_pgid

    def _ours(self, row: Row) -> bool:
        return row.sid != self.own_sid and row.sid not in self.baseline_sids

    def _census(self) -> tuple[dict[int, Row], dict[int, Row]]:
        """(escaped members, owned-group orphans we parent)."""
        rows = _scan()
        escaped: dict[int, Row] = {}
        orphans: dict[int, Row] = {}
        for pid, row in _closure(rows, self.root).items():
            if not self._ours(row):
                continue
            if row.pgid == self.owned_pgid:
                # The leader is the reservation: never reaped here.
                if row.ppid == self.root and pid != self.owned_pgid:
                    orphans[pid] = row
                continue
            escaped[pid] = row
        return escaped, orphans

    def _reap_zombies(self, rows: dict[int, Row]) -> None:
        for pid, row in rows.items():
            if row.ppid != self.root or not row.dead or pid == self.owned_pgid:
                continue
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass

    def reap_orphans(self) -> None:
        """Reap our command's zombie orphans so they never pile up mid-run.

        Best-effort and signal-free: an unreadable census just skips this tick.
        """
        if not self.subreaper or not self._baseline_ok:
            return
        try:
            escaped, orphans = self._census()
        except CensusUnknown:
            return
        self._reap_zombies(escaped)
        self._reap_zombies(orphans)

    def sweep(self, *, deadline: float, signum: int = signal.SIGKILL) -> "SweepResult":
        """Signal every live escaped member until two clean scans prove none.

        Called after the owned group has been killed; ``deadline`` is the run's
        shared cleanup bound. Never raises for a census problem: it reports it.
        """
        result = SweepResult(subreaper=self.subreaper)
        if not self._baseline_ok:
            result.reason = "pre-spawn descendant baseline unreadable"
            return result
        clean_scans = 0
        while True:
            try:
                escaped, orphans = self._census()
            except CensusUnknown as exc:
                result.reason = str(exc)
                clean_scans = 0
                if time.monotonic() >= deadline:
                    return result
                time.sleep(_POLL_SECONDS)
                continue
            self._reap_zombies(escaped)
            self._reap_zombies(orphans)
            live = {pid: row for pid, row in escaped.items() if not row.dead}
            result.remaining = sorted(live)
            if not live:
                clean_scans += 1
                if clean_scans >= 2:
                    result.reason = ""
                    result.proven = self.subreaper
                    if not self.subreaper:
                        result.reason = (
                            "no child subreaper on this host: an orphan re-parented "
                            "elsewhere is invisible to this census"
                        )
                    return result
                # Confirm at once, deadline or not: a clean scan costs one /proc
                # walk, and a caller whose bound is already spent (or a fake
                # clock) must not turn an empty census into "unproven".
                continue
            clean_scans = 0
            for pid, row in live.items():
                # Re-read IMMEDIATELY before the kill: a pid recycled since the
                # scan has a different birth (or parent) and is never signalled.
                try:
                    now = _read_row(pid)
                except CensusUnknown:
                    continue
                if now is None or now.birth != row.birth or now.ppid != row.ppid:
                    continue
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    continue
                except OSError:
                    if pid not in result.unsignalled:
                        result.unsignalled.append(pid)
                    continue
                if pid not in result.signalled:
                    result.signalled.append(pid)
            if time.monotonic() >= deadline:
                result.reason = "escaped descendant(s) still alive at the cleanup deadline"
                return result
            time.sleep(_POLL_SECONDS)


@dataclass
class SweepResult:
    subreaper: bool
    proven: bool = False
    remaining: list[int] = field(default_factory=list)
    signalled: list[int] = field(default_factory=list)
    unsignalled: list[int] = field(default_factory=list)
    reason: str = "not swept"

    def describe(self) -> str:
        if self.proven:
            return "escaped descendants proven gone"
        names = ",".join(str(pid) for pid in self.remaining) or "none-visible"
        return f"escaped descendants unproven ({self.reason}); live={names}"
