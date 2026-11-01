"""Checkout-local, stdlib-only signal authority for the bounded gate runner.

A remembered PGID is NOT ownership. Enrolment needs a living authenticated
creation chain; signalling needs a fresh complete census and matching births,
actual kernel PG/SID and a previously enrolled live anchor. Terminal samples are
not absence or anchors. Refusal deliberately trades cleanup for operator safety.

This is a checked owner-alive USER-SPACE boundary, NOT an atomic kernel handle:
a member/anchor may exit between the final observation and killpg, and canonical
birth tokens have finite resolution. No zero-recipient-race or hard-SIGKILL
owner cleanup guarantee is claimed. No keeper process is introduced.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from local_operator import procstate
from local_operator.mobile import resources


class Unknown(RuntimeError):
    """Incomplete proof, never a spelling of observed absence."""


def strict_darwin(pgid, deadline, *, listpids=None, clock=time.monotonic):
    """A filled libproc buffer is ambiguous, including the zero-size-query race."""
    lib = resources._darwin_libproc() if listpids is None else None
    call = listpids if listpids is not None else getattr(lib, "proc_listpids", None)
    if call is None or pgid <= 1:
        raise Unknown("strict Darwin census unsupported")
    size = ctypes.sizeof(ctypes.c_int)
    needed = call(2, pgid, None, 0)
    if type(needed) is not int or needed < 0 or needed % size:
        raise Unknown("malformed libproc size")
    capacity = max(64, needed // size + 1)
    for _ in range(4):
        if clock() >= deadline or capacity > 262144:
            raise Unknown("strict census deadline/capacity exhausted")
        buffer = (ctypes.c_int * capacity)()
        written = call(2, pgid, buffer, ctypes.sizeof(buffer))
        if type(written) is not int or written < 0 or written > ctypes.sizeof(buffer) or written % size:
            raise Unknown("malformed libproc result length")
        if written == ctypes.sizeof(buffer):
            capacity *= 2
            continue
        pids = list(buffer[:written // size])
        if any(p <= 0 for p in pids) or len(set(pids)) != len(pids):
            raise Unknown("malformed libproc PID rows")
        return sorted(pids)
    raise Unknown("saturated libproc census")


def strict_linux(pgid, deadline, *, listdir=os.listdir, read=None, stat=os.stat,
                 clock=time.monotonic):
    """Refuse omitted /proc rows; only proved ENOENT can exclude a vanished PID.

    Re-listing detects arrivals during the scan, not ownership or an atomic
    snapshot. The accepted census-to-signal race still applies on Linux.
    Existing lenient resource-monitor readers are intentionally unchanged.
    """
    read = read or (lambda p: Path(p).read_text(encoding="utf-8"))
    for _ in range(4):
        if clock() >= deadline:
            raise Unknown("strict /proc census deadline")
        first = {n for n in listdir("/proc") if n.isdigit()}
        result = []
        disappeared = set()
        for entry in first:
            if clock() >= deadline:
                raise Unknown("strict /proc row deadline")
            path = f"/proc/{entry}"
            try:
                raw = read(path + "/stat")
            except FileNotFoundError:
                try:
                    stat(path)
                except FileNotFoundError:
                    disappeared.add(entry)
                    continue
                raise Unknown("stat row missing for existing PID")
            except (OSError, UnicodeError) as exc:
                raise Unknown("unreadable /proc row") from exc
            try:
                left, tail = raw[:raw.index("(")].strip(), raw[raw.rindex(")") + 1:].split()
                if int(left) != int(entry) or len(tail) < 3 or len(tail[0]) != 1:
                    raise ValueError("invalid process stat")
                parent, group = int(tail[1]), int(tail[2])
                if parent < 0 or group < 0:
                    raise ValueError("negative process identity")
            except (ValueError, IndexError) as exc:
                raise Unknown("malformed /proc row") from exc
            if group == pgid:
                result.append(int(entry))
        last = {n for n in listdir("/proc") if n.isdigit()}
        if first - disappeared == last:
            return sorted(result)
    raise Unknown("unstable /proc census")


def strict_census(pgid, deadline):
    try:
        if sys.platform == "darwin":
            return strict_darwin(pgid, deadline)
        if sys.platform.startswith("linux"):
            return strict_linux(pgid, deadline)
    except (OSError, ValueError, TypeError) as exc:
        raise Unknown("strict membership probe failed") from exc
    raise Unknown("strict group census unsupported")


def kernel_identity(pid):
    pair = resources.direct_ppid_pgid(pid)
    if pair is None or not hasattr(os, "getsid"):
        raise Unknown("actual parent/PG/SID unavailable")
    parent, group = pair
    if os.getpgid(pid) != group:
        raise Unknown("kernel group changed while reading parent")
    return parent, group, os.getsid(pid)


@dataclass(frozen=True)
class Receipt:
    pid: int
    scheme: str
    birth: str
    parent: int
    parent_birth: str
    pgid: int
    sid: int
    observed: float


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str
    members: tuple[int, ...] = ()
    evidence_age_s: float | None = None


_SEAL = object()


class OwnedGroup:
    def __init__(self, seal, leader, *, census, samples, identity, scheme, clock, signal_group):
        if seal is not _SEAL or type(leader) is not int or leader <= 1:
            raise ValueError("numeric PGID alone cannot construct signal authority")
        self.leader = self.pgid = leader
        self.sid = leader
        self.receipts = {}
        self.census, self.samples, self.identity = census, samples, identity
        self.scheme, self.clock, self._send = scheme, clock, signal_group
        self.initial_error = "unproven initial identity"
        self.last = Verdict("REFUSED", self.initial_error)

    @classmethod
    def capture(cls, leader, *, deadline, parent_pid=None, census=strict_census,
                samples=None, identity=kernel_identity,
                scheme=procstate.birth_scheme, clock=time.monotonic, signal_group=os.killpg):
        cap = cls(_SEAL, leader, census=census, samples=samples, identity=identity,
                  scheme=scheme, clock=clock, signal_group=signal_group)
        parent = os.getpid() if parent_pid is None else parent_pid
        try:
            members = cap._members(deadline)
            if not members:
                cap.last = Verdict("OBSERVED_EMPTY", "strict initial empty census")
                return cap
            before = cap._samples([leader, parent], deadline)
            view = identity(leader)
            after = cap._samples([leader, parent], deadline)
            stamp = scheme()
            if view != (parent, leader, leader) or identity(leader) != view:
                raise Unknown("Popen leader parent/PG/SID not pinned")
            if not stamp or any(p not in before or p not in after for p in [leader, parent]):
                raise Unknown("leader/parent birth unavailable")
            for pid in [leader, parent]:
                if before[pid].zombie or after[pid].zombie or after[pid].is_birth(stamp, before[pid].birth) is not True:
                    raise Unknown("leader/parent not a matching live creation chain")
            cap.receipts[leader] = Receipt(leader, stamp, before[leader].birth, parent,
                                           before[parent].birth, leader, leader, clock())
            cap.initial_error = ""
            cap.refresh(deadline=deadline)
        except Exception as exc:
            cap.last = Verdict("REFUSED", type(exc).__name__ + ":" + str(exc), tuple(members) if 'members' in locals() else ())
        return cap

    def _members(self, deadline):
        if self.clock() >= deadline:
            raise Unknown("ownership deadline exhausted")
        members = self.census(self.pgid, deadline)
        if members is None or any(type(p) is not int or p <= 0 for p in members) or len(set(members)) != len(members):
            raise Unknown("incomplete/malformed membership")
        return sorted(members)

    def _samples(self, pids, deadline):
        # Canonical ps fallback has a fixed ONE SECOND timeout. Never start it
        # when the caller's remaining allowance cannot cover that bound.
        if deadline - self.clock() < 1.0:
            raise Unknown("remaining deadline cannot cover canonical 1s birth probe")
        return self.samples(sorted(set(pids)))

    def refresh(self, *, deadline, enrol=True):
        members = []
        try:
            members = self._members(deadline)
            if not members:
                self.last = Verdict("OBSERVED_EMPTY", "strict empty census")
                return self.last
            if not self.receipts:
                raise Unknown(self.initial_error)
            known = dict(self.receipts)
            before = self._samples(set(members) | set(known), deadline)
            views = {p: self.identity(p) for p in members}
            stamp = self.scheme()
            def matching(pid, receipt, sample):
                return sample is not None and sample.is_birth(receipt.scheme, receipt.birth) is True
            # Observed reuse is not excused by another otherwise-valid survivor.
            for pid, receipt in known.items():
                if pid in before and not matching(pid, receipt, before[pid]):
                    raise Unknown("recorded PID birth reused/unreadable")
            anchors = {p for p in members if p in known and matching(p, known[p], before.get(p))
                       and not before[p].zombie and views[p][1:] == (self.pgid, self.sid)}
            if not anchors:
                raise Unknown("no previously authenticated LIVE anchor")
            pending = set(members) - set(known)
            if pending and not enrol:
                raise Unknown("unenrolled member before signal")
            while pending:
                progressed = False
                for pid in list(pending):
                    parent, group, sid = views[pid]
                    if parent not in known or parent not in members:
                        continue
                    ps, child = before.get(parent), before.get(pid)
                    if (ps is None or child is None or ps.zombie or child.zombie
                            or not matching(parent, known[parent], ps) or not child.birth
                            or (group, sid) != (self.pgid, self.sid)):
                        raise Unknown("unprovable live ancestry enrolment")
                    known[pid] = Receipt(pid, stamp, child.birth, parent, ps.birth, group, sid, self.clock())
                    pending.remove(pid)
                    progressed = True
                if not progressed:
                    raise Unknown("member has no authenticated live ancestry")
            after = self._samples(set(members) | set(known), deadline)
            for pid in members:
                if (not matching(pid, known[pid], after.get(pid)) or self.identity(pid) != views[pid]
                        or views[pid][1:] != (self.pgid, self.sid)):
                    raise Unknown("member birth/actual PG/SID changed or unreadable")
                if pid not in self.receipts:
                    parent = known[pid].parent
                    if after[pid].zombie or parent not in after or after[parent].zombie or not matching(parent, known[parent], after[parent]):
                        raise Unknown("enrolment chain did not remain live")
            if not any(p in after and not after[p].zombie and matching(p, self.receipts[p], after[p]) for p in anchors):
                raise Unknown("live continuity anchor lost during proof")
            if self._members(deadline) != members:
                raise Unknown("group membership changed during proof")
            self.receipts = known
            self.last = Verdict("VERIFIED", "fresh complete enrolled group", tuple(members), 0.0)
        except Exception as exc:
            self.last = Verdict("REFUSED", type(exc).__name__ + ":" + str(exc), tuple(members))
        return self.last

    def signal(self, signum, *, deadline):
        result = self.refresh(deadline=deadline, enrol=False)
        if result.status != "VERIFIED":
            return result
        checked = self.clock()
        if checked >= deadline:
            self.last = Verdict("REFUSED", "expired before syscall", result.members)
            return self.last
        try:
            self._send(self.pgid, signum)
        except OSError as exc:
            self.last = Verdict("REFUSED", "checked signal failed:" + type(exc).__name__, result.members)
            return self.last
        self.last = Verdict("SIGNALED", "checked owner-alive signal; non-atomic race remains", result.members, self.clock() - checked)
        return self.last
