"""Checkout-local signal authority for the bounded gate runner.

Stdlib plus this checkout's own leaves (``procstate``, ``mobile.resources``); no
third-party import and no installation, so ``python -S`` bootstrap holds.

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
    # More attempts than the four that used to abort a healthy gate: each attempt
    # is a fresh complete scan, so retrying cannot manufacture identity — it only
    # stops ordinary fork churn from being read as lost authority. The deadline,
    # not this count, bounds the work.
    for _ in range(6):
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
                # The directory exists but the row did not read: retry it once.
                # A transient miss during churn is not a malformed row, and only a
                # second miss is genuinely unreadable identity.
                try:
                    raw = read(path + "/stat")
                except FileNotFoundError:
                    raise Unknown("stat row missing for existing PID")
                except (OSError, UnicodeError) as exc:
                    raise Unknown("unreadable /proc row") from exc
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


def bounded_samples(pids, deadline, clock=time.monotonic):
    """Use canonical native tokens first; budget only the fixed ps fallback.

    Native readers are fork-free. An unavailable native reader is NOT permission
    to spend its canonical one-second subprocess fallback in a 0.25s cleanup.
    Missing per-PID samples remain unknown, never invented births or absence.
    """
    wanted = sorted(set(pids))
    if clock() >= deadline:
        raise Unknown("birth sample deadline exhausted")
    if sys.platform == "darwin":
        result = procstate._sysctl_samples(wanted)
        if result is not None:
            if clock() >= deadline:
                raise Unknown("native birth sample exceeded deadline")
            return result
    elif sys.platform.startswith("linux"):
        result = procstate._proc_samples(wanted)
        if clock() >= deadline:
            raise Unknown("native birth sample exceeded deadline")
        return result
    if deadline - clock() < 1.0:
        raise Unknown("remaining deadline cannot cover canonical ps fallback")
    result = procstate.process_samples(wanted)
    if clock() >= deadline:
        raise Unknown("canonical birth fallback exceeded deadline")
    return result


def kernel_identity(pid):
    pair = resources.direct_ppid_pgid(pid)
    if pair is None or not hasattr(os, "getsid"):
        raise Unknown("actual parent/PG/SID unavailable")
    parent, group = pair
    if os.getpgid(pid) != group:
        raise Unknown("kernel group changed while reading parent")
    return parent, group, os.getsid(pid)


def _observed(before, after):
    """What the capability BELIEVED about each pid, for a receipt that must not be empty.

    On the acquisition-failure path ``receipts`` is empty by construction, and that
    is exactly the run whose evidence a caller needs, so the attempts are kept
    separately from the enrolled proof.
    """
    rows = {}
    for stage, samples in (("before", before or {}), ("after", after or {})):
        for pid, sample in samples.items():
            rows.setdefault(str(pid), {})[stage] = {
                "birth": sample.birth, "zombie": sample.zombie}
    return rows


@dataclass(frozen=True)
class Receipt:
    pid: int
    #: Never empty: both enrolment paths refuse a falsy scheme before writing one.
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
        #: Attempted/observed identity, kept even when proof fails.
        self.probe = {"leader": leader, "parent": None, "members": [], "observed": {},
                      "scheme": None}
        self.census, self.samples, self.identity = census, samples, identity
        self.scheme, self.clock, self._send = scheme, clock, signal_group
        self.initial_error = "unproven initial identity"
        self.last = Verdict("REFUSED", self.initial_error)

    @classmethod
    def capture(cls, leader, *, deadline, parent_pid=None, census=strict_census,
                samples=None, identity=kernel_identity,
                scheme=procstate.birth_scheme, clock=time.monotonic, signal_group=None):
        if signal_group is None:
            # Resolved HERE, never as a keyword default: a default is evaluated when
            # the class body runs — at IMPORT — and `os.killpg` does not exist on
            # Windows, so `signal_group=os.killpg` turned an unsupported platform
            # into an ImportError for every caller of this module (CI
            # xplat-probe-windows, job 124804637518). None means "this platform has
            # no checked signalling", which the signal path already refuses.
            signal_group = getattr(os, "killpg", None)
        cap = cls(_SEAL, leader, census=census, samples=samples, identity=identity,
                  scheme=scheme, clock=clock, signal_group=signal_group)
        parent = os.getpid() if parent_pid is None else parent_pid
        cap.probe["parent"] = parent
        try:
            members = cap._members(deadline)
            cap.probe["members"] = list(members)
            if not members:
                cap.last = Verdict("OBSERVED_EMPTY", "strict initial empty census")
                return cap
            before = cap._samples([leader, parent], deadline)
            cap.probe["observed"] = _observed(before, None)
            view = identity(leader)
            after = cap._samples([leader, parent], deadline)
            cap.probe["observed"] = _observed(before, after)
            stamp = scheme()
            cap.probe["scheme"] = stamp or None
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
            cap.probe["error"] = type(exc).__name__ + ":" + str(exc)
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
        if self.clock() >= deadline:
            raise Unknown("ownership sample deadline exhausted")
        if self.samples is None:
            return bounded_samples(pids, deadline, self.clock)
        result = self.samples(sorted(set(pids)))
        if self.clock() >= deadline:
            raise Unknown("injected sample exceeded deadline")
        return result

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
            # Keep the record honest: what the LAST attempt saw, not only what the
            # first capture did. Every refresh replaces these fields.
            self.probe["members"] = list(members)
            self.probe["observed"] = _observed(before, None)
            views = {p: self.identity(p) for p in members}
            # Same rule as `capture`: a scheme this build cannot produce is not
            # proof, and writing a scheme-less receipt would merely read as
            # "unreadable" one step later instead of refusing at the point of proof.
            stamp = self.scheme()
            self.probe["scheme"] = stamp or None
            if not stamp:
                raise Unknown("birth scheme unavailable at enrolment")
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
            self.probe["observed"] = _observed(before, after)
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
            self.probe["error"] = type(exc).__name__ + ":" + str(exc)
        return self.last

    def signal(self, signum, *, deadline):
        result = self.refresh(deadline=deadline, enrol=False)
        if result.status != "VERIFIED":
            return result
        if self._send is None:
            # A platform without the checked primitive: refuse, do not raise. Same
            # bounded refusal the capability reports for any other unusable sender.
            self.last = Verdict("REFUSED", "checked signal unsupported on this platform",
                                result.members)
            return self.last
        checked = self.clock()
        if checked >= deadline:
            self.last = Verdict("REFUSED", "expired before syscall", result.members)
            return self.last
        try:
            self._send(self.pgid, signum)
        except ProcessLookupError:
            # ESRCH from the checked syscall: the kernel says this group is
            # already gone. That is the state a passing gate wants to reach, NOT
            # lost authority — mapping it to REFUSED turned a completed gate into
            # "125: did not run" whenever its residual members exited in the
            # window between the final census and this call.
            self.last = Verdict("OBSERVED_EMPTY", "checked signal: group absent (ESRCH)",
                                result.members)
            return self.last
        except PermissionError:
            # EPERM is genuine doubt: the group exists and we could not reach it.
            self.last = Verdict("REFUSED", "checked signal denied (EPERM)", result.members)
            return self.last
        except OSError as exc:
            self.last = Verdict("REFUSED", "checked signal failed:" + type(exc).__name__, result.members)
            return self.last
        self.last = Verdict("SIGNALED", "checked owner-alive signal; non-atomic race remains", result.members, self.clock() - checked)
        return self.last
