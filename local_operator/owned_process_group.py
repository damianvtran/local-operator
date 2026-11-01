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
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from local_operator import procstate
from local_operator.mobile import resources


class Unknown(RuntimeError):
    """Incomplete proof, never a spelling of observed absence."""


class PinLost(Unknown):
    """The group number is not provably ours any more: refuse, never signal.

    A distinct class, not a message: the wrapper records the refusal reason from
    the TYPE, and a caller that cannot tell "this is not ours" from "we could not
    read it" would retry the wrong one.
    """


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
        if (
            type(written) is not int
            or written < 0
            or written > ctypes.sizeof(buffer)
            or written % size
        ):
            raise Unknown("malformed libproc result length")
        if written == ctypes.sizeof(buffer):
            capacity *= 2
            continue
        pids = list(buffer[: written // size])
        if any(p <= 0 for p in pids) or len(set(pids)) != len(pids):
            raise Unknown("malformed libproc PID rows")
        return sorted(pids)
    raise Unknown("saturated libproc census")


def strict_linux(
    pgid, deadline, *, listdir=os.listdir, read=None, stat=os.stat, clock=time.monotonic
):
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
                left, tail = raw[: raw.index("(")].strip(), raw[raw.rindex(")") + 1 :].split()
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


def _identities(receipts, samples, members):
    """The exact set a signal was proven against: pid, birth, group and session.

    Recorded for BOTH outcomes — what was signalled and what was declined — so a
    later reader never has to reconstruct which pids a run was willing to touch.
    """
    rows = []
    for pid in members:
        receipt = receipts.get(pid)
        sample = samples.get(pid)
        rows.append(
            {
                "pid": pid,
                "birth": (
                    receipt.birth if receipt is not None else (sample.birth if sample else None)
                ),
                "scheme": receipt.scheme if receipt is not None else None,
                "pgid": receipt.pgid if receipt is not None else None,
                "sid": receipt.sid if receipt is not None else None,
            }
        )
    return rows


def _observed(before, after):
    """What the capability BELIEVED about each pid, for a receipt that must not be empty.

    On the acquisition-failure path ``receipts`` is empty by construction, and that
    is exactly the run whose evidence a caller needs, so the attempts are kept
    separately from the enrolled proof.
    """
    rows = {}
    for stage, samples in (("before", before or {}), ("after", after or {})):
        for pid, sample in samples.items():
            rows.setdefault(str(pid), {})[stage] = {"birth": sample.birth, "zombie": sample.zombie}
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
    #: What held the group number at decision time: "leader-reservation" (an
    #: unreaped corpse the census can see) or "leader-live" (no corpse needed).
    anchor: str | None = None
    #: Why authority was refused, as a class the record can carry: "pin_lost" or
    #: "unproven". ``None`` on a VERIFIED/OBSERVED_EMPTY verdict.
    refusal_class: str | None = None


_SEAL = object()


class OwnedGroup:
    def __init__(self, seal, leader, *, census, samples, identity, scheme, clock, signal_group):
        if seal is not _SEAL or type(leader) is not int or leader <= 1:
            raise ValueError("numeric PGID alone cannot construct signal authority")
        self.leader = self.pgid = leader
        self.sid = leader
        self.receipts = {}
        #: Attempted/observed identity, kept even when proof fails.
        self.probe = {
            "leader": leader,
            "parent": None,
            "members": [],
            "observed": {},
            "scheme": None,
            #: The exact proven set and the exact declined set, carried into the
            #: run record's `ownership` object on every run.
            "proof": [],
            "residuals": [],
        }
        self.census, self.samples, self.identity = census, samples, identity
        self.scheme, self.clock, self._send = scheme, clock, signal_group
        self.initial_error = "unproven initial identity"
        self.last = Verdict("REFUSED", self.initial_error)

    @classmethod
    def capture(
        cls,
        leader,
        *,
        deadline,
        parent_pid=None,
        census=strict_census,
        samples=None,
        identity=kernel_identity,
        scheme=procstate.birth_scheme,
        clock=time.monotonic,
        signal_group=None,
    ):
        if signal_group is None:
            # Resolved HERE, never as a keyword default: a default is evaluated when
            # the class body runs — at IMPORT — and `os.killpg` does not exist on
            # Windows, so `signal_group=os.killpg` turned an unsupported platform
            # into an ImportError for every caller of this module (CI
            # xplat-probe-windows, job 124804637518). None means "this platform has
            # no checked signalling", which the signal path already refuses.
            signal_group = getattr(os, "killpg", None)
        cap = cls(
            _SEAL,
            leader,
            census=census,
            samples=samples,
            identity=identity,
            scheme=scheme,
            clock=clock,
            signal_group=signal_group,
        )
        parent = os.getpid() if parent_pid is None else parent_pid
        cap.probe["parent"] = parent
        # SIGCHLD=SIG_IGN means the kernel reaps our children for us: there is no
        # corpse, so the reservation this design rests on simply does not exist.
        # Proceeding would be a pin that is never held. Refuse with a name.
        sigchld = getattr(signal, "SIGCHLD", None)
        if sigchld is not None and signal.getsignal(sigchld) == signal.SIG_IGN:
            cap.last = Verdict(
                "REFUSED",
                "SIGCHLD is ignored: no corpse, so no reservation",
                refusal_class="no_corpse_sigchld",
            )
            cap.probe["error"] = "SIGCHLD is ignored: no corpse, so no reservation"
            return cap
        # Bound before the try: the refusal verdict reports it even when the census
        # itself raised, which is exactly the case a reader needs to see.
        members: list[int] = []
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
                if (
                    before[pid].zombie
                    or after[pid].zombie
                    or after[pid].is_birth(stamp, before[pid].birth) is not True
                ):
                    raise Unknown("leader/parent not a matching live creation chain")
            # A receipt carries PROVEN births only. The predicate above already
            # refuses an unreadable birth; saying it here makes the refusal
            # explicit instead of relying on `is_birth(...) is not True` to imply it.
            ledger_birth, parent_birth = before[leader].birth, before[parent].birth
            if ledger_birth is None or parent_birth is None:
                raise Unknown("leader/parent birth unavailable")
            cap.receipts[leader] = Receipt(
                leader,
                stamp,
                ledger_birth,
                parent,
                parent_birth,
                leader,
                leader,
                clock(),
            )
            cap.initial_error = ""
            cap.refresh(deadline=deadline)
        except Exception as exc:
            cap.probe["error"] = type(exc).__name__ + ":" + str(exc)
            cap.last = Verdict(
                "REFUSED",
                type(exc).__name__ + ":" + str(exc),
                tuple(members) if "members" in locals() else (),
            )
        return cap

    def _members(self, deadline):
        if self.clock() >= deadline:
            raise Unknown("ownership deadline exhausted")
        members = self.census(self.pgid, deadline)
        if (
            members is None
            or any(type(p) is not int or p <= 0 for p in members)
            or len(set(members)) != len(members)
        ):
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

    def leader_terminal(self, deadline):
        """Whether the leader is TERMINAL-or-gone, WITHOUT reaping it.

        THE SEAM that replaces ``Popen.poll()``. ``poll()`` reaps, and reaping the
        leader is the one operation that frees the group number this design is
        built on: it literally performs the reservation the capability is supposed
        to keep. Asking "is there still a corpse there?" leaves the reservation
        intact until cleanup deliberately gives it up.

        Terminal samples are not ABSENCE and they are not a LIVENESS claim — but
        terminal-and-unreaped IS the reservation witness: the group number cannot
        be handed to another process while this corpse holds it.

        One sample, no census: this runs every poll tick and must not cost two
        strict scans. The census belongs to the signal path, where authority is
        actually spent.
        """
        sample = self._samples([self.leader], deadline).get(self.leader)
        return sample is None or sample.zombie

    def _leader_reservation(self, deadline):
        """The leader's current sample if it still holds THIS receipt, else None."""
        anchor = self.receipts.get(self.leader)
        if anchor is None:
            return None
        sample = self._samples([self.leader], deadline).get(self.leader)
        if sample is None or sample.is_birth(anchor.scheme, anchor.birth) is not True:
            return None
        return sample

    def refresh(self, *, deadline):
        """Prove the group is still ours by RESERVATION, not by membership.

        Ownership is not derived from who is in the group. The leader is held
        unreaped, so the group number stays reserved in the kernel (Linux by the
        struct-pid's group reference, Darwin by pid-number reservation) and every
        member of that number is therefore ours by construction. Membership is a
        census of what the reservation covers, never a proof in its own right.

        Two complete strict censuses are UNIONED rather than compared for
        equality: a pid seen in either scan was seen inside our number, and
        dropping one would be a silent amnesty. A member that is not in both scans
        is recorded, not treated as a reason to keep a live group alive.
        """
        members = []
        try:
            anchor = self.receipts.get(self.leader)
            if anchor is None:
                raise Unknown("no leader receipt")
            first = self._members(deadline)
            second = self._members(deadline)
            members = sorted(set(first) | set(second))
            self.probe["members"] = list(members)
            if not members:
                self.last = Verdict("OBSERVED_EMPTY", "two strict censuses: no members")
                return self.last
            lead = self._leader_reservation(deadline)
            if lead is None:
                raise PinLost("leader reaped or birth mismatch: reservation released")
            if lead.zombie:
                if self.leader not in first and self.leader not in second:
                    # Platform scope, stated plainly: the pin needs the corpse to
                    # be VISIBLE in the group census. Where it is not (Darwin is the
                    # measured case), this build claims no reservation and stays
                    # live-leader-only rather than pretending to a pin it cannot see.
                    raise PinLost(
                        "zombie leader not reported by the group census: "
                        "live-leader-only on this platform"
                    )
                pin = "leader-reservation"
            else:
                pin = "leader-live"
            samples = self._samples(set(members) | {self.leader}, deadline)
            self.probe["observed"] = _observed(samples, None)
            live = [pid for pid in members if pid in samples and not samples[pid].zombie]
            if not live:
                raise Unknown("no live member inside the reserved group")
            for pid in members:
                parent, group, sid = self.identity(pid)
                if (group, sid) != (self.pgid, self.sid):
                    raise Unknown("member outside the recorded group/session")
            # The reservation must still hold at the END of the proof, not only at
            # its start: a reap mid-proof would make the census describe a number
            # that is no longer ours.
            if self._leader_reservation(deadline) is None:
                raise PinLost("reservation lost during proof")
            self.probe["proof"] = _identities(self.receipts, samples, members)
            self.probe["residuals"] = []
            self.last = Verdict(
                "VERIFIED",
                "leader pin holds; members are the reservation's contents",
                tuple(members),
                0.0,
                anchor=pin,
            )
        except Exception as exc:
            self.last = Verdict(
                "REFUSED",
                type(exc).__name__ + ":" + str(exc),
                tuple(members),
                refusal_class="pin_lost" if isinstance(exc, PinLost) else "unproven",
            )
            self.probe["error"] = type(exc).__name__ + ":" + str(exc)
            self.probe["residuals"] = list(members)
        return self.last

    def signal(self, signum, *, deadline):
        result = self.refresh(deadline=deadline)
        if result.status != "VERIFIED":
            return result
        if self._send is None:
            # A platform without the checked primitive: refuse, do not raise. Same
            # bounded refusal the capability reports for any other unusable sender.
            self.last = Verdict(
                "REFUSED", "checked signal unsupported on this platform", result.members
            )
            return self.last
        # ABSOLUTE EXCLUSIONS, checked at the one place a signal leaves this module:
        # never pid 1, never our own process or process group, never a number whose
        # members were not proven to sit in the recorded group AND session.
        if self.pgid <= 1 or self.pgid == os.getpgrp() or self.pgid == os.getpid():
            self.last = Verdict(
                "REFUSED",
                "refuses pid 1, this process and its own group",
                result.members,
                refusal_class="excluded",
            )
            return self.last
        checked = self.clock()
        if checked >= deadline:
            self.last = Verdict("REFUSED", "expired before syscall", result.members)
            return self.last
        # RESIDENCY RE-CHECK IMMEDIATELY BEFORE THE SYSCALL. The census above
        # proved the reservation held a moment ago; between it and killpg the only
        # thing that could free the number is a reap of this corpse. One sample
        # answers that, and it is the difference between "proven" and "proven a
        # while ago" — the non-atomicity this module discloses rather than hides.
        if self._leader_reservation(deadline) is None:
            self.last = Verdict(
                "REFUSED",
                "reservation lost between census and syscall",
                result.members,
                anchor=result.anchor,
                refusal_class="pin_lost",
            )
            self.probe["residuals"] = list(result.members)
            return self.last
        try:
            self._send(self.pgid, signum)
        except ProcessLookupError:
            # ESRCH from the checked syscall: the kernel says this group is
            # already gone. That is the state a passing gate wants to reach, NOT
            # lost authority — mapping it to REFUSED turned a completed gate into
            # "125: did not run" whenever its residual members exited in the
            # window between the final census and this call.
            self.last = Verdict(
                "OBSERVED_EMPTY", "checked signal: group absent (ESRCH)", result.members
            )
            return self.last
        except PermissionError:
            # EPERM is genuine doubt: the group exists and we could not reach it.
            self.last = Verdict("REFUSED", "checked signal denied (EPERM)", result.members)
            return self.last
        except OSError as exc:
            self.last = Verdict(
                "REFUSED", "checked signal failed:" + type(exc).__name__, result.members
            )
            return self.last
        self.last = Verdict(
            "SIGNALED",
            "checked owner-alive signal; non-atomic race remains",
            result.members,
            self.clock() - checked,
            anchor=result.anchor,
        )
        return self.last
