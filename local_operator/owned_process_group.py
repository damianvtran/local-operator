"""Checkout-local signal authority for the bounded gate runner.

Stdlib plus this checkout's own leaves (``procstate``, ``mobile.resources``); no
third-party import and no installation, so ``python -S`` bootstrap holds.

OWNERSHIP IS A RESERVATION, NOT A MEMBER SET: the leader is held UNREAPED, which
keeps its process-group number reserved in the kernel, so every process carrying
that number is ours. Two strict censuses describe what the reservation holds
rather than proving ancestry member by member; the birth of the leader is
checked against the recorded one because a reused pid is the one thing the
reservation cannot survive. Members are re-verified against the group and session
before any signal, and the reservation is re-checked between that check and the
send. A reaped leader is not a reservation, and is refused by name.

The residual risk this rests on, in one sentence: no platform frees a PID or a
group number while a member or a corpse still holds it -- Linux by the struct-pid's
group reference, Darwin by pid-number reservation.

WHAT THE PIN DOES NOT COVER, in one sentence: it is owner-alive plus
exclusive-reap plus SECOND-RESOLUTION birth identity, so a group that disappears
and is recreated inside one birth-token tick is UNSUPPORTED -- indistinguishable
here, and named rather than claimed safe. No supervisor, keeper or system-wide
owner is introduced; escaped or new-session descendants stay outside this group's
authority.

CENSUS RESIDENCE IS NOT THE PREDICATE. A census lists targets; continuity is the
leader's creation receipt (a readable sample whose birth matches, live or
zombie-unreaped). Whether an enumeration lists a corpse is a diagnostic, and its
completeness is UNKNOWN on any platform not directly measured.

This is a checked owner-alive USER-SPACE boundary, NOT an atomic kernel handle:
a member may exit between the final observation and the send, and birth tokens
have finite resolution. No zero-recipient-race or hard-SIGKILL guarantee is
claimed, and no keeper process is introduced.
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
    """Incomplete proof, never a spelling of observed absence.

    THE MODEL, in one paragraph: ownership is not derived from the members. The
    leader is held UNREAPED, which keeps its process-group number reserved in the
    kernel, and every process carrying that number is therefore ours by
    construction. A census then describes what the reservation covers instead of
    trying to prove who belongs. The residual risk, and the boundary this is not,
    are stated once in the module docstring.

    Scope, MEASURED rather than assumed: the pin needs the corpse to be VISIBLE in
    the group census, and it is on both platforms this ships for — Darwin and
    Linux both report an unreaped zombie leader in the group census (measured:
    `strict_census -> [pid]`, `ps` row `Z <defunct>`). The reservation is
    therefore claimed on both. A REAPED leader is `pin_lost` on both platforms
    all the same: once the corpse is given up the number is releasable, which is
    exactly why nothing reaps before the terminal census. Where SIGCHLD is
    SIG_IGN there is no corpse at all, so the premise is void and capture refuses
    with a named reason.

    Exclusions enforced at the one place a signal leaves this module: never pid 1,
    never this process or its own process group, never a group whose members are
    not proven to sit in the recorded group AND session.
    """


class PinLost(Unknown):
    """The group number is not provably ours any more: refuse, never signal.

    A distinct class, not a message: the wrapper records the refusal reason from
    the TYPE, and a caller that cannot tell "this is not ours" from "we could not
    read it" would retry the wrong one.
    """


#: Reuse inside one birth-token tick is UNSUPPORTED, not safe: the token has
#: second resolution (``ps`` lstart), so two processes with the same pid and the
#: same parent in the same second are indistinguishable here -- named so the
#: authority claim stays honest rather than implying a discriminator we lack.
SECOND_RESOLUTION_REUSE = "UNSUPPORTED"

#: ``proc_listpids``'s "every live pid" selector. Unlike ``PROC_PGRP_ONLY`` (2),
#: this is not restricted to the caller's session, which is the whole reason the
#: group census uses it: ``run_bounded`` spawns with ``start_new_session=True``, so
#: the group we must see ALWAYS lives in another session, and ``PROC_PGRP_ONLY``
#: answers an empty list for it rather than an error -- measured on macOS, where a
#: live leader's capture came back OBSERVED_EMPTY.
_ALL_PIDS = 1


#: Facts from the last Darwin enumeration, filled in by ``_darwin_all_pids`` and
#: reported only by a census that misses. Diagnostic, never an input to a verdict.
_LAST_ENUMERATION: dict[str, object] = {}


#: The bound on one internal ownership phase (one leader sample), deliberately
#: independent of any user-supplied timeout.
_TERMINAL_PHASE_SECONDS = 0.5


def _darwin_all_pids(deadline, clock):
    """Every live pid on Darwin, through libproc, or ``Unknown``.

    ``os.getpgid`` then filters by group: it answers for any pid this user may
    ask about, across sessions, and is the same verified interface the module
    already uses for identity. An unreadable answer raises rather than guessing,
    because a silent omission is indistinguishable from an empty group.
    """
    lib = resources._darwin_libproc()
    call = getattr(lib, "proc_listpids", None) if lib is not None else None
    if call is None:
        raise Unknown("strict Darwin census unsupported")
    size = ctypes.sizeof(ctypes.c_int)
    required = call(_ALL_PIDS, 0, None, 0)
    if type(required) is not int or required <= 0 or required % size:
        raise Unknown("malformed libproc size")
    capacity = required // size + 64
    for _ in range(4):
        if clock() >= deadline or capacity > 262144:
            raise Unknown("strict census deadline/capacity exhausted")
        buffer = (ctypes.c_int * capacity)()
        written = call(_ALL_PIDS, 0, buffer, ctypes.sizeof(buffer))
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
        pids = [pid for pid in buffer[: written // size] if pid > 0]
        if len(set(pids)) != len(pids):
            raise Unknown("malformed libproc PID rows")
        # ONE LINE, ONLY WHEN A CENSUS COMES BACK WITHOUT THE GROUP IT WAS ASKED
        # ABOUT. A healthy run pays nothing; a miss carries the discriminator for
        # the candidates that look identical from outside (restricted table vs a
        # short read vs a capacity retry vs the group simply not being there).
        _LAST_ENUMERATION.update(
            {
                "required": required,
                "capacity": capacity,
                "buffer_bytes": ctypes.sizeof(buffer),
                "written": written,
                "parsed": len(pids),
                "retries": _,
            }
        )
        _LAST_ENUMERATION["_first"] = pids[:5]
        return pids
    raise Unknown("saturated libproc census")


def strict_darwin(pgid, deadline, *, listpids=None, clock=time.monotonic):
    """Members of ``pgid``, enumerated WITHOUT the caller's-session restriction.

    ``proc_listpids(PROC_PGRP_ONLY, ...)`` cannot see a group that lives in a
    different session and returns an EMPTY list rather than an error, so it was
    the wrong instrument for a process the runner spawns with
    ``start_new_session=True``. Enumeration is now every pid, filtered by
    ``os.getpgid``; an unreadable pid raises rather than being dropped.
    ``listpids`` stays as the injection seam for tests.
    """
    if listpids is None:
        members = []
        for pid in _darwin_all_pids(deadline, clock):
            if clock() >= deadline:
                raise Unknown("strict census deadline")
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                continue  # exited between the listing and the query: gone, not ours
            except PermissionError:
                raise Unknown("pid outside this user: census cannot be complete")
            except OSError as exc:
                raise Unknown("unreadable process group") from exc
            if group == pgid:
                members.append(pid)
        if pgid not in members:
            _report_census_miss(pgid, deadline)
        return sorted(members)
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


def _report_census_miss(pgid, deadline):
    """One stderr line explaining WHY a Darwin census missed its group.

    Nothing here is asserted on: it exists so the next CI run answers "was the
    leader alive, was the enumeration short, or was the pid unfilterable?" from
    the record instead of from a hypothesis. ``os.getpgid`` outcomes for the first
    few raw pids are the discriminator between "not listed" (kernel restriction)
    and "listed but unresolvable" (our own silent drop).
    """
    import json

    sample = None
    try:
        sample = bounded_samples([pgid], deadline).get(pgid)
    except Exception:  # noqa: BLE001 - the diagnostic must never change a verdict
        sample = None
    state = "unknown"
    if sample is not None:
        state = "zombie" if sample.zombie else "live"
    recorded = _LAST_ENUMERATION.get("_first")
    raw = [int(pid) for pid in recorded] if isinstance(recorded, list) else []
    facts = dict(_LAST_ENUMERATION)
    facts.pop("_first", None)
    facts.update(
        asked_pgid=pgid,
        leader_state=state,
        leader_birth_readable=bool(sample is not None and sample.birth is not None),
        matched=0,
        probed=[{"pid": pid, "getpgid": (lambda q: _pgid_outcome(q))(pid)} for pid in raw[:5]],
    )
    sys.stderr.write("DARWIN_CENSUS_MISS " + json.dumps(facts, default=str) + "\n")
    sys.stderr.flush()


def _pgid_outcome(pid):
    try:
        return os.getpgid(pid)
    except Exception as exc:  # noqa: BLE001 - the exception TYPE is the evidence
        return type(exc).__name__


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
                raise Unknown("strict initial empty census: no observable group to own")
            before = cap._samples([leader, parent], deadline)
            cap.probe["observed"] = _observed(before, None)
            # C1 -- EXCLUSIVE REAP OWNERSHIP, split honestly into what is ENFORCED
            # here and what is ASSUMED of the caller. Enforced: the kernel's observed
            # parent must be THIS process, and SIGCHLD must not be ignored (checked
            # above) -- either failure refuses by name. Assumed: that no competing
            # waiter or consuming handler could take the corpse, which this module
            # cannot observe. That third is a PRECONDITION OF THE CALLER, not a
            # claim, and it is why the witness is only as strong as the wrapper's
            # own no-reap discipline.
            view = identity(leader)
            if view[0] != os.getpid():
                # The kernel's answer about who this child's parent IS, not what a
                # caller claims: only our own child can be held as a witness, and a
                # competing reaper would take the corpse before we could bracket it.
                cap.last = Verdict(
                    "REFUSED",
                    "leader is not our child: exclusive reap ownership absent",
                    (),
                    refusal_class="not_exclusive_reaper",
                )
                cap.probe["error"] = "not_exclusive_reaper"
                return cap
            after = cap._samples([leader, parent], deadline)
            cap.probe["observed"] = _observed(before, after)
            stamp = scheme()
            cap.probe["scheme"] = stamp or None
            if view != (parent, leader, leader) or identity(leader) != view:
                raise Unknown("Popen leader parent/PG/SID not pinned")
            if not stamp or any(p not in before or p not in after for p in [leader, parent]):
                raise Unknown("leader/parent birth unavailable")
            for pid in [leader, parent]:
                # A HELD, UNREAPED LEADER IS A VALID WITNESS even as a corpse: the
                # number it holds is the reservation this design is built on, so
                # "terminal" must not be read as "no receipt". The PARENT is the
                # process asking, so it must genuinely be live.
                terminal_ok = pid == leader
                if (
                    (before[pid].zombie and not terminal_ok)
                    or (after[pid].zombie and not terminal_ok)
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

        THE CORPSE IS NEVER READ THROUGH A LIVE SESSION QUERY, and that is a
        correctness requirement, not an optimisation: on Darwin `getsid` fails on
        a zombie, so `kernel_identity` raises `Unknown: actual parent/PG/SID
        unavailable` for exactly the process this design exists to keep as its
        witness. The terminal path therefore reads the corpse from the group
        census plus the RECORDED birth token (`_leader_witness`), never from
        a function that asks the kernel about a live session.
        """
        # EVERY INTERNAL PHASE CARRIES ITS OWN FINITE BUDGET. The caller's
        # deadline is the USER's contract -- a `--timeout`-less run may wait
        # forever for the child -- but deciding OWNERSHIP must never inherit that
        # infinity: a wedged sampler would otherwise spin here for as long as the
        # child lives, which is how a proof phase became a local hang.
        phase = self.clock() + _TERMINAL_PHASE_SECONDS
        if deadline is not None:
            phase = min(deadline, phase)
        sample = self._samples([self.leader], phase).get(self.leader)
        return sample is None or sample.zombie

    def _leader_witness(self, deadline):
        """The leader as a CREATION-RECEIPT witness: state plus its sample.

        CONTINUITY IS THE RECEIPT, NOT CENSUS RESIDENCE. A readable leader sample
        whose birth matches the recorded receipt is our witness whether it is LIVE
        or ZOMBIE-UNREAPED; whether a census lists the corpse is irrelevant to the
        predicate (it is a diagnostic, not evidence). The retained process lifetime
        is what forbids a new process taking PID N, and POSIX forbids a group N
        being recreated without a process PID N.

        THREE-VALUED ON PURPOSE: ``is_birth(...) is None`` is NOT False. Unreadable
        is a fact about the READER (missing entry, missing birth, unknown scheme,
        provider error, deadline) and refuses as unreadable; a readable mismatched
        birth is a fact about the WORLD (reuse) and refuses as ``pin_lost``.

        Returns ``(state, sample)`` with state in ``held`` / ``reaped`` /
        ``unreadable``.
        """
        anchor = self.receipts.get(self.leader)
        if anchor is None:
            return "unreadable", None
        try:
            samples = self._samples([self.leader], deadline)
        except Exception:  # noqa: BLE001 - a failed sample is unreadable, never absent
            return "unreadable", None
        sample = samples.get(self.leader)
        if sample is None:
            # NO SAMPLE IS TWO DIFFERENT FACTS. The contract separates "positively
            # reaped" (the number was given up -> pin_lost) from "the reader could
            # not read it" (unreadable); a missing sample alone cannot tell them
            # apart, so ask the census. A COMPLETED enumeration that does not list
            # the leader means the process is gone, which is loss. A census that
            # lists it, or that cannot complete, is a reader failure.
            try:
                listed = self._members(deadline)
            except Exception:  # noqa: BLE001 - an unreadable census proves nothing
                return "unreadable", sample
            if self.leader not in listed:
                return "reaped", sample
            return "unreadable", sample
        if sample.birth is None:
            return "unreadable", sample
        held = sample.is_birth(anchor.scheme, anchor.birth)
        if held is True:
            return "held", sample
        if held is None:
            return "unreadable", sample
        return "reaped", sample

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
        # ONLY THE DECLINED SET IS PER-VERDICT. `residuals` and `error` describe
        # THIS check, so a later VERIFIED verdict must not inherit a refusal's.
        # `proof` and `anchor` are RUN-LEVEL evidence: the last successful proof is
        # what a caller binds QA to, and the final refresh of a run normally returns
        # OBSERVED_EMPTY -- popping them there published `anchor: null` and
        # `proof: []` on every successful run.
        for stale in ("residuals", "error", "vanished"):
            self.probe.pop(stale, None)
        members = []
        samples = {}
        try:
            anchor = self.receipts.get(self.leader)
            if anchor is None:
                raise Unknown("no leader receipt")
            # THE WITNESS IS READ BEFORE THE FIRST CENSUS. The creation receipt is
            # the predicate and the census is only target evidence, so a census that
            # fails -- or that runs at all -- cannot pre-empt it. That ordering is
            # also what gives `pin_lost` precedence: a readable foreign birth is
            # positive proof of loss, and a coincident census failure must not
            # degrade it to a generic "could not read".
            state, witness = self._leader_witness(deadline)
            self.probe["witness"] = state
            if state == "reaped":
                raise PinLost("leader reaped or birth mismatch: reservation released")
            if state == "unreadable":
                raise Unknown("leader witness unreadable")
            try:
                first = self._members(deadline)
                second = self._members(deadline)
            except Exception:
                # Re-read the witness exactly once: if it has become a readable
                # foreign birth in the meantime, the loss still wins over the
                # unreadable census.
                late, _ = self._leader_witness(deadline)
                if late == "reaped":
                    raise PinLost("reservation lost during a failed census") from None
                raise
            members = sorted(set(first) | set(second))
            self.probe["members"] = list(members)
            if not members:
                # EMPTY-COMPLETE, bracketed on both sides: a retained matching
                # witness before AND after, or no absence may be claimed.
                if witness is not None and witness.zombie:
                    after_state, _ = self._leader_witness(deadline)
                    if after_state == "reaped":
                        raise PinLost("reservation lost during the empty proof")
                    if after_state != "held":
                        raise Unknown("witness unreadable during the empty proof")
                    self.probe["anchor"] = "leader-reservation"
                    self.last = Verdict(
                        "OBSERVED_EMPTY", "retained corpse and complete no-live-member census"
                    )
                    return self.last
                raise Unknown("live leader absent from a complete census: contradictory")
            samples = self._samples(set(members) | {self.leader}, deadline)
            self.probe["observed"] = _observed(samples, None)
            # UNRESOLVED MEMBER SAMPLES ARE REFUSED BEFORE ANY EMPTINESS VERDICT.
            # A missing map entry cannot prove departure, so certifying "no live
            # members" while one member is unresolved would be false cleanup through
            # a second door.
            absent = [pid for pid in members if pid not in samples]
            if absent:
                self.probe["unresolved"] = absent
                raise Unknown("census member sample unreadable")
            live = [pid for pid in members if not samples[pid].zombie]
            if not live:
                # Corpse-only: allowed ONLY under a retained matching witness, and a
                # witness that has become a READABLE FOREIGN BIRTH is loss, not an
                # unreadable detail.
                still, _ = self._leader_witness(deadline)
                if still == "reaped":
                    raise PinLost("reservation lost before the corpse-only proof")
                if still != "held":
                    raise Unknown("no live member and no readable witness")
                self.probe["anchor"] = "leader-reservation"
                self.last = Verdict(
                    "OBSERVED_EMPTY", "reserved group holds no live member", tuple(members)
                )
                return self.last
            for pid in live:
                parent, group, sid = self.identity(pid)
                if (group, sid) != (self.pgid, self.sid):
                    raise Unknown("member outside the recorded group/session")
            # The witness must still hold at the END of the proof.
            final_state, final_witness = self._leader_witness(deadline)
            if final_state == "reaped":
                raise PinLost("reservation lost during proof")
            if final_state == "unreadable":
                raise Unknown("witness unreadable during proof")
            # An unreadable witness at the end of the proof is a REFUSAL, never a
            # defaulted anchor: defaulting would publish a pin this run never held.
            if final_witness is None:
                raise Unknown("witness unreadable at the end of the proof")
            pin = "leader-reservation" if final_witness.zombie else "leader-live"
            self.probe["proof"] = _identities(self.receipts, samples, members)
            self.probe["residuals"] = []
            # Sticky: the record must be able to say HOW ownership was established
            # even though the run's final verdict is the terminal OBSERVED_EMPTY.
            self.probe["anchor"] = pin
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
            # The DECLINED set carries the same fields as the proven set — pid,
            # birth, scheme, group, session — because "who did we refuse to touch"
            # needs the same evidence as "who did we signal". The samples this check
            # already paid for supply the birth for a member that was never enrolled.
            self.probe["residuals"] = _identities(self.receipts, samples, members)
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
        # WITNESS RE-CHECK IMMEDIATELY BEFORE THE SYSCALL -- THE MUTATION ANCHOR.
        # This exact block is what a deletion control removes; `refresh` carries its
        # own end-of-proof check for a different reason (a reap during the proof),
        # so a mutation names this one and cannot take both by accident. The census
        # proved the witness held a moment ago; between it and the send the only
        # thing that could free the number is a reap of this process, and one sample
        # answers that. It is the difference between "proven" and "proven a while
        # ago" -- the non-atomicity this module discloses rather than hides.
        final_state, _ = self._leader_witness(deadline)
        if final_state != "held":
            self.last = Verdict(
                "REFUSED",
                (
                    "witness lost between census and syscall"
                    if final_state == "reaped"
                    else "witness unreadable between census and syscall"
                ),
                result.members,
                anchor=result.anchor,
                refusal_class="pin_lost" if final_state == "reaped" else "unproven",
            )
            self.probe["residuals"] = _identities(self.receipts, {}, result.members)
            self.probe["anchor"] = result.anchor
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
