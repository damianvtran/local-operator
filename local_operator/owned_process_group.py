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

import copy
import ctypes
import errno
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
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

    census_failure: dict[str, object] | None = None


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
        raw = list(buffer[: written // size])
        # A FILTER THAT TURNS "I COULD NOT SEE" INTO "THERE IS NOTHING" IS FORBIDDEN.
        # A non-positive pid in the returned rows is not padding to be skipped: it
        # means the read is not the enumeration this census claims it is, so the
        # whole read is unreadable rather than quietly shorter (review round 8, R3).
        if any(pid <= 0 for pid in raw):
            raise Unknown("malformed libproc PID rows: non-positive entry")
        pids = raw
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
        # THE COVERAGE TRAVELS WITH THE READ (review round 5, R6). Reading the
        # process-global ledger after this returns let a second census in another
        # thread overwrite it, so a partial read could inherit a complete
        # certificate. The global below stays DIAGNOSTIC ONLY and is never read
        # for a decision.
        return pids, required, written, len(pids)
    raise Unknown("saturated libproc census")


def _libproc_coverage(required, written, rows):
    """Coverage the libproc read can actually CLAIM, from the size-query contract.

    `required` is the kernel's own row budget in bytes and `rows` is what we
    parsed. A short (or empty) answer against a non-zero budget is a PARTIAL read,
    never an empty group. One calculation, shared by the production path and the
    injected `listpids` seam -- review round 4 found the guard living only in the
    seam branch, so production still certified a population nobody measured.
    """
    size = ctypes.sizeof(ctypes.c_int)
    if type(required) is not int or required <= 0:
        return False
    return written == required and rows * size == required


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
        unknown = []
        # THE RAW NAMESPACE, before any group filter: `leader_seen` must answer
        # "was this pid in the enumeration at all", not "was it a member".
        raw_pids, required, written, parsed = _darwin_all_pids(deadline, clock)
        leader_seen = pgid in raw_pids
        for pid in raw_pids:
            if clock() >= deadline:
                raise Unknown("strict census deadline")
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                # ESRCH is the only tolerated outcome: the pid went away between
                # the listing and the query, so it cannot be a member.
                if pid == pgid:
                    # ...unless it is the pid we were ASKED about. Then the answer
                    # is unknown, not absent: an unprovable membership must never
                    # be folded into an empty group.
                    unknown.append(pid)
                continue
            except PermissionError:
                unknown.append(pid)
                continue
            except OSError:
                unknown.append(pid)
                continue
            if group == pgid:
                members.append(pid)
        if unknown:
            raise Unknown(
                "strict census membership unknown: %d pid(s) could not be asked" % len(unknown)
            )
        if pgid not in members:
            _report_census_miss(pgid, deadline)
        # THE CERTIFICATE COMES FROM THE READ, even here: `_darwin_all_pids`
        # recorded the size query and what it parsed, and anything short of the
        # full budget is incomplete.
        complete = _libproc_coverage(required, written, parsed)
        return sorted(members), complete, leader_seen
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
        # COMPLETENESS IS EARNED FROM THE READ. `needed` is the kernel's own row
        # budget from the size query; a shorter answer -- including zero bytes from
        # a non-zero budget -- is a PARTIAL read, never an empty group. Asserting
        # True here (review round 3, R3) certified a population nobody measured.
        complete = _libproc_coverage(needed, written, len(pids))
        # THE SEAM IS TYPED HERE TOO: this branch serves the injected `listpids`
        # reader as well as the live libproc one. leader_seen is the RAW namespace
        # read, before any group filter, for the same reason as everywhere else.
        return sorted(pids), complete, pgid in pids
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


# Closing-delta relists allowed inside ONE strict_linux attempt. The deadline is
# the real bound; this cap only guarantees termination under an injected or
# stalled clock. Exhausting it is an incomplete attempt (stricter), never a pass.
_LINUX_CENSUS_RELISTS = 8

# Full census->census->sample cycles one `refresh` may run before refusing.
# CI run 41748872018 (job 126359450123) refused a HEALTHY fast-forking command
# with exit 126 "census member sample unreadable": censuses
# [[2425,2443,2444,2445,2446],[2425,2444,2445,2446]], and 2444 -- present in
# BOTH -- died before its sample (ENOENT). That shape cannot be amnestied (a
# member seen in census 2 could have forked a live child before dying), so the
# fix re-runs the proof instead of relaxing it. INVARIANT: a verdict rests on
# exactly one cycle's evidence; earlier cycles only decide whether another
# starts. The loop never carries evidence across cycles -- censuses, samples,
# failures and departures are all rebuilt -- and a retry is attempted only when
# `_rerunnable` holds and another cycle fits the deadline. Three cycles keep
# the cost bounded; a trigger persisting through all of them still refuses.
#
# A SECOND RETRY TRIGGER shares the same cap: a corpse-only cycle (every member
# exited at sample time) takes a post-sample CENSUS 3 (`_late_members`), and if
# that census shows a new pid -- or a member that is no longer the exited
# process sampled -- the cycle is discarded exactly like the ENOENT shape above.
# One 3-cycle budget covers both triggers together, and still no evidence is
# carried across cycles.
_SAMPLE_CYCLES = 3


def strict_linux(
    pgid,
    deadline,
    *,
    listdir=os.listdir,
    read=None,
    stat=os.stat,
    clock=time.monotonic,
    diagnostics=None,
):
    """Refuse omitted /proc rows; only proved ENOENT can exclude a vanished PID.

    Optional diagnostics describe existing reads, never authorise omission or
    alter scan/retry/deadline policy. Changed-PID relation is known only from a
    successful stat parse in the SAME attempt; a closing-only arrival is unknown.
    """
    read = read or (lambda p: Path(p).read_text(encoding="utf-8"))
    attempts, attempt_count, remaining_ids = [], 0, 16
    phase = "opening-deadline"

    def classify(entry):
        """Return the parsed group of /proc/<entry>, or None when ENOENT proves it gone.

        Raises Unknown for every other outcome (EACCES, EIO, a short or
        unparseable row, a stat row missing beside a live directory). The
        message carries the errno name so a witness names the real cause.

        ESRCH is treated like ENOENT on the FIRST read only, and only as a
        trigger for the same stat(path) proof: on Linux, read() of
        /proc/<pid>/stat for a task that exits after open() returns ESRCH, not
        ENOENT (witness: PR #3879 gate, xdist churn, census refused with
        "unreadable /proc row", 170 PIDs scanned). ESRCH never excuses a PID by
        itself: gone is still proven only by stat(path) raising ENOENT; a live
        directory gets the one retry, and a second ESRCH there stays Unknown.
        To extend: add exception types to the first `except` only if the kernel documents them
        as "task no longer exists"; EACCES/EIO must never join. The
        opening scan and the closing-delta reconciliation share this one
        proof so an added PID can never be excused by a weaker rule.
        """
        nonlocal phase
        path = f"/proc/{entry}"
        phase = "row-read"

        def unreadable(exc):
            code = getattr(exc, "errno", None)
            name = errno.errorcode.get(code) if isinstance(code, int) else None
            return Unknown(f"unreadable /proc row ({name})" if name else "unreadable /proc row")

        try:
            raw = read(path + "/stat")
        except (FileNotFoundError, ProcessLookupError):
            try:
                stat(path)
            except FileNotFoundError:
                return None
            # The directory exists but the row did not read: retry it once.
            # A transient miss during churn is not a malformed row, and only a
            # second miss is genuinely unreadable identity.
            try:
                raw = read(path + "/stat")
            except FileNotFoundError:
                raise Unknown("stat row missing for existing PID")
            except (OSError, UnicodeError) as exc:
                raise unreadable(exc) from exc
        except (OSError, UnicodeError) as exc:
            raise unreadable(exc) from exc
        phase = "row-parse"
        try:
            left, tail = (
                raw[: raw.index("(")].strip(),
                raw[raw.rindex(")") + 1 :].split(),
            )
            if int(left) != int(entry) or len(tail) < 3 or len(tail[0]) != 1:
                raise ValueError("invalid process stat")
            parent, group = int(tail[1]), int(tail[2])
            if parent < 0 or group < 0:
                raise ValueError("negative process identity")
        except (ValueError, IndexError) as exc:
            raise Unknown("malformed /proc row") from exc
        return group

    def scan():
        nonlocal phase, attempt_count
        # Keep the existing six attempts and exact deadline/read cadence. These
        # observations measure namespace churn; they do not excuse it, extend
        # the allowance, or convert a partial group census into authority.
        for attempt in range(6):
            phase = "opening-deadline"
            if clock() >= deadline:
                raise Unknown("strict /proc census deadline")
            attempt_count = attempt + 1
            first, last = None, None
            result, disappeared, parsed = [], set(), set()
            relists = 0
            reconciled = {"different_group": 0, "same_group": 0, "gone": 0, "departed": 0}
            unclassified, listed, gone_members = None, set(), set()
            phase = "opening-scan"
            try:
                first = {n for n in listdir("/proc") if n.isdigit()}
                listed = set(first)
                phase = "row-scan"
                for entry in first:
                    if clock() >= deadline:
                        phase = "row-deadline"
                        raise Unknown("strict /proc row deadline")
                    group = classify(entry)
                    if group is None:
                        disappeared.add(entry)
                        continue
                    parsed.add(entry)
                    if group == pgid:
                        result.append(int(entry))
                # RECONCILE THE CLOSING DELTA TO A FIXPOINT (PR3269, CI run
                # 41657372783 job 125993075099; architect ruling 641ae72fe5d1,
                # "Unstable /proc census at refresh"). The old completeness test
                # was a machine-wide zero delta (opening minus proven-gone ==
                # closing), so ANY unrelated fork or exit on a busy runner
                # failed all six attempts and refresh() refused with
                # "unstable /proc census" while our group was perfectly stable.
                # The promise that test made is "every PID in the final list is
                # classified", and that is what is kept here, not weakened:
                # - a REMOVED PID was already read (parsed) or proven gone in
                #   this attempt, so it is classified; our-group ones stay in
                #   `result` and refresh() samples them;
                # - an ADDED PID (including a proven-gone PID listed again, i.e.
                #   possible PID reuse, because it is not in `parsed`) is read
                #   with the SAME proof as the opening scan: another pgid or
                #   ENOENT-gone is harmless, our pgid joins `result`, and every
                #   other outcome (EACCES, EIO, short/unparseable row) leaves
                #   the attempt INCOMPLETE -- never a silent omission;
                # - then /proc is listed again until a listing adds nothing
                #   unclassified. Relists are capped per attempt and every read
                #   checks the deadline, so the six attempts and the caller's
                #   deadline still bound the whole census; an unsettled delta
                #   is Unknown, never the last partial attempt.
                #
                # - an OUR-group PID that a listing no longer contains is NOT
                #   "already classified" (review round 1, R1): keeping it would
                #   hand refresh()/_terminal_absence a corpse and refuse exactly
                #   the post-SIGKILL census this change exists to complete, and
                #   would defeat #3360's in-census-1-not-in-census-2 rule. It is
                #   re-read: ENOENT on row AND directory drops it (departed);
                #   a positive read while absent from the listing is a
                #   contradiction, and any other outcome is unknown -- both
                #   leave the attempt incomplete. A departure also forces one
                #   more listing, so a child it forked before exiting is seen.
                #   Foreign removals need no re-read: they never enter result.
                # - PID-reuse blind spot (review round 1, m1): a PID parsed as
                #   foreign is not re-read if a later listing still shows it, so
                #   a reuse of that PID by our group between relists is missed.
                #   The old zero-delta rule had the same blind spot within one
                #   attempt; relists widen the window by at most
                #   _LINUX_CENSUS_RELISTS listings inside the same deadline.
                while True:
                    phase = "closing-scan"
                    last = {n for n in listdir("/proc") if n.isdigit()}
                    listed |= last
                    departed = sorted({str(pid) for pid in result} - last, key=int)
                    pending = sorted(last - parsed, key=int)
                    if not pending and not departed:
                        # RAW NAMESPACE, before the group filter, for the same reason as Darwin.
                        leader_seen = str(pgid) in listed
                        phase = "complete"
                        return sorted(result), True, leader_seen
                    if clock() >= deadline:
                        phase = "row-deadline"
                        raise Unknown("strict /proc row deadline")
                    work = [(entry, True) for entry in departed]
                    work += [(entry, False) for entry in pending]
                    if relists >= _LINUX_CENSUS_RELISTS:
                        # unclassified_count: PIDs this attempt still had to
                        # classify when it stopped (m2: one meaning everywhere).
                        unclassified = len(work)
                        phase = "namespace-changed"
                        break
                    relists += 1
                    phase = "reconcile"
                    for index, (entry, was_ours) in enumerate(work):
                        if clock() >= deadline:
                            phase = "row-deadline"
                            raise Unknown("strict /proc row deadline")
                        try:
                            group = classify(entry)
                        except Unknown:
                            unclassified = len(work) - index
                            phase = "reconcile-unclassified"
                            break
                        if was_ours:
                            if group is not None:
                                # Readable while the listing omits it: the two
                                # facts contradict, so nothing here is proven.
                                unclassified = len(work) - index
                                phase = "reconcile-contradiction"
                                break
                            # A filter, not list.remove: the session-deletion
                            # guard flags every `.remove` call shape. Same
                            # effect -- result holds each PID once.
                            result = [pid for pid in result if pid != int(entry)]
                            parsed.discard(entry)
                            disappeared.add(entry)
                            gone_members.add(entry)
                            reconciled["departed"] += 1
                            continue
                        if group is None:
                            disappeared.add(entry)
                            reconciled["gone"] += 1
                            continue
                        parsed.add(entry)
                        disappeared.discard(entry)
                        if group == pgid:
                            result.append(int(entry))
                            reconciled["same_group"] += 1
                        else:
                            reconciled["different_group"] += 1
                    if unclassified is not None:
                        break
            finally:
                # A diagnostic must not replace any original read/parse/deadline
                # refusal. Record only facts already obtained in THIS attempt.
                try:
                    opening, closing = first or set(), last or set()

                    def entries(pids):
                        nonlocal remaining_ids
                        selected = sorted(pids, key=int)[:remaining_ids]
                        remaining_ids -= len(selected)
                        return [
                            {
                                "pid": int(pid),
                                "relation": (
                                    "departed-same-group"
                                    if pid in gone_members
                                    else (
                                        "unobserved"
                                        if pid not in parsed
                                        else (
                                            "same-group"
                                            if int(pid) in result
                                            else "different-group"
                                        )
                                    )
                                ),
                            }
                            for pid in selected
                        ]

                    # m3: arrivals accumulate across every relist of the
                    # attempt; removals compare the opening and final listings.
                    added = listed - opening if last is not None else set()
                    removed = opening - closing if last is not None else set()
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "phase": phase,
                            "opening_count": len(first) if first is not None else None,
                            "closing_count": len(last) if last is not None else None,
                            "added_count": len(added),
                            "removed_count": len(removed),
                            "confirmed_disappeared_count": len(disappeared),
                            "relist_count": relists,
                            "reconciled": dict(reconciled),
                            "unclassified_count": unclassified,
                            "added": entries(added),
                            "removed": entries(removed),
                            "confirmed_disappeared": entries(disappeared),
                        }
                    )
                except Exception:  # noqa: BLE001 - evidence is never an authority input
                    attempts.append({"diagnostic_unavailable": True})
        phase = "attempt-limit"
        raise Unknown("unstable /proc census")

    def snapshot():
        return copy.deepcopy({"phase": phase, "attempt_count": attempt_count, "attempts": attempts})

    try:
        return scan()
    except Exception as exc:
        # Per-exception copy: a caller mutating its optional sink or a later
        # census must not rewrite evidence already attached to this refusal.
        try:
            setattr(exc, "census_failure", snapshot())
        except Exception:  # noqa: BLE001 - never replace the original refusal
            pass
        raise
    finally:
        if diagnostics is not None:
            try:
                diagnostics.clear()
                diagnostics.update(snapshot())
            except Exception:  # noqa: BLE001 - a broken sink cannot change the verdict
                pass


def strict_census(pgid, deadline):
    """Members of ``pgid`` as a CensusResult, with completeness stated not inferred.

    `complete` is True only because every refusal route in `strict_darwin` /
    `strict_linux` raises rather than dropping a row: an unreadable pid, a
    non-positive row, a saturated retry and a malformed length are all `Unknown`
    before this point. `leader_seen` records whether the requested number appeared
    in the namespace AT ALL, which is what separates "we reaped it" from "the
    enumeration never saw it".
    """
    try:
        if sys.platform == "darwin":
            pids, complete, leader_seen = strict_darwin(pgid, deadline)
        elif sys.platform.startswith("linux"):
            pids, complete, leader_seen = strict_linux(pgid, deadline)
        else:
            raise Unknown("strict group census unsupported")
        if complete is not True:
            # FAIL CLOSED: a reader that cannot state its coverage supplies no
            # certificate, and none is manufactured for it here.
            return CensusResult(tuple(pids), False, bool(leader_seen), "reader stated no coverage")
        # Coverage belongs to this backend's accounting argument, not to a
        # libproc byte count or the legacy complete boolean. This instrument
        # on this installed OS has no authoritative Darwin constructor.
        if sys.platform.startswith("linux"):
            return AuthoritativeCensus(tuple(pids), True, bool(leader_seen))
        return CensusResult(
            tuple(pids),
            False,
            bool(leader_seen),
            "this instrument cannot certify coverage on this installed OS",
        )
    except (OSError, ValueError, TypeError) as exc:
        refused = Unknown("strict membership probe failed")
        refused.census_failure = copy.deepcopy(getattr(exc, "census_failure", None))
        raise refused from exc


def bounded_samples(pids, deadline, clock=time.monotonic, *, diagnostics=None):
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
        result = procstate._proc_samples(wanted, diagnostics=diagnostics)
        if clock() >= deadline:
            raise Unknown("native birth sample exceeded deadline")
        return result
    if deadline - clock() < 1.0:
        raise Unknown("remaining deadline cannot cover canonical ps fallback")
    result = procstate.process_samples(wanted)
    if clock() >= deadline:
        raise Unknown("canonical birth fallback exceeded deadline")
    return result


def _exited(sample):
    """Whether ``sample`` is a process that can no longer fork: a single-threaded corpse.

    A ``Z`` state is not enough on Linux. When a thread-group leader's MAIN thread
    exits while other threads live, the leader lingers as a zombie until the group
    empties (kernel/exit.c:1505 and ``delay_group_leader`` in
    include/linux/sched/signal.h:776), ``/proc/<pid>/stat`` reports the LEADER's
    state (fs/proc/array.c:491) -- so ``Z`` -- and its surviving threads can still
    fork into our group. ``num_threads`` (field 20, fs/proc/array.c:609) tells the
    two apart. ``threads is None`` (platforms without a count) keeps the old
    meaning; the Linux reader omits a row with an unparseable count instead of
    producing ``None`` there.
    """
    return sample.zombie and not (sample.threads is not None and sample.threads > 1)


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
            row = {"birth": sample.birth, "zombie": sample.zombie}
            # Only a multi-threaded count is recorded (security S3): it is the one
            # value that explains why a `zombie: true` member counted as LIVE (see
            # `_exited`), and omitting the ordinary case keeps every other row --
            # and the record's size -- exactly as before.
            if sample.threads is not None and sample.threads > 1:
                row["threads"] = sample.threads
            rows.setdefault(str(pid), {})[stage] = row
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
    # Acquisition is deliberately not a proof verdict. Keep the positive
    # creation observation typed and separate from any population/cleanup claim.
    acquisition_kind: str | None = None
    creation_receipt: Receipt | None = None
    population_coverage: str | None = None


_SEAL = object()


@dataclass(frozen=True)
class CensusResult:
    """What a census SAW and whether it can vouch for the namespace it saw.

    A successful call is not a complete enumeration. `complete` is conservative
    and total: any dropped row, unreadable pid, saturation or miss makes it False,
    and an incomplete census is UNKNOWN VISIBILITY -- never emptiness.

    `leader_seen` separates two worlds that used to look identical: a leader pid
    absent from the namespace because WE reaped it (expected, and consistent with
    an empty group), versus a leader pid that never appeared in an enumeration
    claiming to be complete (a contradiction, and a refusal).
    """

    pids: tuple[int, ...]
    complete: bool
    leader_seen: bool
    #: Why coverage could not be claimed; empty when it could.
    reason: str = ""


@dataclass(frozen=True)
class AuthoritativeCensus(CensusResult):
    """Coverage-bearing result; production construction is strict Linux only.

    Synthetic positive controls must explicitly choose an authoritative backend;
    returning legacy complete=True is deliberately insufficient.
    """


class OwnedGroup:
    def __init__(
        self,
        seal,
        leader,
        *,
        census,
        samples,
        identity,
        scheme,
        clock,
        signal_group,
        departure=None,
        proc_self=None,
    ):
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
        #: EVERY SIGNAL THIS CAPABILITY ACTUALLY DELIVERED. Appended only at the
        #: success site in `signal()` (after the checked syscall returns SIGNALED),
        #: never on a refusal path. The terminal report is allowed to rest on the
        #: census precisely because this run ENDED the group; without this record a
        #: terminal verdict would describe an event that never happened (review
        #: round 8, R2), and it lives here rather than in the runner because the
        #: runner is a different object and would leave this door open.
        self.terminating_sends = []
        self.census, self.samples, self.identity = census, samples, identity
        self.scheme, self.clock, self._send = scheme, clock, signal_group
        self.initial_error = "unproven initial identity"
        self.last = Verdict("REFUSED", self.initial_error)
        self._revoked = False
        self._sample_failures = []
        self._terminal_result = None
        #: THE DEPARTURE PROBES (see `_departed`). Kept on the `identity`
        #: pattern so synthetic histories can inject them, but `None` is
        #: meaningful: it means "the product default", which is only consulted
        #: on the native sampling path. An injected `samples` reader produces no
        #: per-PID ENOENT diagnostics we can trust, so a fixture that wants a
        #: departure must inject the probe explicitly (positive control).
        self._departure = departure
        self._proc_self = proc_self

    def revoke(self):
        # Reaping releases the reservation even if wait times out or raises.
        # Revocation is monotone; no refresh or cached VERIFIED restores it.
        self._revoked = True

    def accepted_terminal(self, verdict):
        return (
            not self._revoked
            and verdict is self._terminal_result
            and verdict is self.last
            and verdict.status == "OBSERVED_EMPTY"
        )

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
        departure=None,
        proc_self=None,
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
            departure=departure,
            proc_self=proc_self,
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
            if sys.platform.startswith("linux") and cap._acquire_live_creation(parent, deadline):
                return cap
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

    def _acquire_live_creation(self, parent, deadline):
        """Observe only our live child's creation, never population or sends.

        A machine-wide strict census can race ordinary fork/exit churn before
        the owner even records its own child. Linux may retain that positive
        creation receipt without claiming a complete population. Every later
        signal still executes the unchanged strict refresh and final witness
        guards; this status cannot certify cleanup or rescue a failed proof.

        Exclusive wait/reap ownership remains the caller's explicit precondition:
        SIG_DFL rules out a handler we can observe, not an unobservable rival
        waiter. A corpse at the first sample uses the legacy strict path instead.
        """

        def prerequisites():
            if self._revoked:
                raise PinLost("signal authority irreversibly revoked during creation")
            if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
                raise Unknown("live creation requires default SIGCHLD disposition")
            if self.pgid <= 1 or self.pgid in (os.getpid(), os.getpgrp()):
                raise Unknown("live creation excludes pid 1, this process and its own group")
            if parent != os.getpid():
                raise Unknown("live creation parent is not this process")

        checked = self.clock()
        if not math.isfinite(deadline) or not math.isfinite(checked) or checked >= deadline:
            raise Unknown("live creation requires a finite unexpired deadline")
        if parent != os.getpid():
            # Preserve the existing named refusal when caller/observed parent
            # cannot be our exclusive-reap chain. This is not an acquisition.
            self.last = Verdict(
                "REFUSED",
                "leader is not our child: exclusive reap ownership absent",
                refusal_class="not_exclusive_reaper",
            )
            self.probe["error"] = "not_exclusive_reaper"
            return True
        prerequisites()
        before = self._samples([self.leader, parent], deadline)
        self.probe["observed"] = _observed(before, None)
        first = before.get(self.leader)
        if first is None:
            raise Unknown("live creation leader sample unavailable")
        if first.zombie:
            return False
        expected = (parent, self.leader, self.leader)
        view = self.identity(self.leader)
        after = self._samples([self.leader, parent], deadline)
        self.probe["observed"] = _observed(before, after)
        stamp = self.scheme()
        self.probe["scheme"] = stamp or None
        if view != expected or self.identity(self.leader) != expected:
            raise Unknown("live creation parent/PG/SID not pinned")
        if not stamp or any(pid not in before or pid not in after for pid in (self.leader, parent)):
            raise Unknown("live creation leader/parent sample unavailable")
        for pid in (self.leader, parent):
            if (
                before[pid].zombie
                or after[pid].zombie
                or not before[pid].birth
                or not after[pid].birth
                or after[pid].is_birth(stamp, before[pid].birth) is not True
            ):
                raise Unknown("live creation requires two matching live births")
        observed = self.clock()
        if not math.isfinite(observed) or observed >= deadline:
            raise Unknown("live creation deadline expired before publication")
        birth, parent_birth = before[self.leader].birth, before[parent].birth
        if birth is None or parent_birth is None:
            raise Unknown("live creation birth unavailable")
        receipt = Receipt(
            self.leader,
            stamp,
            birth,
            parent,
            parent_birth,
            self.pgid,
            self.sid,
            observed,
        )
        # Prepare the observational view before the last precondition/expiry
        # checks; even local serialization work cannot spend the live deadline
        # and then publish an expired (or meanwhile revoked) positive receipt.
        creation_view = asdict(receipt)
        prerequisites()
        publishing = self.clock()
        if not math.isfinite(publishing) or publishing >= deadline:
            raise Unknown("live creation deadline expired before publication")
        self.receipts[self.leader] = receipt
        self.initial_error = ""
        self.probe["acquisition_kind"] = "linux-live-creation"
        # The ledger/verdict retain the frozen typed receipt; the legacy probe
        # is a JSON-ready observational view and is never an authority source.
        self.probe["creation_receipt"] = creation_view
        self.probe["population_coverage"] = "not-established"
        self.probe["exclusive_reap_precondition"] = "caller retains exclusive wait/reap ownership"
        self.probe["anchor"] = "leader-live-creation"
        self.last = Verdict(
            "ACQUIRED_POSITIVE",
            "live creation receipt only; population coverage not established",
            anchor="leader-live-creation",
            acquisition_kind="linux-live-creation",
            creation_receipt=receipt,
            population_coverage="not-established",
        )
        return True

    def _members(self, deadline):
        self.probe.pop("census_failure", None)
        if self.clock() >= deadline:
            raise Unknown("ownership deadline exhausted")
        try:
            seen = self.census(self.pgid, deadline)
        except Unknown as exc:
            if exc.census_failure is not None:
                self.probe["census_failure"] = copy.deepcopy(exc.census_failure)
            raise
        if isinstance(seen, AuthoritativeCensus):
            if not seen.complete:
                # INCOMPLETE VISIBILITY IS NOT EMPTINESS. This is the door R3 came
                # through: a successful call that could not vouch for the namespace
                # must never become "there is nothing".
                raise Unknown("census incomplete: " + (seen.reason or "coverage not stated"))
            if not seen.leader_seen and not self.terminating_sends:
                # A namespace that never contained the leader, while we hold an
                # unreaped reservation and have sent nothing that could end it, is a
                # contradiction rather than an absence.
                raise Unknown("census never saw the leader and nothing was sent")
            members = list(seen.pids)
        else:
            # A READER THAT SUPPLIES NO COVERAGE CLAIM IS NOT COMPLETE. Optimism
            # belongs in fixtures, which return CensusResult explicitly; the product
            # fails closed here so a future third backend cannot inherit a hole.
            raise Unknown("census reader supplied no coverage claim")
        if (
            members is None
            or any(type(p) is not int or p <= 0 for p in members)
            or len(set(members)) != len(members)
        ):
            raise Unknown("incomplete/malformed membership")
        return sorted(members)

    def _samples(self, pids, deadline):
        # Never attach an earlier witness/census attempt's error to a later
        # unresolved PID. Publish only this attempt's bounded diagnostic facts.
        self._sample_failures = []
        if self.clock() >= deadline:
            raise Unknown("ownership sample deadline exhausted")
        if self.samples is None:
            current = []
            try:
                return bounded_samples(pids, deadline, self.clock, diagnostics=current)
            finally:
                self._sample_failures = current
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
            return "unverifiable", None
        try:
            samples = self._samples([self.leader], deadline)
        except PermissionError:
            # ROUTED, NOT RENAMED (review round 3, R1): an unreadable row is
            # identity uncertainty and must take the same branch as every other
            # unverifiable witness.
            return "unverifiable", None
        except Exception:  # noqa: BLE001 - a failed sample is unreadable, never absent
            return "unverifiable", None
        sample = samples.get(self.leader)
        if sample is None:
            # A MISSING SAMPLE IS NOT PROOF OF ABSENCE (review round 3, R3b): the
            # sampling contract keeps it unknown, and `_proc_samples` omits rows on
            # OSError. `gone` is established POSITIVELY in `refresh`, from a census
            # that is complete AND did not contain this number.
            return "unseen", None
        if sample.birth is None:
            # A MISSING ROW IS NOT PROOF OF REAPING. It is the ambiguous
            # observation: the census demonstrably omits processes on macOS, so
            # "not listed" and "the reader failed" are indistinguishable here.
            # Only a READABLE FOREIGN birth below may be `pin_lost`; everything
            # this branch can see is `unreadable`.
            return "unverifiable", sample
        held = sample.is_birth(anchor.scheme, anchor.birth)
        if held is True:
            return "held", sample
        if held is None:
            return "unverifiable", sample
        return "reaped", sample

    def _terminal_absence(self, deadline):
        """The END of a run we killed: complete-and-empty, or a refusal.

        COMPLETE means complete: a census that raises, is truncated, or cannot
        enumerate is an UNKNOWN population, and an unknown population is never
        laundered into absence -- the same rule the admission refusal and the
        probe-streak rule enforce. Only two successful strict censuses whose union
        is empty may report OBSERVED_EMPTY here.
        """
        # NO KILL, NO TERMINAL ABSENCE. The census may decide only for a group this
        # run actually ended: a run that never sent anything has proved nothing
        # about an end, and empty-therefore-clean is exactly the laundering the
        # reviewer reproduced (R2).
        if not self.terminating_sends:
            raise Unknown("terminal absence requires a recorded terminating send")
        try:
            first = self._members(deadline)
            second = self._members(deadline)
        except Exception as exc:  # noqa: BLE001 - unreadable is never absent
            # PRESERVE THE CAUSE: the classifier's own words must reach the verdict,
            # so a caller can tell "census incomplete" from "malformed".
            raise Unknown("terminal census unreadable: " + str(exc)) from None
        members = sorted(set(first) | set(second))
        if members:
            raise Unknown("terminal census still lists members")
        self.probe["members"] = []
        self.last = Verdict(
            "OBSERVED_EMPTY",
            "terminal census complete and empty after a terminating send by this run",
            (),
        )
        self._terminal_result = self.last
        return self.last

    def refresh(self, *, deadline, terminal=False):
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
        # A new refusal (including irreversible revocation) must not relabel
        # an earlier attempt's sample diagnostics as evidence gathered now.
        for stale in (
            "sample_failure",
            "unresolved",
            "census_failure",
            "vanished",
            "sample_retries",
        ):
            self.probe.pop(stale, None)
        if self._revoked:
            self.last = Verdict(
                "REFUSED", "signal authority irreversibly revoked", refusal_class="pin_lost"
            )
            return self.last
        self._terminal_result = None
        for stale in ("residuals", "error"):
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
            if state == "unverifiable":
                raise Unknown("leader witness unreadable: identity of the leader unproven")
            if state == "unseen":
                # `gone` IS POSITIVELY ESTABLISHED, never inferred from a missing
                # sample: only a census that is COMPLETE and did not contain this
                # number proves the namespace no longer holds it.
                try:
                    survey = self.census(self.pgid, deadline)
                except Exception:  # noqa: BLE001 - an unreadable survey is not absence
                    raise Unknown("leader unseen and the namespace could not be surveyed") from None
                if survey is None:
                    # A reader that states nothing states no coverage (round 9): the
                    # reason names that, so callers can tell it from a partial read.
                    raise Unknown("leader unseen and the census reader supplied no coverage claim")
                if not isinstance(survey, AuthoritativeCensus) or survey.complete is not True:
                    raise Unknown(
                        "census incomplete: leader unseen and no complete survey omits it"
                    )
                if getattr(survey, "leader_seen", True):
                    raise Unknown("leader unseen but no complete survey omits it")
                state = "gone"
            if state == "gone":
                if terminal:
                    # THE ONLY PLACE THE CENSUS MAY DECIDE. A send never reaches
                    # here: `signal()` calls `refresh(deadline=...)` without
                    # `terminal`, so REFUSED/unproven/pin_lost stay unreachable as a
                    # send. The witness authorises a SEND; the census reports an END.
                    # After a kill we ourselves performed, the leader is legitimately
                    # gone and the stronger evidence cannot exist any more -- so the
                    # terminal report may rest on the weaker one, and on nothing else.
                    return self._terminal_absence(deadline)
                raise Unknown("leader witness unreadable")
            # ONE VERDICT, ONE CYCLE'S EVIDENCE (see `_SAMPLE_CYCLES`). Every name
            # the proof below reads -- the censuses, members, samples, failures and
            # departures -- is rebuilt inside the cycle that decides; an earlier
            # cycle only decides whether another one starts.
            for cycle in range(_SAMPLE_CYCLES):
                started = self.clock()
                members = []
                samples = {}
                # A cycle that refuses before re-publishing these (its census
                # raises or is incomplete, its sample overruns) must not show a
                # DISCARDED cycle's members/observations beside its refusal; the
                # `sample_retries` rows are the only record of earlier cycles.
                # `vanished` too (review R1/S1): the late-member door below is the
                # first way to discard a cycle AFTER it pruned departures, and those
                # departures authorised nothing the deciding cycle proved.
                for stale in ("members", "observed", "vanished"):
                    self.probe.pop(stale, None)
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
                        self._terminal_result = self.last
                        return self.last
                    raise Unknown("live leader absent from a complete census: contradictory")
                samples = self._samples(set(members) | {self.leader}, deadline)
                # SNAPSHOT NOW (security review C11): `_samples` resets
                # `_sample_failures` on every call and `_leader_witness` calls it, so
                # any later witness read would erase this attempt's member evidence
                # -- or, after a refactor, substitute the leader's.
                failures = {row["pid"]: row for row in self._sample_failures}
                self.probe["observed"] = _observed(samples, None)
                # UNRESOLVED MEMBER SAMPLES ARE REFUSED BEFORE ANY EMPTINESS VERDICT.
                # A missing map entry cannot prove departure, so certifying "no live
                # members" while one member is unresolved would be false cleanup through
                # a second door.
                absent = [pid for pid in members if pid not in samples]
                departed, unresolved = self._departed(absent, first, second, failures)
                if departed and not unresolved:
                    kept = [pid for pid in members if pid not in departed]
                    if not kept:
                        # Never let a pruned set reach an emptiness verdict: the
                        # empty-census branch above is bracketed by its own witness
                        # reads and was not designed for "everything left". The
                        # refusal keeps the unpruned members as its residuals.
                        raise Unknown("every census member departed: no member to prove")
                    # Pruned BEFORE `live`, so identities, Verdict members and
                    # residuals all describe the members that remain.
                    members = kept
                    self.probe["vanished"] = [departed[pid] for pid in sorted(departed)]
                    absent = []
                if absent:
                    # THE RE-RUN DOOR (see `_SAMPLE_CYCLES`). It never removes a
                    # member: it discards this whole cycle and starts a fresh one,
                    # and only when one more cycle -- at the pace this one ran --
                    # still fits inside the deadline with room to spare.
                    now = self.clock()
                    if (
                        cycle < _SAMPLE_CYCLES - 1
                        and now + 2 * (now - started) < deadline
                        and self._rerunnable(unresolved, second, failures)
                    ):
                        self.probe.setdefault("sample_retries", []).append(
                            {
                                "cycle": cycle,
                                "rerun": unresolved[:16],
                                "rerun_count": len(unresolved),
                                "censuses": [first[:16], second[:16]],
                            }
                        )
                        continue
                    self.probe["unresolved"] = absent
                    # These are observations from the existing reads, not a new
                    # authority predicate. ENOENT ALONE is not amnesty for a PID seen
                    # by either complete census -- `/proc` visibility is not the
                    # kernel's process table (hidepid, setuid). The one exception is
                    # `_departed`'s conjunction, and it applies only when EVERY
                    # absent member qualifies; otherwise this refusal is unchanged.
                    self.probe["sample_failure"] = {
                        "phase": "terminal-member-sample" if terminal else "member-sample",
                        "witness": state,
                        "censuses": [first[:16], second[:16]],
                        "census_counts": [len(first), len(second)],
                        "unresolved_count": len(absent),
                        "unresolved": [
                            failures.get(pid, {"pid": pid, "reason": "unavailable", "errno": None})
                            for pid in absent[:16]
                        ],
                        # The cycles already discarded, so CI stderr shows what was tried
                        # before this refusal (empty when none was eligible).
                        "sample_retries": list(self.probe.get("sample_retries", [])),
                    }
                    raise Unknown("census member sample unreadable")
                live = [pid for pid in members if not _exited(samples[pid])]
                if not live:
                    # CENSUS 3, the zombie late-fork guard. Censuses 1 and 2 both
                    # PRECEDE the sample, so a member can fork a child after census
                    # 2 and then die: at sample time every member reads exited, and
                    # the live child was seen by neither census. Measured on main
                    # before this guard: OBSERVED_EMPTY with members (10, 21) while
                    # pid 30 ran -- false cleanup. Only a census taken AFTER the
                    # sample can see that child, so a corpse-only verdict waits for
                    # one; a live-member cycle needs none (its members are
                    # signalled and re-proved anyway).
                    third, late = self._late_members(members, samples, deadline)
                    if late:
                        now = self.clock()
                        if cycle < _SAMPLE_CYCLES - 1 and now + 2 * (now - started) < deadline:
                            self.probe.setdefault("sample_retries", []).append(
                                {
                                    "cycle": cycle,
                                    "late": late[:16],
                                    "late_count": len(late),
                                    "censuses": [first[:16], second[:16], third[:16]],
                                }
                            )
                            continue
                        self.probe["sample_failure"] = {
                            "phase": "post-sample-census",
                            "witness": state,
                            "censuses": [first[:16], second[:16], third[:16]],
                            "census_counts": [len(first), len(second), len(third)],
                            "late": late[:16],
                            "late_count": len(late),
                            "sample_retries": list(self.probe.get("sample_retries", [])),
                        }
                        # Loss outranks the unproven refusal (security S2), as in
                        # `_late_members`' own failure path: a leader that re-reads
                        # as a foreign birth is a pin lost, and the capability must
                        # be revoked. Anything short of `reaped` -- including an
                        # unreadable witness -- stays the fail-closed `Unknown`.
                        lost, _ = self._leader_witness(deadline)
                        if lost == "reaped":
                            raise PinLost("reservation lost after the corpse-only sample")
                        raise Unknown("group changed after the corpse-only sample")
                # This cycle produced a complete, resolved sample: it alone is the
                # evidence the proof below rests on.
                break
            else:
                # Unreachable: the last cycle may not `continue` (both retry doors
                # require `cycle < _SAMPLE_CYCLES - 1`), so it breaks or raises.
                # Spelled out FAIL-CLOSED rather than pre-binding `live = []`,
                # which would default an exhausted loop to the corpse-only verdict.
                raise Unknown("sample cycles exhausted without a decided cycle")
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
                self._terminal_result = self.last
                return self.last
            for pid in live:
                parent, group, sid = self.identity(pid)
                if (group, sid) != (self.pgid, self.sid):
                    raise Unknown("member outside the recorded group/session")
            # The witness must still hold at the END of the proof.
            final_state, final_witness = self._leader_witness(deadline)
            if final_state == "reaped":
                raise PinLost("reservation lost during proof")
            if final_state != "held":
                # A REQUIREMENT, NOT A BLACKLIST (review round 3, R4): `VERIFIED`
                # needs the witness to still HOLD at the end; unknown names must
                # not fall through to it by not being on a list.
                raise Unknown("witness %s during proof" % final_state)
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
            if isinstance(exc, PinLost):
                self.revoke()
            self.last = Verdict(
                "REFUSED",
                type(exc).__name__ + ":" + str(exc),
                tuple(members),
                refusal_class="pin_lost" if isinstance(exc, PinLost) else "unproven",
            )
            self.probe["error"] = type(exc).__name__ + ":" + str(exc)
            # Departures are evidence FOR a proof; a run that refused must not
            # publish them beside its refusal as if they had authorised anything.
            self.probe.pop("vanished", None)
            # The DECLINED set carries the same fields as the proven set — pid,
            # birth, scheme, group, session — because "who did we refuse to touch"
            # needs the same evidence as "who did we signal". The samples this check
            # already paid for supply the birth for a member that was never enrolled.
            self.probe["residuals"] = _identities(self.receipts, samples, members)
        return self.last

    def _late_members(self, members, samples, deadline):
        """Census 3 for a corpse-only cycle: return ``(third, late)``.

        ``late`` empty means the corpse-only sample still describes the group.
        Anything else is a pid the sample did not certify as exited:

        * a pid in ``third`` that censuses 1 and 2 never listed -- a child forked
          after census 2 (the measured hole);
        * a member whose re-sample is missing, no longer exited, or carries a
          missing or different birth -- the pid now hosts something else (PID
          reuse can put a live late child at a member's number, so a census that
          is a subset of ``members`` proves nothing on its own).

        A member ABSENT from ``third`` is fine: it was sampled exited and has
        since been reaped. The leader is excluded from the re-sample because
        ``_leader_witness`` owns its identity and runs after this anyway.

        The census uses the same strict ``_members`` reader as censuses 1 and 2;
        a failure re-reads the witness once so a reaped pin still reports as
        loss rather than as an unreadable census.
        """
        try:
            third = self._members(deadline)
        except Exception:
            lost, _ = self._leader_witness(deadline)
            if lost == "reaped":
                raise PinLost("reservation lost during the post-sample census") from None
            raise
        sampled = set(members)
        late = [pid for pid in third if pid not in sampled]
        if late:
            return third, late
        rest = [pid for pid in third if pid != self.leader]
        if not rest:
            return third, []
        again = self._samples(rest, deadline)
        for pid in rest:
            row = again.get(pid)
            if (
                row is None
                or not _exited(row)
                or row.birth is None
                or row.birth != samples[pid].birth
            ):
                late.append(pid)
        return third, late

    def _departed(self, absent, first, second, failures):
        """Split unsampled members into ``(departed, unresolved)``.

        WHY THIS EXISTS. A healthy churn gate's child can exit between the two
        censuses: CI 41665666039 (job 126020333214) recorded censuses
        ``[[2716,2739,2740,2741],[2716,2741]]`` with 2739/2740 failing their
        sample at ENOENT. The union-of-censuses rule then refused every such run
        with exit 126 ("census member sample unreadable") although nothing of
        ours was alive. Architect ruling 641ae72fe5d1 and security review
        e8f8f83bc578 approved dropping such a PID only under ALL of:

        - both censuses were authoritative and complete (guaranteed by
          `_members`, which raises otherwise, so we never get here);
        - P was in census 1 and NOT in census 2. LOAD-BEARING, not redundant
          with ESRCH: a complete strict scan requires start list minus
          proven-gone == end list, so a P absent from census 2 was already gone
          during that scan, and any child it forked into our group before dying
          is in census 2 or broke completeness. "In census 2, then ESRCH" would
          let a P fork a live child after census 2 and then be dropped;
        - P is neither the leader nor the pgid (the witness is never amnestied);
        - this attempt's sample diagnostic for P says ``read-error``/ENOENT.
          Any other reason, no entry, or past the 16-row cap refuses;
        - AFTER the sample, ``getpgid(P)`` raises ProcessLookupError. That is the
          kernel's answer, independent of `/proc` visibility: a zombie still
          answers getpgid, hidepid cannot fake ESRCH, and a reused PID returns a
          number. ONLY ProcessLookupError counts -- a value, EPERM/EACCES (LSM
          denial) or any other error refuses;
        - `/proc` is our PID namespace (C10): `/proc/self` must name our pid,
          because getpgid resolves in OUR namespace and a foreign `/proc` would
          make "ENOENT there + ESRCH here" describe two different processes.

        The witness state is already "held" (the only branch that reaches the
        member sample) and every later witness check is unchanged. Returns
        ``departed`` as ``{pid: evidence}`` for the run record.
        """
        if not absent:
            return {}, []
        probe = self._departure
        if probe is None:
            if self.samples is not None:
                # An injected sampler carries no trustworthy per-PID errno.
                return {}, list(absent)
            probe = os.getpgid
        try:
            own = (self._proc_self or (lambda: os.readlink("/proc/self")))()
        except Exception:  # noqa: BLE001 - an unreadable namespace grants nothing
            return {}, list(absent)
        if own != str(os.getpid()):
            return {}, list(absent)
        seen_first, seen_second = set(first), set(second)
        departed, unresolved = {}, []
        for pid in absent:
            row = failures.get(pid)
            if (
                pid in (self.leader, self.pgid)
                or pid not in seen_first
                or pid in seen_second
                or row is None
                or row.get("reason") != "read-error"
                or row.get("errno") != errno.ENOENT
            ):
                unresolved.append(pid)
                continue
            try:
                probe(pid)
            except ProcessLookupError:
                departed[pid] = {
                    "pid": pid,
                    "reason": "read-error",
                    "errno": errno.ENOENT,
                    "getpgid": "ProcessLookupError",
                    "census": "first-only",
                }
                continue
            except Exception:  # noqa: BLE001 - anything but ESRCH is not departure
                pass
            unresolved.append(pid)
        return departed, unresolved

    def _rerunnable(self, unresolved, second, failures):
        """Whether this cycle's unresolved members justify discarding it and re-running.

        WHY THIS EXISTS BESIDE `_departed`. `_departed` deliberately refuses a
        member that is in BOTH censuses and then reads ENOENT: dropping it would
        let a member fork a live child after census 2 and then be amnestied (the
        late-fork hole). But that is also exactly what a healthy fast-forking
        command looks like when a child exits between census 2 and its sample
        (run 41748872018, job 126359450123: censuses
        ``[[2425,2443,2444,2445,2446],[2425,2444,2445,2446]]``, 2444 ENOENT).
        Neither amnesty nor refusal is right for that shape; a fresh cycle is,
        because any child forked before the death is in the next census.

        This predicate ONLY decides whether another full census->census->sample
        cycle starts. It never removes a member and never feeds evidence into the
        next cycle, so it may be looser than `_departed` only in admitting the
        in-both shape; every other conjunct is the same one `_departed` uses.
        The census-2 and non-empty conjuncts never decide today -- `_departed`
        runs first and either prunes a census-1-only pid or leaves it unresolved
        for a reason re-checked here, and `unresolved` is non-empty whenever
        `absent` is -- so they are a defence-in-depth net for a later reordering,
        not dead code:

        - ``unresolved`` is non-empty and EVERY pid in it qualifies (one
          unexplained member is a refusal, not a retry);
        - the pid is in census 2 (the in-census-1-only shape is `_departed`'s);
        - it is neither the leader nor the pgid;
        - this cycle's diagnostic says ``read-error``/ENOENT;
        - ``getpgid`` raises ProcessLookupError (the kernel agrees it is gone);
        - `/proc/self` names our pid (C10), and the native sampler is in use
          (an injected sampler carries no per-PID errno).
        """
        if not unresolved or self.samples is not None:
            return False
        probe = self._departure or os.getpgid
        try:
            own = (self._proc_self or (lambda: os.readlink("/proc/self")))()
        except Exception:  # noqa: BLE001 - an unreadable namespace grants nothing
            return False
        if own != str(os.getpid()):
            return False
        seen_second = set(second)
        for pid in unresolved:
            row = failures.get(pid)
            if (
                pid in (self.leader, self.pgid)
                or pid not in seen_second
                or row is None
                or row.get("reason") != "read-error"
                or row.get("errno") != errno.ENOENT
            ):
                return False
            try:
                probe(pid)
            except ProcessLookupError:
                continue
            except Exception:  # noqa: BLE001 - anything but ESRCH is not departure
                return False
            return False
        return True

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
            if final_state == "reaped":
                self.revoke()
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
        # Check at the mutation boundary as well as refresh entry: a cached
        # VERIFIED verdict cannot lend authority after a consuming wait.
        if self._revoked:
            self.last = Verdict(
                "REFUSED", "signal authority irreversibly revoked", refusal_class="pin_lost"
            )
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
        if signum in (signal.SIGKILL, signal.SIGTERM):
            # THE FACT IS NARROW ON PURPOSE: a checked send happened AND it is one
            # this code's ladder uses to end a group. `0`, SIGCONT, SIGSTOP and the
            # rest prove delivery, not an ending.
            self.terminating_sends.append(signum)
        self.last = Verdict(
            "SIGNALED",
            "checked owner-alive signal; non-atomic race remains",
            result.members,
            self.clock() - checked,
            anchor=result.anchor,
        )
        return self.last
