"""The TARGET-side receipt of a termination signal: when, which, and whether anyone
had staged a stop for it.

WHY THIS MODULE EXISTS. The stop marker (``control._stop_marker_payload``) is the
ACTING party's statement, written before it signals; it can only exist when a
cooperating party did the signalling. The 2026-09-30 18:14 wave (twelve exec
workers and two runtimes signalled inside seventeen seconds) was the other kind:
nobody staged anything, so the victims had nothing to pair a signal with, and the
only surviving facts were an exit code and one journal row a later boot
overwrote. This module is what a victim writes AT ARRIVAL so that gap becomes a
statement ("SIGTERM arrived at HH:MM:SS; no stop was staged for it") instead of
silence.

WHAT IT CAN AND CANNOT SAY (stated here once, repeated in ``docs/EXEC.md``):

* signal name and number, arrival time, whether work was in flight, what the
  handler did, and the receiver's own identity: YES.
* the SENDER's pid and uid: YES, where ``local_operator.siginfo`` could arm its
  one-shot ``SA_SIGINFO`` sensor over the handler that owns the signal (Darwin,
  today). The stdlib exposes no siginfo of its own (``signal.sigwaitinfo`` is
  absent on this host) and asyncio's ``add_signal_handler`` discards it, which is
  why the capture needed a module of its own rather than a flag; where it is not
  armed, or the platform cannot capture, the receipt still says
  ``sender: unavailable`` and claims nothing. The 2026-10-30 21:57 wave — twelve
  exec workers, at least five runtimes and two daemons signalled inside 1.33 s —
  is the reason that distinction is worth a whole module: every one of those rows
  said ``no-siginfo``, so the sender was unnameable and the incident
  unfalsifiable.
  The ``sender`` dict carries ``state``/``reason`` on both arms, plus ``pid``,
  ``uid`` and ``code`` when captured, and a pid of ``0`` there means the KERNEL
  delivered it (no sending process), never "process zero".
* SIGKILL, a crash or a power loss: NOTHING. The target is not executing. Those
  stay on the acting party's marker (SIGKILL rung) or on no evidence at all.

STILL NO SENDER, BUT DIRECTION GAINS ONE HONEST FACT. The spawn chain this
process was born into (``LOP_SPAWN_CHAIN``, recorded by ``macos_disclaim``)
carries each member's liveness as recorded at spawn (``alive_at_spawn``), and
this module adds a second, arrival reading (``alive_now``) when a signal lands.
When the app at the root of that chain was RECORDED ALIVE and is GONE at
arrival, the renderer says exactly that — the app was running when the runtime
started and was no longer running when the signal arrived, with the limit
stated in the sentence itself ("when it exited is not recorded") — and says
nothing when the readings do not conspire, so the clause cannot read as the
cause of a signal or as a named sender.

PAIRING IS DECIDED AT WRITE TIME. The marker is staged BEFORE the signal by
contract (control's "marker first, then the signal"), so it is on disk at receipt
iff the signal was sanctioned. Reading it at that instant and snapshotting a
compact subset into the receipt means a later reader needs the receipt alone to
answer "was this signal asked for" — and a marker staged AFTER the signal can
never retroactively sanction it. The residual false-sanction risk (a stale
in-window marker adjacent to an unrelated signal) is bounded by
:data:`PAIR_WINDOW_S` and is a stated limit, not a hidden one.

Stdlib plus ``registry``, ``types`` and ``macos_disclaim`` (itself stdlib-only,
and imported for the spawn-chain snapshot): this is imported from the runtime's
signal callback and from the exec worker, and neither may pay for anything
heavier. Every function that runs on a signal-arrival path NEVER RAISES.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from local_operator.macos_disclaim import snapshot_spawn_chain
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SIGNAL_DRAIN_S

logger = logging.getLogger(__name__)

#: Schema version of the receipt, bumped only on an incompatible change.
SCHEMA_VERSION = 1

#: The name of the refusals record this module writes — re-exported so a caller or a
#: guard test names it from the module that OWNS the behaviour, while the declaration
#: itself lives in the stdlib-only leaf (``runtime.artefact_names``) that the two
#: enumerating registries can import without a cycle (see that module).
SIGNAL_REFUSALS_NAME = registry.SIGNAL_REFUSALS_NAME

#: How far BEFORE the signal a marker may have been staged and still sanction it.
#: The runtime's own drain bound plus a minute: the ladder stages a marker, then
#: signals, and a rung-1 marker (socket) can precede a rung-2 signal by the
#: whole drain wait. Anything older is a different act.
PAIR_WINDOW_S = SIGNAL_DRAIN_S + 60.0

#: A receipt keeps at most this many signal entries; ``count`` keeps the true
#: total. Bounded because a repeating sender must not grow a file on the signal
#: path without limit.
MAX_SIGNALS = 8

#: Tolerance when comparing two readings of one run's ``started_at`` (the same
#: rounding allowance ``attention._RUN_KEY_TOLERANCE_S`` applies to a marker).
_RUN_KEY_TOLERANCE_S = 1.0

#: What the receipt says about the sender, and why. ``could_be`` is kill(2)'s
#: permission rule (same uid, or root) — a true, INFERRED fact, labelled as such
#: by its key — never an observation of who signalled.
#:
#: THIS IS THE NOT-CAPTURED SHAPE, and it stays BYTE-IDENTICAL to what every
#: receipt written before ``local_operator.siginfo`` existed says: readers and
#: tests downstream of a persist-and-render chain already key off exactly these
#: three fields, so the sender that could not be named keeps its old spelling
#: rather than acquiring a new one.
SENDER_UNAVAILABLE: dict[str, str] = {
    "state": "unavailable",
    "reason": "no-siginfo",
    "could_be": "same-uid-or-root",
}

#: The sender the KERNEL filled in: ``siginfo_t.si_pid`` is 0, which means "no
#: sending process" (measured on this host: a ``setitimer`` SIGALRM arrives with
#: ``si_pid 0, si_uid 0, si_code 0``), NOT "process zero". Kept distinct from
#: :data:`SENDER_UNAVAILABLE` because the two are different facts — one is "the
#: capture never happened", this one is "the capture happened and named no
#: process" — and rendered the same way, because neither names a party.
SENDER_KERNEL: dict[str, str] = {
    "state": "unavailable",
    "reason": "kernel-originated",
}

#: The sender that WAS named. ``pid``/``uid`` are ``si_pid``/``si_uid`` as the
#: kernel handed them over, and ``code`` is ``si_code`` verbatim — recorded, not
#: interpreted, because its meaning depends on the origin (``kill(2)`` and a
#: kernel timer both measured ``0`` on Darwin). ``reason`` stays on this arm so a
#: reader that only looks at ``reason`` never has to special-case the shape.
SENDER_SIGINFO_REASON = "siginfo"

#: The ``action`` a signal entry carries when the VICTIM REFUSED the arrival.
#:
#: A terminating signal with no covering sanction is recorded and then not acted
#: on (``process._route_signal``, ``exec_worker``'s handler), and this is the word
#: the durable row says instead of the ``stop``/``drain``/``repeat-absorbed`` it
#: would have carried. It is a distinct ACTION rather than a flag because the
#: question a reader of the row has is "what did the handler do?", and "it kept
#: running" must not be confused with "it will stop at its boundary".
ACTION_REFUSED = "refused"

#: ``stop_class`` vocabulary shared by the exec ledger and the readers.
CLASS_DELIBERATE = "deliberate"
CLASS_ATTRIBUTED = "attributed-involuntary"
CLASS_UNATTRIBUTED_SIGNAL = "unattributed-signal"
CLASS_UNATTRIBUTED_DEATH = "unattributed-death"


def receiver_facts() -> dict[str, Any]:
    """The receiving process's own identity. Never raises.

    ``ppid`` is the receiver's PARENT — lineage (who spawned it), NOT the sender —
    and is named for what it is so a reader does not mistake it for one. The uid
    getters are guarded because they do not exist on Windows.
    """
    facts: dict[str, Any] = {
        "ppid": os.getppid(),
        "argv0": os.path.basename(sys.argv[0] or "") or sys.executable,
    }
    for key, getter in (
        ("uid", "getuid"),
        ("euid", "geteuid"),
        ("gid", "getgid"),
        ("pgid", "getpgrp"),
    ):
        fn = getattr(os, getter, None)
        if callable(fn):
            try:
                facts[key] = fn()
            except OSError:
                pass
    return facts


def _stamp(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value == 0:
        return None
    return float(value)


def marker_subset(marker: dict[str, Any] | None) -> dict[str, Any] | None:
    """The compact part of a stop marker a receipt snapshots, or ``None``."""
    if not isinstance(marker, dict):
        return None
    killer = marker.get("killer")
    killer = killer if isinstance(killer, dict) else {}
    subset: dict[str, Any] = {
        "at": marker.get("at"),
        "rung": marker.get("rung"),
        "deliberate": marker.get("deliberate"),
        "killer_pid": killer.get("pid"),
        "killer_command": killer.get("command") or killer.get("argv0"),
    }
    # ``mechanism``/``actor`` exist only on an involuntary act, and ``sweep_id``
    # only on a sweep's markers; absent stays absent so the snapshot mirrors the
    # source rather than inventing empties.
    for key in ("mechanism", "actor", "sweep_id"):
        if marker.get(key):
            subset[key] = marker[key]
    return subset


def marker_covers_signal(
    marker: dict[str, Any] | None,
    *,
    session_id: str,
    pid: int,
    started_at: float | None,
    at: float,
) -> bool:
    """Whether ``marker`` was staged FOR a signal arriving now at this run.

    All four must hold: the marker names this run (session, pid, and
    ``started_at`` when both sides know it) AND was staged no earlier than
    :data:`PAIR_WINDOW_S` before ``at`` and not after it. The last bound is the
    ordering invariant: a marker staged after the signal did not sanction it.
    """
    if not isinstance(marker, dict):
        return False
    try:
        if str(marker.get("session_id") or "") != session_id:
            return False
        if int(marker.get("pid") or -1) != int(pid):
            return False
        theirs = _stamp(marker.get("started_at"))
        if theirs is not None and started_at is not None:
            if abs(theirs - float(started_at)) >= _RUN_KEY_TOLERANCE_S:
                return False
        staged = _stamp(marker.get("at"))
        if staged is None:
            return False
        return 0.0 <= at - staged <= PAIR_WINDOW_S
    except (TypeError, ValueError):
        return False


def sender_facts(number: int) -> dict[str, Any]:
    """What the receipt can say about WHO sent signal ``number``. Never raises.

    THE ONE PLACE THE CAPTURE IS TURNED INTO THE PERSISTED SHAPE, shared by both
    writers this receipt has (the runtime's signal callback and the exec worker)
    so the two cannot describe one delivery differently.

    Three arms, and the difference between the last two is the point:

    * nothing captured -> :data:`SENDER_UNAVAILABLE`, byte-identical to every
      receipt written before the capture existed (an unwritten fact, not a gap
      anybody can widen later);
    * captured with ``si_pid == 0`` -> :data:`SENDER_KERNEL`: the kernel
      delivered it and no process did, so there is nobody to name;
    * captured with a positive ``si_pid`` -> the named arm, carrying the pid, the
      uid and the raw ``si_code``.

    An implausible capture (a negative pid, a value that is not an int at all)
    falls back to :data:`SENDER_UNAVAILABLE` rather than persisting a guess: the
    wrong pid in an incident report is worse than the old "unavailable".
    """
    try:
        from local_operator import siginfo

        facts = siginfo.take(number)
    except Exception:  # noqa: BLE001 — an instrument must never fail a signal path
        return dict(SENDER_UNAVAILABLE)
    if not isinstance(facts, dict):
        return dict(SENDER_UNAVAILABLE)
    pid = facts.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return dict(SENDER_UNAVAILABLE)
    if pid == 0:
        return dict(SENDER_KERNEL)
    if pid < 0:
        return dict(SENDER_UNAVAILABLE)
    uid = facts.get("uid")
    code = facts.get("code")
    known: dict[str, Any] = {
        "state": "known",
        "reason": SENDER_SIGINFO_REASON,
        "pid": int(pid),
        "uid": int(uid) if isinstance(uid, int) and not isinstance(uid, bool) else None,
        "code": int(code) if isinstance(code, int) and not isinstance(code, bool) else None,
    }
    return known


def build_signal(
    name: str,
    number: int,
    *,
    at: float,
    in_flight: bool | None,
    action: str,
    marker: dict[str, Any] | None,
    covered: bool,
) -> dict[str, Any]:
    """One signal entry. ``covered`` is the pairing verdict, decided by the caller
    from :func:`marker_covers_signal`; ``marker`` is snapshotted only when covered.

    The sender is read from the capture AT THIS INSTANT (:func:`sender_facts`),
    not passed in: the capture is the arrival's own evidence and it is already
    stored by the time any Python-level handler runs (``siginfo`` restores the
    disposition and re-raises, so the platform's delivery is what gets here).
    """
    entry: dict[str, Any] = {
        "name": name,
        "number": number,
        "at": at,
        "in_flight": in_flight,
        "action": action,
        "sender": sender_facts(number),
        "stop_marker": marker_subset(marker) if covered else None,
        "sanction": "marker" if covered else "none",
    }
    return entry


def _same_run(existing: dict[str, Any], run_key: dict[str, Any]) -> bool:
    if existing.get("session_id") != run_key.get("session_id"):
        return False
    if existing.get("pid") != run_key.get("pid"):
        return False
    mine, theirs = _stamp(run_key.get("started_at")), _stamp(existing.get("started_at"))
    if mine is not None and theirs is not None:
        return abs(mine - theirs) < _RUN_KEY_TOLERANCE_S
    return True


def merge_into(
    existing: dict[str, Any] | None, run_key: dict[str, Any], signal_entry: dict[str, Any]
) -> dict[str, Any]:
    """The receipt after ``signal_entry`` arrives: appended within a run, replaced
    by a fresh receipt when ``existing`` belongs to a different run.

    A receipt for another run is NOT carried forward — the run key is what stops
    an earlier run's signal from narrating a later run's death, exactly as it does
    for the stop marker.

    A REFUSED ARRIVAL NEVER REACHES THIS FUNCTION. Refusals live in their own file
    (``registry.SIGNAL_REFUSALS_NAME``) so that "this receipt exists" keeps meaning
    "a signal reached this run and its pairing is in here" — the invariant
    ``covers_run`` and ``rescue._death_class_receipt`` both read. See that constant
    for the terminal ``unclassified/receipt-no-signals`` state a signals-less
    receipt would otherwise create in the rescue lane.
    """
    signals: list[dict[str, Any]] = []
    count = 0
    if isinstance(existing, dict) and _same_run(existing, run_key):
        raw = existing.get("signals")
        if isinstance(raw, list):
            signals = [item for item in raw if isinstance(item, dict)]
        count = int(existing.get("count") or len(signals))
    if len(signals) < MAX_SIGNALS:
        signals.append(signal_entry)
    count += 1
    receipt: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "kind": run_key.get("kind"),
        "session_id": run_key.get("session_id"),
        "pid": run_key.get("pid"),
        "started_at": run_key.get("started_at"),
        "signals": signals,
        "count": count,
        "receiver": receiver_facts(),
        "platform": sys.platform,
    }
    return receipt


def rearm_sender_capture(number: int) -> bool:
    """Put ``siginfo``'s one-shot sensor back after a REFUSED arrival spent it. Never raises.

    WHY A REFUSAL HAS TO DO THIS (sanction gate x ``local_operator.siginfo``). The
    sensor is spent by the FIRST delivery (it restores the platform's action before
    re-raising) and ``siginfo.arm`` on an already-armed signal does not re-install it.
    On the old path the first terminating signal ended the process, so a spent sensor
    cost nothing. Under the gate a REFUSED signal leaves the process running, so
    without this the sensor is gone for the rest of its life and the sanctioned stop,
    whose receipt is the one an incident is reconstructed from, would say
    ``no-siginfo`` about a capturable sender.

    ONLY ``siginfo.rearm`` IS SAFE HERE, never ``disarm`` + ``arm``: that pair
    un-publishes the saved action for a moment, and a delivery inside the window runs
    the sensor with nothing to restore, which re-raises at the process forever (0.2 ms
    arrivals hung 4 of 8 runs in review). ``rearm`` keeps the state published, so the
    worst a racing signal can suffer is the ordinary path and a ``no-siginfo`` receipt.

    Only a signal this process ARMED is re-armed: SIGINT and the residual levers are
    not armed by anyone, so this is a no-op for them. Called AFTER the arrival has
    claimed its capture (``observe``), so it can never take the capture that arrival
    is still owed. A failed re-arm is LOGGED (debug) because the receipt it degrades
    is otherwise silent.
    """
    try:
        from local_operator import siginfo

        if not siginfo.armed(int(number)):
            return False
        ok = bool(siginfo.rearm(int(number)))
        if not ok:
            logger.debug(
                "sender capture for signal %s could not be re-armed after a refusal; "
                "the next receipt will say no-siginfo",
                number,
            )
        return ok
    except Exception:  # noqa: BLE001 — an instrument must never fail a signal path
        return False


def _named_sender_text(sender: object) -> str:
    """``"pid 7 (uid 0)"`` for a sender that names a process, ``""`` otherwise.

    ONE IMPLEMENTATION, in ``incidents`` — the module that already renders this
    receipt's sentences (:func:`incidents.render_named_sender`) — so the log line
    and the transcript cannot disagree about whether a party was named. The
    import is local because ``incidents`` is the heavier module and this function
    runs on the exit path; ``cut_off_verdict`` imports it the same way and for the
    same reason.
    """
    try:
        from local_operator.incidents import render_named_sender

        return render_named_sender(sender)
    except Exception:  # noqa: BLE001 — a log suffix must never fail an exit path
        return ""


def refusal_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A refusal record built from the arrival entry.

    ``disposition: refused`` is the class in one word, and the arrival's own facts
    (name, number, time, sender, spawn chain, in-flight state) ride along so the
    refusal is as evidential as the signal it refused. ``sanction`` is ``none`` by
    construction: an arrival a marker covered is never refused.
    """
    refusal: dict[str, Any] = {
        "disposition": "refused",
        "name": entry.get("name"),
        "number": entry.get("number"),
        "at": entry.get("at"),
        "first_at": entry.get("at"),
        "count": 1,
        "in_flight": entry.get("in_flight"),
        "sender": dict(entry.get("sender") or SENDER_UNAVAILABLE),
        "sanction": "none",
    }
    chain = entry.get("spawn_chain")
    if chain is not None:
        refusal["spawn_chain"] = chain
    return refusal


def merge_refusal(
    existing: dict[str, Any] | None,
    run_key: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    """The refusals document after a REFUSED arrival: ONE coalesced entry per run.

    COALESCED, NOT APPENDED, and that is a requirement rather than tidiness: a sender
    looping SIGTERM at one runtime must not be able to grow a file on the signal path
    without limit (``MAX_SIGNALS`` bounds the receipt for the same reason). A document
    for a DIFFERENT run is not carried forward at all — the run key is what stops an
    earlier run's refusals from narrating a later run.
    """
    refusals: list[dict[str, Any]] = []
    if isinstance(existing, dict) and _same_run(existing, run_key):
        raw = existing.get("refusals")
        if isinstance(raw, list):
            refusals = [item for item in raw if isinstance(item, dict)]
    fresh = refusal_entry(entry)
    if refusals:
        previous = refusals[0]
        previous["count"] = int(previous.get("count") or 1) + 1
        previous["at"] = fresh["at"]
        previous["name"] = fresh["name"]
        previous["number"] = fresh["number"]
        # SCHEMA: ``sender`` is the FIRST refused arrival's (who tried first) and never
        # changes; ``last_sender`` is the LATEST's and is ABSENT while ``count`` is 1.
        # With the sensor re-armed after each refusal a repeat can name its own
        # sender, and a second party signalling the same run is exactly the fact a
        # coalesced entry would otherwise lose. ``describe()`` and the incident
        # renderers read the signal receipt, not this file, so they show neither.
        previous["last_sender"] = fresh["sender"]
        refusals = [previous]
    else:
        refusals = [fresh]
    return {
        "v": SCHEMA_VERSION,
        "kind": run_key.get("kind"),
        "session_id": run_key.get("session_id"),
        "pid": run_key.get("pid"),
        "started_at": run_key.get("started_at"),
        "refusals": refusals,
        "receiver": receiver_facts(),
        "platform": sys.platform,
    }


def describe(receipt: dict[str, Any] | None) -> str:
    """One line naming the LATEST signal, its pairing and its sender, for a log line.

    ``sender=`` stays the word ``unavailable`` where nothing was captured, so a
    log reader's existing grep keeps working, and becomes ``pid N (uid U)`` when
    the capture named a party — the same two facts the rendered sentence carries.
    Never raises.
    """
    try:
        if not receipt:
            return ""
        signals = receipt.get("signals") or []
        last = signals[-1]
        when = time.strftime("%H:%M:%S", time.localtime(float(last["at"])))
        sanction = last.get("sanction") or "none"
        extra = (
            f" (+{int(receipt.get('count') or 1) - 1} more)"
            if (receipt.get("count") or 1) > 1
            else ""
        )
        who = _named_sender_text(last.get("sender")) or "unavailable"
        return f"{last.get('name')} at {when}, sanction={sanction}, sender={who}{extra}"
    except Exception:  # noqa: BLE001 — a log suffix must never fail an exit path
        return ""


def observe(
    directory: Path | None,
    *,
    session_id: str,
    pid: int,
    started_at: float | None,
    name: str,
    number: int,
    in_flight: bool | None,
    action: str,
    refused_action: str | None = None,
) -> dict[str, Any]:
    """Build the signal entry for a signal arriving NOW, pairing it with the marker.

    The marker is read at this instant (see the module docstring on why pairing is
    decided at write time). May raise only on programmer error; callers on a signal
    path wrap it (:func:`record`, ``exec_worker``).

    ``refused_action`` IS HOW THE VICTIM RECORDS A REFUSAL IN ONE WRITE. The
    caller cannot know ``covered`` before this call — pairing is decided here, by
    reading the marker — while ``action`` is written into the same row, so a
    caller that owns the gate passes :data:`ACTION_REFUSED` and the row this
    arrival leaves on disk already says the runtime kept running. Passing it is
    also what keeps the write single: a second pass over the receipt to correct an
    action would be a second window in which a SIGKILL loses the correction.
    """
    at = time.time()
    marker = registry.read_stop_marker(directory) if directory is not None else None
    covered = marker_covers_signal(
        marker, session_id=session_id, pid=pid, started_at=started_at, at=at
    )
    # AN UNCOVERED ARRIVAL IS REFUSED, so its row says so (see ACTION_REFUSED).
    # ``covered`` is the same verdict the caller's gate reads back off this entry,
    # so the action and the gate cannot disagree about one signal.
    if refused_action is not None and not covered:
        action = refused_action
    entry = build_signal(
        name, number, at=at, in_flight=in_flight, action=action, marker=marker, covered=covered
    )
    # The spawn chain with the arrival half of the renderer's gate: each
    # member already carries its liveness as recorded at spawn
    # (``macos_disclaim``); this snapshot adds the reading taken NOW, and
    # ``incidents`` renders the clause only when the two conspire (recorded
    # alive, gone at arrival). ``None`` when this process was not spawned with
    # a chain — absent stays absent, and the renderer says nothing extra.
    chain = snapshot_spawn_chain()
    if chain is not None:
        entry["spawn_chain"] = chain
    return entry


def _run_key(kind: str, session_id: str, pid: int, started_at: float | None) -> dict[str, Any]:
    """The run identity a receipt, a stop marker and a refusal all agree on."""
    return {"kind": kind, "session_id": session_id, "pid": pid, "started_at": started_at}


def _write(
    directory: Path | None,
    *,
    run_key: dict[str, Any],
    entry: dict[str, Any],
    merge: Callable[..., dict[str, Any]],
    read: Callable[[Path], dict[str, Any] | None],
    write: Callable[[Path, dict[str, Any]], object],
    what: str,
) -> dict[str, Any] | None:
    """Write one arrival into its own file, merging it with what is already there.

    NEVER RAISES, and that contract is the whole of this function: the two writers
    below differ only in WHICH FILE they write (the receipt's ``signals``, or the
    refusals document next to it) and both sit on a signal path where an instrument
    must not delay or fail anything.

    The directory is NOT created: evidence must not conjure a session directory.
    Where there is none the same facts are logged at WARNING and the line says
    "logged only", so the evidence is at worst in the runtime log.
    """
    try:
        present = directory is not None and directory.is_dir()
        existing = read(directory) if present and directory else None
        receipt = merge(existing, run_key, entry)
        if present and directory:
            write(directory, receipt)
        else:
            logger.warning(
                "session runtime: %s receipt logged only (no conversation directory): %s",
                what,
                describe(receipt),
            )
        return receipt
    except Exception:  # noqa: BLE001 — an instrument must never delay or fail a signal path
        # LOUD, not debug: a receipt that cannot be written is the evidence gap this
        # module exists to close, and a silent loss here is indistinguishable from a
        # runtime that never got the signal. The same facts the file would have held.
        logger.warning(
            "session runtime: %s receipt could not be written (signal %s/%s, pid %s, "
            "session %s, started_at %s, directory %s)",
            what,
            entry.get("name"),
            entry.get("number"),
            run_key.get("pid"),
            run_key.get("session_id"),
            run_key.get("started_at"),
            directory,
            exc_info=True,
        )
        return None


def record_signal(
    directory: Path | None,
    *,
    kind: str,
    session_id: str,
    pid: int,
    started_at: float | None,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Write a PERMITTED arrival into the receipt's ``signals`` list.

    Taking the already-built ``entry`` (from :func:`observe`) rather than the facts
    is what makes the gate honest: the pairing verdict the caller acted on and the
    row it wrote are ONE evaluation, so no second marker read can disagree with the
    decision a few statements earlier.
    """
    return _write(
        directory,
        run_key=_run_key(kind, session_id, pid, started_at),
        entry=entry,
        merge=merge_into,
        read=registry.read_signal_receipt,
        write=registry.write_signal_receipt,
        what="signal",
    )


def record_refusal(
    directory: Path | None,
    *,
    kind: str,
    session_id: str,
    pid: int,
    started_at: float | None,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Write a REFUSED arrival into its OWN file, coalesced per run.

    Deliberately not the receipt: see ``registry.SIGNAL_REFUSALS_NAME`` for the
    terminal ``unclassified/receipt-no-signals`` state a signals-less receipt creates
    in the rescue lane, and :func:`merge_refusal` for the coalescing rule.
    """
    return _write(
        directory,
        run_key=_run_key(kind, session_id, pid, started_at),
        entry=entry,
        merge=merge_refusal,
        read=registry.read_signal_refusals,
        write=registry.write_signal_refusals,
        what="refusal",
    )


def record(
    directory: Path | None,
    *,
    kind: str,
    session_id: str,
    pid: int,
    started_at: float | None,
    name: str,
    number: int,
    in_flight: bool | None,
    action: str,
    refused_action: str | None = None,
) -> dict[str, Any] | None:
    """Observe one arrival and write it as a signal. NEVER RAISES.

    The convenience form for a caller that has no gate to run (and the shape every
    test helper uses). Callers that DO own the gate call :func:`observe` themselves,
    decide, and then write through :func:`record_signal` or :func:`record_refusal` —
    so the decision and the row cannot come from two different marker reads.

    ``refused_action`` is forwarded to :func:`observe` (see its docstring). Returns
    the receipt it built, or ``None`` when even building it failed.
    """
    try:
        entry = observe(
            directory,
            session_id=session_id,
            pid=pid,
            started_at=started_at,
            name=name,
            number=number,
            in_flight=in_flight,
            action=action,
            refused_action=refused_action,
        )
    except Exception:  # noqa: BLE001 — an instrument must never delay or fail a signal path
        logger.warning(
            "session runtime: signal receipt could not be built "
            "(signal %s/%s, pid %s, session %s, started_at %s, directory %s)",
            name,
            number,
            pid,
            session_id,
            started_at,
            directory,
            exc_info=True,
        )
        return None
    return record_signal(
        directory,
        kind=kind,
        session_id=session_id,
        pid=pid,
        started_at=started_at,
        entry=entry,
    )


def covers_run(receipt: dict[str, Any] | None, directory_name: str, dead: Any | None) -> bool:
    """Whether ``receipt`` attests to the RUN being classified (reader side).

    The same refusal ``attention._stop_marker_covers_run`` applies to a marker: the
    conversation id must match, and when a dead record survives its pid and start
    time must too. With no record to compare, the receipt is NOT accepted blind —
    unlike the marker's permissive no-evidence answer, a receipt claims a specific
    signal at a specific time, so an unverifiable one is refused.
    """
    if not isinstance(receipt, dict) or not receipt.get("signals"):
        return False
    if str(receipt.get("session_id") or "") != directory_name:
        return False
    if dead is None:
        return False
    try:
        if int(receipt.get("pid") or -1) != int(getattr(dead, "pid", -2) or -2):
            return False
        theirs, ours = _stamp(receipt.get("started_at")), _stamp(getattr(dead, "started_at", 0.0))
        if theirs is not None and ours is not None:
            return abs(theirs - ours) < _RUN_KEY_TOLERANCE_S
    except (TypeError, ValueError):
        return False
    return True


def is_sanctioned(signal_entry: dict[str, Any] | None) -> bool:
    """Whether a durable sanction covers this arrival — the victim gate's ONE question.

    THE GATE, STATED POSITIVELY SO THE TWO CALLERS CANNOT DISAGREE. A session
    runtime and an exec worker each decide whether a terminating signal may take
    the run away, and the answer is exactly "a stop marker was staged for this run
    before the signal arrived".

    THE PERMIT IS ``stop_class_of`` ∈ {deliberate, attributed-involuntary} — the
    contract's own predicate (``docs/design-surface-independence.md`` §7 step 2 at
    ``3ba837e6d9e``), which is ``sanction == "marker"`` seen through the classifier
    that decides what a cut turn SAYS. It is deliberately NOT "``cut_off_verdict``
    returned something": that function answers "what does a signal-cut turn say", so
    a covering involuntary marker — ``reclaim``'s own sanctioned sweep — returns a
    ``runtime-killed`` verdict and a gate keyed on it would refuse the very sweeps
    the design certifies as compliant.

    BOTH KINDS OF MARKER SANCTION A SIGNAL, deliberately. ``deliberate`` is a stop
    a person asked for (the ladder stages one before every rung); an involuntary
    marker (``control.note_involuntary_stop``, ``deliberate: false``) is a harness
    act that names itself and its actor. What the gate exists to make INERT is the
    third case — no marker at all — which is the measured 2026-10-30 shape: twelve
    exec workers and at least five runtimes ended inside 1.33 s, nothing staged a
    stop for any of them, and no ``logs/stop-sweeps.jsonl`` existed, so no
    in-product stop was ever asked for. ``deliberate`` still decides what the
    CUT-OFF says (:func:`cut_off_verdict`); it never decides whether a stop is
    obeyed.

    A ``None`` or malformed entry answers ``False``. That is the fail-closed
    direction on purpose: an entry this build cannot read is not evidence of a
    sanction, and the cost of being wrong is a refused signal — which SIGKILL
    still ends — rather than a killed turn nobody can attribute.
    """
    return bool(stop_class_of(signal_entry) in (CLASS_DELIBERATE, CLASS_ATTRIBUTED))


def stop_class_of(signal_entry: dict[str, Any] | None) -> str:
    """The ``stop_class`` a signal entry earns from its pairing.

    ``marker`` + ``deliberate`` -> deliberate; ``marker`` with ``deliberate: false``
    -> attributed-involuntary; anything else (no covering marker, or a marker that
    predates the flag is treated as deliberate exactly as the classifier does)
    -> unattributed-signal.
    """
    if not isinstance(signal_entry, dict) or signal_entry.get("sanction") != "marker":
        return CLASS_UNATTRIBUTED_SIGNAL
    marker = signal_entry.get("stop_marker")
    if isinstance(marker, dict) and marker.get("deliberate") is False:
        return CLASS_ATTRIBUTED
    return CLASS_DELIBERATE


def cut_off_verdict(
    signal_entry: dict[str, Any] | None, *, count: int = 1
) -> tuple[str, str] | None:
    """``(cause, detail)`` a turn cut by this signal earns, or ``None`` when it was asked for.

    THE ONE SHARED DISCRIMINATOR for "what does a signal-cut turn say", used by the
    in-process writer (``exec_worker``'s SIGTERM handler, before it aborts the turn) and
    by the boot-time reader (``attention._classify_orphaned_run``) so the two cannot
    tell different stories about one signal. WHY IT EXISTS (wave B, 2026-09-30 20:00):
    the aborted turn's own end publishes BEFORE any dispose rung can note a cause, and
    an aborted end with no cause is the taxonomy's default for the USER's stop — so an
    external SIGTERM nobody staged anything for was recorded ``interrupted/user-stop``
    eleven times. The receipt's pairing is the evidence that settles it:

    * covering DELIBERATE marker -> ``None`` (a stop somebody asked for; the caller
      records positive evidence of it and nothing changes from before);
    * covering involuntary marker -> ``runtime-killed`` naming the recorded actor;
    * no covering marker -> ``runtime-shutdown`` ("terminated while this turn was
      running") with the sender sentence: the unidentified one where nothing was
      captured (byte-identical to what this arm always said), or the named one
      ("from pid N (uid U)") where ``local_operator.siginfo`` captured the
      delivery. The CAUSE TOKENS are the existing ones — a new token would ripple
      across every surface that switches on the taxonomy — so only the detail
      carries the new fact.

    LATEST SIGNAL WINS, and the rule is stated here because three callers share
    this function: the runtime's signal path, the boot-time reader and the exec
    worker all pass the LAST entry of a run's signals (``count`` is the run's
    total, for the rendered sentence). The reason it is the latest rather than
    the first: pairing is decided per signal, so a later signal carrying a
    covering deliberate marker is a stop somebody asked for, while an early
    unmarked SIGTERM is only evidence that nobody had asked YET. Deciding on the
    first would let an unexplained signal outrank the user's own later stop on
    one surface and not on another, which is the disagreement this rule removes
    (agent review round 1, MINOR 3). The earlier arrivals are not lost:
    ``signals`` keeps them, and ``count`` says how many there were.
    """
    from local_operator.incidents import (
        involuntary_kill_detail,
        render_signal_receipt_detail,
    )

    entry = signal_entry if isinstance(signal_entry, dict) else {}
    klass = stop_class_of(entry)
    if klass == CLASS_DELIBERATE:
        return None
    marker = entry.get("stop_marker")
    if klass == CLASS_ATTRIBUTED and isinstance(marker, dict):
        return (
            "runtime-killed",
            involuntary_kill_detail(
                mechanism=str(marker.get("mechanism") or ""),
                actor=str(marker.get("actor") or marker.get("killer_command") or ""),
                killer_pid=marker.get("killer_pid"),
            ),
        )
    return (
        "runtime-shutdown",
        render_signal_receipt_detail(
            signal_name=str(entry.get("name") or ""),
            at=entry.get("at"),
            count=int(count or 1),
            spawn_chain=entry.get("spawn_chain"),
            # The receipt's own captured sender, passed through verbatim: the
            # renderer decides whether it can name a party (see
            # ``incidents.render_signal_receipt_detail``), and a receipt written
            # before the capture existed simply has the unavailable shape.
            sender=entry.get("sender"),
        ),
    )
