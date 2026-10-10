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
* the SENDER's pid or uid: NO, on macOS. The stdlib exposes no siginfo
  (``signal.sigwaitinfo`` is absent) and asyncio's ``add_signal_handler``
  discards it, so the receipt says ``sender: unavailable`` rather than guessing.
  The only path that NAMES a caller is a covering stop marker, and that comes from
  the acting party's own attestation, not from this module observing anything.
* SIGKILL, a crash or a power loss: NOTHING. The target is not executing. Those
  stay on the acting party's marker (SIGKILL rung) or on no evidence at all.

STILL NO SENDER, BUT DIRECTION GAINS ONE HONEST FACT. The spawn chain this
process was born into (``LOP_SPAWN_CHAIN``, recorded by ``macos_disclaim``)
carries each member's liveness as recorded at spawn (``alive_at_spawn``), and
this module adds a second, arrival reading (``alive_now``) when a signal lands.
When the app at the root of that chain was RECORDED ALIVE and is GONE at
arrival, the renderer says exactly that — the app was running when the runtime
started and was no longer running when the signal arrived (2026-10-09
incident) — and says nothing when the readings do not conspire, so the clause
cannot read as the cause of a signal or as a named sender.

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
from typing import Any

from local_operator.macos_disclaim import snapshot_spawn_chain
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SIGNAL_DRAIN_S

logger = logging.getLogger(__name__)

#: Schema version of the receipt, bumped only on an incompatible change.
SCHEMA_VERSION = 1

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
SENDER_UNAVAILABLE: dict[str, str] = {
    "state": "unavailable",
    "reason": "no-siginfo",
    "could_be": "same-uid-or-root",
}

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
    """
    entry: dict[str, Any] = {
        "name": name,
        "number": number,
        "at": at,
        "in_flight": in_flight,
        "action": action,
        "sender": dict(SENDER_UNAVAILABLE),
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


def describe(receipt: dict[str, Any] | None) -> str:
    """One line naming the LATEST signal and its pairing, for a log line. Never raises."""
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
        return f"{last.get('name')} at {when}, sanction={sanction}, sender=unavailable{extra}"
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
) -> dict[str, Any]:
    """Build the signal entry for a signal arriving NOW, pairing it with the marker.

    The marker is read at this instant (see the module docstring on why pairing is
    decided at write time). May raise only on programmer error; callers on a signal
    path wrap it (:func:`record`, ``exec_worker``).
    """
    at = time.time()
    marker = registry.read_stop_marker(directory) if directory is not None else None
    covered = marker_covers_signal(
        marker, session_id=session_id, pid=pid, started_at=started_at, at=at
    )
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
) -> dict[str, Any] | None:
    """Write the receipt for a signal that has just arrived. NEVER RAISES.

    Returns the receipt it built (written, or logged when the conversation
    directory is not there yet), or ``None`` when even building it failed. The
    directory is NOT created (a receipt must not conjure a session directory);
    a runtime whose directory is not materialised yet logs the same facts at
    WARNING and says "receipt logged only", so the evidence is at worst in the
    runtime log rather than nowhere.
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
        )
        run_key = {"kind": kind, "session_id": session_id, "pid": pid, "started_at": started_at}
        present = directory is not None and directory.is_dir()
        existing = registry.read_signal_receipt(directory) if present and directory else None
        receipt = merge_into(existing, run_key, entry)
        if present and directory:
            registry.write_signal_receipt(directory, receipt)
        else:
            logger.warning(
                "session runtime: signal receipt logged only (no conversation directory): %s",
                describe(receipt),
            )
        return receipt
    except Exception:  # noqa: BLE001 — an instrument must never delay or fail a signal path
        # LOUD, not debug: a receipt that cannot be written is the evidence gap this
        # module exists to close, and a silent loss here is indistinguishable from a
        # runtime that never got the signal. The same facts the file would have held.
        logger.warning(
            "session runtime: signal receipt could not be written "
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
      running") with the unidentified-sender sentence. The CAUSE TOKENS are the
      existing ones — a new token would ripple across every surface that switches
      on the taxonomy — so only the detail carries the new fact.

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
        ),
    )
