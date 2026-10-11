"""S3a of the mesh rolling updates: the rollout record and the serial pass.

WHAT THIS FILE OWNS (RU §8.2 item 3, first half; the trigger note §4/§6):

- **The record** — ``<config>/network/rollouts/ro_<id>.json``, one per
  (origin, network), in RU §6.1's exact shape, written under the approvals
  store's discipline (atomic replace, a per-record flock, reader-never-
  creates) and superseding an older ``active`` record for the same network.
- **The driver thread** — ``mesh-rollout``, registered as a start hook so a
  suite that builds relays without starting them pays for no thread. It scans
  the rollouts directory every :data:`SCAN_INTERVAL_S`, and per member runs
  the pass the note freezes: one ask at a time, skip-then-retry, a ``busy``
  member parked on the catch-up cadence (never waited on, never forced), a
  failed member backed off, an unreachable member caught on its next contact,
  a ``no_grant`` member parked for the rollout with the remedy recorded.
  Every state re-enters from the record; the record is the only state.
- **The origin-side audit rows** — ``update_rollout_started`` (durable, written
  by S3b's hook through :func:`open_record`'s callers), ``update_member_state``
  on a member's semantic change (never per poll), ``update_rollout_done`` when
  the pass ends, the window lapses, or ``--cancel`` closes it.
- **The mutations the CLI verbs use** — :func:`resume_record` (clear the
  ``next_ask_at`` floors; the driver's scan is the wake, because the record is
  the only channel between processes and a second signal channel would be a
  second source of truth) and :func:`cancel_record` (state ``abandoned`` plus
  the closing row).

DORMANCY, BY CONSTRUCTION — and this is the S3a contract, not an accident:
**no production path opens a record in this slice.** The trigger
(``open_after_update``, called by ``lop update`` / ``/update``) is S3b's; the
only callers of :func:`open_record` in this tree are its tests. A relay with
the driver registered and an empty rollouts directory scans, finds nothing and
writes nothing — no record, no audit row, no ask — so S3a ships inert in
production and becomes reachable exactly when S3b's hook lands. The dormancy
cell in ``tests/unit/network/test_rollouts.py`` observes that directly.

WHAT IS DELIBERATELY NOT HERE (S3b's, named so the halves stay honest): the
``open_after_update`` hook and its front-end call sites; the
``network.rollout_on_update`` setting and ``lop update --no-roll``;
``update_rollout_suppressed``; the summary PRINT (the summary TEXT lives here,
as :func:`summary`). And, per the approved split and the note's §9.3: the
link-open nudge is deferred — the cadence is the contact detector.

WHY THE DRIVER IS A THREAD OF ITS OWN (note §4): an apply is a slow call
bounded at 900 s plus its reply margin; driving it on the definitions
syncer's tick would stall definitions sync for every member meanwhile. The
CLI never drives: a foreground driver dies with its terminal and two drivers
would need a lease.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

#: The record's own directory, under the network root (RU §6.1).
ROLLOUTS_DIRNAME = "rollouts"

#: How often the driver scans for records (seconds). A directory listing — no
#: wake-up op, no new local op — and the bound on how stale every catch-up
#: path can be: ``--resume`` clears floors and the next scan picks them up.
SCAN_INTERVAL_S = 15.0

#: The catch-up cadence (seconds): the re-ask floor for a member parked
#: ``busy``/``unreachable``. One number, because "next contact" is one
#: mechanism (note §3/§4; the note's "every 5 minutes").
CADENCE_S = 300.0

#: The failure backoff ladder (seconds), then its cap: 5, 10, 20, 40 min, then
#: 2 h. Per member, from its consecutive-failure count — the retry-storm guard
#: (RU §10). There is no give-up counter; the record's window is the bound.
FAIL_BACKOFF_S: tuple[float, ...] = (300.0, 600.0, 1200.0, 2400.0, 7200.0)

#: The record's own window (seconds; RU §6.1) — the one bound that ends
#: automatic retries.
WINDOW_S = 86400.0

#: After a drop whose last known state was ``applying``, the re-probe floor
#: (seconds): the member's last act of a successful pass is rolling its own
#: relay, so the ``done`` reply may never arrive — the bounded reconnect plus
#: one re-ask turns it into ``done`` (or ``already_on_target``). (Note §4, Q4.)
DROPPED_AFTER_APPLY_RETRY_S = 120.0

#: Receipt coalescing (note §6): identical consecutive answers fold into one
#: entry with a count and a last-seen time, capped at this many per member.
RECEIPT_CAP = 20
#: What a receipt's detail (and any audit ``reason``) may carry of the
#: member's sentence (chars; the audit detail value cap is 200).
DETAIL_CAP = 200

#: The member states that mean "nothing more will happen to this member
#: without a new rollout" (the §3 vocabulary map's record column). ``failed``
#: is NOT terminal: the window is its bound.
DONE_STATES: frozenset[str] = frozenset({"done", "already_on_target", "ahead_of_target"})
TERMINAL_STATES: frozenset[str] = DONE_STATES | {"refused", "skipped"}

#: The audit kinds this half emits (registered in ``audit.py`` with the rest).
ROLLOUT_STARTED = "update_rollout_started"
MEMBER_STATE = "update_member_state"
ROLLOUT_DONE = "update_rollout_done"


# ---------------------------------------------------------------------------
# Paths and record I/O (the approvals store's discipline, not its code)
# ---------------------------------------------------------------------------


def rollouts_dir(root: Path | None = None) -> Path:
    """``<config>/network/rollouts`` — the path, never created by a reader.

    A listing over a directory that is not there answers "no rollouts", which
    is the truth on every device until S3b's hook opens one; writers use
    :func:`ensure_rollouts_dir`.
    """
    from local_operator.network.identity import network_root

    return network_root(root) / ROLLOUTS_DIRNAME


def ensure_rollouts_dir(root: Path | None = None) -> Path:
    """``<config>/network/rollouts``, created 0700 — WRITERS ONLY."""
    from local_operator.network.identity import ensure_network_root

    ensure_network_root(root)
    path = rollouts_dir(root)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def record_path(record_id: str, root: Path | None = None) -> Path:
    return rollouts_dir(root) / f"{record_id}.json"


def lock_path(record_id: str, root: Path | None = None) -> Path:
    """The per-record flock target. One name per record — the whole mechanism."""
    return rollouts_dir(root) / f"{record_id}.lock"


@contextmanager
def record_lock(record_id: str, root: Path | None = None) -> Iterator[None]:
    """Hold one record's lock for a read-modify-write.

    ``fcntl`` IS IMPORTED HERE, NOT AT MODULE SCOPE — the same lesson the
    member pass learned (S2 review round 1, B1): a POSIX-only import at module
    scope makes the slice unloadable on Windows, which ``_install_slices``
    swallows silently. On a platform without it a record cannot be written
    under its lock at all, which is a refusal BY NAME rather than a racy write.
    """
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover — exercised by the blocked-import probe
        raise MeshRefusal(
            "rollout_lock_unsupported",
            "this platform has no advisory file locking (no ``fcntl`` here), so a "
            "rollout record cannot be written safely; nothing was changed",
        ) from exc
    ensure_rollouts_dir(root)
    handle = open(lock_path(record_id, root), "a+", encoding="utf-8")  # noqa: SIM115
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        handle.close()


def read_record(record_id: str, root: Path | None = None) -> dict[str, Any] | None:
    """One record, or ``None``. A READER: never creates, never locks."""
    path = record_path(record_id, root)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def active_record_paths(root: Path | None = None) -> list[Path]:
    """Every record file, sorted; the caller filters on ``state``.

    A read of a directory that does not exist is ``[]`` — reader-never-create.
    """
    try:
        return sorted(rollouts_dir(root).glob("ro_*.json"))
    except OSError:
        return []


def _write_record(record: Mapping[str, Any], root: Path | None = None) -> None:
    """Atomic replace, inside the caller's lock (or for a fresh record)."""
    path = record_path(str(record["id"]), root)
    payload = json.dumps(record, sort_keys=True, indent=2)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def mutate_record(
    record_id: str, root: Path | None, fn: Callable[[dict[str, Any]], bool]
) -> dict[str, Any] | None:
    """Lock, load, ``fn(record)`` (truthy = persist), unlock. Returns the record."""
    with record_lock(record_id, root):
        record = read_record(record_id, root)
        if record is None:
            return None
        if fn(record):
            _write_record(record, root)
        return record


# ---------------------------------------------------------------------------
# The record itself (RU §6.1's shape)
# ---------------------------------------------------------------------------


def new_record_id() -> str:
    """``ro_`` + 12 hex chars. Local, unique; not a wire identifier."""
    return f"ro_{secrets.token_hex(6)}"


def open_record(
    *,
    network: str,
    origin: str,
    target: Mapping[str, str],
    members: Sequence[Mapping[str, str]],
    root: Path | None = None,
    window_s: float = WINDOW_S,
    now: float | None = None,
) -> dict[str, Any]:
    """Open one rollout record, superseding any older ``active`` one for the network.

    S3b CALLS THIS — from ``open_after_update`` at the two update exits and
    ``/update``. In S3a the tests are the only callers (the dormancy contract;
    see the module docstring). ``members`` is the snapshot, in network order,
    each ``{"device": …, "name": …}``; every member starts ``pending``.

    ``unsupported_kind`` (the §3 exclusion of pool-kind members "before any
    call") has NO PRODUCING PATH in S3a (round 1, M3): it is S3b's snapshot
    call — an initial-state field in this mapping, or a documented
    one-mutate-after-open — and is named here so nothing builds on an arm
    that cannot fire.

    The supersede is part of the open (RU §6.3): a second update mid-pass
    re-snapshots, and the older record keeps its history, marked
    ``superseded``. Readers never see a half-open record: the new record is
    written first, then the old ones are marked, both under their own locks.
    """
    moment = time.time() if now is None else now
    record: dict[str, Any] = {
        "id": new_record_id(),
        "state": "active",
        "created_at": moment,
        "window_s": float(window_s),
        "network": str(network),
        "origin": str(origin),
        "target": {
            "version": str(target.get("version") or ""),
            "source_ref": str(target.get("source_ref") or ""),
        },
        "members": [
            {
                "device": str(member.get("device") or ""),
                "name": str(member.get("name") or ""),
                "state": "pending",
                "code": "",
                "detail": "",
                "version": "",
                "next_ask_at": 0.0,
                "fails": 0,
                "receipts": [],
            }
            for member in members
        ],
    }
    ensure_rollouts_dir(root)
    _write_record(record, root)
    for path in active_record_paths(root):
        other_id = path.stem
        if other_id == record["id"]:
            continue
        other = read_record(other_id, root)
        if (
            other is None
            or other.get("state") != "active"
            or str(other.get("network")) != str(network)
        ):
            continue
        if float(other.get("created_at") or 0.0) >= float(record.get("created_at") or 0.0):
            # NOT OURS TO RETIRE (round 1, M1): a CONCURRENT open whose record
            # is newer (or same-second) stays active. Without this guard two
            # racing opens could supersede each other — zero active records —
            # and the older racer could retire the newer record's pass.
            continue
        mutate_record(other_id, root, lambda rec: rec.update({"state": "superseded"}) or True)
    return record


def member_of(record: Mapping[str, Any], device_id: str) -> dict[str, Any] | None:
    for member in record.get("members", []):
        if str(member.get("device")) == str(device_id):
            return member
    return None


def _mark_applying(record: dict[str, Any], device: str) -> bool:
    """Set one member to the in-flight ``applying`` state before its ask.

    Not audit-rowed: it is a marker the NEXT outcome supersedes, and a row per
    ask is the per-poll shape note §6 forbids. It exists so a drop or crash
    mid-ask resumes as "an apply may have begun" rather than as ``pending``.
    """
    member = member_of(record, device)
    if member is None or str(member.get("state")) == "applying":
        return False
    member["state"] = "applying"
    member["code"] = ""
    member["next_ask_at"] = 0.0
    return True


def summary(record: Mapping[str, Any]) -> str:
    """One line naming who is queued and who is already skipped (RU §7.3).

    The text S3b prints at open; here it is also the read a `--json` consumer
    (and a test) gets without parsing the record.
    """
    groups: dict[str, list[str]] = {}
    for member in record.get("members", []):
        state = str(member.get("state") or "pending")
        code = str(member.get("code") or "")
        # RU §7.3's shape, taken literally (round 1, M4): the summary NAMES the
        # queued set and the skips known without a call — the operator's first
        # sight of a rollout must say "not cloud-node-2, it never granted this
        # device `update`" without a second command.
        if state == "pending":
            label = "queued"
        elif state == "skipped" and code:
            label = f"skipped ({code})"
        else:
            label = state
        groups.setdefault(label, []).append(str(member.get("name") or member.get("device") or "?"))
    parts = "; ".join(f"{label}: {', '.join(names)}" for label, names in sorted(groups.items()))
    count = len(record.get("members", []))
    line = f"rollout {record.get('id')} is {record.get('state')} to {count} peers ({parts})"
    if record.get("state") == "done" and any(
        str(member.get("state")) not in DONE_STATES for member in record.get("members", [])
    ):
        line += (
            "; this rollout is no longer active — re-run `lop network update` to roll what remains"
        )
    return line


# ---------------------------------------------------------------------------
# The serial pass (the note §4 loop, one step at a time)
# ---------------------------------------------------------------------------


def _backoff_s(fails: int) -> float:
    index = min(max(fails, 1), len(FAIL_BACKOFF_S)) - 1
    return FAIL_BACKOFF_S[index]


def classify_answer(answer: Mapping[str, Any], *, fails: int, now: float) -> dict[str, Any]:
    """One member answer → the record's next state (the §3 map's record column).

    Deliberately a pure function of the answer, the member's prior failure
    count and the clock: the pass's outcomes must be reviewable without a
    relay in the room, which is also what lets the cells drive every row with
    a scripted answer. (A ``prior_state`` parameter lived here until round 1
    (N1): the driver hard-wired it, so its non-matching branch was dead — the
    one fact a silent ask needs, "an apply may have begun", is already in the
    record's own ``applying`` marker.)
    """
    code = str(answer.get("code") or "")
    state = str(answer.get("state") or "")
    version = str(answer.get("version") or "")
    method = str(answer.get("method") or "")
    message = str(answer.get("message") or "")

    def moved(
        new_state: str,
        *,
        new_code: str = "",
        floor: float = 0.0,
        cause: str = "",
        outcome: str = "ok",
    ) -> dict[str, Any]:
        return {
            "state": new_state,
            "code": new_code,
            "next_ask_at": now + floor,
            "cause": cause,
            "outcome": outcome,
            "detail": message[:DETAIL_CAP],
            "version": version,
            "method": method,
            # CONSECUTIVE-failure count, so every non-failure answer resets it
            # (round 1, N2): a busy streak between two failures used to advance
            # the ladder anyway, and the ladder's own comment claims
            # "consecutive". The failure arms override this with fails + 1.
            "fails": 0,
        }

    if answer.get("ok") and state in DONE_STATES:
        # The member's own receipt sentence rides ``detail`` — the count line
        # (`this device moved from … ; N retired, M kept`) is authored prose
        # for a reader, and the record is where it belongs.
        return moved(state)
    if state == "busy" or code == "update_in_progress":
        # Skip-then-retry: parked on the cadence, a FRESH probe each time, and
        # nothing was touched on the member (its own sentence rides `detail`).
        return moved(
            "deferred", new_code="busy" if state == "busy" else code, floor=CADENCE_S, cause="busy"
        )
    if state == "failed":
        next_fails = fails + 1
        return moved(
            "failed",
            new_code=code,
            floor=_backoff_s(next_fails),
            cause="internal",
            outcome="failed",
        ) | {"fails": next_fails}
    if code == "no_grant":
        # The member's own refusal to grant THIS peer `update`: parked for the
        # rollout, never re-asked on the cadence (RU §2's remedy is in the
        # message the CLI already prints); the member's durable
        # `authorisation_refused` row is its own record.
        return moved("skipped", new_code="no_grant", cause="capability_denied", outcome="refused")
    if code == "refused":
        # THE #2179 RESOLUTION, MADE REAL (round 1, Q-1): the authoriser's
        # refusal crosses the wire CODELESS by design (which guard fired is
        # withheld — ``wire.refusal_frame``), and for THIS op the codeless
        # refusal has exactly one reading in the design: the requester does
        # not hold `update` on this member (``meshupdate``'s open question
        # names the two ways to a coded frame; until one lands, this arm is
        # the honest classification — the pre-round-1 fallback parked the
        # member on the failure ladder instead). Parked `skipped (no_grant)`
        # with the member's own sentence (and RU §2's remedy) in ``detail``.
        return moved("skipped", new_code="no_grant", cause="capability_denied", outcome="refused")
    if code == "predates_rolling_updates":
        return moved("skipped", new_code="predates_rolling_updates", cause="capability_denied")
    if code in ("target_not_published", "editable_install"):
        # A member-side terminal refusal (the ack carries `state: refused`) or —
        # in the rare case the origin's own build cannot be offered — the same
        # class of fact: this target cannot be installed anywhere, so the
        # member is terminal for this rollout rather than retried on a timer.
        return moved("refused", new_code=code, cause="policy", outcome="refused")
    if code in ("unreachable", "no_answer"):
        if code == "no_answer":
            # REQUEST SENT, THEN SILENCE: the member's relay may be mid-roll
            # (its last act of a successful pass), so the re-ask is delayed by
            # the bounded reconnect and then asks once more — the reply turns
            # it into done/already_on_target (note §4, Q4). A DIAL failure
            # (``unreachable``) proves nothing was sent, so it never claims an
            # apply began; the record's ``applying`` marker (written before
            # every ask) is what distinguishes the two readings.
            return moved(
                "deferred", new_code="no_answer", floor=DROPPED_AFTER_APPLY_RETRY_S, cause="timeout"
            )
        return moved("unreachable", new_code=code, floor=CADENCE_S, cause="peer_unreachable")
    return moved(
        "failed",
        new_code=code or "failed",
        floor=_backoff_s(fails + 1),
        cause="internal",
        outcome="failed",
    ) | {"fails": fails + 1}


def note_receipt(member: dict[str, Any], *, state: str, detail: str, now: float) -> None:
    """Coalesce identical consecutive answers; cap the list (note §6)."""
    receipts: list[dict[str, Any]] = member.setdefault("receipts", [])
    if receipts and receipts[-1].get("state") == state and receipts[-1].get("detail") == detail:
        receipts[-1]["count"] = int(receipts[-1].get("count", 1)) + 1
        receipts[-1]["at"] = now
    else:
        receipts.append({"at": now, "state": state, "detail": detail, "count": 1})
    del receipts[:-RECEIPT_CAP]


def _audit_member_state(
    server: "RelayServer | None",
    record: Mapping[str, Any],
    member: Mapping[str, Any],
    update: Mapping[str, Any],
    *,
    root: Path | None,
) -> None:
    """One origin-side row per member CHANGE, best-effort (the audit cost rule)."""
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        detail = {
            "rollout": str(record.get("id") or ""),
            "state": str(member.get("state") or ""),
            "code": str(member.get("code") or ""),
            "reason": str(member.get("detail") or "")[:DETAIL_CAP],
            "method": str(update.get("method") or ""),
            "version": str(update.get("version") or ""),
        }
        event = AuditEvent(
            event=MEMBER_STATE,
            actor="self",
            subject=str(record.get("id") or ""),
            outcome=str(update.get("outcome") or "ok"),
            network_id=str(record.get("network") or ""),
            cause=str(update.get("cause") or ""),
            detail=detail,
        )
        if server is not None and getattr(server, "audit", None) is not None:
            server.audit.record(event)
        else:
            # THE WRITER IS CLOSED, not dropped (round 1, Q-2): the log batches
            # a row and without a close the buffer dies with this object —
            # measured on --cancel's closing row, which flipped the record
            # `abandoned` while the row never landed.
            log = AuditLog.from_config(root)
            log.record(event)
            log.close()  # ``close`` is ``flush``; the row is on disk or reported
    except Exception:  # noqa: BLE001 — a record is not a gate (the mcpdefs rule)
        pass


def _close_record(
    *,
    record: dict[str, Any],
    state: str,
    outcome: str,
    server: "RelayServer | None" = None,
    root: Path | None = None,
) -> None:
    """End a record (done/abandoned) and write its closing row. Best-effort row."""
    record["state"] = state
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        done = sum(1 for m in record.get("members", []) if str(m.get("state")) in DONE_STATES)
        total = len(record.get("members", []))
        counts = f"{done} of {total} moved"
        if state == "done" and done < total:
            counts += (
                "; this rollout is no longer active — re-run `lop network update` "
                "to roll what remains"
            )
        event = AuditEvent(
            event=ROLLOUT_DONE,
            actor="self",
            subject=str(record.get("id") or ""),
            outcome=outcome,
            network_id=str(record.get("network") or ""),
            detail={"rollout": str(record.get("id") or ""), "counts": counts[:DETAIL_CAP]},
        )
        if server is not None and getattr(server, "audit", None) is not None:
            server.audit.record(event)
        else:
            # THE WRITER IS CLOSED, not dropped (round 1, Q-2): the log batches
            # a row and without a close the buffer dies with this object —
            # measured on --cancel's closing row, which flipped the record
            # `abandoned` while the row never landed.
            log = AuditLog.from_config(root)
            log.record(event)
            log.close()  # ``close`` is ``flush``; the row is on disk or reported
    except Exception:  # noqa: BLE001 — best-effort, same as the member rows
        pass


def step_record(server: "RelayServer", record_id: str, *, now: float | None = None) -> None:
    """One pass over one record: ask every member that is due, in order. NEVER raises.

    The loop is the note's §4 pseudocode: terminal members are skipped, a
    member inside its ``next_ask_at`` floor is skipped, and every ask is a
    FRESH ``net_update`` probe — an apply returns, or times out, before the
    next member is asked (serial by construction). The record is re-read
    before each member so ``--cancel`` (and a supersede) stops the pass at
    the next boundary without any other command.
    """
    from local_operator.network import meshupdate

    root = Path(server.root) if getattr(server, "root", None) is not None else None
    moment = time.time() if now is None else now
    record = read_record(record_id, root)
    if record is None or record.get("state") != "active":
        return
    if moment - float(record.get("created_at") or 0.0) > float(record.get("window_s") or WINDOW_S):
        mutate_record(
            record_id,
            root,
            lambda rec: rec.update({"state": "done"}) or True,
        )
        closed = read_record(record_id, root)
        if closed is not None:
            _close_record(record=closed, state="done", outcome="partial", server=server, root=root)
            _write_record(closed, root)
        return

    for member in list(record.get("members", [])):
        current = read_record(record_id, root)
        if current is None or current.get("state") != "active":
            return
        member = member_of(current, str(member.get("device"))) or member
        if str(member.get("state")) in TERMINAL_STATES:
            continue
        if float(member.get("next_ask_at") or 0.0) > moment:
            continue
        device = str(member.get("device") or "")
        # The PRE-ASK outcome state, kept for the audit comparison below: the
        # ``applying`` marker is an in-flight fact, not an outcome, so comparing
        # against it would write a row for every ask of a member parked on the
        # cadence — the per-poll row note §6 forbids.
        prior_state = str(member.get("state") or "")
        prior_code = str(member.get("code") or "")
        fails = int(member.get("fails") or 0)
        # THE IN-FLIGHT STATE IS WRITTEN BEFORE THE ASK (note §3: "the slow call
        # itself, installing" is ``applying``). Without it a drop after the
        # member began its install could not be told from a drop before the
        # request left — the one distinction the re-probe path is built on —
        # and a crash mid-ask used to resume from a stale ``pending`` that
        # claimed nothing had been sent.
        mutate_record(
            record_id,
            root,
            lambda rec: _mark_applying(rec, device),
        )
        try:
            answer = meshupdate.update_peer(server, device, rollout=record_id)
        except Exception as exc:  # noqa: BLE001 — a driver must never die on one member
            answer = {"ok": False, "code": "no_answer", "message": str(exc), "state": ""}
        update = classify_answer(answer, fails=fails, now=moment)

        def _apply(
            rec: dict[str, Any], _update: Mapping[str, Any] = update, _device: str = device
        ) -> bool:
            target_member = member_of(rec, _device)
            if target_member is None:
                return False
            changed = str(target_member.get("state")) != str(_update["state"]) or str(
                target_member.get("code") or ""
            ) != str(_update.get("code") or "")
            target_member["state"] = str(_update["state"])
            target_member["code"] = str(_update.get("code") or "")
            target_member["detail"] = str(_update.get("detail") or "")
            target_member["version"] = str(_update.get("version") or "")
            target_member["fails"] = int(_update.get("fails") or 0)
            target_member["next_ask_at"] = float(_update.get("next_ask_at") or 0.0)
            note_receipt(
                target_member,
                state=str(_update["state"]),
                detail=str(_update.get("detail") or ""),
                now=moment,
            )
            return changed

        assert update["state"] != "applying", "classify_answer only returns outcomes"
        mutate_record(record_id, root, _apply)
        after = read_record(record_id, root)
        if after is not None:
            updated_member = member_of(after, device)
            if updated_member is not None and (
                str(updated_member.get("state") or "") != prior_state
                or str(updated_member.get("code") or "") != prior_code
            ):
                _audit_member_state(server, after, updated_member, update, root=root)

    final = read_record(record_id, root)
    if (
        final is not None
        and final.get("state") == "active"
        and all(str(m.get("state")) in TERMINAL_STATES for m in final.get("members", []))
    ):
        done = sum(1 for m in final.get("members", []) if str(m.get("state")) in DONE_STATES)
        outcome = "ok" if done == len(final.get("members", [])) else "partial"
        mutate_record(record_id, root, lambda rec: rec.update({"state": "done"}) or True)
        closed = read_record(record_id, root)
        if closed is not None:
            _close_record(record=closed, state="done", outcome=outcome, server=server, root=root)
            _write_record(closed, root)


def tick(server: "RelayServer") -> None:
    """One scan: drive every active record's pass. NEVER raises."""
    try:
        root = Path(server.root) if getattr(server, "root", None) is not None else None
        for path in active_record_paths(root):
            try:
                step_record(server, path.stem)
            except Exception:  # noqa: BLE001 — one bad record must not stop the scan
                continue
    except Exception:  # noqa: BLE001 — a scan is housekeeping, never fatal
        pass


# ---------------------------------------------------------------------------
# The driver thread
# ---------------------------------------------------------------------------

_DRIVERS: dict[int, "RolloutDriver"] = {}
_DRIVER_LOCK = threading.Lock()


class RolloutDriver(threading.Thread):
    """The ``mesh-rollout`` thread: scan, step, sleep, repeat.

    A daemon thread, started by the relay's start hook (``on_start``), so a
    suite that builds relays without starting them pays for no thread — the
    same rule ``definitions.install`` states for its syncer.
    """

    def __init__(self, server: "RelayServer") -> None:
        super().__init__(name="mesh-rollout", daemon=True)
        self.server = server

    def run(self) -> None:  # pragma: no cover — the loop is driven by tick() in cells
        while True:
            time.sleep(SCAN_INTERVAL_S)
            tick(self.server)


def ensure_driver(server: "RelayServer") -> RolloutDriver:
    """Start this relay's rollout driver if it is not already running."""
    with _DRIVER_LOCK:
        existing = _DRIVERS.get(id(server))
        if existing is not None and existing.is_alive():
            return existing
        driver = RolloutDriver(server)
        _DRIVERS[id(server)] = driver
        driver.start()
        return driver


def resume_record(
    record_id: str, root: Path | None = None, *, now: float | None = None
) -> dict[str, Any] | None:
    """``lop network update --resume``: clear every non-terminal member's floor.

    It does not drive: the driver's scan is the wake (≤ :data:`SCAN_INTERVAL_S`),
    because the record is the only channel between the CLI's process and the
    relay's and a second signal channel would be a second source of truth.

    Only an ``active`` record can be resumed (round 1, M2): a terminal id is
    refused with a sentence instead of silently rewritten.
    """
    current = read_record(record_id, root)
    if current is None or str(current.get("state")) != "active":
        state = str((current or {}).get("state") or "missing")
        raise MeshRefusal(
            "rollout_not_active",
            f"rollout {record_id} is {state}; only an active rollout can be resumed",
        )

    def _clear(rec: dict[str, Any]) -> bool:
        cleared = False
        for member in rec.get("members", []):
            if str(member.get("state")) in TERMINAL_STATES:
                continue
            if float(member.get("next_ask_at") or 0.0) > 0.0:
                member["next_ask_at"] = 0.0
                cleared = True
            # A resume is also the operator saying "try now": the failure
            # ladder restarts too, so a member stuck at the 2 h rung gets its
            # next attempt on this scan.
            if int(member.get("fails") or 0):
                member["fails"] = 0
                cleared = True
        return cleared

    return mutate_record(record_id, root, _clear)


def cancel_record(
    record_id: str, root: Path | None = None, *, server: "RelayServer | None" = None
) -> dict[str, Any] | None:
    """``lop network update --cancel``: mark a record ``abandoned`` + closing row.

    An apply already in flight finishes — an install cannot be recalled — and
    is recorded by the driver's own step; this only ends the pass's future.
    Only an ``active`` record can be cancelled (round 1, M2): a ``done`` id
    would otherwise be rewritten to ``abandoned`` with a contradicting second
    closing row.
    """
    current = read_record(record_id, root)
    if current is None or str(current.get("state")) != "active":
        state = str((current or {}).get("state") or "missing")
        raise MeshRefusal(
            "rollout_not_active",
            f"rollout {record_id} is {state}; only an active rollout can be cancelled",
        )
    updated = mutate_record(record_id, root, lambda rec: rec.update({"state": "abandoned"}) or True)
    if updated is not None:
        _close_record(
            record=updated, state="abandoned", outcome="partial", server=server, root=root
        )
        _write_record(updated, root)
    return updated


def latest_active(  # noqa: SIM108 — the loop is clearer than a comprehension here
    *network: str, root: Path | None = None
) -> dict[str, Any] | None:
    """The most recently created ``active`` record, optionally for one network."""
    best: dict[str, Any] | None = None
    for path in active_record_paths(root):
        record = read_record(path.stem, root)
        if record is None or record.get("state") != "active":
            continue
        if network and str(record.get("network")) not in network:
            continue
        if best is None or float(record.get("created_at") or 0.0) > float(
            best.get("created_at") or 0.0
        ):
            best = record
    return best


def install(server: "RelayServer") -> None:
    """Register the driver as a start hook. This slice serves no ops of its own.

    ``register_ops`` with an empty op table is exactly the shape wanted: no
    peer or local op, no slow slot, no capability table change — one hook,
    run once when the relay starts, on its own daemon thread (relay.py's
    ``_run_start_hooks``). The driver is what S3b's trigger will wake into
    work; until then it scans empty directories.
    """
    server.register_ops({}, on_start={"mesh-rollout": lambda: ensure_driver(server)})
