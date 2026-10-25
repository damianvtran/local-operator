"""S3a of the mesh rolling updates: the rollout record and the serial pass.

WHAT THIS FILE OWNS (RU §8.2 item 3, first half; the trigger note §4/§6):

- **The record** — ``<config>/network/rollouts/ro_<id>.json``, one per
  (origin, network), in RU §6.1's exact shape, written under the approvals
  store's discipline (atomic replace, a per-record flock, reader-never-
  creates) and superseding an older ``active`` record for the same network.
- **The driver thread** — ``mesh-rollout``, registered as a start hook (with
  a matching stop hook that reaps it at ``RelayServer.stop()`` — round 5,
  Q-R5-1) so a suite that builds relays without starting them pays for no
  thread. It scans
  the rollouts directory every :data:`SCAN_INTERVAL_S`, and per member runs
  the pass the note freezes: one ask at a time, skip-then-retry, a ``busy``
  member parked on the catch-up cadence (never waited on, never forced), a
  failed member backed off, an unreachable member caught on its next contact,
  a codeless refusal retried on the cadence (the wire's "not now", never a
  claim of why — round 2, R2-2), and a CODED ``no_grant`` member parked for
  the rollout with the remedy recorded.
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

THE TRIGGER (S3b) — :func:`open_after_update`, called by ``lop update``'s two
successful-install exits and by the TUI's ``/update``. It reads the store and
writes files, never the network: the policy (``network.rollout_on_update``,
default :data:`DEFAULT_ROLLOUT_ON_UPDATE`), one record per network this device
belongs to with at least one other active member, and the durable
``update_rollout_started`` row. The relay's driver is still the only thing
that calls a peer. A device with no members — or a move that is not strictly
newer, or a suppressed run — opens no record: the dormancy the S3a cell
observed stays true wherever there is nothing to roll.

WHAT IS DELIBERATELY NOT HERE: the summary PRINT (the summary TEXT lives
here, as :func:`summary`, and the front ends print what
:func:`open_after_update` returns); the manual verbs' ``--all``/``--status``
and the peers-row rollout segment (the visibility slice's). And, per the
approved split and the note's §9.3: the link-open nudge is deferred — the
cadence is the contact detector. The coded ``not_authorised`` frame the r2
errata names stays a follow-up: a codeless refusal is recorded non-claiming
and retried (:func:`classify_answer`), never guessed as the missing grant.

WHY THE DRIVER IS A THREAD OF ITS OWN (note §4): an apply is a slow call
bounded at 900 s plus its reply margin; driving it on the definitions
syncer's tick would stall definitions sync for every member meanwhile. The
CLI never drives: a foreground driver dies with its terminal and two drivers
would need a lease.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

from local_operator.network.types import MeshRefusal

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: The record's own directory, under the network root (RU §6.1).
ROLLOUTS_DIRNAME = "rollouts"

#: How often the driver scans for records (seconds). A directory listing — no
#: wake-up op, no new local op — and the bound on how stale every catch-up
#: path can be: ``--resume`` clears floors and the next scan picks them up.
SCAN_INTERVAL_S = 15.0

#: Bounded join for ``stop_driver`` (round 5, Q-R5-1). The page loop's join in
#: relay.py carries the same shape: a tick may be mid-dial when the stop lands,
#: and ``stop()`` must not inherit a peer's latency — the thread is a daemon, so
#: this bound is about promptness, not about leaking a thread nobody can reach.
DRIVER_STOP_JOIN_S = 5.0

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
#: is NOT terminal: the window is its bound. ``refused`` SPLITS (round 2,
#: R2-2): a rule refusal is terminal, a codeless authoriser refusal is retried
#: — :func:`member_terminal` is the one place that decides.
DONE_STATES: frozenset[str] = frozenset({"done", "already_on_target", "ahead_of_target"})
TERMINAL_STATES: frozenset[str] = DONE_STATES | {"refused", "skipped"}

#: The codes that make a ``refused`` member TERMINAL: a member-side RULE said
#: this target cannot be installed anywhere (RU §3's rule refusals).
RULE_REFUSAL_CODES: tuple[str, ...] = ("target_not_published", "editable_install")

#: The audit kinds this half emits (registered in ``audit.py`` with the rest).
ROLLOUT_STARTED = "update_rollout_started"
MEMBER_STATE = "update_member_state"
ROLLOUT_DONE = "update_rollout_done"
ROLLOUT_SUPPRESSED = "update_rollout_suppressed"

#: The trigger's policy default (``network.rollout_on_update``): roll after an
#: update moves to a strictly newer version. ``off`` is the other value; the
#: registry's enum carries both (settings_io's ``mesh_updates`` section), and
#: ``_consumer_defaults()`` in tests/unit/test_settings_io.py pins this
#: constant against the registry default.
DEFAULT_ROLLOUT_ON_UPDATE = "auto"
ROLLOUT_POLICY_VALUES: frozenset[str] = frozenset({"auto", "off"})

#: The closed authority token the trigger writes (note §6): the primary
#: decided under the operator's standing instruction. The manual verbs' token
#: is ``operator_command`` (a future path); a record opened without one carries
#: no authority object, and the driver reads that as "not policy-driven".
AUTHORITY_STANDING_INSTRUCTION = "standing_instruction"


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
    authority: Mapping[str, str] | None = None,
    root: Path | None = None,
    window_s: float = WINDOW_S,
    now: float | None = None,
) -> dict[str, Any]:
    """Open one rollout record, superseding any older ``active`` one for the network.

    :func:`open_after_update` (the S3b trigger) is the production caller; the
    tests call this directly. ``members`` is the snapshot, in network order,
    each ``{"device": …, "name": …}`` — plus the member's INITIAL STATE when
    the snapshot already knows it without a call (round 1, M3): a pool member
    arrives ``{"state": "skipped", "code": "unsupported_kind"}``, so that arm
    is produced here rather than being a dead branch, and a reader never sees
    the pod as ``pending`` for the moment before the pass would have learned
    it. Everything else starts ``pending``.

    ``authority`` is WHO DECIDED (note §3/§6), written into the record in the
    same step that creates it: ``{"authority": "standing_instruction",
    "policy": "auto", "via": "update"|"snapshot"|"tui"}`` on the trigger
    path; ``None`` when no authority object applies (the tests, a future
    operator-command path). The driver reads it before every ask — a
    ``standing_instruction`` record halts when the policy is no longer
    ``auto`` (note §5).

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
                "state": str(member.get("state") or "pending"),
                "code": str(member.get("code") or ""),
                "detail": "",
                "version": "",
                "next_ask_at": 0.0,
                "fails": 0,
                "receipts": [],
            }
            for member in members
        ],
    }
    if authority is not None:
        record["authority"] = dict(authority)
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


def member_terminal(member: Mapping[str, Any]) -> bool:
    """Is this member done being asked, or merely refused FOR NOW? (round 2, R2-2)

    ``refused`` splits in two: a RULE refusal (:data:`RULE_REFUSAL_CODES`) is
    terminal for the rollout, while a CODELESS authoriser refusal (``code
    "refused"``) is the wire saying "not now" without saying why — a
    previous-epoch or reconcile-phase refusal self-heals — so it is retried on
    the cadence until the window closes (note §3's ``not_authorised`` row,
    REALIGNED by this slice's errata). Everything else terminal is terminal.
    """
    state = str(member.get("state") or "")
    if state not in TERMINAL_STATES:
        return False
    if state == "refused":
        return str(member.get("code") or "") in RULE_REFUSAL_CODES
    return True


#: The ``skipped`` codes no later re-run can move (round 1, R1-2): the
#: member's kind cannot run updates at all, or its build predates the frames.
#: ``no_grant`` is deliberately NOT one — a later grant makes that member
#: rollable, so closing with the nudge stays honest for it.
UNROLLABLE_SKIP_CODES: frozenset[str] = frozenset({"unsupported_kind", "predates_rolling_updates"})


def _rollable_remainder(record: Mapping[str, Any]) -> bool:
    """Could a later ``lop network update`` still move a remaining member? (R1-2)

    The closing nudge ("re-run ``lop network update`` to roll what remains")
    must not fire for a remainder that only holds members no re-run can move —
    a ``skipped (unsupported_kind)`` pool member, or a build that predates the
    frames — or the sentence promises an action that cannot work. The skip
    itself stays NAMED in the summary's sets; only the nudge is withheld.

    A ``refused`` member SPLITS the same way ``member_terminal`` splits it
    (round 2, R2-1): a RULE refusal (:data:`RULE_REFUSAL_CODES` —
    ``target_not_published`` / ``editable_install``, origin-side facts
    evaluated at ask time) is terminal for the rollout and a re-run from the
    same origin cannot move it, so it is exempt here too; a CODELESS refusal
    (code ``refused``) is the wire's "not now", self-heals, and stays
    rollable — the split R2-2 drew for termination is the same split the nudge
    needs.
    """
    for member in record.get("members", []):
        state = str(member.get("state") or "")
        if state in DONE_STATES:
            continue
        if state == "skipped" and str(member.get("code") or "") in UNROLLABLE_SKIP_CODES:
            continue
        if state == "refused" and str(member.get("code") or "") in RULE_REFUSAL_CODES:
            continue
        return True
    return False


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
    state = str(record.get("state") or "")
    # The count reads "(s)" (the repo's convention), and the OPEN line says
    # what the roll is about to DO — it is an ask, not a status adjective
    # (round 1, D1/D2: "is active to 1 peers" was neither plural nor speech a
    # person uses). The closing states keep a plain reader-only fallback.
    lead = (
        f"will ask {count} peer(s) to move to this build"
        if state == "active"
        else f"is {state}: {count} peer(s)"
    )
    line = f"rollout {record.get('id')} {lead} ({parts})"
    if state == "done" and _rollable_remainder(record):
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
        # THE CODED GATE (round 2/3): only an answer that NAMES the grant's
        # absence parks the member for the rollout. Today that code arrives
        # through ``update_peer``'s coded ``not_authorised`` branch, whose own
        # docstring calls it the FUTURE coded authoriser frame — the
        # security-reviewed change this PR's follow-ups name as the S3b
        # trigger. Parked, never re-asked on the cadence; RU §2's remedy is in
        # the message the CLI already prints, and the member's durable
        # `authorisation_refused` row is its own record. (ONE arm: a second
        # identical ``if`` here was dead code — round 3, R3-2.)
        return moved("skipped", new_code="no_grant", cause="capability_denied", outcome="refused")
    if code == "refused":
        # A CODELESS AUTHORISER REFUSAL — THE WIRE'S "NOT NOW", NEVER A CLAIM
        # OF WHY (round 2, R2-2). Every authoriser refusal (missing grant, a
        # previous epoch, a reconcile-phase reconciliation) crosses codeless by
        # design (``wire.refusal_frame``) and some of them self-heal, so
        # claiming `skipped (no_grant)` + `capability_denied` here was a false
        # statement about the member — and S1's own docstring says it "never
        # guesses no_grant". The honest record: state ``refused`` with code
        # ``refused``, NO cause claimed, and a cadence floor — retried like any
        # other not-now answer, with the rollout-scoped row dedupe (the audit
        # comparison writes one row per change, so ≥3 ticks of re-asks stay
        # one row). Only a future CODED ``not_authorised`` frame parks
        # ``skipped (no_grant)`` (the S3b/follow-up item this PR names).
        return moved("refused", new_code="refused", floor=CADENCE_S, outcome="refused")
    if code == "predates_rolling_updates":
        return moved("skipped", new_code="predates_rolling_updates", cause="capability_denied")
    if code in RULE_REFUSAL_CODES:
        # A member-side terminal refusal (the ack carries `state: refused`) or —
        # in the rare case the origin's own build cannot be offered — the same
        # class of fact: this target cannot be installed anywhere, so the
        # member is terminal for this rollout (:func:`member_terminal`) rather
        # than retried on a timer.
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
        if state == "done" and _rollable_remainder(record):
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
    the next member boundary.

    THE SWITCH IS RE-READ BEFORE EVERY ASK (note §5): a policy-driven roll
    (``standing_instruction``) halts at the next member boundary when
    ``network.rollout_on_update`` is no longer ``auto`` — no other command
    needed. The record stays ``active`` (the manual verbs still work under
    either policy, and a later ``auto`` resumes it); an apply already in
    flight inside ``update_peer`` finishes and is recorded — an install
    cannot be recalled.
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
        if member_terminal(member):
            continue
        if float(member.get("next_ask_at") or 0.0) > moment:
            continue
        # THE SWITCH, RE-READ BEFORE EVERY ASK (note §5): only a
        # policy-driven record halts — a manual/operator_command one (or one
        # with no authority object) is not this switch's to stop.
        authority = current.get("authority") or {}
        if (
            str(authority.get("authority") or "") == AUTHORITY_STANDING_INSTRUCTION
            and rollout_policy(root) != "auto"
        ):
            return
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
        and all(member_terminal(m) for m in final.get("members", []))
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
        #: The stop/reap signal (round 5, Q-R5-1). NOT ``_stop``: ``Thread``
        #: owns that attribute internally (``_wait_for_tstate_lock`` CALLS it),
        #: so shadowing it broke ``join`` with "'Event' object is not callable"
        #: — found by this fix's own cells. The loop WAITS on this event rather
        #: than calling ``time.sleep``, which fixes two things at once: a stop
        #: lands within one boundary instead of never, and a process-global
        #: ``time.sleep`` patch in a co-resident test (the WAL cell in
        #: ``test_attention_lock_contention`` patches ``time.sleep``
        #: module-wide) cannot capture this driver's cadence at all. The old
        #: ``time.sleep(15.0)`` loop was captured that way and took
        #: ``test (3.12, 0)`` red twice on the S3a head.
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Ask the loop to end; the next wake returns immediately."""
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.wait(SCAN_INTERVAL_S):
            tick(self.server)


def ensure_driver(server: "RelayServer") -> RolloutDriver | None:
    """Start this relay's rollout driver if it is not already running.

    Returns ``None`` when the relay is already stopping or stopped (round 6,
    R6-1): the start hooks are fire-and-forget daemon threads (relay.py's
    ``_run_start_hooks``), so a hook can legally land AFTER ``stop()`` ran the
    slice stop hooks — and a driver started then is the one thread no reaper
    will ever visit. The guard is the relay's own ``_stop`` event, read under
    the registry lock so the two seams cannot interleave.
    """
    with _DRIVER_LOCK:
        stop = getattr(server, "_stop", None)
        if stop is not None and stop.is_set():
            return None
        existing = _DRIVERS.get(id(server))
        if existing is not None and existing.is_alive():
            return existing
        driver = RolloutDriver(server)
        _DRIVERS[id(server)] = driver
        driver.start()
        return driver


def stop_driver(server: "RelayServer", *, timeout: float = DRIVER_STOP_JOIN_S) -> None:
    """Stop and reap this relay's driver, if it has one (round 5, Q-R5-1).

    Wired into ``RelayServer.stop()`` via the slice's ``on_stop`` hook, so the
    driver does not outlive the relay: ``stop()``'s post-condition is that the
    device is quiet, and an immortal scan thread broke that — the same class of
    leak the page-loop join closes there. The join is BOUNDED for the page
    loop's reason: a tick may be mid-dial, and a stop that waited on a peer
    would make every caller's stop cost depend on another machine. The thread is
    a daemon, so the bound is a promptness claim, not an unreachable leak.

    The registry entry is what makes a survivor findable, so it is removed only
    once the thread is known stopped: ``Thread.join(timeout)`` returns nothing
    either way, and popping before the join left a still-running driver with no
    handle any caller could reach. A driver still alive at the deadline is
    reported in a warning and its entry is KEPT, so a subsequent
    ``ensure_driver`` reuses the survivor rather than starting a second scan. A
    thread that ignores its stop cannot be killed from here; this makes the
    condition visible instead of impossible.
    """
    with _DRIVER_LOCK:
        driver = _DRIVERS.get(id(server))
    if driver is None:
        return
    driver.stop()
    driver.join(timeout=timeout)
    if driver.is_alive():
        logger.warning(
            "mesh rollouts: driver thread did not stop within the %.1fs timeout "
            "and is still running — it was not killed; its registry entry is "
            "kept so ensure_driver reuses it rather than starting a second one",
            timeout,
        )
        return
    with _DRIVER_LOCK:
        # Identity-checked: a concurrent ensure_driver may have replaced a driver
        # that looked dead, and this must not unregister its replacement.
        if _DRIVERS.get(id(server)) is driver:
            del _DRIVERS[id(server)]


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
            if member_terminal(member):
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


# ---------------------------------------------------------------------------
# The trigger (S3b): a completed update opens the records
# ---------------------------------------------------------------------------


def rollout_policy(root: Path | None = None) -> str:
    """``network.rollout_on_update``, re-read fresh; default ``auto``.

    THE ONE READER, because two consumers must agree: the HOOK (a fresh update
    process, which reads it once at its own start — note Q5) and the DRIVER
    (which re-reads it before every ask, so ``off`` halts a policy-driven roll
    at the next member boundary — note §5; the reason the setting's registry
    section is LIVE). Through ``store.read_config``, the one reader this
    package uses, so the key a front end reads and the key the relay reads
    cannot land in different places.

    AN UNRECOGNISED VALUE READS AS THE DEFAULT — the registry refuses anything
    else on write, and a hand-edited typo must not read as a silent ``off``.
    But a hand-edit of the WORD ``off`` is not a typo, and YAML parses the
    unquoted spelling as a boolean: the registry's own writer quotes it (so a
    `/settings` write round-trips as the string), while ``rollout_on_update:
    off`` typed into ``config.yml`` arrives here as ``False``. A boolean reads
    as the obvious word — off for ``False``, the default for ``True`` — because
    falling back to ``auto`` there would be the unsafe direction for the one
    key whose job is stopping a roll.
    """
    from local_operator.network import store

    raw = store.read_config(("network", "rollout_on_update"), DEFAULT_ROLLOUT_ON_UPDATE, root=root)
    if isinstance(raw, bool):
        return DEFAULT_ROLLOUT_ON_UPDATE if raw else "off"
    value = str(raw or "")
    return value if value in ROLLOUT_POLICY_VALUES else DEFAULT_ROLLOUT_ON_UPDATE


@dataclass(frozen=True)
class RolloutOpen:
    """What :func:`open_after_update` did: the lines to show, and any failure.

    ``lines`` carries one :func:`summary` per opened record — the update's own
    summary is the operator's first sight of the roll (RU §7.3), so the TEXT
    is owned here and each front end renders it on its own surface (the CLI
    prints it; ``/update`` shows it as a notice). ``failure`` is the one
    sentence a front end shows when the hook could not queue anything — with
    the next step its siblings carry ("updated, but the mesh roll could not be
    queued: … — run `lop network update` to roll by hand"; round 1, D3) —
    because the install is already in and nothing here may change an exit code.
    """

    lines: tuple[str, ...] = ()
    failure: str = ""


def _member_snapshot(record: Any) -> list[dict[str, str]]:
    """One network's roll set from the store record, in network order.

    Every other ACTIVE member, self out; a pool member rides the mapping
    already ``skipped``/``unsupported_kind`` — note §3's exclusion "before any
    call", recorded so the summary can say why the pod was not rolled (RU
    §7.3's own example). Names may be empty; the summary falls back to the
    device id. A member's grant and its feature string are deliberately NOT
    filtered here: the requester cannot read the receiver's row, so an absent
    grant is learned from the member's own answer, never guessed (the §3
    scope sentence; ``definitions.unholdable_capability``'s docstring is the
    same rule, and pre-filtering with the local row would skip every member
    whose grant lives only on the other end).
    """
    snapshot: list[dict[str, str]] = []
    for member in record.active_members():
        if str(member.device_id) == str(record.self_device_id):
            continue
        row = {"device": str(member.device_id), "name": str(member.name or "")}
        if str(member.kind) == "pool":
            row["state"] = "skipped"
            row["code"] = "unsupported_kind"
        snapshot.append(row)
    return snapshot


def open_after_update(
    before: str,
    after: str,
    *,
    via: str,
    no_roll: bool = False,
    root: Path | None = None,
    now: float | None = None,
) -> RolloutOpen:
    """The update front ends' hook: open the mesh's rollout records. NEVER raises.

    Called between the services tail and the machine report line on ``lop
    update``'s two successful-install exits (PyPI ``via="update"``, snapshot
    ``via="snapshot"``) and by the TUI's ``/update`` (``via="tui"``). It reads
    the store and writes files — no network: the relay is the only driver
    (note §3), and the services tail may have just rolled it, which is why
    this is a file write rather than a call.

    FIRES ONLY WHEN ALL HOLD (note §3):
    (1) the install moved to a STRICTLY newer version — ``update.parse_version``
        on both sides, the comparator ``readiness.compare_builds`` uses. A
        snapshot that keeps the last tag's version has nothing for members to
        do, and an unparsable side proves no move, so both fire nothing;
    (2) the policy is ``auto`` and ``--no-roll`` was not given;
    (3) this device has at least one other active member, else no record and
        no row.
    One record per active network with members, in the store's stable order
    (the walk of ``DefinitionsSyncer._targets``), the target the version just
    installed — members move to that version exactly, never "latest" (RU §3).
    ``source_ref`` stays empty on this path: it is provenance the member never
    fetches (RU §2 rule 4), the wire target is built live from the asking
    relay's own stamp (``meshupdate.update_peer``), and this process's marker
    names the tree it is superseding.

    A suppressed run (``off`` or ``--no-roll``, with ≥1 member) writes ONE
    ``update_rollout_suppressed`` row and opens nothing (note §5/§6). A failure
    of the hook itself must not touch the exit code, so every exception is
    caught and reported as the one sentence (note §3).
    """
    # Hoisted above the ``try`` (R1-1, round 1): the ``except`` returns the
    # lines already opened, and a failure BEFORE the loop must find it bound.
    lines: list[str] = []
    try:
        # Function-local, not module-level: this module is on the relay's
        # import path and ``update`` pulls httpx and the installer machinery.
        from local_operator.update import parse_version

        parsed_after = parse_version(str(after))
        parsed_before = parse_version(str(before))
        if parsed_after is None or parsed_before is None or parsed_after <= parsed_before:
            return RolloutOpen()

        from local_operator.network import store

        networks: list[tuple[Any, list[dict[str, str]]]] = []
        for record in store.list_networks(root):
            if str(record.trust) != "active" or not str(record.self_device_id):
                continue
            snapshot = _member_snapshot(record)
            if snapshot:
                networks.append((record, snapshot))
        if not networks:
            return RolloutOpen()

        policy = rollout_policy(root)
        if no_roll or policy != "auto":
            _audit_rollout_suppressed(via=via, policy=policy, no_roll=no_roll, root=root)
            return RolloutOpen()

        moment = time.time() if now is None else now
        for record, snapshot in networks:
            opened = open_record(
                network=str(record.network_id),
                origin=str(record.self_device_id),
                target={"version": str(after), "source_ref": ""},
                members=snapshot,
                authority={
                    "authority": AUTHORITY_STANDING_INSTRUCTION,
                    "policy": policy,
                    "via": str(via),
                },
                root=root,
                now=moment,
            )
            _audit_rollout_started(opened, before=str(before), root=root)
            lines.append(summary(opened))
        return RolloutOpen(lines=tuple(lines))
    except Exception as exc:  # noqa: BLE001 — the install already succeeded (note §3)
        # R1-1 (round 1): lines opened before the failure are RETURNED, not
        # discarded — those networks' records are on disk and will be driven,
        # so both front ends render lines-then-sentence and the operator is
        # never told "could not be queued" for a network where it was.
        return RolloutOpen(
            lines=tuple(lines),
            failure=(
                f"updated, but the mesh roll could not be queued: {exc} — "
                "run `lop network update` to roll by hand"
            ),
        )


def _audit_rollout_started(
    record: Mapping[str, Any], *, before: str, root: Path | None = None
) -> None:
    """The record's OPENING row — durable, best-effort (note §6).

    The authority decision, written in the same step that opens the record:
    ``authority`` is a closed token, ``via`` names the front end, ``target``
    and ``from`` are the move, ``members`` is the snapshot's size. Written
    through its own short-lived writer rather than a relay's: the relay may be
    mid-restart at this moment (the services tail just rolled it), which is the
    same reason the record is a file.
    """
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        authority = record.get("authority") or {}
        event = AuditEvent(
            event=ROLLOUT_STARTED,
            actor="self",
            subject=str(record.get("id") or ""),
            outcome="ok",
            network_id=str(record.get("network") or ""),
            detail={
                "rollout": str(record.get("id") or ""),
                "authority": str(authority.get("authority") or ""),
                "policy": str(authority.get("policy") or ""),
                "via": str(authority.get("via") or ""),
                "target": str((record.get("target") or {}).get("version") or ""),
                "from": str(before),
                "members": str(len(record.get("members", []))),
            },
        )
        log = AuditLog.from_config(root)
        log.record(event)
        log.close()
    except Exception:  # noqa: BLE001 — best-effort, same as every other row
        pass


def _audit_rollout_suppressed(
    *, via: str, policy: str, no_roll: bool, root: Path | None = None
) -> None:
    """The suppressed row (note §5/§6): an update moved, a switch stopped the roll.

    Batched, not durable — it records a decision NOT to act, remade at the next
    update. ``flag`` names the per-run override when that is what stopped it;
    ``policy`` is what the key said at the time.
    """
    try:
        from local_operator.network.audit import AuditEvent, AuditLog

        event = AuditEvent(
            event=ROLLOUT_SUPPRESSED,
            actor="self",
            outcome="ok",
            cause="policy",
            detail={
                "via": str(via),
                "policy": str(policy),
                "flag": "--no-roll" if no_roll else "",
            },
        )
        log = AuditLog.from_config(root)
        log.record(event)
        log.close()
    except Exception:  # noqa: BLE001 — best-effort, same as every other row
        pass


def install(server: "RelayServer") -> None:
    """Register the driver's start and stop hooks. This slice serves no ops.

    ``register_ops`` with an empty op table is exactly the shape wanted: no
    peer or local op, no slow slot, no capability table change — one start
    hook, run once when the relay starts, on its own daemon thread (relay.py's
    ``_run_start_hooks``), and its matching stop hook, run inline by
    ``RelayServer.stop()`` (round 5, Q-R5-1: a driver that outlived the relay
    kept ticking a root the caller believed quiet, and its ``time.sleep``
    boundaries were visible to any process-global patch of ``time.sleep``).
    The driver is what the trigger wakes with work (:func:`open_after_update`
    writes the record; this thread asks the peers), and it still scans empty
    directories quietly on a device with no records.
    """
    server.register_ops(
        {},
        on_start={"mesh-rollout": lambda: ensure_driver(server)},
        on_stop={"mesh-rollout": lambda: stop_driver(server)},
    )
