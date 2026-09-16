"""Durable record of a wake fire that could not be delivered yet.

``<config_dir>/wakes/deliveries/<session_id>.json`` answers the one question
the supervisor could not answer before this file existed: *is there a fire we
attempted, failed to deliver, and still owe?*

**Why the schedule index is not that record.** :mod:`local_operator.wakes.store`
keeps a wake's ``next_due_at`` in the past while nothing has fired it, so an
overdue schedule looks like an owed fire. It is not the same statement, and the
difference is the defect this module closes: the supervisor engages what is
DUE, and anything that removes a due time from that set — the session advancing
it, a rewrite, or simply the schedule ageing past ::data:`~local_operator.wakes
.supervisor.STALE_AFTER_S` and being skipped forever — takes the owed fire with
it, silently. A failed engage left only a WARNING line in
``wake-supervisor.log`` (510 lifetime on the operator's machine, 128 in a single
day), and nothing an operator could look at, so scheduled work that did not run
was invisible from every surface but that log.

**This is the supervisor's own state, not schedule state.** The supervisor
remains the sole reader and never advances, retires or persists a schedule
(docs in :mod:`local_operator.wakes.supervisor`); the entries written here
describe the supervisor's own delivery attempts and nothing else. The session's
transcript is still the only source of truth for what a schedule *is*, and a
store that disagrees with this ledger is repaired the way the index is: the
owning session rewrites the schedule, the supervisor reconciles the record away
(see ``_reconcile_deliveries``).

**Import-light for the same reason as the store**: the supervisor is an
always-on ~40 MB process whose justification is that it does not load the
harness, so this module is stdlib-only and is pinned that way by
``tests/unit/test_import_graph.py``.

**Atomic, like the store.** Staged write plus ``os.replace`` so a reader never
sees a torn file; the temp name starts with ``.`` so a scan skips it. Every
write is best-effort from the caller's point of view: a ledger that cannot be
written must never stop a wake from being engaged, because the engage is the
thing that fires it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Subdirectory of the wakes dir holding one ``<session_id>.json`` per fire the
#: supervisor has attempted and not yet handed to a runtime. Deliberately a
#: SUBdirectory of ``wakes/``: ``store.read_index`` lists that directory and
#: skips anything not ending in ``.json``, so an index scan cannot pick a
#: ledger entry up as a schedule.
DELIVERIES_DIRNAME = "deliveries"

#: Bumped only on an incompatible change to the entry shape, exactly as
#: ``store.INDEX_SCHEMA`` is: a reader that does not understand a record must
#: treat it as absent rather than guess.
DELIVERY_SCHEMA = 1

#: Backoff before the second attempt, in seconds. Small enough that a wake
#: delayed by a transient cold-start failure still fires promptly, because the
#: operator is waiting on the wake and not on the supervisor's arithmetic.
RETRY_BASE_S = 15.0

#: Multiplier per consecutive failure.
RETRY_FACTOR = 2.0

#: Ceiling on the backoff. This is the constant that fixes the starvation the
#: old code documented as a known residual: every retry used to cost a FULL
#: engage deadline (``WAKE_DEADLINE_S``, 180 s) out of
#: ``_MAX_CONCURRENT_ENGAGES`` (2) slots on every pass, so one session that
#: could not be reached occupied half the fleet's engagement capacity
#: indefinitely and fresh wakes queued behind it. A 15-minute cadence spends
#: 180 s of one slot per 15 minutes (20%) and bounds how long a newly armed
#: wake waits behind a permanently failing one.
RETRY_CAP_S = 900.0

#: How many failed attempts before an owed fire is REPORTED as undelivered
#: rather than merely retrying.
#:
#: A REPORT, NOT A STOP. The fire keeps being retried at the capped backoff:
#: giving up permanently is the defect, not the fix. What changes at this
#: threshold is that the condition is logged once at ERROR and rendered on
#: ``lop wake status`` as a stalled fire the operator has to know about, rather
#: than as one more routine retry line in a file nothing rotates.
UNDELIVERED_AFTER_ATTEMPTS = 5

#: States a record can carry. Both are RETRIED — the difference is what the
#: operator is told, not whether the supervisor keeps trying.
#:
#: ``retrying``: a recent failure, still inside the ordinary backoff run.
#: ``undelivered``: :data:`UNDELIVERED_AFTER_ATTEMPTS` consecutive failures —
#: the fire is stalled and is named as such on ``lop wake status``.
STATE_RETRYING = "retrying"
STATE_UNDELIVERED = "undelivered"


def deliveries_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "wakes" / DELIVERIES_DIRNAME


def delivery_path(config_dir: Path, session_id: str) -> Path:
    return deliveries_dir(config_dir) / f"{session_id}.json"


def _now_ms() -> int:
    return int(time.time() * 1000)


def read_delivery(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """One session's record, or ``None``. Defensive: a torn or foreign-schema
    file is treated as absent, and the next failure rewrites it."""
    path = delivery_path(config_dir, session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning(
            "wake deliveries: unreadable record %s; treating as absent", path, exc_info=True
        )
        return None
    if not isinstance(data, dict) or data.get("schema") != DELIVERY_SCHEMA:
        logger.warning("wake deliveries: skipping record %s with unknown schema", path)
        return None
    if not isinstance(data.get("occurrence_ms"), int) or isinstance(
        data.get("occurrence_ms"), bool
    ):
        return None
    return data


def read_deliveries(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every readable record keyed by session id. A missing directory is an
    empty ledger — the common case on a machine whose wakes all fire."""
    directory = deliveries_dir(config_dir)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}
    except OSError:
        logger.warning("wake deliveries: cannot list %s", directory, exc_info=True)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if not name.endswith(".json") or name.startswith("."):
            continue  # staged temp files and stray dotfiles are never records
        session_id = name[: -len(".json")]
        record = read_delivery(config_dir, session_id)
        if record is None:
            continue
        record["session_id"] = session_id
        out[session_id] = record
    return out


def write_delivery(
    config_dir: Path, session_id: str, record: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Write (replace) one session's record, atomically.

    Returns the record as stored — including the schema and session id, so a
    caller that logs or returns it is describing the file rather than a
    pre-image of it — or ``None`` when the write failed. Best-effort BY
    CONTRACT: the ledger is observability and retry bookkeeping, and a partly
    failed filesystem must not turn a wake that could have fired into one that
    did not.
    """
    path = delivery_path(config_dir, session_id)
    body = dict(record)
    body["schema"] = DELIVERY_SCHEMA
    body["session_id"] = session_id
    directory = path.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{session_id}.", suffix=".tmp")
    except OSError:
        logger.warning("wake deliveries: cannot write a record for %s", session_id, exc_info=True)
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        logger.warning("wake deliveries: cannot write a record for %s", session_id, exc_info=True)
        return None
    return body


def remove_delivery(config_dir: Path, session_id: str) -> bool:
    """Delete one record. Idempotent: absent is success."""
    try:
        delivery_path(config_dir, session_id).unlink()
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning(
            "wake deliveries: cannot remove the record for %s", session_id, exc_info=True
        )
        return False
    return True


def backoff_s(attempts: int) -> float:
    """Seconds to wait before the next attempt, given the failures so far.

    Exponential from :data:`RETRY_BASE_S`, capped at :data:`RETRY_CAP_S`. The
    cap is the load-bearing half (see that constant): without it the wait grows
    past the point where the operator would still call the wake late rather
    than lost.
    """
    if attempts < 1:
        return RETRY_BASE_S
    return min(RETRY_CAP_S, RETRY_BASE_S * (RETRY_FACTOR ** (attempts - 1)))


def next_attempt_ms(attempts: int, *, now_ms: int) -> int:
    return now_ms + int(backoff_s(attempts) * 1000)


def note_failure(
    config_dir: Path,
    session_id: str,
    occurrence_ms: int,
    *,
    error: str,
    now_ms: int | None = None,
) -> dict[str, Any] | None:
    """Record one failed attempt on an owed fire, and schedule the next.

    Returns the record as STORED, or ``None`` when the ledger would not take it
    (review round 1, MINOR 2 — the caller must not claim a durability that did
    not happen: the old shape returned the in-memory record either way, so a
    full disk produced the "STILL OWED … 'lop wake status' reports it" line
    while that surface had nothing to report).

    ``attempts`` counts consecutive failures for THIS occurrence; a record for
    a different occurrence is replaced rather than incremented, because the one
    the supervisor engages is the earliest due one and an occurrence it moves
    to is a different fire.
    """
    moment = now_ms if now_ms is not None else _now_ms()
    existing = read_delivery(config_dir, session_id)
    first = None
    attempts = 0
    if existing is not None and existing.get("occurrence_ms") == occurrence_ms:
        prior_first = existing.get("first_attempt_ms")
        first = (
            prior_first
            if isinstance(prior_first, int) and not isinstance(prior_first, bool)
            else None
        )
        prior = existing.get("attempts")
        attempts = prior if isinstance(prior, int) and not isinstance(prior, bool) else 0
    attempts += 1
    record = {
        "occurrence_ms": occurrence_ms,
        "state": STATE_UNDELIVERED if attempts >= UNDELIVERED_AFTER_ATTEMPTS else STATE_RETRYING,
        "attempts": attempts,
        "first_attempt_ms": first if isinstance(first, int) else moment,
        "last_attempt_ms": moment,
        "next_attempt_ms": next_attempt_ms(attempts, now_ms=moment),
        # Truncated for the same reason every other log-bound string here is:
        # an exception's text can carry a child's whole stderr, and this field
        # is rendered in a table.
        "last_error": error[:400],
    }
    # ``write_delivery`` answers with the stored body or ``None``; the caller
    # needs to know which (see the docstring).
    return write_delivery(config_dir, session_id, record)


def note_delivered(config_dir: Path, session_id: str, occurrence_ms: int) -> int:
    """Clear an owed fire's record now that a runtime has taken it.

    Returns how many failed attempts it took, so the caller can log the
    recovery with the figure that makes it worth reading ("delivered after 4
    attempts" is what tells an operator the host was struggling; a bare
    "delivered" does not).

    SCOPED RESIDUAL (review round 1, MINOR 3): this clears on HANDOVER, which is
    where the supervisor's job ends — the runtime exists, and by design the
    session's own scheduler is what fires the occurrence. A runtime that then
    dies before advancing the schedule leaves the occurrence due with no record,
    so the next pass re-engages at full deadline cost: the backoff paces failed
    ATTEMPTS, not a handover that did not run. That is the same shape and the
    same cost a first-time due wake has (and the shape ``main`` has for every
    attempt), so it is not a regression this record introduced; removing it needs
    a record that outlives the handover and a rule for "the occurrence actually
    ran", which is a design change rather than a constant.

    THE RECORD IS DELETED rather than kept as a completed row: success is
    visible on the surfaces that already answer it — the record's absence from
    ``lop wake status``, and the session's own ``last fired`` stamp on the entry
    once the runtime actually runs the occurrence. A record kept here would need
    a lifetime, a grace, and a re-arm rule for the case where the runtime died
    before advancing the schedule — all of which the existing store already gets
    right.
    """
    existing = read_delivery(config_dir, session_id)
    attempts = 0
    if existing is not None and existing.get("occurrence_ms") == occurrence_ms:
        prior = existing.get("attempts")
        attempts = prior if isinstance(prior, int) and not isinstance(prior, bool) else 0
        remove_delivery(config_dir, session_id)
    return attempts
