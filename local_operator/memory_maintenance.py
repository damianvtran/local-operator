"""Unattended memory-store maintenance: journal compaction and capture expiry.

WHY THIS EXISTS. The memory journal (``ops``) only shrinks when someone runs
``lop memory compact --yes --expect …``; nobody did, and the store refilled its
byte cap and refused writes (design 2026-10-27, F6/F7). Captured candidates
never expired either (F10): 7,883 of 8,050 live facts were unreviewed captures,
1,911 of them owned by sessions whose directories were already deleted. This
module is the one pass that keeps both bounded without an operator.

WHAT IT RUNS, IN ORDER (one pass, at most every :data:`PASS_INTERVAL_S`):

1. **Expiry (operator decision D2).** Captured candidates — ``source.author``
   ``deterministic-capture`` AND ``status`` ``candidate`` — whose
   ``expires_at`` has passed are forgotten with ordinary tombstones
   (``MemoryStore.forget_many``), at most :data:`EXPIRE_LIMIT` per pass. Read
   paths already hide them (``search``/``tagged`` filter ``expires_at``); the
   tombstone is what makes the expiry real on disk and on synced peers. Rows
   captured before stamping existed carry ``expires_at = null``, so their
   deadline is derived from ``source_time`` (else ``created_at``) + the same
   TTL; without that every pre-existing capture would live forever. Deliberate
   facts (any other author, or a non-candidate status) are never selected.
2. **Compaction (D6).** When the cheap probe crosses a threshold, the EXISTING
   ``journal_plan()`` → ``compact(expect=fingerprint)`` runs with
   ``reason="auto"``. No deletion logic lives here, so #2882's invariants
   (all heads, tombstones kept, cross-entity pins, persisted seq floor) hold by
   construction.

WHERE IT RUNS, AND WHERE IT MUST NOT. The wake supervisor seats it (one
long-lived process beside every runtime, so no session pays the 212 MB plan
RSS); each session RUNTIME process also offers it once, off-thread, after its
record is published, because the supervisor retires when no wake is fireable.
Only long-lived hosts offer it (QA round 1, Q1/Q2): a one-shot process
(``lop exec``, ``lop memory status``) would either hold its exit for the pass
or kill it mid-plan on every invocation, re-paying the plan's RSS without ever
converging. NEVER in ``MemoryStore.connection()``: a 0.7 s plan on every store
open would tax every tool call.

SAFETY. One pass at a time across processes (``fcntl`` lock ``memory/.compact.lock``, non-blocking:
a second caller skips rather than queues). ``compact`` takes ``BEGIN IMMEDIATE`` before planning, so
concurrent writers wait on the store's 5 s ``busy_timeout`` — measured lock hold 1.28 s — rather
than fail. A hard :data:`PASS_TIMEOUT_S` bounds how long the CALLER waits, not the work: a worker
that overruns is abandoned by the caller but, in a long-lived process, keeps running to completion
and commits — Python cannot stop a thread inside SQLite. It holds the lock until it ends, so no
second pass can overlap it. Only if the whole process exits first is its open transaction rolled
back by SQLite (the session exports it rewrites are replaced atomically, ``tmp`` + ``os.replace``,
so they are never half-written either). Expiry gets its own smaller budget (:data:`EXPIRE_BUDGET_S`)
so compaction still fits. VACUUM is skipped when free disk is below ``max(1 GiB, 2 × db bytes)``: it
rewrites the whole file, and the DELETE already returns pages to the freelist that every table
reuses.

MERGE ORDER: this must not ship before the peer-echo ingress filter (B0) —
without it a peer pushes the dropped ops straight back on its next sync.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Captured candidates expire this long after their LAST revision (D2). 14 days
#: covers multi-week lanes; 7 would have cut long programmes (measured: ~380
#: facts / 57k tokens of the after-state). Refreshed on every capture revision,
#: so a fact the session keeps re-capturing stays alive while the session works.
CAPTURE_TTL_S = 14 * 86400

#: Minimum spacing between two passes, measured from the later of the newest
#: ``journal_prune`` row (an operator pass counts) and this module's stamp file.
#: The stamp is what stops a below-threshold store being re-probed every tick.
PASS_INTERVAL_S = 6 * 3600

#: Compaction thresholds (design B1). Any one fires the plan.
RECLAIMABLE_OPS_THRESHOLD = 5000
PAGE_CAP_FRACTION = 0.5
FREELIST_FRACTION = 0.25

#: Hard bound on one pass as the CALLER sees it (design B1).
PASS_TIMEOUT_S = 60.0

#: Expired candidates tombstoned per pass; the rest wait for the next pass.
#: Read paths hide them meanwhile, so the bound costs bytes, never correctness.
#: 500, not the design's 2,000: review round 1 measured ~7 ms per forgotten
#: fact on the live clone (each per-owner chunk re-imports and re-exports that
#: session and checkpoints), so 2,000 took 19.1 s of a 60 s pass on a fast disk.
EXPIRE_LIMIT = 500

#: Wall-clock budget for the expiry step inside one pass. Checked between
#: per-owner chunks, so the overrun is one chunk; what is left waits for the next
#: pass and compaction always starts with most of :data:`PASS_TIMEOUT_S` left.
EXPIRE_BUDGET_S = 15.0

#: VACUUM needs about the file's size again in free space; below this it is
#: skipped (D6). The 1 GiB floor keeps a small store from consuming the last
#: of a nearly full disk.
VACUUM_MIN_FREE_BYTES = 1024 * 1024 * 1024

CAPTURE_AUTHOR = "deterministic-capture"
LOCK_NAME = ".compact.lock"
STAMP_NAME = ".compact.stamp"

#: Captured candidates whose deadline has passed. ``expires_at`` when stamped,
#: else the legacy deadline derived from the row's own time (see module doc).
#: ``current`` holds only ``heads[0]``; ``forget_many(capture_reason=...)``
#: re-checks EVERY head inside its transaction and skips a conflicted entity.
#: ``source.session = owner`` mirrors ``memory.unreviewed_capture``: a capture
#: moved into another session is deliberate and must not be re-selected (and
#: then skipped) on every pass.
_EXPIRED_SQL = (
    "SELECT entity_id, owner, op_id FROM current "
    "WHERE scope='session' "
    "AND json_extract(fact,'$.source.session')=owner "
    "AND json_extract(fact,'$.source.author')=? "
    "AND json_extract(fact,'$.status')='candidate' "
    "AND coalesce(json_extract(fact,'$.expires_at'), "
    "coalesce(json_extract(fact,'$.source_time'), json_extract(fact,'$.created_at')) + ?) <= ? "
    "ORDER BY owner, entity_id LIMIT ?"
)

_background: threading.Thread | None = None
_background_guard = threading.Lock()


def _store(config_dir: Path, owner: str = "") -> Any:
    from local_operator.memory import MemoryIdentity, MemoryStore

    store = MemoryStore(config_dir, MemoryIdentity(owner))
    if owner:
        # The session's portable export is rewritten by ``forget_many`` only when
        # the store knows its directory; without it a moved session's export
        # would still carry the forgotten puts and resurrect them elsewhere.
        store.session_dir = store._session_directory(owner)
    return store


def _forget(
    config_dir: Path, rows: list[tuple[str, str, str]], deadline: float, reason: str
) -> int:
    """Tombstone ``(entity, owner, head op)`` rows, grouped per owner, bounded.

    Goes through ``forget_many(capture_reason=reason)`` rather than writing
    tombstones here, so the quota relief, export rewrite and op shape are the
    store's own, and so the "still ONE unreviewed capture head" check runs
    inside the write transaction: a row edited, promoted or conflicted since it
    was read is skipped there, never forgotten (review round 1, M2).
    """
    from local_operator.memory import MAX_BULK_FORGET, MemoryError

    by_owner: dict[str, dict[str, str]] = {}
    for entity, owner, op_id in rows:
        by_owner.setdefault(owner, {})[entity] = op_id
    forgotten = 0
    for owner, expected in sorted(by_owner.items()):
        store = _store(config_dir, owner)
        items = sorted(expected.items())
        for start in range(0, len(items), MAX_BULK_FORGET):
            if time.monotonic() >= deadline:
                return forgotten
            chunk = dict(items[start : start + MAX_BULK_FORGET])
            try:
                forgotten += len(
                    store.forget_many("session", chunk, capture_reason=reason)["forgotten"]
                )
            except MemoryError:
                logger.info("memory maintenance: a capture chunk changed concurrently; next pass")
    return forgotten


def forget_session_captures(config_dir: Path, session_id: str) -> int:
    """Forget a deleted session's captured candidates; keep its deliberate facts (D2).

    Called by ``session.cleanup.delete_session`` after the directory is gone.
    Only unreviewed captures go: a fact an agent or the operator stated (or a
    candidate someone promoted) stays, listable with
    ``lop memory list --session ID`` and movable. Returns the number forgotten.
    """
    from local_operator.memory import MemoryIdentity, MemoryStore

    store = MemoryStore(config_dir, MemoryIdentity())
    if not store.path.exists():
        return 0
    with store._reader() as db:
        rows = db.execute(
            "SELECT entity_id, owner, op_id FROM current WHERE scope='session' AND owner=? "
            "AND json_extract(fact,'$.source.session')=owner "
            "AND json_extract(fact,'$.source.author')=? "
            "AND json_extract(fact,'$.status')='candidate'",
            (session_id, CAPTURE_AUTHOR),
        ).fetchall()
    return _forget(
        config_dir,
        [tuple(row) for row in rows],
        time.monotonic() + PASS_TIMEOUT_S,
        "session-deleted",
    )


def _last_pass(db: sqlite3.Connection, root: Path) -> float:
    try:
        row = db.execute("SELECT max(at) FROM journal_prune").fetchone()
        pruned = float(row[0]) if row and row[0] is not None else 0.0
    except sqlite3.OperationalError:
        # A store written before journal_prune existed; the probe must not create it.
        pruned = 0.0
    try:
        stamped = float((root / STAMP_NAME).read_text().strip() or 0)
    except (OSError, ValueError):
        stamped = 0.0
    return max(pruned, stamped)


def probe(config_dir: Path) -> dict[str, Any]:
    """The cheap, read-only trigger reading. ~20 ms on a 75 MB store.

    ``reclaimable`` is ``count(ops) - count(DISTINCT entity_id)``: every op
    beyond one per entity, answered from the ``ops_entity`` index. It is an
    UPPER bound on what the plan drops (a live conflict keeps every head, a pin
    keeps its op) — measured equal to the plan's 22,906 on the operator's
    store. The exact number is recomputed by ``journal_plan`` inside the pass;
    this only decides whether that 0.7 s / 212 MB plan is worth running. A
    write-path ``meta`` counter was the alternative and was rejected: it needs
    increments in ``apply_ops`` and capture and can drift, while this read
    cannot.
    """
    from local_operator.memory import (
        CAP_PAGE_SIZE,
        MAX_DB_BYTES,
        MemoryIdentity,
        MemoryStore,
    )

    store = MemoryStore(config_dir, MemoryIdentity())
    if not store.path.exists():
        return {"exists": False}
    with store._reader() as db:
        reclaimable = int(
            db.execute(
                "SELECT (SELECT count(*) FROM ops) - (SELECT count(DISTINCT entity_id) FROM ops)"
            ).fetchone()[0]
        )
        pages = int(db.execute("PRAGMA page_count").fetchone()[0])
        free = int(db.execute("PRAGMA freelist_count").fetchone()[0])
        last = _last_pass(db, store.root)
    cap_pages = MAX_DB_BYTES // CAP_PAGE_SIZE
    reasons = []
    if reclaimable >= RECLAIMABLE_OPS_THRESHOLD:
        reasons.append("reclaimable")
    if pages >= cap_pages * PAGE_CAP_FRACTION:
        reasons.append("page-cap")
    if pages and free >= pages * FREELIST_FRACTION:
        reasons.append("freelist")
    return {
        "exists": True,
        "reclaimable": reclaimable,
        "pages": pages,
        "freelist": free,
        "cap_pages": cap_pages,
        "last_pass": last,
        "triggers": reasons,
    }


def _vacuum_allowed(store: Any) -> tuple[bool, int]:
    try:
        free = shutil.disk_usage(store.root).free
        size = store.path.stat().st_size
    except OSError:
        return False, 0
    return free >= max(VACUUM_MIN_FREE_BYTES, 2 * size), free


def _run(config_dir: Path, now: float, deadline: float, report: dict[str, Any]) -> None:
    from local_operator.memory import MemoryIdentity, MemoryStore

    store = MemoryStore(config_dir, MemoryIdentity())
    with store._reader() as db:
        expired = [
            tuple(row)
            for row in db.execute(_EXPIRED_SQL, (CAPTURE_AUTHOR, CAPTURE_TTL_S, now, EXPIRE_LIMIT))
        ]
    started = time.monotonic()
    budget = min(deadline, started + EXPIRE_BUDGET_S)
    report["expired"] = _forget(config_dir, expired, budget, "expired") if expired else 0
    report["expiry_seconds"] = round(time.monotonic() - started, 3)
    reading = probe(config_dir)
    report["probe"] = reading
    if not reading.get("triggers") and not report["expired"]:
        # Below every threshold: nothing was done, so no stamp either — the
        # next tick re-reads the ~20 ms probe and a store that crosses a
        # threshold an hour from now is not made to wait out a 6 h interval.
        report["status"] = "idle"
        return
    plan = store.journal_plan()
    if not (plan["ops"]["dropped"] or plan["storage"]["freelist_pages"]):
        return
    vacuum, disk_free = _vacuum_allowed(store)
    report["vacuum_skipped_low_disk"] = not vacuum
    report["disk_free"] = disk_free
    done = store.compact(expect=plan["fingerprint"], reason="auto", vacuum=vacuum)
    report["compacted_ops"] = done["ops"]["dropped"]
    report["vacuumed"] = done["vacuumed"]
    report["bytes_before"] = done["storage"]["bytes_before"]
    report["bytes_after"] = done["storage"]["bytes_after"]


def run_pass(
    config_dir: Path, *, now: float | None = None, timeout: float = PASS_TIMEOUT_S
) -> dict[str, Any]:
    """One maintenance pass if due. Blocking up to ``timeout``; never raises.

    The report's ``status`` is one of ``disabled``, ``absent``, ``unsupported``
    (no ``fcntl``), ``busy`` (another process holds the pass), ``not-due``
    (inside the 6 h interval), ``idle`` (due, below every threshold, nothing
    expired), ``ran``, ``timeout`` or ``failed``.
    """
    from local_operator.memory_capture import policy

    started = time.monotonic()
    clock = time.time() if now is None else now
    report: dict[str, Any] = {"status": "ran"}
    try:
        if not policy(config_dir)["enabled"]:
            return {"status": "disabled"}
        root = Path(config_dir) / "memory"
        if not (root / "memory.db").exists():
            return {"status": "absent"}
        try:
            import fcntl
        except ImportError:
            # Without advisory locking two processes could compact at once;
            # refuse by name rather than race.
            return {"status": "unsupported"}
        handle = open(root / LOCK_NAME, "a+", encoding="utf-8")  # noqa: SIM115 — held by the worker
    except Exception as exc:  # noqa: BLE001 — maintenance never fails its host
        logger.warning("memory maintenance could not start", exc_info=True)
        return {"status": "failed", "error": type(exc).__name__}
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return {"status": "busy"}

    def work() -> None:
        try:
            from local_operator.memory import MemoryIdentity, MemoryStore

            with MemoryStore(config_dir, MemoryIdentity())._reader() as db:
                last = _last_pass(db, root)
            if clock - last < PASS_INTERVAL_S:
                report["status"] = "not-due"
                report["last_pass"] = last
                return
            try:
                _run(config_dir, clock, started + timeout, report)
            finally:
                # Stamped whenever the pass did (or attempted) real work, failure
                # included: a pass that raises must not retry its 212 MB plan
                # every tick; it gets the next interval like a successful one.
                if report["status"] != "idle":
                    try:
                        (root / STAMP_NAME).write_text(f"{clock}\n", encoding="utf-8")
                    except OSError:
                        logger.debug("memory maintenance stamp not written", exc_info=True)
        except Exception as exc:  # noqa: BLE001 — maintenance never fails its host
            logger.warning("memory maintenance pass failed", exc_info=True)
            report["status"] = "failed"
            report["error"] = type(exc).__name__
        finally:
            # The lock lives exactly as long as the work, including a worker the
            # caller abandoned on timeout, so no second pass can overlap it.
            handle.close()

    worker = threading.Thread(target=work, name="memory-maintenance", daemon=True)
    worker.start()
    worker.join(max(0.0, timeout - (time.monotonic() - started)))
    if worker.is_alive():
        logger.warning("memory maintenance pass exceeded %.0fs; abandoned", timeout)
        return {**report, "status": "timeout"}
    report["seconds"] = round(time.monotonic() - started, 3)
    # DEBUG here: the supervisor seat logs the outcome at INFO, and the CLI
    # doors (`lop memory status`) must not print a maintenance line to stderr.
    logger.debug("memory maintenance: %s", report)
    return report


def start_background(config_dir: Path) -> threading.Thread | None:
    """Offer one pass on a daemon thread, at most once per process.

    The opportunistic door for long-lived session runtimes only (see the module
    doc for why one-shot processes do not offer it): the supervisor is the
    primary seat but retires when no wake is fireable. The pass itself is gated
    by the cross-process lock and the 6 h interval, so a fleet of runtime starts
    costs one stamp read each.
    """
    global _background
    with _background_guard:
        if _background is not None:
            return None
        _background = threading.Thread(
            target=run_pass, args=(Path(config_dir),), name="memory-maintenance-offer", daemon=True
        )
        _background.start()
        return _background


def last_auto_pass(config_dir: Path) -> dict[str, Any] | None:
    """The newest ``journal_prune`` row with ``reason='auto'``, for status JSON."""
    from local_operator.memory import MemoryIdentity, MemoryStore

    store = MemoryStore(config_dir, MemoryIdentity())
    if not store.path.exists():
        return None
    try:
        with store._reader() as db:
            row = db.execute(
                "SELECT at, ops, bytes FROM journal_prune WHERE reason='auto' "
                "ORDER BY at DESC LIMIT 1"
            ).fetchone()
    except sqlite3.Error:
        return None
    return {"at": row[0], "ops": row[1], "bytes": row[2]} if row else None


def main() -> int:  # pragma: no cover — operator diagnostic
    import json

    from local_operator.paths import config_dir

    print(json.dumps(run_pass(config_dir()), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
