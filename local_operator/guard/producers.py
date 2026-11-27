"""The Guard background producers and uploader (Guard RFC v0.9 §5.3, slice L3).

WHAT. One :meth:`GuardProducers.tick`, run on a timer inside ``lop serve``
(:func:`run_forever`), that

1. reads the member's LIVE consents (``GuardConsentHost.live_consents``) --
   ``revoke_pending``, ``revoked``, declined and absent are all "stopped";
2. installs or removes the process-wide decision sink (L1's seam) and task
   sink (``guard/activity.py``) so records are kept only while some org holds
   a live consent for that family;
3. observes posture (``guard/posture.py``) and runs the review-cycle adapter;
4. drains every closed-day record to each live org as ONE signed batch per org
   (``guard/ingest.py``), with exponential backoff and idempotent batch ids;
5. on a revoke -- local, from Console, or a refusal saying so -- drops what is
   queued for that org, and when no org is left for a family, purges it.

ZERO COST WITHOUT CONSENT (§5.1, the L3 zero-call row). The FIRST thing a tick
does is ask whether ``guard/consent.sqlite3`` exists, which only an admitted
Guard prompt ever creates. If it does not -- every non-Team install, every
Team member who never answered -- the tick returns before any file, network
call, credential-store read or sink install. With a store but no live
consent it touches only Guard's own files (to purge them) and still sends
nothing.

AUTHORITY PER BATCH (§6.2, §22). Each org has its own high-water FLOORS over
the queues, set when its current consent is first seen live. A record queued
before that consent is never sent under it, so a later grant cannot carry an
old queue, and a re-grant after a revoke starts empty. A paused consent
(org off, plan lapsed, membership ended, device revoked) stays paused until
the member grants again; nothing is re-sent under the new grant.

WHAT LEAVES. Only what ``guard/ingest.py`` lets through: enums, UTC days or ISO
weeks, random ids, a randomised posture bit, the install id and the consent's
revisions. The friction counter never leaves (v0.3 §9.4).

THE MEMO OUTLIVES EVERYTHING HERE. No path in this module deletes or edits
``posture_memo.sqlite3``: revoke, pause, ``namespace_reset`` and a re-grant
all keep it (§10'), so the next report after off->on carries the same bit.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import random
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from . import activity as act
from . import decisions as gd
from . import ingest
from . import posture as pst
from ._sqlite import open_store
from .consent import GuardConsentHost, GuardConsentStore
from .outbox import GuardOutbox

logger = logging.getLogger(__name__)

PRODUCERS_SCHEMA_VERSION: Final = 1

#: How often ``lop serve`` ticks. Event families send closed UTC days only and
#: live 72 h in the queue (§9.2'), so a quarter hour is far inside any deadline
#: while costing one ``stat`` per tick on an install with no consent.
TICK_S: Final = 900.0
#: First tick after boot: long enough that a daemon start is never slowed by it.
FIRST_TICK_S: Final = 30.0
#: Backoff after a retryable failure: 60 s doubling, capped at 6 h (the
#: 6-hourly admission cadence, §5.2.1), with 50-100 % jitter so a fleet that
#: failed together does not retry together.
BACKOFF_BASE_S: Final = 60.0
BACKOFF_CAP_S: Final = 6 * 3600.0
#: Batches per org per tick: drains a full outbox (2048 rows at 500 per batch).
MAX_BATCHES_PER_TICK: Final = 5
#: Posture is reported for the week that just closed. A week older than this is
#: past the server's weekly fold (≤ 8 days, §10') and is dropped, not sent.
POSTURE_MAX_LAG_WEEKS: Final = 2

_SCOPE_FAMILIES: Final = {
    "decisions": ingest.FAMILY_DECISIONS,
    "task_time": ingest.FAMILY_TASK_TIME,
    "review_cycles": ingest.FAMILY_REVIEW_CYCLES,
    "posture": ingest.FAMILY_POSTURE,
}


def producers_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "producers.sqlite3"


_DDL: Final = (
    "CREATE TABLE tenant_state (tenant TEXT PRIMARY KEY, collection_rev INTEGER NOT NULL,"
    " granted_at REAL NOT NULL, floor_decisions INTEGER NOT NULL, floor_task INTEGER NOT NULL,"
    " floor_review INTEGER NOT NULL, next_at REAL NOT NULL DEFAULT 0,"
    " failures INTEGER NOT NULL DEFAULT 0, paused_code TEXT)",
    "CREATE TABLE delivered (tenant TEXT NOT NULL, family TEXT NOT NULL,"
    " record_id TEXT NOT NULL, PRIMARY KEY (tenant, family, record_id))",
    # The member's TRUE posture value per closed week. Local only, never sent.
    "CREATE TABLE posture_obs (week TEXT NOT NULL, fact TEXT NOT NULL, value TEXT NOT NULL,"
    " PRIMARY KEY (week, fact))",
    "CREATE TABLE posture_weeks (tenant TEXT NOT NULL, week TEXT NOT NULL,"
    " sent INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (tenant, week))",
)


@dataclass(frozen=True)
class LiveConsent:
    tenant: str
    collection_rev: int
    revocation_gen: int
    scopes: tuple[str, ...]
    granted_at: float

    @property
    def families(self) -> frozenset[str]:
        return frozenset(_SCOPE_FAMILIES[s] for s in self.scopes if s in _SCOPE_FAMILIES)


@dataclass
class TickReport:
    """What one tick did; for tests and the debug log. Holds no record content."""

    live: int = 0
    sent: dict[str, int] = field(default_factory=dict)
    results: list[str] = field(default_factory=list)
    skipped: str = ""


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


class GatedDecisionSink:
    """L1's decision sink for an install with a live ``decisions`` or ``reports`` consent.

    ``decisions`` -> the outbox; ``reports`` -> the local friction counter on
    every ``block``. The flags are set by the tick, so a revoke stops new
    records within one tick and the uploader's own live re-check stops sends
    at once.
    """

    def __init__(self, outbox: GuardOutbox, store: act.ActivityStore) -> None:
        self.outbox = outbox
        self.store = store
        self.decisions = False
        self.reports = False

    def emit(self, record: gd.GuardDecisionRecord) -> None:
        if self.decisions:
            self.outbox.append(record)
        if self.reports and record.outcome == gd.OUTCOME_BLOCK:
            self.store.count_block(record.rule_class, day=record.day)


# ---------------------------------------------------------------------------
# The producers
# ---------------------------------------------------------------------------


class GuardProducers:
    """Everything one tick does, for one config root."""

    def __init__(
        self,
        config_dir: Path | str,
        *,
        transport_factory: Callable[[], tuple[Any, str | None] | None] | None = None,
        key_store: Any = None,
        clock: Callable[[], float] = time.time,
        rng: Callable[[], float] | None = None,
        install_sinks: bool = True,
    ) -> None:
        # No I/O here: a non-Team install constructs this and leaves no file.
        self._config_dir = Path(config_dir)
        self._transport_factory = transport_factory
        self._key_store = key_store
        self._clock = clock
        self._jitter = rng or random.SystemRandom().random
        self._rng = rng
        self._install_sinks = install_sinks
        self._sink: GatedDecisionSink | None = None
        self._task_sink: act.StoreTaskSink | None = None
        self._resolved: tuple[Any, str | None] | None = None
        #: Per-org record ceiling after a 413 ``guard_batch_too_large``: half the
        #: largest family of the refused batch, halved again per refusal and
        #: cleared by the next accepted or dropped batch.
        #: In memory on purpose: with the caps at or under A3's a 413 is not
        #: expected at all, and a daemon restart simply starts at full size.
        self._split: dict[str, int] = {}
        self._lock = threading.Lock()

    # -- paths ---------------------------------------------------------------

    @property
    def outbox(self) -> GuardOutbox:
        return GuardOutbox(self._config_dir)

    @property
    def activity(self) -> act.ActivityStore:
        return act.ActivityStore(self._config_dir)

    def _state(self):  # -> sqlite3.Connection
        return open_store(
            producers_path(self._config_dir), version=PRODUCERS_SCHEMA_VERSION, ddl=_DDL
        )

    def _host(self) -> GuardConsentHost:
        return GuardConsentHost(
            self._config_dir, transport=None, mirror=False, key_store=self._key_store
        )

    def live(self) -> list[LiveConsent]:
        """Live consents only; ``[]`` without touching disk when no consent store exists."""

        if not GuardConsentStore(self._config_dir).exists():
            return []
        return [LiveConsent(**row) for row in self._host().live_consents()]

    # -- the tick -------------------------------------------------------------

    def tick(self) -> TickReport:
        """One pass. Never raises."""

        with self._lock:
            try:
                return self._tick()
            except Exception:  # noqa: BLE001 -- background telemetry never fails the daemon
                logger.debug("guard producers tick failed", exc_info=True)
                return TickReport(skipped="error")

    def _tick(self) -> TickReport:
        report = TickReport()
        # THE ZERO-COST GATE: a stat, before anything else (module docstring).
        if not GuardConsentStore(self._config_dir).exists():
            self._set_sinks(decisions=False, reports=False, tasks=False)
            report.skipped = "no_consent_store"
            return report
        consents = self.live()
        report.live = len(consents)
        families = frozenset().union(*(c.families for c in consents)) if consents else frozenset()
        reports_on = any("reports" in c.scopes for c in consents)
        self._set_sinks(
            decisions=ingest.FAMILY_DECISIONS in families,
            reports=reports_on,
            tasks=ingest.FAMILY_TASK_TIME in families,
        )
        self._reconcile(consents, families)
        if not consents:
            report.skipped = "no_live_consent"
            return report
        now = self._clock()
        today = _dt.datetime.fromtimestamp(now, _dt.UTC).date()
        self.activity.prune(today=today)
        # L1's ``prune`` opens (and so would create) the outbox; an install
        # whose orgs never consented to ``decisions`` must not grow that file.
        if self.outbox.path.exists():
            self.outbox.prune(today=today)
        if ingest.FAMILY_POSTURE in families:
            self._observe_posture(consents, today)
        if ingest.FAMILY_REVIEW_CYCLES in families:
            self.activity.scan_reviews(today=today.isoformat())
        transport: tuple[Any, str | None] | None = None
        for consent in consents:
            state = self._tenant_state(consent.tenant)
            if state is None or state["paused_code"] or state["next_at"] > now:
                continue
            resigned = False
            for _ in range(MAX_BATCHES_PER_TICK):
                built = self._build(consent, state, today)
                if built is None:
                    break
                if transport is None:
                    transport = self._resolve_transport()
                    if transport is None:
                        report.skipped = "signed_out"
                        return report
                outcome = self._send(consent, built, transport, now, resigned=resigned)
                report.results.append(outcome.kind)
                if outcome.kind == "accepted":
                    for family, rows in built.families.items():
                        report.sent[family] = report.sent.get(family, 0) + len(rows)
                if outcome.kind == "device_revoked":
                    # The key is dead for every org on this install; nothing
                    # else this tick may sign with it.
                    self._settle(self.live())
                    return report
                # A3 ``guard_batch_too_large`` (smaller batch next) and the
                # FIRST ``guard_request_expired`` (re-sign now) are sent again
                # at once; every other non-accept ends this org's tick.
                if outcome.kind == "resign" and not resigned:
                    resigned = True
                    continue
                if outcome.kind not in ("accepted", "split"):
                    break
                state = self._tenant_state(consent.tenant)
                if state is None:
                    break
        self._settle(self.live())
        return report

    # -- sinks -----------------------------------------------------------------

    def _set_sinks(self, *, decisions: bool, reports: bool, tasks: bool) -> None:
        if not self._install_sinks:
            return
        if decisions or reports:
            if self._sink is None:
                self._sink = GatedDecisionSink(self.outbox, self.activity)
            self._sink.decisions = decisions
            self._sink.reports = reports
            if gd.get_decision_sink() is not self._sink:
                gd.set_decision_sink(self._sink)
        elif self._sink is not None:
            self._sink.decisions = self._sink.reports = False
            if gd.get_decision_sink() is self._sink:
                gd.set_decision_sink(None)
            self._sink = None
        if tasks:
            if self._task_sink is None:
                self._task_sink = act.StoreTaskSink(self.activity)
            if act.get_task_sink() is not self._task_sink:
                act.set_task_sink(self._task_sink)
        elif self._task_sink is not None:
            if act.get_task_sink() is self._task_sink:
                act.set_task_sink(None)
            self._task_sink = None

    def close(self) -> None:
        """Remove this instance's sinks (daemon shutdown)."""
        with self._lock:
            self._set_sinks(decisions=False, reports=False, tasks=False)

    # -- per-tenant authority ---------------------------------------------------

    def _reconcile(self, consents: list[LiveConsent], families: frozenset[str]) -> None:
        """Floors for new grants; drop state and queues for consents that stopped."""

        live = {c.tenant: c for c in consents}
        conn = self._state()
        try:
            conn.execute("BEGIN IMMEDIATE")
            known = {r["tenant"]: r for r in conn.execute("SELECT * FROM tenant_state").fetchall()}
            for tenant, row in known.items():
                consent = live.get(tenant)
                if consent is None or (
                    row["collection_rev"] != consent.collection_rev
                    or row["granted_at"] != consent.granted_at
                ):
                    # Stopped, or a different grant: nothing queued under the
                    # old authority may be sent under this one.
                    self._forget(conn, tenant)
            for tenant, consent in live.items():
                exists = conn.execute(
                    "SELECT 1 FROM tenant_state WHERE tenant = ?", (tenant,)
                ).fetchone()
                if exists is None:
                    conn.execute(
                        "INSERT INTO tenant_state (tenant, collection_rev, granted_at,"
                        " floor_decisions, floor_task, floor_review) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            tenant,
                            consent.collection_rev,
                            consent.granted_at,
                            self.outbox.max_seq(),
                            self._activity_seq(),
                            self._activity_seq(),
                        ),
                    )
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
        # Drop on revoke: a family no live org collects is purged locally.
        if ingest.FAMILY_DECISIONS not in families:
            self.outbox.purge()
        for family in act.FAMILIES:
            if family not in families:
                self.activity.purge(family)

    def _activity_seq(self) -> int:
        return self.activity.max_seq()

    @staticmethod
    def _forget(conn: Any, tenant: str) -> None:
        conn.execute("DELETE FROM tenant_state WHERE tenant = ?", (tenant,))
        conn.execute("DELETE FROM delivered WHERE tenant = ?", (tenant,))
        conn.execute("DELETE FROM posture_weeks WHERE tenant = ?", (tenant,))

    def _tenant_state(self, tenant: str) -> Mapping[str, Any] | None:
        conn = self._state()
        try:
            found = conn.execute(
                "SELECT * FROM tenant_state WHERE tenant = ?", (tenant,)
            ).fetchone()
            return dict(found) if found is not None else None
        finally:
            conn.close()

    # -- posture -----------------------------------------------------------------

    def _observe_posture(self, consents: list[LiveConsent], today: _dt.date) -> None:
        week = pst.iso_week(today)
        values = pst.read_facts(self._config_dir)
        conn = self._state()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Last true value of the week wins (§10': "the memoised randomised
            # answer for their last true value that week").
            for fact, value in values.items():
                conn.execute(
                    "INSERT OR REPLACE INTO posture_obs (week, fact, value) VALUES (?, ?, ?)",
                    (week, fact, value),
                )
            for consent in consents:
                if ingest.FAMILY_POSTURE in consent.families:
                    conn.execute(
                        "INSERT OR IGNORE INTO posture_weeks (tenant, week) VALUES (?, ?)",
                        (consent.tenant, week),
                    )
            oldest = pst.iso_week(today - _dt.timedelta(weeks=POSTURE_MAX_LAG_WEEKS + 1))
            conn.execute("DELETE FROM posture_obs WHERE week < ?", (oldest,))
            conn.execute("DELETE FROM posture_weeks WHERE week < ?", (oldest,))
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()

    def _posture_week(
        self, consent: LiveConsent, today: _dt.date
    ) -> tuple[str, dict[str, str]] | None:
        """The oldest closed, unsent, still-useful week for ``consent`` and its true values."""

        current = pst.iso_week(today)
        floor = pst.iso_week(today - _dt.timedelta(weeks=POSTURE_MAX_LAG_WEEKS))
        conn = self._state()
        try:
            row = conn.execute(
                "SELECT week FROM posture_weeks WHERE tenant = ? AND sent = 0 AND week < ?"
                " AND week >= ? ORDER BY week LIMIT 1",
                (consent.tenant, current, floor),
            ).fetchone()
            if row is None:
                return None
            values = {
                r["fact"]: r["value"]
                for r in conn.execute(
                    "SELECT fact, value FROM posture_obs WHERE week = ?", (row["week"],)
                ).fetchall()
            }
            return row["week"], values
        finally:
            conn.close()

    # -- building and sending ----------------------------------------------------------

    @dataclass
    class _Batch:
        families: dict[str, list[dict[str, Any]]]
        seqs: dict[str, int]
        posture_week: str | None

    def _build(
        self, consent: LiveConsent, state: Mapping[str, Any], today: _dt.date
    ) -> GuardProducers._Batch | None:
        before = today.isoformat()
        delivered = self._delivered(consent.tenant)
        families: dict[str, list[dict[str, Any]]] = {}
        seqs: dict[str, int] = {}
        # A3's per-family caps, lowered after a 413 (``_split``).
        ceiling = self._split.get(consent.tenant)

        def cap(family: str) -> int:
            base = ingest.FAMILY_CAPS.get(family, ingest.MAX_RECORDS_PER_FAMILY)
            return base if ceiling is None else max(1, min(base, ceiling))

        limit = cap(ingest.FAMILY_DECISIONS)
        if ingest.FAMILY_DECISIONS in consent.families:
            rows = [
                (seq, rec)
                for seq, rec in self.outbox.pending_after(
                    int(state["floor_decisions"]), before_day=before, limit=limit * 2
                )
                if rec.record_id not in delivered.get(ingest.FAMILY_DECISIONS, ())
            ][:limit]
            if rows:
                families[ingest.FAMILY_DECISIONS] = [
                    {
                        "record_id": r.record_id,
                        "schema": r.schema,
                        "day": r.day,
                        "outcome": r.outcome,
                        "rule_class": r.rule_class,
                        "tool_class": r.tool_class,
                    }
                    for _, r in rows
                ]
        for family, floor_key in (
            (ingest.FAMILY_TASK_TIME, "floor_task"),
            (ingest.FAMILY_REVIEW_CYCLES, "floor_review"),
        ):
            if family not in consent.families:
                continue
            limit = cap(family)
            rows2 = [
                (seq, body)
                for seq, body in self.activity.pending_after(
                    family, int(state[floor_key]), before_day=before, limit=limit * 2
                )
                if body.get("record_id") not in delivered.get(family, ())
            ][:limit]
            if rows2:
                families[family] = [body for _, body in rows2]
                seqs[family] = rows2[-1][0]
        posture_week: str | None = None
        if ingest.FAMILY_POSTURE in consent.families:
            found = self._posture_week(consent, today)
            if found is not None:
                week, values = found
                account = self._account() if values else None
                if account:
                    memo = pst.PostureMemo(self._config_dir, rng=self._rng)
                    families[ingest.FAMILY_POSTURE] = [
                        r.wire() for r in pst.reports_for(memo, account, week, values)
                    ]
                    # Only a week whose bits are IN this batch is marked sent by it.
                    posture_week = week
                elif not values:
                    self._mark_posture_sent(consent.tenant, week)
        if not families:
            return None
        return GuardProducers._Batch(families, seqs, posture_week)

    def _delivered(self, tenant: str) -> dict[str, set[str]]:
        conn = self._state()
        try:
            out: dict[str, set[str]] = {}
            for r in conn.execute(
                "SELECT family, record_id FROM delivered WHERE tenant = ?", (tenant,)
            ).fetchall():
                out.setdefault(r["family"], set()).add(r["record_id"])
            return out
        finally:
            conn.close()

    def _account(self) -> str | None:
        resolved = self._resolve_transport()
        return None if resolved is None else resolved[1]

    def reset_transport(self) -> None:
        """Forget the resolved client, so the next send re-reads the sign-in."""
        self._resolved = None

    def _person_client(self) -> tuple[Any, str | None] | None:
        from local_operator.hub_org import eligibility

        return eligibility.person_client(self._config_dir)

    def _resolve_transport(self) -> tuple[Any, str | None] | None:
        if self._resolved is not None:
            return self._resolved
        factory = self._transport_factory or self._person_client

        try:
            self._resolved = factory()
        except Exception:  # noqa: BLE001 -- signed out / no network: nothing sent
            logger.debug("guard producers could not resolve a transport", exc_info=True)
            self._resolved = None
        return self._resolved

    def _send(
        self,
        consent: LiveConsent,
        batch: GuardProducers._Batch,
        transport: tuple[Any, Any],
        now: float,
        *,
        resigned: bool = False,
    ) -> ingest.IngestResult:
        from local_operator.guard import install_key

        # Re-read authority immediately before the send: a revoke that landed
        # since the tick began stops this batch (§5.4 revoke always wins locally).
        if not any(
            c.tenant == consent.tenant
            and c.collection_rev == consent.collection_rev
            and c.granted_at == consent.granted_at
            for c in self.live()
        ):
            return ingest.IngestResult("revoked", "guard_revoked_locally")
        host = self._host()
        install_id = host.enrolled_install_id()
        if install_id is None:
            return ingest.IngestResult("retry", "guard_not_enrolled")
        store = self._key_store or install_key.default_store(self._config_dir)
        try:
            key = install_key.load(store, install_id)  # NEVER creates a key
        except install_key.GuardKeyStoreError:
            key = None
        if key is None:
            self._backoff(consent.tenant, now)
            return ingest.IngestResult("retry", "guard_key_unavailable")
        payload = ingest.build_payload(
            tenant=consent.tenant,
            install_id=install_id,
            collection_rev=consent.collection_rev,
            revocation_gen=consent.revocation_gen,
            families=batch.families,
            # The clock at SIGNING, not the tick start: A3 refuses a ts more
            # than 15 min off, and a re-sign after ``guard_request_expired``
            # must carry the time it is actually sent.
            ts=int(self._clock()),
        )
        result = ingest.send(transport[0], ingest.sign(key, payload))
        if result.kind == "resign" and resigned:
            # Re-signed with the current clock and refused again: the local
            # clock is off. Back off instead of looping.
            result = ingest.IngestResult("retry", result.code)
        self._apply(consent, batch, result, now)
        return result

    def _apply(
        self,
        consent: LiveConsent,
        batch: GuardProducers._Batch,
        result: ingest.IngestResult,
        now: float,
    ) -> None:
        tenant = consent.tenant
        if result.kind in ("accepted", "drop"):
            self._split.pop(tenant, None)
            # Accepted, field-refused or uncountable (day closed / stale): none
            # of these records is ever sent to this org again.
            conn = self._state()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for family, rows in batch.families.items():
                    if family == ingest.FAMILY_POSTURE:
                        continue
                    conn.executemany(
                        "INSERT OR IGNORE INTO delivered (tenant, family, record_id)"
                        " VALUES (?, ?, ?)",
                        [(tenant, family, row["record_id"]) for row in rows],
                    )
                if batch.posture_week is not None:
                    conn.execute(
                        "UPDATE posture_weeks SET sent = 1 WHERE tenant = ? AND week = ?",
                        (tenant, batch.posture_week),
                    )
                conn.execute(
                    "UPDATE tenant_state SET failures = 0, next_at = 0 WHERE tenant = ?", (tenant,)
                )
                conn.execute("COMMIT")
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                conn.close()
            if result.kind == "drop":
                logger.info("guard ingest dropped a batch (%s)", result.code)
            elif result.rejected:
                # Per-record ``guard_day_closed`` and the like (inside the 200):
                # retired with the batch above, because they can never count.
                logger.info("guard ingest: %d records rejected one by one", result.rejected)
            return
        if result.kind == "revoked":
            if result.code != "guard_revoked_locally":
                # Console (or another surface) revoked first: the same sticky
                # "Stopped" state a stale grant refusal records (§5.2.6).
                self._host().observe_console_revoke(tenant)
            self._drop_tenant(tenant)
            return
        if result.kind == "paused":
            conn = self._state()
            try:
                conn.execute(
                    "UPDATE tenant_state SET paused_code = ? WHERE tenant = ?",
                    (result.code[:64], tenant),
                )
            finally:
                conn.close()
            logger.info("guard ingest paused for an org (%s)", result.code)
            return
        if result.kind == "device_revoked":
            # Every live consent on this install stops and the dead key is
            # retired; only a fresh person-admitted accept enrols a new one.
            stopped = {c.tenant for c in self.live()} | {tenant}
            self._host().observe_device_revoked()
            for other in sorted(stopped):
                self._drop_tenant(other)
            return
        if result.kind == "split":
            largest = max(len(rows) for rows in batch.families.values())
            if largest <= 1:
                # One record per family and still too large: these records can
                # never be sent; retire them rather than loop.
                self._apply(consent, batch, ingest.IngestResult("drop", result.code), now)
                return
            self._split[tenant] = largest // 2
            return
        if result.kind == "resign":
            return  # the tick re-signs at once (one time)
        if result.kind == "unavailable":
            # A3: "no retry loop". Wait the whole cap, not the doubling ladder.
            self._backoff(tenant, now, delay=BACKOFF_CAP_S)
            return
        self._backoff(tenant, now)

    def _backoff(self, tenant: str, now: float, *, delay: float | None = None) -> None:
        conn = self._state()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT failures FROM tenant_state WHERE tenant = ?", (tenant,)
            ).fetchone()
            failures = (int(row["failures"]) if row is not None else 0) + 1
            if delay is None:
                delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** min(failures - 1, 20)))
                delay *= 0.5 + 0.5 * self._jitter()
            conn.execute(
                "UPDATE tenant_state SET failures = ?, next_at = ? WHERE tenant = ?",
                (failures, now + delay, tenant),
            )
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()

    def _drop_tenant(self, tenant: str) -> None:
        conn = self._state()
        try:
            self._forget(conn, tenant)
        finally:
            conn.close()

    def _mark_posture_sent(self, tenant: str, week: str) -> None:
        conn = self._state()
        try:
            conn.execute(
                "UPDATE posture_weeks SET sent = 1 WHERE tenant = ? AND week = ?", (tenant, week)
            )
        finally:
            conn.close()

    def namespace_reset(self) -> None:
        """The server's ``namespace_reset`` (§16'): discard every queue. KEEP the posture memo."""

        self.outbox.purge()
        self.activity.purge()
        path = producers_path(self._config_dir)
        if path.exists():
            conn = self._state()
            try:
                conn.execute("BEGIN IMMEDIATE")
                for table in ("tenant_state", "delivered", "posture_weeks", "posture_obs"):
                    conn.execute(f"DELETE FROM {table}")  # noqa: S608 -- fixed names
                conn.execute("COMMIT")
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                conn.close()

    def _settle(self, consents: Iterable[LiveConsent]) -> None:
        """Mark records delivered to EVERY live org collecting them; forget the rest."""

        conn = self._state()
        try:
            paused = {
                r[0]
                for r in conn.execute(
                    "SELECT tenant FROM tenant_state WHERE paused_code IS NOT NULL"
                ).fetchall()
            }
            # A paused org will never take these records (its queue was dropped
            # with the pause), so it must not hold them back from the others.
            consents = [c for c in consents if c.tenant not in paused]
            for family in (ingest.FAMILY_DECISIONS, *act.FAMILIES):
                tenants = [c.tenant for c in consents if family in c.families]
                if not tenants:
                    continue
                marks = ",".join("?" for _ in tenants)
                done = [
                    r[0]
                    for r in conn.execute(
                        # ``marks`` is only "?" placeholders; values are bound.
                        "SELECT record_id FROM delivered WHERE family = ?"
                        f" AND tenant IN ({marks})"
                        " GROUP BY record_id HAVING COUNT(DISTINCT tenant) = ?",
                        (family, *tenants, len(tenants)),
                    ).fetchall()
                ]
                if not done:
                    continue
                if family == ingest.FAMILY_DECISIONS:
                    self.outbox.mark_sent(done)
                else:
                    self.activity.delete(done)
                conn.executemany(
                    "DELETE FROM delivered WHERE family = ? AND record_id = ?",
                    [(family, rid) for rid in done],
                )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# The daemon loop
# ---------------------------------------------------------------------------


async def run_forever(
    config_dir: Path | str,
    *,
    first_tick_s: float = FIRST_TICK_S,
    tick_s: float = TICK_S,
    producers: GuardProducers | None = None,
) -> None:
    """``lop serve``'s Guard loop. Cancelling it removes its sinks."""

    worker = producers or GuardProducers(config_dir)
    try:
        await asyncio.sleep(first_tick_s)
        while True:
            # A fresh transport per tick: a sign-out or token refresh between
            # ticks must not be served from a cached client.
            worker.reset_transport()
            await asyncio.to_thread(worker.tick)
            await asyncio.sleep(tick_s)
    finally:
        worker.close()


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_CAP_S",
    "FIRST_TICK_S",
    "GatedDecisionSink",
    "GuardProducers",
    "LiveConsent",
    "TICK_S",
    "TickReport",
    "producers_path",
    "run_forever",
]
