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
consent it creates NOTHING (no ``producers.sqlite3`` for a prompt nobody
answered or a decline) and touches only Guard's own EXISTING files, to purge
or prune them; it still sends nothing.

REVOKE STOPS COLLECTION AT ONCE. The sinks are installed per tick, but every
record they would keep first passes :class:`_LiveGate`, a stat-keyed re-read of
the consent store (one ``stat`` per decision or settled job; a SQLite read only
after the file changed). A revoke from any surface or process therefore stops
collection on the next record, not the next tick; the tick then purges what was
kept (queues, the friction counter, the true posture values and the review
dedup marks) once no live consent covers it, and prunes on every tick whether or
not anything is live.

ONE ACCOUNT PER CONSENT (S-4). L2a's consent row does not record who granted
it, so the producers bind each consent to the signed-in account the first time
they see it live (the org-hub record's ``account_id``, zero network, else the
resolved bearer's at the first send). A different account on the same
computer -- a shared machine, a sign-out and sign-in -- never inherits it:
the org is paused (``account_changed``, queue purged) and sends nothing until
the member grants again. The signing account is also re-checked at every send.

AUTHORITY PER BATCH (§6.2, §22). Each org has its own high-water FLOORS over
the queues, set when its current consent is first seen live. A record queued
before that consent is never sent under it, so a later grant cannot carry an
old queue, and a re-grant after a revoke starts empty. A paused consent
(org off, plan lapsed, membership ended, device revoked) stays paused until
the member grants again; nothing is re-sent under the new grant.

POSTURE IS SENT DURING THE WEEK. The server takes a posture week only until 24 h
after it ends, and the last report of a week wins, so the tick sends the OPEN
ISO week's memo (UTC) and sends it again only when a reported BIT changed since
the last successful send for that org. A week is never held back until it
closes, and a ``guard_day_closed`` on posture is "too late", not "delivered":
that week is retired unsent (``posture_weeks.closed``). Day records (decisions,
task time, review cycles) still go once their UTC day has closed, inside the
server's 72 h.

WHAT LEAVES. Only what ``guard/ingest.py`` lets through: enums, UTC days or ISO
weeks, per-org random ids, a randomised posture bit, the install id and the
consent's revisions. The friction counter never leaves (v0.3 §9.4). Orgs can
link records only through the one ``install_id``/``key_id`` the A3 contract
gives a computer (see ``ingest.py``, "What orgs can link"); a record's id is
derived per org with a local salt, so the same record is not joinable on it.

THE MEMO OUTLIVES EVERYTHING HERE. No path in this module deletes or edits
``posture_memo.sqlite3``: revoke, pause, ``namespace_reset`` and a re-grant
all keep it (§10'), so the next report after off->on carries the same bit.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import hmac
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
#: A ``delivered`` mark outlives its record only by this many days past the queue's
#: own TTL (``activity.EVENT_TTL_DAYS`` = 3 days, the decisions outbox 72 h).
DELIVERED_KEEP_DAYS: Final = act.EVENT_TTL_DAYS + 1
#: Local posture rows (true values and per-org send marks) older than this many
#: weeks are pruned. Only the current and the just-closed week can still be
#: sent (``posture.week_is_open``); the rest is slack for a skewed clock.
POSTURE_KEEP_WEEKS: Final = 2
#: ``tenant_state.paused_code`` when the signed-in account is not the one that
#: held the consent. Sticky like every pause: only a new grant clears it.
ACCOUNT_CHANGED: Final = "account_changed"

_SCOPE_FAMILIES: Final = {
    "decisions": ingest.FAMILY_DECISIONS,
    "task_time": ingest.FAMILY_TASK_TIME,
    "review_cycles": ingest.FAMILY_REVIEW_CYCLES,
    "posture": ingest.FAMILY_POSTURE,
}


def producers_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "producers.sqlite3"


_DDL: Final = (
    # ``account_id`` binds the consent to the account that was signed in when it
    # was first seen live (S-4); NULL until one can be named.
    "CREATE TABLE tenant_state (tenant TEXT PRIMARY KEY, collection_rev INTEGER NOT NULL,"
    " granted_at REAL NOT NULL, floor_decisions INTEGER NOT NULL, floor_task INTEGER NOT NULL,"
    " floor_review INTEGER NOT NULL, next_at REAL NOT NULL DEFAULT 0,"
    " failures INTEGER NOT NULL DEFAULT 0, paused_code TEXT, account_id TEXT)",
    # ``seq``/``day`` are the record's queue position and UTC day: ``seq`` lets
    # ``_settle`` ask only the orgs whose floor the record is above, and ``day``
    # lets a row whose record the queue's own TTL already removed be dropped.
    "CREATE TABLE delivered (tenant TEXT NOT NULL, family TEXT NOT NULL,"
    " record_id TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0, day TEXT NOT NULL DEFAULT '',"
    " PRIMARY KEY (tenant, family, record_id))",
    # The per-install salt of the per-org record ids (S-5). Never leaves.
    "CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK (id = 1), salt TEXT NOT NULL)",
    "INSERT INTO meta (id, salt) VALUES (1, lower(hex(randomblob(16))))",
    # The member's TRUE posture value per ISO week (the open week's is re-read each
    # tick; last value wins). Local only, never sent.
    "CREATE TABLE posture_obs (week TEXT NOT NULL, fact TEXT NOT NULL, value TEXT NOT NULL,"
    " PRIMARY KEY (week, fact))",
    # What this org's server holds for the week, as two local digests: the TRUE
    # values and the REPORTED bits last sent (NULL = nothing sent yet). ``closed``
    # = the server answered ``guard_day_closed``: too late, never tried again.
    "CREATE TABLE posture_weeks (tenant TEXT NOT NULL, week TEXT NOT NULL,"
    " sent_values TEXT, sent_bits TEXT, closed INTEGER NOT NULL DEFAULT 0,"
    " PRIMARY KEY (tenant, week))",
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


def _stat_key(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


class _LiveGate:
    """Which families (and ``reports``) a LIVE consent covers right now, stat-cached.

    WHY. A revoke has to stop collection at once, from any surface or process,
    and the sinks run on the tool-call path and the job-settle path, so the
    answer must cost one ``stat`` when nothing changed. The consent store is a
    rollback-journal SQLite file (no WAL), so a commit rewrites it and moves its
    mtime/size; only then is it re-read. A failed read fails CLOSED (nothing
    collected) and is retried on the next record.
    """

    def __init__(self, producers: GuardProducers) -> None:
        self._producers = producers
        self._lock = threading.Lock()
        self._key: tuple[int, int] | None = None
        self._families: frozenset[str] = frozenset()
        self._reports = False

    def __call__(self) -> tuple[frozenset[str], bool]:
        path = self._producers.consent_path
        key = _stat_key(path)
        with self._lock:
            if key is None:
                self._key, self._families, self._reports = None, frozenset(), False
            elif key != self._key:
                try:
                    consents = self._producers.live()
                except Exception:  # noqa: BLE001 -- unreadable: collect nothing
                    return frozenset(), False
                self._key = key
                self._families = (
                    frozenset().union(*(c.families for c in consents)) if consents else frozenset()
                )
                self._reports = any("reports" in c.scopes for c in consents)
            return self._families, self._reports


class GatedDecisionSink:
    """L1's decision sink for an install with a live ``decisions`` or ``reports`` consent.

    ``decisions`` -> the outbox; ``reports`` -> the local friction counter on
    every ``block``. The flags are what the tick decided (live AND not paused);
    ``gate`` re-asks the consent store on every record so a revoke cuts
    collection immediately, and the uploader's own live re-check stops sends.
    """

    def __init__(
        self,
        outbox: GuardOutbox,
        store: act.ActivityStore,
        gate: Callable[[], tuple[frozenset[str], bool]] | None = None,
    ) -> None:
        self.outbox = outbox
        self.store = store
        self.decisions = False
        self.reports = False
        self._gate = gate

    def emit(self, record: gd.GuardDecisionRecord) -> None:
        decisions, reports = self.decisions, self.reports
        if self._gate is not None and (decisions or reports):
            families, live_reports = self._gate()
            decisions = decisions and ingest.FAMILY_DECISIONS in families
            reports = reports and live_reports
        if decisions:
            self.outbox.append(record)
        if reports and record.outcome == gd.OUTCOME_BLOCK:
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
        #: Serialises ticks. ``close`` deliberately does NOT take it (a tick in a
        #: worker thread holds it through its HTTP call; shutdown must not wait).
        self._lock = threading.Lock()
        #: Guards the two sink slots against a tick racing ``close``.
        self._sink_lock = threading.Lock()
        self._closed = False
        self._gate = _LiveGate(self)

    # -- paths ---------------------------------------------------------------

    @property
    def outbox(self) -> GuardOutbox:
        return GuardOutbox(self._config_dir)

    @property
    def activity(self) -> act.ActivityStore:
        return act.ActivityStore(self._config_dir)

    @property
    def consent_path(self) -> Path:
        return GuardConsentStore(self._config_dir).path

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
        now = self._clock()
        today = _dt.datetime.fromtimestamp(now, _dt.UTC).date()
        # ACTIVE = live AND not paused (org off, membership gone, another
        # account signed in...): a paused org's records are never sent, so they
        # are not collected for it either (review m4).
        consents = self._reconcile(self.live())
        report.live = len(consents)
        families = frozenset().union(*(c.families for c in consents)) if consents else frozenset()
        reports_on = any("reports" in c.scopes for c in consents)
        self._set_sinks(
            decisions=ingest.FAMILY_DECISIONS in families,
            reports=reports_on,
            tasks=ingest.FAMILY_TASK_TIME in families,
        )
        # Prune on every tick, consent or not: retention (72 h events, 30 days
        # friction) must not stop with the last revoke (security S-3). Each
        # prune is a no-op when its file does not exist, so none is created.
        self.activity.prune(today=today)
        # L1's ``prune`` opens (and so would create) the outbox; an install
        # whose orgs never consented to ``decisions`` must not grow that file.
        if self.outbox.path.exists():
            self.outbox.prune(today=today)
        self._prune_delivered(today)
        if not consents:
            report.skipped = "no_live_consent"
            return report
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
                if built.retired:
                    self._retire(consent.tenant, built.retired)
                if not built.families:
                    break  # only damaged rows were found; nothing to send
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
        with self._sink_lock:
            if self._closed:
                # ``close`` won the race with a running tick: never reinstall.
                decisions = reports = tasks = False
            self._set_sinks_locked(decisions=decisions, reports=reports, tasks=tasks)

    def _set_sinks_locked(self, *, decisions: bool, reports: bool, tasks: bool) -> None:
        if decisions or reports:
            if self._sink is None:
                self._sink = GatedDecisionSink(self.outbox, self.activity, gate=self._gate)
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
                self._task_sink = act.StoreTaskSink(
                    self.activity,
                    permit=lambda: ingest.FAMILY_TASK_TIME in self._gate()[0],
                )
            if act.get_task_sink() is not self._task_sink:
                act.set_task_sink(self._task_sink)
        elif self._task_sink is not None:
            if act.get_task_sink() is self._task_sink:
                act.set_task_sink(None)
            self._task_sink = None

    def close(self) -> None:
        """Remove this instance's sinks (daemon shutdown).

        Takes only the sink lock, never the tick lock: ``close`` runs on the
        event loop and a tick in a worker thread can hold the tick lock through
        an HTTP call that cancelling the awaiting task does not stop (review m5).
        ``_closed`` keeps that in-flight tick from reinstalling a sink.
        """
        with self._sink_lock:
            self._closed = True
            self._set_sinks_locked(decisions=False, reports=False, tasks=False)

    # -- per-tenant authority ---------------------------------------------------

    def _signed_in_account(self) -> str | None:
        """The account the org-hub record says is signed in. ZERO network, never raises."""

        from local_operator.hub_org import eligibility

        try:
            record = eligibility.read_record(self._config_dir)
        except Exception:  # noqa: BLE001 -- unknown account: bound at the first send
            return None
        account = record.get("account_id") if record else None
        return account if isinstance(account, str) and account else None

    def _reconcile(self, consents: list[LiveConsent]) -> list[LiveConsent]:
        """Floors for new grants, account binding, purge for what stopped.

        Returns the ACTIVE consents: live, and neither paused by the server nor
        bound to a different account than the one signed in now.
        """

        if not consents and not producers_path(self._config_dir).exists():
            # Nothing live and no state of ours: create nothing (security S-2),
            # but still purge a queue an earlier consent left behind.
            self._purge_uncovered(frozenset(), reports=False)
            return []
        live = {c.tenant: c for c in consents}
        account = self._signed_in_account() if consents else None
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
                row = conn.execute(
                    "SELECT paused_code, account_id FROM tenant_state WHERE tenant = ?", (tenant,)
                ).fetchone()
                if row is None:
                    conn.execute(
                        "INSERT INTO tenant_state (tenant, collection_rev, granted_at,"
                        " floor_decisions, floor_task, floor_review, account_id)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            tenant,
                            consent.collection_rev,
                            consent.granted_at,
                            self.outbox.max_seq(),
                            self._activity_seq(),
                            self._activity_seq(),
                            account,
                        ),
                    )
                elif row["account_id"] is None and account is not None:
                    conn.execute(
                        "UPDATE tenant_state SET account_id = ? WHERE tenant = ?", (account, tenant)
                    )
                elif (
                    row["account_id"] is not None
                    and account is not None
                    and row["account_id"] != account
                    and not row["paused_code"]
                ):
                    # Another account is signed in: this consent is not theirs.
                    conn.execute(
                        "UPDATE tenant_state SET paused_code = ? WHERE tenant = ?",
                        (ACCOUNT_CHANGED, tenant),
                    )
            paused = {
                r["tenant"]
                for r in conn.execute(
                    "SELECT tenant FROM tenant_state WHERE paused_code IS NOT NULL"
                ).fetchall()
            }
            active = [c for c in consents if c.tenant not in paused]
            families = frozenset().union(*(c.families for c in active)) if active else frozenset()
            if ingest.FAMILY_POSTURE not in families:
                # The member's TRUE posture values exist to serve a posture
                # consent; with none live they are not kept (security S-3). The
                # memo of randomised bits is kept on purpose (§10').
                conn.execute("DELETE FROM posture_obs")
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()
        self._purge_uncovered(families, reports=any("reports" in c.scopes for c in active))
        return active

    def _purge_uncovered(self, families: frozenset[str], *, reports: bool) -> None:
        """Drop locally what no active consent covers any more (revoke, pause, narrowing)."""

        if ingest.FAMILY_DECISIONS not in families:
            self.outbox.purge()
        for family in act.FAMILIES:
            if family not in families:
                self.activity.purge(family)
        if not reports:
            self.activity.purge_friction()

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
            oldest = pst.iso_week(today - _dt.timedelta(weeks=POSTURE_KEEP_WEEKS))
            conn.execute("DELETE FROM posture_obs WHERE week < ?", (oldest,))
            conn.execute("DELETE FROM posture_weeks WHERE week < ?", (oldest,))
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            conn.close()

    def _posture_pending(
        self, consent: LiveConsent, today: _dt.date
    ) -> list[tuple[str, dict[str, str], str, str | None]]:
        """Open weeks whose true values differ from what this org last received.

        ``(week, values, values_digest, sent_bits)``, oldest first. A week the
        server has refused as too late (``closed``), one past its window, and
        one whose true values are unchanged since the last successful send are
        all absent: an idle tick finds nothing here and never touches the
        sign-in, the memo or the network.
        """

        conn = self._state()
        try:
            out: list[tuple[str, dict[str, str], str, str | None]] = []
            for row in conn.execute(
                "SELECT week, sent_values, sent_bits FROM posture_weeks"
                " WHERE tenant = ? AND closed = 0 ORDER BY week",
                (consent.tenant,),
            ).fetchall():
                if not pst.week_is_open(row["week"], today):
                    continue
                values = {
                    r["fact"]: r["value"]
                    for r in conn.execute(
                        "SELECT fact, value FROM posture_obs WHERE week = ?", (row["week"],)
                    ).fetchall()
                }
                if not values:
                    continue
                digest = pst.values_digest(values)
                if digest != row["sent_values"]:
                    out.append((row["week"], values, digest, row["sent_bits"]))
            return out
        finally:
            conn.close()

    def _record_posture(
        self, tenant: str, week: str, values_digest: str, bits_digest: str | None, *, closed: bool
    ) -> None:
        """Remember what the server now holds for ``week`` (or that it is too late)."""

        conn = self._state()
        try:
            if closed:
                conn.execute(
                    "UPDATE posture_weeks SET closed = 1 WHERE tenant = ? AND week = ?",
                    (tenant, week),
                )
            else:
                conn.execute(
                    "UPDATE posture_weeks SET sent_values = ?, sent_bits = ?"
                    " WHERE tenant = ? AND week = ?",
                    (values_digest, bits_digest, tenant, week),
                )
        finally:
            conn.close()

    # -- building and sending ----------------------------------------------------------

    @dataclass
    class _Batch:
        families: dict[str, list[dict[str, Any]]]
        seqs: dict[str, int]
        #: ``(week, values_digest, bits_digest)`` of the posture memo in this batch.
        posture: tuple[str, str, str] | None
        #: ``(family, record_id) -> queue seq`` of every record in ``families``,
        #: kept so ``delivered`` can say which orgs a record is still owed to.
        record_seqs: dict[tuple[str, str], int] = field(default_factory=dict)
        #: ``(family, record_id, seq, day)`` of stored rows that failed
        #: ``ingest.validate_record``: retired for this org rather than signed, so
        #: one damaged row never blocks (or is sent with) the batch.
        retired: list[tuple[str, str, int, str]] = field(default_factory=list)

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

        record_seqs: dict[tuple[str, str], int] = {}
        retired: list[tuple[str, str, int, str]] = []

        def take(
            family: str,
            fetch: Callable[[int, int], list[tuple[int, dict[str, Any]]]],
            floor: int,
        ) -> None:
            """Up to ``cap(family)`` valid, undelivered rows past ``floor``.

            Pages through the queue instead of reading a fixed window: rows
            already delivered to THIS org (but still owed to another) would
            otherwise fill the window and starve it of newer rows (review m2).
            """

            limit = cap(family)
            page = limit * 2
            after = floor
            rows: list[tuple[int, dict[str, Any]]] = []
            while len(rows) < limit:
                batch = fetch(after, page)
                for seq, body in batch:
                    rid = body.get("record_id")
                    if rid in delivered.get(family, ()):
                        continue
                    if not ingest.validate_record(family, body):
                        # Damaged or hand-edited: never signed (security S-1).
                        if isinstance(rid, str):
                            retired.append((family, rid, seq, str(body.get("day", ""))[:10]))
                        continue
                    rows.append((seq, body))
                    if len(rows) >= limit:
                        break
                if len(batch) < page:
                    break
                after = batch[-1][0]
            if rows:
                families[family] = [body for _, body in rows]
                seqs[family] = rows[-1][0]
                for seq, body in rows:
                    record_seqs[(family, body["record_id"])] = seq

        def decision_rows(after: int, n: int) -> list[tuple[int, dict[str, Any]]]:
            return [
                (
                    seq,
                    {
                        "record_id": r.record_id,
                        "schema": r.schema,
                        "day": r.day,
                        "outcome": r.outcome,
                        "rule_class": r.rule_class,
                        "tool_class": r.tool_class,
                    },
                )
                for seq, r in self.outbox.pending_after(after, before_day=before, limit=n)
            ]

        if ingest.FAMILY_DECISIONS in consent.families:
            take(ingest.FAMILY_DECISIONS, decision_rows, int(state["floor_decisions"]))
        for family, floor_key in (
            (ingest.FAMILY_TASK_TIME, "floor_task"),
            (ingest.FAMILY_REVIEW_CYCLES, "floor_review"),
        ):
            if family in consent.families:
                take(
                    family,
                    lambda after, n, family=family: self.activity.pending_after(
                        family, after, before_day=before, limit=n
                    ),
                    int(state[floor_key]),
                )
        posture: tuple[str, str, str] | None = None
        if ingest.FAMILY_POSTURE in consent.families:
            pending = self._posture_pending(consent, today)
            account = self._account() if pending else None
            if account:
                memo = pst.PostureMemo(self._config_dir, rng=self._rng)
                for week, values, v_digest, sent_bits in pending:
                    reports = pst.reports_for(memo, account, week, values)
                    b_digest = pst.bits_digest(reports)
                    if b_digest == sent_bits:
                        # A true value moved but its memoised bit did not (or
                        # flipped back): the server already holds this memo.
                        self._record_posture(consent.tenant, week, v_digest, b_digest, closed=False)
                        continue
                    if reports:
                        families[ingest.FAMILY_POSTURE] = [r.wire() for r in reports]
                        # One week per batch; the tick's loop sends the next.
                        posture = (week, v_digest, b_digest)
                        break
        if not families and not retired:
            return None
        return GuardProducers._Batch(families, seqs, posture, record_seqs, retired)

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
        refused = self._check_account(consent.tenant, transport[1])
        if refused is not None:
            return refused
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
            # Per-org record ids (security S-5); the raw ids stay local, in
            # ``delivered``, which is keyed on them.
            families=self._for_org(consent.tenant, batch.families),
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

    def _check_account(self, tenant: str, bearer_account: str | None) -> ingest.IngestResult | None:
        """``None`` when the signed-in bearer is the account this consent is bound to.

        Bound at first sight (``_reconcile``); a consent still unbound because
        the org-hub record named no account then is bound here, to the account
        that is about to sign. A bearer of ANOTHER account is refused before anything
        is sent and the org is paused until the member grants again; one that
        names no account is refused without a pause (security S-4).
        """

        state = self._tenant_state(tenant)
        bound = None if state is None else state["account_id"]
        if bound is None:
            if bearer_account:
                conn = self._state()
                try:
                    conn.execute(
                        "UPDATE tenant_state SET account_id = ? WHERE tenant = ?"
                        " AND account_id IS NULL",
                        (bearer_account, tenant),
                    )
                finally:
                    conn.close()
            return None
        if bearer_account is None:
            # The bearer names no account: it cannot be shown to be the granting
            # one, so nothing is sent, but nothing is paused either -- a token
            # that does not say who it is is not evidence of a different person.
            return ingest.IngestResult("retry", "guard_account_unknown")
        if bearer_account == bound:
            return None
        conn = self._state()
        try:
            conn.execute(
                "UPDATE tenant_state SET paused_code = ? WHERE tenant = ?",
                (ACCOUNT_CHANGED, tenant),
            )
        finally:
            conn.close()
        logger.info("guard producers paused an org: another account is signed in")
        return ingest.IngestResult("paused", ACCOUNT_CHANGED)

    def _salt(self) -> str:
        conn = self._state()
        try:
            return str(conn.execute("SELECT salt FROM meta WHERE id = 1").fetchone()[0])
        finally:
            conn.close()

    def _for_org(
        self, tenant: str, families: Mapping[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """``families`` with every ``record_id`` replaced by this org's own id.

        HMAC-SHA-256 over (tenant, raw id) keyed with a local random salt,
        truncated to the wire's 32 hex: stable per org (a retry is the same
        batch), unrelated across orgs (two orgs cannot join one record on its
        id), and not reversible to the raw id (security S-5).
        """

        salt = self._salt().encode()
        out: dict[str, list[dict[str, Any]]] = {}
        for family, rows in families.items():
            out[family] = [
                (
                    {**row, "record_id": self._wire_record_id(salt, tenant, row["record_id"])}
                    if "record_id" in row
                    else dict(row)
                )
                for row in rows
            ]
        return out

    @staticmethod
    def _wire_record_id(salt: bytes, tenant: str, record_id: str) -> str:
        return hmac.new(salt, f"{tenant}\x00{record_id}".encode(), hashlib.sha256).hexdigest()[:32]

    def _retire(self, tenant: str, rows: list[tuple[str, str, int, str]]) -> None:
        """Mark damaged queue rows delivered for ``tenant`` so they are never rebuilt for it."""

        conn = self._state()
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO delivered (tenant, family, record_id, seq, day)"
                " VALUES (?, ?, ?, ?, ?)",
                [(tenant, family, rid, seq, day) for family, rid, seq, day in rows],
            )
        finally:
            conn.close()

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
                        "INSERT OR IGNORE INTO delivered (tenant, family, record_id, seq, day)"
                        " VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                tenant,
                                family,
                                row["record_id"],
                                batch.record_seqs.get((family, row["record_id"]), 0),
                                str(row.get("day", ""))[:10],
                            )
                            for row in rows
                        ],
                    )
                conn.execute(
                    "UPDATE tenant_state SET failures = 0, next_at = 0 WHERE tenant = ?", (tenant,)
                )
                conn.execute("COMMIT")
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                conn.close()
            if batch.posture is not None:
                week, v_digest, b_digest = batch.posture
                too_late = (ingest.FAMILY_POSTURE, "guard_day_closed") in result.rejected_codes
                # ``guard_day_closed`` on a posture week means the bits are NOT
                # held: "too late", not delivered. Retire the week so it is
                # never retried, and never record it as sent (A3 GUARD.md,
                # Deltas for lop #6). Anything else the 200 or a dropped batch
                # said (accepted, not designated, record conflict, scope
                # refused, replay) leaves nothing a re-send of the SAME bits
                # could change, so the week waits for its next bit change.
                self._record_posture(tenant, week, v_digest, b_digest, closed=too_late)
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
        """Mark records delivered to every org they are OWED to; forget the rest.

        A record is owed to an org that collects its family AND whose floor is
        below the record's queue seq: an org that consented later never receives
        the older rows, so counting it would keep them (and starve the earlier
        org's window) until the 72 h prune (review m2). Paused orgs are excluded:
        their queue was dropped with the pause.
        """

        conn = self._state()
        try:
            paused = {
                r[0]
                for r in conn.execute(
                    "SELECT tenant FROM tenant_state WHERE paused_code IS NOT NULL"
                ).fetchall()
            }
            floors = {
                r["tenant"]: {
                    ingest.FAMILY_DECISIONS: int(r["floor_decisions"]),
                    ingest.FAMILY_TASK_TIME: int(r["floor_task"]),
                    ingest.FAMILY_REVIEW_CYCLES: int(r["floor_review"]),
                }
                for r in conn.execute("SELECT * FROM tenant_state").fetchall()
            }
            active = [c for c in consents if c.tenant not in paused and c.tenant in floors]
            for family in (ingest.FAMILY_DECISIONS, *act.FAMILIES):
                collecting = [c.tenant for c in active if family in c.families]
                if not collecting:
                    continue
                held: dict[str, tuple[int, set[str]]] = {}
                for r in conn.execute(
                    "SELECT tenant, record_id, seq FROM delivered WHERE family = ?", (family,)
                ).fetchall():
                    seq, tenants = held.setdefault(r["record_id"], (int(r["seq"]), set()))
                    tenants.add(r["tenant"])
                done = [
                    rid
                    for rid, (seq, tenants) in held.items()
                    if all(t in tenants for t in collecting if floors[t][family] < seq)
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

    def _prune_delivered(self, today: _dt.date) -> None:
        """Drop marks whose record the queue's own TTL has certainly removed."""

        if not producers_path(self._config_dir).exists():
            return
        oldest = (today - _dt.timedelta(days=DELIVERED_KEEP_DAYS)).isoformat()
        conn = self._state()
        try:
            conn.execute("DELETE FROM delivered WHERE day < ?", (oldest,))
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
