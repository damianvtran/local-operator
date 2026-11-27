"""Task timing, review cycles and the friction counter (Guard RFC v0.9 slice L3).

WHAT. Three local producers that feed the ``task_time`` and ``review_cycles``
grid families (RFC §10') and the local-only friction counter (v0.3 §9.4):

* :func:`active_seconds` -- the INTERVAL UNION of work spans. RFC §23: "count
  active time by the union of work intervals, not the sum of nested or
  parallel spans". A task whose subagents ran in parallel is busy once, not
  twice; a child nested inside its parent adds nothing.
* :class:`TaskTimeRecord` + :func:`record_from_job` -- one record per settled
  ``task`` job (a subagent), reduced to closed enums: a workflow class folded
  from the BUILT-IN role name (a custom agent's name is user text and becomes
  ``other``), the terminal state, and -- for completed tasks only -- one
  duration BAND. Open tasks are never recorded; non-completed tasks carry no
  duration (the grid's disjoint ``terminal_state`` partition).
* :func:`scan_review_cycles` -- the review-cycle adapter over the harness's OWN
  ``code_requests`` disk cache (the one the ``code_requests`` tool and routes
  fill). It counts distinct agent-lane review ROUNDS from the parsed
  convention passes the cache already stores, never comment bodies. A change
  is recorded once, only if this install saw it OPEN after collection began
  and later saw it terminal ("never backfill pre-enrollment history"); a
  change with no parsed round is incomplete and not recorded ("unsupported
  signals are incomplete, not zero").
* the friction counter -- blocks per rule class per UTC day, 30-day rolling,
  never uploaded by anything; a confirm-each report may read its band.

REDACTION. Nothing here stores or emits a name, path, argument, prompt, forge
id, title or timestamp finer than the UTC day. The review adapter's dedup key
is a salted local hash of the change key that never leaves this file.

WHERE IT IS STORED. ``<config_dir>/guard/activity.sqlite3``, created only when
a producer records something under a live consent (``guard/producers.py``
decides that); constructing :class:`ActivityStore` does no I/O.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import secrets
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from ._sqlite import open_store

ACTIVITY_SCHEMA_VERSION: Final = 1

TASK_SCHEMA: Final = "guard_task_time_v1"
REVIEW_SCHEMA: Final = "guard_review_cycle_v1"

# -- task_time vocabulary (RFC §10' grid v2) --------------------------------
TERMINAL_DONE: Final = "done"
TERMINAL_CANCELLED: Final = "cancelled"
TERMINAL_DROPPED: Final = "dropped"
TERMINAL_BLOCKED: Final = "blocked"
TERMINAL_STATES: Final = frozenset(
    {TERMINAL_DONE, TERMINAL_CANCELLED, TERMINAL_DROPPED, TERMINAL_BLOCKED}
)

#: A job's status folded to the grid's terminal states. ``failed`` and
#: ``interrupted`` are work that ended without its result: ``dropped``.
#: ``running`` is absent on purpose -- open tasks are never released.
_JOB_TERMINAL: Final = {
    "completed": TERMINAL_DONE,
    "cancelled": TERMINAL_CANCELLED,
    "failed": TERMINAL_DROPPED,
    "interrupted": TERMINAL_DROPPED,
}

#: Built-in delegation roles (``agent`` tool). Any other agent name is
#: operator-authored text and folds to ``other``; no name is ever carried.
_WORKFLOW_BY_ROLE: Final = {
    "coder": "coder",
    "reviewer": "reviewer",
    "architect": "architect",
    "manager": "manager",
    "designer": "designer",
    "scout": "scout",
}
WORKFLOW_GENERAL: Final = "general"
WORKFLOW_OTHER: Final = "other"
WORKFLOW_CLASSES: Final = frozenset({*_WORKFLOW_BY_ROLE.values(), WORKFLOW_GENERAL, WORKFLOW_OTHER})

#: RFC §10' ``active_duration_band``; upper bounds in seconds, exclusive.
DURATION_BANDS: Final = (
    ("lt_1m", 60.0),
    ("1m_5m", 300.0),
    ("5m_15m", 900.0),
    ("15m_60m", 3600.0),
    ("1h_4h", 14400.0),
    ("gt_4h", float("inf")),
)
DURATION_BAND_NAMES: Final = frozenset(name for name, _ in DURATION_BANDS)

# -- review_cycles vocabulary ------------------------------------------------
REVIEW_OUTCOMES: Final = frozenset({"merged", "closed"})
ROUNDS_BANDS: Final = frozenset({"1", "2", "3", "4_plus"})
#: "source is an adapter attribute, not a dimension" (RFC §10').
REVIEW_SOURCE_HARNESS: Final = "harness_code_requests"

#: Friction is kept 30 days (v0.3 §9.4/§11 table), then pruned locally.
FRICTION_RETENTION_DAYS: Final = 30
FRICTION_BANDS: Final = ("0", "1_5", "6_20", "gt_20")

#: Event-family records live as long as a decision batch (§9.2': 72 h); a
#: record that could not be sent by then is discarded, never sent late.
EVENT_TTL_DAYS: Final = 3

#: Bound on cache entries one review scan reads. The code_requests cache keeps
#: one file per tracked change and sweeps at 30 days; 2000 is far above what
#: one install tracks and only bounds a pathological directory.
MAX_REVIEW_SCAN: Final = 2000


def _utc_day(at: float | None = None) -> str:
    moment = _dt.datetime.now(_dt.UTC) if at is None else _dt.datetime.fromtimestamp(at, _dt.UTC)
    return moment.date().isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Interval union
# ---------------------------------------------------------------------------


def interval_union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge ``[start, end)`` intervals; empty or inverted ones are ignored."""

    ordered = sorted((s, e) for s, e in intervals if e > s)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def active_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    """Total covered time of ``intervals``: nested/parallel spans count once."""

    return sum(end - start for start, end in interval_union(intervals))


def duration_band(seconds: float) -> str:
    for name, upper in DURATION_BANDS:
        if seconds < upper:
            return name
    return DURATION_BANDS[-1][0]


def rounds_band(rounds: int) -> str:
    if rounds >= 4:
        return "4_plus"
    return str(rounds)


def friction_band(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 5:
        return "1_5"
    if count <= 20:
        return "6_20"
    return "gt_20"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskTimeRecord:
    """One settled task, enum-only. ``active_duration_band`` only when ``done``."""

    workflow_class: str
    terminal_state: str
    active_duration_band: str | None
    day: str = field(default_factory=_utc_day)
    record_id: str = field(default_factory=_new_id)
    schema: str = TASK_SCHEMA

    def __post_init__(self) -> None:
        if self.workflow_class not in WORKFLOW_CLASSES:
            raise ValueError("unknown guard workflow class")
        if self.terminal_state not in TERMINAL_STATES:
            raise ValueError("unknown guard terminal state")
        if self.terminal_state == TERMINAL_DONE:
            if self.active_duration_band not in DURATION_BAND_NAMES:
                raise ValueError("a completed task needs a duration band")
        elif self.active_duration_band is not None:
            # The grid's non-completed partition has no duration dimension.
            raise ValueError("only a completed task carries a duration band")
        _dt.date.fromisoformat(self.day)

    def wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "record_id": self.record_id,
            "schema": self.schema,
            "day": self.day,
            "workflow_class": self.workflow_class,
            "terminal_state": self.terminal_state,
        }
        if self.active_duration_band is not None:
            out["active_duration_band"] = self.active_duration_band
        return out


@dataclass(frozen=True)
class ReviewCycleRecord:
    """One change that reached a terminal state, enum-only."""

    terminal_outcome: str
    rounds_band: str
    day: str = field(default_factory=_utc_day)
    record_id: str = field(default_factory=_new_id)
    source: str = REVIEW_SOURCE_HARNESS
    schema: str = REVIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.terminal_outcome not in REVIEW_OUTCOMES:
            raise ValueError("unknown guard review outcome")
        if self.rounds_band not in ROUNDS_BANDS:
            raise ValueError("unknown guard rounds band")
        if self.source != REVIEW_SOURCE_HARNESS:
            raise ValueError("unknown guard review source")
        _dt.date.fromisoformat(self.day)

    def wire(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "schema": self.schema,
            "day": self.day,
            "terminal_outcome": self.terminal_outcome,
            "rounds_band": self.rounds_band,
            "source": self.source,
        }


def workflow_class_of(agent_id: str | None) -> str:
    if not agent_id:
        return WORKFLOW_GENERAL
    return _WORKFLOW_BY_ROLE.get(agent_id.strip().lower(), WORKFLOW_OTHER)


def _job_children(job: Any) -> list[Any]:
    manager = getattr(job, "child_jobs", None)
    lister = getattr(manager, "list", None)
    if not callable(lister):
        return []
    try:
        children = lister()
        return list(children) if isinstance(children, Iterable) else []
    except Exception:  # noqa: BLE001 -- a child roster we cannot read adds no span
        return []


def _work_spans(job: Any, *, depth: int = 0) -> list[tuple[float, float]]:
    """Run intervals of ``job`` and its descendant jobs (bounded depth)."""

    spans: list[tuple[float, float]] = []
    started = getattr(job, "started_at", None)
    ended = getattr(job, "settled_at", None)
    if isinstance(started, (int, float)) and isinstance(ended, (int, float)):
        spans.append((float(started), float(ended)))
    if depth < 8:
        for child in _job_children(job):
            spans.extend(_work_spans(child, depth=depth + 1))
    return spans


def record_from_job(job: Any) -> TaskTimeRecord | None:
    """The record for one settled ``task`` job, or ``None`` (not a task / not terminal).

    Active time is the union of the job's own run window and every descendant
    job's run window; the queue wait before ``started_at`` is not active time.
    """

    if getattr(job, "type", None) != "task":
        return None
    terminal = _JOB_TERMINAL.get(str(getattr(job, "status", "")))
    if terminal is None:
        return None
    settled = getattr(job, "settled_at", None)
    day = _utc_day(float(settled)) if isinstance(settled, (int, float)) else _utc_day()
    band: str | None = None
    if terminal == TERMINAL_DONE:
        band = duration_band(active_seconds(_work_spans(job)))
    return TaskTimeRecord(
        workflow_class=workflow_class_of(getattr(job, "agent_id", None)),
        terminal_state=terminal,
        active_duration_band=band,
        day=day,
    )


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def activity_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / "activity.sqlite3"


_DDL: Final = (
    "CREATE TABLE meta (id INTEGER PRIMARY KEY CHECK (id = 1), high_water INTEGER NOT NULL,"
    " salt TEXT NOT NULL)",
    "INSERT INTO meta (id, high_water, salt) VALUES (1, 0, lower(hex(randomblob(16))))",
    "CREATE TABLE records (record_id TEXT PRIMARY KEY, family TEXT NOT NULL,"
    " day TEXT NOT NULL, body TEXT NOT NULL, seq INTEGER NOT NULL)",
    "CREATE INDEX records_family ON records (family, seq)",
    "CREATE TABLE review_seen (change_key TEXT PRIMARY KEY, state TEXT NOT NULL)",
    "CREATE TABLE friction (day TEXT NOT NULL, rule_class TEXT NOT NULL,"
    " count INTEGER NOT NULL, PRIMARY KEY (day, rule_class))",
)

FAMILY_TASK: Final = "task_time"
FAMILY_REVIEW: Final = "review_cycles"
FAMILIES: Final = (FAMILY_TASK, FAMILY_REVIEW)


class ActivityStore:
    """The local queue for ``task_time`` / ``review_cycles`` plus the friction counter."""

    def __init__(self, config_dir: Path | str) -> None:
        self._path = activity_path(config_dir)
        self._config_dir = Path(config_dir)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def _connect(self):  # -> sqlite3.Connection
        return open_store(self._path, version=ACTIVITY_SCHEMA_VERSION, ddl=_DDL)

    def append(self, record: TaskTimeRecord | ReviewCycleRecord) -> None:
        family = FAMILY_TASK if isinstance(record, TaskTimeRecord) else FAMILY_REVIEW
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE meta SET high_water = high_water + 1 WHERE id = 1")
            seq = conn.execute("SELECT high_water FROM meta WHERE id = 1").fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO records (record_id, family, day, body, seq)"
                " VALUES (?, ?, ?, ?, ?)",
                (record.record_id, family, record.day, json.dumps(record.wire()), seq),
            )
            conn.execute("COMMIT")
        finally:
            conn.close()

    def max_seq(self) -> int:
        if not self.exists():
            return 0
        conn = self._connect()
        try:
            return int(conn.execute("SELECT high_water FROM meta WHERE id = 1").fetchone()[0])
        finally:
            conn.close()

    def pending_after(
        self, family: str, seq_floor: int, *, before_day: str, limit: int = 500
    ) -> list[tuple[int, dict[str, Any]]]:
        """``(seq, wire body)`` past ``seq_floor`` on closed days (see the outbox twin)."""
        if not self.exists():
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT seq, body FROM records WHERE family = ? AND seq > ? AND day < ?"
                " ORDER BY seq LIMIT ?",
                (family, seq_floor, before_day, limit),
            ).fetchall()
            return [(int(r["seq"]), json.loads(r["body"])) for r in rows]
        finally:
            conn.close()

    def delete(self, record_ids: Iterable[str]) -> None:
        ids = list(record_ids)
        if not ids or not self.exists():
            return
        conn = self._connect()
        try:
            conn.executemany("DELETE FROM records WHERE record_id = ?", [(i,) for i in ids])
        finally:
            conn.close()

    def purge(self, family: str | None = None) -> None:
        """Drop queued records (all, or one family). The friction counter is kept."""
        if not self.exists():
            return
        conn = self._connect()
        try:
            if family is None:
                conn.execute("DELETE FROM records")
            else:
                conn.execute("DELETE FROM records WHERE family = ?", (family,))
        finally:
            conn.close()

    def prune(self, *, today: _dt.date) -> None:
        if not self.exists():
            return
        events = (today - _dt.timedelta(days=EVENT_TTL_DAYS)).isoformat()
        friction = (today - _dt.timedelta(days=FRICTION_RETENTION_DAYS)).isoformat()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM records WHERE day < ?", (events,))
            conn.execute("DELETE FROM friction WHERE day < ?", (friction,))
        finally:
            conn.close()

    # -- friction (local only) ------------------------------------------------

    def count_block(self, rule_class: str, *, day: str | None = None) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO friction (day, rule_class, count) VALUES (?, ?, 1)"
                " ON CONFLICT (day, rule_class) DO UPDATE SET count = count + 1",
                (day or _utc_day(), rule_class),
            )
        finally:
            conn.close()

    def friction(self, rule_class: str, *, today: _dt.date | None = None) -> str:
        """The 30-day rolling block-count band for ``rule_class`` (``0`` when none)."""
        if not self.exists():
            return "0"
        floor = (
            (today or _dt.datetime.now(_dt.UTC).date())
            - _dt.timedelta(days=FRICTION_RETENTION_DAYS)
        ).isoformat()
        conn = self._connect()
        try:
            total = conn.execute(
                "SELECT COALESCE(SUM(count), 0) FROM friction WHERE rule_class = ? AND day >= ?",
                (rule_class, floor),
            ).fetchone()[0]
        finally:
            conn.close()
        return friction_band(int(total))

    # -- review dedup -----------------------------------------------------------

    def _change_key(self, conn, key: str) -> str:
        salt = conn.execute("SELECT salt FROM meta WHERE id = 1").fetchone()[0]
        return hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()

    def scan_reviews(self, *, today: str | None = None) -> int:
        """Run the review-cycle adapter once; returns records appended."""
        return scan_review_cycles(self, self._config_dir, today=today)


# ---------------------------------------------------------------------------
# Review-cycle adapter
# ---------------------------------------------------------------------------

_OPEN: Final = frozenset({"open", "draft"})


def agent_rounds(entry: Mapping[str, Any]) -> int:
    """Distinct agent-lane REVIEW round numbers in a cache entry's stored parse.

    Reads ``entry["convention"]["pieces"]`` (``ReviewPass`` payloads the
    service stored at fetch time), never the bodies. Remediations, passes of
    the same round (``round 1 (fix verification)``), other lanes and passes
    with no round number add nothing: parallel reviewers and re-posts do not
    add rounds (RFC §23).
    """

    convention = entry.get("convention")
    pieces = convention.get("pieces") if isinstance(convention, Mapping) else None
    if not isinstance(pieces, Mapping):
        return 0
    seen: set[int] = set()
    for payloads in pieces.values():
        if not isinstance(payloads, list):
            continue
        for item in payloads:
            if not isinstance(item, Mapping):
                continue
            if item.get("lane") != "agent" or item.get("kind") != "review":
                continue
            number = item.get("round")
            if isinstance(number, int) and not isinstance(number, bool) and number > 0:
                seen.add(number)
    return len(seen)


def _cache_entries(config_dir: Path) -> Iterable[Mapping[str, Any]]:
    root = Path(config_dir) / "cache" / "code_requests"
    if not root.is_dir():
        return
    read = 0
    for host_dir in sorted(root.iterdir()):
        if not host_dir.is_dir() or host_dir.name.startswith("."):
            continue
        for path in sorted(host_dir.glob("*.json")):
            if read >= MAX_REVIEW_SCAN:
                return
            read += 1
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, Mapping):
                yield data


def scan_review_cycles(store: ActivityStore, config_dir: Path, *, today: str | None = None) -> int:
    """One pass of the adapter: track open changes, record those that went terminal."""

    entries = [
        e for e in _cache_entries(config_dir) if isinstance(e.get("key"), str) and e.get("state")
    ]
    if not entries:
        return 0
    appended = 0
    conn = store._connect()
    try:
        for entry in entries:
            state = str(entry.get("state"))
            key = store._change_key(conn, str(entry["key"]))
            row = conn.execute(
                "SELECT state FROM review_seen WHERE change_key = ?", (key,)
            ).fetchone()
            if row is None:
                # First sight. Open: track it. Already terminal: it finished
                # before collection could watch it -- never backfilled.
                seen = "tracking" if state in _OPEN else "skipped"
                conn.execute(
                    "INSERT INTO review_seen (change_key, state) VALUES (?, ?)", (key, seen)
                )
                continue
            if row["state"] != "tracking" or state not in REVIEW_OUTCOMES:
                continue
            rounds = agent_rounds(entry)
            conn.execute(
                "UPDATE review_seen SET state = ? WHERE change_key = ?",
                ("emitted" if rounds else "incomplete", key),
            )
            if not rounds:
                continue
            record = ReviewCycleRecord(
                terminal_outcome=state, rounds_band=rounds_band(rounds), day=today or _utc_day()
            )
            conn.execute("UPDATE meta SET high_water = high_water + 1 WHERE id = 1")
            seq = conn.execute("SELECT high_water FROM meta WHERE id = 1").fetchone()[0]
            conn.execute(
                "INSERT INTO records (record_id, family, day, body, seq) VALUES (?, ?, ?, ?, ?)",
                (record.record_id, FAMILY_REVIEW, record.day, json.dumps(record.wire()), seq),
            )
            appended += 1
    finally:
        conn.close()
    return appended


# ---------------------------------------------------------------------------
# The process-wide task sink (mirrors guard/decisions.py's sink)
# ---------------------------------------------------------------------------


class TaskSink(Protocol):
    def job_settled(self, job: Any) -> None: ...


_task_lock = threading.Lock()
_task_sink: TaskSink | None = None


def set_task_sink(sink: TaskSink | None) -> TaskSink | None:
    """Install ``sink`` process-wide (``None`` = off, the default); returns the previous."""

    global _task_sink
    with _task_lock:
        previous = _task_sink
        _task_sink = sink
    return previous


def get_task_sink() -> TaskSink | None:
    return _task_sink


def note_job_settled(job: Any) -> None:
    """Called from ``AsyncJobManager._settle``. A no-op unless a sink is installed.

    Never raises: timing telemetry must not break a job's settle.
    """

    sink = _task_sink
    if sink is None:
        return
    try:
        sink.job_settled(job)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # noqa: BLE001
        return


class StoreTaskSink:
    """Appends a :class:`TaskTimeRecord` per settled ``task`` job."""

    def __init__(self, store: ActivityStore) -> None:
        self._store = store

    def job_settled(self, job: Any) -> None:
        record = record_from_job(job)
        if record is not None:
            self._store.append(record)


def random_hex(n: int = 16) -> str:
    return secrets.token_hex(n)


__all__ = [
    "ACTIVITY_SCHEMA_VERSION",
    "DURATION_BANDS",
    "EVENT_TTL_DAYS",
    "FAMILIES",
    "FAMILY_REVIEW",
    "FAMILY_TASK",
    "FRICTION_BANDS",
    "ActivityStore",
    "ReviewCycleRecord",
    "StoreTaskSink",
    "TaskTimeRecord",
    "active_seconds",
    "activity_path",
    "agent_rounds",
    "duration_band",
    "friction_band",
    "interval_union",
    "note_job_settled",
    "record_from_job",
    "rounds_band",
    "scan_review_cycles",
    "set_task_sink",
    "workflow_class_of",
]
