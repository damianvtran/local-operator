"""The in-session monitor scheduler (contract §5, §11).

In-process only: no supervisor, no daemon, no cold engagement (§10.4 — a
session that goes cold has dormant monitors; they resume with it). The
scheduler owns the armed timer, the per-monitor runtime state, and the
tick → run → normalize → compare → classify → deliver loop. Everything with a
session in it — resolving the tool, executing the call, delivering the delta,
persisting the list — arrives as a callback, so this module is testable
without a session and the session is testable without timers.

The classify leg (§8) is one of those callbacks and the only OPTIONAL one:
absent, every detected change is delivered; present, its verdict runs the
§8.4 fork (deliver, or suppress + count), and a call that raises or answers
``None`` fails OPEN to a delivery. The call is issued OUTSIDE the write lock
and serialised across the session's monitors — see
:meth:`MonitorScheduler._classify_change`.

It preserves the ``WakeScheduler`` load-bearing properties because they were
paid for once already:

1. **A single armed asyncio timer**, re-armed after every pump, capped at
   ``MAX_ARM_MS`` so sleep/clock skew is absorbed by re-reading the wall clock.
2. **``dispose()`` cancels the armed handle and every in-flight check task** —
   a pending tick must never keep the event loop alive.
3. **``needs_rearm``** for construction without a running loop; the session's
   async init re-arms by calling ``pump()`` once.
4. **A write lock** around pump/update mutations so an arm landing inside a
   pump cannot be overwritten by the pump's stale snapshot.
5. **A delivery (or any tick-side failure) still advances** — a broken check
   must never become a hot loop.

Two ceilings, both contract numbers: a global semaphore of 2 bounds
simultaneous checks (a deferred check stays due and runs on the next pass —
not counted as skipped), and the failure ladder backs off
``every_ms × 2^(n-1)`` capped at 15 minutes before auto-disabling at
``maxConsecutiveFailures``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from local_operator.monitors import state as monitor_state
from local_operator.monitors.classify import (
    MonitorClassify,
    gate_state,
    suppressed_counter,
)
from local_operator.monitors.delivery import MonitorDelivery, MonitorNotice
from local_operator.monitors.diff import (
    beyond_window_text,
    content_hash,
    has_line_difference,
    is_pure_addition,
    is_record_log,
    legacy_masked,
    normalize,
    render_delta,
)
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec, next_monitor_seq, spec_identity

logger = logging.getLogger(__name__)

#: Never arm further out than this; long waits re-check the wall clock.
MAX_ARM_MS = 60_000
#: No zero-delay re-entry loop.
MIN_ARM_MS = 25
#: The first check after arm/resume runs within this window, so the baseline
#: is captured promptly and a broken spec surfaces as a health line early
#: (§5.4).
FIRST_CHECK_MIN_MS = 1_000
FIRST_CHECK_MAX_MS = 3_000
#: The failure ladder's ceiling (§11.3).
RETRY_CAP_MS = 900_000
#: Which counters key latches each notice kind once it has actually been SENT.
#: One table rather than two branches, because the rule is the same for both:
#: the latch records a DELIVERY, never an intention (review round 1, R2).
_NOTICE_LATCH: dict[str, str] = {
    "disabled": "disable_notified",
    "stalled": "unavailable_notified",
    "blind": "blind_notified",
    "failing": "failing_notified",
    "striking": "striking_notified",
    "empty": "empty_notified",
}

#: The delivery rate window (§9.4).
DELIVERY_WINDOW_MS = 3_600_000
#: The diff inputs are compared with ``difflib``'s quadratic matcher, which the
#: stored snapshot used to bound at ``snapshotMaxChars``. The full-body baseline
#: removes that implicit bound, so the matcher gets an explicit one: past either
#: limit the change is delivered as ``beyond_window_text`` (with ``hidden``
#: set) rather than diffed. Fail open, and never a multi-second stall inside a
#: check.
MAX_DIFF_CHARS = 1_048_576
MAX_DIFF_LINES = 8_000
#: How long a monitor's tool may stay out of reach before the operator is told
#: (§D3). One notice per episode, not one per tick: the point is that a stalled
#: watch stops being silent, not that it becomes noisy.
UNAVAILABLE_STALL_MS = 30 * 60_000
#: How long "out of reach" may last before the monitor is disabled (§D3). This
#: is how "genuinely gone" is settled without a strike count: a server that is
#: reconnecting comes back, one that is uninstalled does not.
UNAVAILABLE_GONE_MS = 24 * 3_600_000
#: Consecutive failing checks that earn the operator a notice, ONCE THE WATCH HAS
#: SUCCEEDED AT LEAST ONCE (D3). A failure that persists this many consecutive
#: checks is not a blip: the incident behind this rule ran 18 identical failures
#: over 90 minutes (a ``gh pr view`` whose tick cwd was not a git repository) and
#: never said a word, because bash reports a non-zero exit as an ordinary
#: SUCCESS result and two identical error texts diff to "no change".
#:
#: N=2, not N=1: a single failure of a watch that has worked is exactly what a
#: transient network blip, a gh rate limit or a restarting server looks like, and
#: paging on it would replace silence with noise — the reviewer's named attack on
#: this fix. Two consecutive failures cost one interval of doubt (5 minutes for
#: the reported watch) and no real change stays hidden through them.
FAILING_CHECKS_AFTER_BASELINE = 2
#: Consecutive successful-but-empty results before the operator is told the call
#: returns nothing (D4). Before any baseline there is no second strike: a watch
#: that has NEVER worked is broken on its first failed check, not blipping — the
#: incident's very first tick already failed, and its very first tick is when the
#: operator needed to know.
FAILING_CHECKS_WITHOUT_BASELINE = 1
#: Consecutive empty results that earn the "the call returns nothing" notice.
#: Empty output is NOT a failure (a ``grep`` with no match, a file still being
#: written): it is only a signal once the call has never produced anything, and
#: three checks is enough to tell "quiet right now" from "watching nothing".
EMPTY_CHECKS_BEFORE_NOTICE = 3
#: How long a ``failing`` episode may continue before the watch is disabled (D3).
#: A failing call charges NO strike (see ``_apply_failing``), so "genuinely gone"
#: has to be settled some other way, and the ``unavailable`` episode next door
#: already settled it this way: a resource that is missing right now comes back,
#: one that has been missing for a day does not. 24 h is that same horizon, for
#: the same reason, and it replaces the five-strike disable that killed a watch
#: on a not-yet-existing file after ~25 minutes of a legitimate wait (review
#: round 1, BLOCKER-1).
FAILING_GONE_MS = 24 * 3_600_000
#: Consecutive ORDINARY failures that earn the operator a first notice (§D3).
#:
#: THE OTHER CLASS OF FAULT, and it must not be confused with ``failing`` above.
#: A timeout, a tool error or anything else without a ``kind`` is a check that
#: really is heading for the five-strike disable — the fault is in the watch, not
#: in a resource that may still appear — so it keeps charging strikes. What it
#: must not do is stay silent for all five: "35 minutes of nothing and then dead"
#: is exactly the hiding shape the operator named. So it earns ONE notice at the
#: second consecutive failure, three strikes before the disable it announces.
#: Two, not one, for the reason ``FAILING_CHECKS_AFTER_BASELINE`` gives: one
#: transient blip is not news, and no real change stays hidden through two.
STRIKING_CHECKS_BEFORE_NOTICE = 2


def _skipped_checks_since(last_check_at: int, every_ms: int, now_ms: int) -> int:
    """Checks that came due and never ran, counted from ``last_check_at``.

    Rounded to the nearest interval so jitter and a slightly-late resume do
    not invent a skip, minus the one the current check itself supersedes — a
    resume whose only lapse is "this check is a touch late" reports zero
    (contract §9.3; the count is named once, in the first delivery).
    """
    if last_check_at <= 0 or every_ms <= 0:
        return 0
    gap = now_ms - last_check_at
    if gap <= 0:
        return 0
    intervals = (gap + every_ms // 2) // every_ms
    return max(0, intervals - 1)


def _read_spill_body(handle: str) -> str | None:
    """The full text an earlier check spilled, or ``None`` when unusable.

    A missing/evicted entry and an entry stored ``complete=False`` are both
    unusable: neither reproduces the whole result, and a partial body could
    hide the very change the diff exists to show.
    """
    from local_operator.tools.spill import get_store

    store = get_store()
    meta = store.stat(handle)
    if meta is None or not meta.complete:
        return None
    loaded = store.read_lines(handle, 1, None)
    if loaded is None:
        return None
    lines, _total = loaded
    return "\n".join(lines)


class CheckOutcome(TypedDict, total=False):
    """What one check produced: model-visible text, or a failure reason.

    ``kind`` is the runner's classification of a failure, and it exists so the
    ladder can tell "the tool is not reachable right now" from "the tool
    returned an error". Absent means an ordinary error and keeps the strike
    policy unchanged: five consecutive failures disable the monitor.

    - ``unavailable``: the call never ran. No strike is charged (a reconnect
      window must not disable a monitor), it is retried on the same backoff
      ladder, and a stall notice goes out after ``UNAVAILABLE_STALL_MS``;
    - ``blind``: the call RAN but its result could not be compared honestly —
      the spill that held the full body is gone, or it was stored incompletely.
      No strike is charged and the baseline is NOT advanced (advancing it would
      make the missed change unrecoverable); ``error`` carries the sentence the
      operator is shown, and the blind notice goes out once per episode;
    - ``fatal``: the failure is deterministic and cannot self-heal (the tool's
      own schema rejected the arguments). Disabled on the first occurrence.
    - ``failing``: the call RAN and the tool said it could not do the job — a
      non-zero exit with a complaint on stderr, which ``bash`` reports as an
      ordinary success result (see ``execute_bash``'s ``exit_code``/
      ``stderr_head`` details and ``Session._run_monitor_check``). It charges NO
      strike: it keeps the capped backoff, stays armed, earns ONE latched
      ``failing`` notice per episode (on the first failure if the watch has no
      baseline yet, on the second consecutive one if it has — see
      ``FAILING_CHECKS_AFTER_BASELINE``), and is disabled only after
      ``FAILING_GONE_MS`` of continuous failure (see ``_apply_failing``). A
      TIMEOUT is deliberately NOT this kind: the check itself cannot complete, so
      it belongs to the ordinary ladder (which now speaks once per episode — see
      ``STRIKING_CHECKS_BEFORE_NOTICE``), because that watch really is heading for
      the disable.
    """

    text: str | None
    error: str | None
    kind: str | None
    #: The result carried NO comparable output — an empty body, or (for bash) a
    #: command that printed nothing at all. Not a failure: see
    #: ``EMPTY_CHECKS_BEFORE_NOTICE``.
    empty: bool | None
    #: The structured process status the tool published, when it has one (bash).
    #: Read instead of parsing an ``exit code: N`` head line out of the result.
    exit_code: int | None
    #: The ``spill://`` handle whose full body became ``text``, when the result
    #: overflowed the tool-output limit. The scheduler keeps it as the next
    #: diff's baseline: a snapshot above ``snapshotMaxChars`` is a clipped
    #: prefix, and diffing against that prefix hides a change past the cut.
    handle: str | None


@dataclass
class _Entry:
    """One monitor's live runtime: its spec plus its counters/health."""

    spec: MonitorSpec
    counters: dict[str, Any]
    generation: int = 0


@dataclass(frozen=True)
class _Change:
    """One detected change, between the diff and the gate's verdict (§8).

    :meth:`MonitorScheduler._apply_success` commits the diff state machine
    and hands this back; the scheduler then classifies it OUTSIDE the write
    lock (a network call may not hold a lock that the pump and the arm flow
    need) and settles the outcome in :meth:`MonitorScheduler._settle_change`.
    The fields are snapshots taken under the lock, and the spec rides along
    so a settle that lands after an edit still names the row it settled.
    """

    spec: MonitorSpec
    changes: int
    delta_text: str
    skipped: int
    checks: int
    at_ms: int
    #: The delta only ADDED content (an appended log line, a new row). Such a
    #: change is delivered without asking the gate — see
    #: :func:`~local_operator.monitors.diff.is_pure_addition` for why.
    pure_addition: bool = False
    #: The rendered previews do not show what changed (see
    #: :attr:`~local_operator.monitors.diff.DeltaRender.hidden`). The gate is
    #: skipped: a model cannot judge a change the delta does not show, and the
    #: live failure was the gate calling the only visible difference (a spill
    #: footer's char count) non-material metadata.
    hidden: bool = False
    #: The watched result is a RECORD LOG (see
    #: :func:`~local_operator.monitors.diff.is_record_log`), so the record view
    #: has already separated records from bookkeeping: the gate is skipped.
    record_view: bool = False


@dataclass
class MonitorRuntime:
    """Read-only view of one monitor for list surfaces (tests, TUI later)."""

    spec: MonitorSpec
    counters: dict[str, Any] = field(default_factory=dict)


def fresh_counters(monitor_id: str, next_due_at: int | None) -> dict[str, Any]:
    """A new monitor's counters/health file content (§10.3 shape)."""
    return {
        "schema": 1,
        "monitor_id": monitor_id,
        "content_hash": "",
        "last_check_at": 0,
        "last_change_at": 0,
        "next_due_at": next_due_at,
        "checks": 0,
        "deliveries": 0,
        "suppressed": {"non_material_metadata": 0, "ignorable": 0, "rate_cap": 0},
        "rate_window_start": 0,
        "rate_window_count": 0,
        "rate_cap_held": 0,
        "consecutive_failures": 0,
        "disabled": False,
        "disabled_reason": "",
        #: WHICH ladder produced the disable — ``checks`` (the strike ladder),
        #: ``fatal`` (a deterministic failure, first occurrence) or
        #: ``unreachable`` (24 hours out of reach, no strike charged). The
        #: notice's wording depends on it, and a retro-announce reads it off an
        #: older file that has no key, hence the empty default.
        "disabled_kind": "",
        "last_error": "",
        "last_note": "",
        "skipped_overlap": 0,
        # The unavailable-episode fields (§D3). Absent from an older file and
        # read through ``.get`` everywhere, so the schema does not move: a
        # monitor loaded from a pre-upgrade counters file simply has no episode
        # in progress.
        "unavailable_since": 0,
        "unavailable_ticks": 0,
        "unavailable_notified": False,
        # The blind-episode fields: the call runs, but its full result is not
        # comparable (a spill gone or stored incompletely). While set, the
        # baseline does not advance and the row must never render as healthy.
        "blind": False,
        "blind_reason": "",
        "blind_notified": False,
        # The disable notice's exactly-once latch (§D4), set only after the
        # notice is handed to the session — so a crash between the disable and
        # the announce still tells the operator on the next open.
        "disable_notified": False,
    }


class MonitorScheduler:
    """Owns the monitor specs, their runtime state, and one armed asyncio timer."""

    def __init__(
        self,
        *,
        now: Callable[[], int],
        config_dir: Path,
        session_id: str,
        settings: MonitorSettings,
        validate: Callable[[str, Mapping[str, Any]], str | None],
        run_check: Callable[[MonitorSpec], Awaitable[CheckOutcome]],
        deliver: Callable[[MonitorDelivery], Awaitable[None] | None],
        persist: Callable[[list[MonitorSpec]], Awaitable[None] | None],
        on_change: Callable[[], None] | None = None,
        index_writable: Callable[[], bool] | None = None,
        #: The §8 classifier gate: bounded delta → the materiality class, or
        #: ``None`` for "no classifier". Absent (``None``) means every change
        #: is delivered — the scheduler never requires a classifier, and the
        #: gate failing OPEN is the contract (§8.4), not a fallback.
        classify: MonitorClassify | None = None,
        #: The lifecycle-notice sink (§D4). Absent (``None``) means a scheduler
        #: with no session behind it: notices are then not sent, and the
        #: ``disable_notified`` latch is NOT set — so the notice goes out the
        #: first time a scheduler that has one runs. Every existing test
        #: construction keeps working unchanged.
        announce: Callable[[MonitorNotice], Awaitable[None] | None] | None = None,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._now = now
        self._config_dir = Path(config_dir)
        self._session_id = session_id
        self._settings = settings
        self._validate = validate
        self._check_runner = run_check
        self._deliver = deliver
        self._persist = persist
        self._on_change = on_change
        self._index_writable = index_writable
        self._classify = classify
        self._announce = announce
        self._uniform = uniform

        self._entries: dict[str, _Entry] = {}
        self._next_seq = 1
        self._timer: asyncio.TimerHandle | None = None
        # Every timer-created tick and every spawned check is tracked, so
        # dispose() cancels all of them (the wake scheduler's _tick_tasks
        # rationale: a re-arm must not orphan a previous pump).
        # ``set[Task[Any]]``, not ``Task[None]``: the tick task wraps pump()
        # (an int) and the check tasks wrap _run_check (None) — disposal only
        # ever cancels them.
        self._tick_tasks: set[asyncio.Task[Any]] = set()
        self._check_tasks: set[asyncio.Task[Any]] = set()
        self._inflight: set[str] = set()
        # pump()/update() mutate the entry table across await points; without
        # mutual exclusion an arm landing inside a pump is overwritten by the
        # pump's pre-update snapshot.
        self._write_lock = asyncio.Lock()
        #: Serialises gate calls across the session's monitors (§8.1: "calls
        #: are issued sequentially within the pump pass" — hits are rare, and
        #: the cascade's cheapest leg has a documented 0.5 req/s limit). Held
        #: WITHOUT the write lock: a vendor call may take ``timeoutMs``, and a
        #: network call inside the write lock would stall the pump and every
        #: arm landing behind it.
        self._classify_lock = asyncio.Lock()
        # Bounds simultaneous checks across the session's monitors (the wake
        # supervisor's concurrency ceiling). A deferred check stays due.
        self._sem = asyncio.Semaphore(2)
        self._disposed = False
        #: Set when ``_arm`` runs without a running loop; the session's async
        #: init re-arms by calling ``pump()`` once.
        self.needs_rearm = False

    # -- read surface -------------------------------------------------------

    @property
    def monitors(self) -> tuple[MonitorSpec, ...]:
        return tuple(entry.spec for entry in self._ordered_entries())

    @property
    def settings(self) -> MonitorSettings:
        return self._settings

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def next_seq(self) -> int:
        """The next monitor sequence number (persisted beside the rows)."""
        return self._next_seq

    def runtime(self, monitor_id: str) -> MonitorRuntime | None:
        entry = self._entries.get(monitor_id)
        if entry is None:
            return None
        return MonitorRuntime(spec=entry.spec, counters=dict(entry.counters))

    def next_monitor_due_at(self) -> int | None:
        """Earliest ``next_due_at`` across ARMED monitors, or ``None``.

        Disabled and expired rows are excluded: they cannot fire, and the
        pristine probe must not pin a runtime on a schedule that will not run.
        """
        now = self._now()
        due: int | None = None
        for entry in self._entries.values():
            if not self._is_active(entry, now):
                continue
            candidate = entry.counters.get("next_due_at")
            if not isinstance(candidate, int) or isinstance(candidate, bool):
                continue
            if due is None or candidate < due:
                due = candidate
        return due

    def index_rows(self) -> list[dict[str, Any]]:
        """The rows the session writes into the derived index (§10.2)."""
        rows: list[dict[str, Any]] = []
        for entry in self._ordered_entries():
            counters = entry.counters
            rows.append(
                {
                    **entry.spec.model_dump(),
                    "next_due_at": counters.get("next_due_at"),
                    "last_check_at": counters.get("last_check_at", 0),
                    "checks": counters.get("checks", 0),
                    "deliveries": counters.get("deliveries", 0),
                    "consecutive_failures": counters.get("consecutive_failures", 0),
                    "disabled": bool(counters.get("disabled")),
                    "disabled_reason": counters.get("disabled_reason", ""),
                    # The health fields every surface derives a hint from
                    # (§D6). ``arm._compose_index_rows`` composes the same key
                    # set for an external arm — the two must not drift.
                    "unavailable_since": counters.get("unavailable_since", 0),
                    "last_error": counters.get("last_error", ""),
                    # The suppression counts and the blind state ride the row
                    # because ``store.health_hint`` is rendered from the INDEX
                    # by readers with no counters file (the CLI, the desktop
                    # route, the TUI band); a row that dropped them could only
                    # ever say "nothing has changed".
                    "suppressed": (
                        dict(counters["suppressed"])
                        if isinstance(counters.get("suppressed"), Mapping)
                        else {}
                    ),
                    "blind": bool(counters.get("blind")),
                    "blind_reason": str(counters.get("blind_reason") or ""),
                    # A row in a FAILING episode must not render as a quiet one:
                    # ``store.health_hint`` turns these two into "WATCH IS
                    # FAILING: …" for every surface that has no counters file
                    # open (D8). ``empty_checks`` is the "returns nothing"
                    # counterpart.
                    "failing": bool(counters.get("failing")),
                    # ...and HOW FAR INTO the episode it is, with the policy's
                    # own threshold beside it. The reader needs both to tell a
                    # single failed check (deliberately not paged) from an
                    # episode that has reached the threshold and was announced
                    # — and it must not re-derive the threshold from
                    # ``consecutive_failures``, which a failing tick no longer
                    # charges (QA round 1, Q3; review round 1, BLOCKER-1).
                    "failing_checks": counters.get("failing_ticks", 0),
                    "failing_notice_checks": counters.get("failing_notice_checks", 0),
                    # The strike-class counterpart: how many consecutive failures
                    # the ordinary ladder has, and the limit it is heading for. A
                    # reader needs both to say "one failed check, below the
                    # threshold" instead of claiming the watch is failing (QA
                    # round 1, Q3), and to name the disable it is heading for.
                    "failure_limit": counters.get("failure_limit", 0),
                    "empty_checks": counters.get("empty_checks", 0),
                }
            )
        return rows

    # -- lifecycle ----------------------------------------------------------

    def load(self, specs: Sequence[MonitorSpec], *, next_seq: Any = None) -> None:
        """Adopt persisted specs at session open and arm.

        Runtime state (next due, failures, disabled, rate window) comes from
        each monitor's counters file; an absent/unreadable one is rebuilt —
        and an overdue `next_due_at` (the process was down) re-arms within the
        first-check window, which is what makes the resume delivery land
        promptly instead of a full interval later.
        """
        now = self._now()
        adopted: list[_Entry] = []
        for raw in specs:
            try:
                # Persisted rows are untrusted input (a hand-edited transcript
                # can carry values the field constraints reject): drop, don't
                # let a later tick die (the wake load() contract).
                spec = MonitorSpec.model_validate(raw.model_dump())
            except Exception:
                logger.warning("dropping invalid monitor spec %r", getattr(raw, "id", raw))
                continue
            counters = monitor_state.read_counters(self._config_dir, self._session_id, spec.id)
            if counters is None or counters.get("monitor_id") != spec.id:
                counters = fresh_counters(spec.id, None)
            next_due = counters.get("next_due_at")
            if (
                not counters.get("disabled")
                and (not isinstance(next_due, int) or next_due <= now)
                and (spec.until_at is None or now < spec.until_at)
            ):
                next_due = now + self._first_check_delay_ms()
            counters["next_due_at"] = next_due
            adopted.append(_Entry(spec=spec, counters=counters))
        adopted.sort(key=lambda entry: entry.spec.created_at)
        self._entries = {entry.spec.id: entry for entry in adopted}
        self._next_seq = next_monitor_seq([entry.spec.id for entry in adopted], next_seq)
        self._arm()

    async def update(self, monitors: list[MonitorSpec]) -> None:
        """Caller-driven full-list update: persist, then re-arm and notify.

        The parameter is named ``monitors`` because the protocol (and the
        ``WakeScheduler`` precedent) name it that way: pyright treats a
        protocol's parameter NAME as part of conformance, and a keyword-call
        mismatch is a real defect, not a lint.
        """
        async with self._write_lock:
            now = self._now()
            synced: dict[str, _Entry] = {}
            for spec in sorted(monitors, key=lambda item: item.created_at):
                existing = self._entries.get(spec.id)
                if existing is not None:
                    existing.spec = spec.model_copy(deep=True)
                    synced[spec.id] = existing
                else:
                    synced[spec.id] = _Entry(
                        spec=spec.model_copy(deep=True),
                        counters=fresh_counters(spec.id, now + self._first_check_delay_ms()),
                    )
            for removed_id in set(self._entries) - set(synced):
                monitor_state.remove_monitor_state(self._config_dir, self._session_id, removed_id)
            self._entries = synced
            self._next_seq = max(
                self._next_seq, next_monitor_seq([entry.spec.id for entry in synced.values()], 0)
            )
            try:
                await self._maybe_await(self._persist([entry.spec for entry in synced.values()]))
            except Exception:
                # The rows are already live in memory; a failed persist (disk
                # full, transcript I/O) must not kill the scheduler — the next
                # successful write re-records them.
                logger.warning("monitor persist failed", exc_info=True)
            self._arm()
            self._notify_change()

    # -- the tool's create/cancel flows ------------------------------------

    async def create(self, request: Mapping[str, Any], *, cwd: str) -> dict[str, Any]:
        """The whole arm flow; returns a typed outcome for the tool to phrase.

        Order is load-bearing: shape+read-only validation first (so a bad
        request is a sentence, never an armed half-monitor), then dedupe
        (an identical spec is an answer, not an error), then the cap, then
        the storm guard (a third monitor with one name).
        """
        from local_operator.monitors.spec import build_monitor_spec

        now = self._now()
        if self._index_writable is not None and not self._index_writable():
            return {
                "error": (
                    "the monitor index cannot be written right now (the last write "
                    "failed); arming now would lose the monitor. Fix the config "
                    "directory and retry."
                ),
                "malformed": False,
            }

        entries = list(self._entries.values())
        # The id is computed (not yet issued) so an omitted name can derive
        # from it; the sequence only advances when the arm actually lands
        # below, so a failed request never issues an id — and an issued id is
        # never reused (the high-water rule).
        monitor_id = f"m{self._next_seq}"
        outcome = build_monitor_spec(
            request,
            monitor_id=monitor_id,
            now_ms=now,
            settings=self._settings,
            cwd=cwd,
            validate=self._validate,
        )
        if "error" in outcome:
            # TypedDict → dict: the tool-facing return type is deliberately the
            # loose mapping, and a TypedDict is not assignable to dict[str, Any].
            return dict(outcome)

        spec = outcome["spec"]
        identity = spec_identity(spec.tool, spec.arguments)
        for entry in entries:
            if spec_identity(entry.spec.tool, entry.spec.arguments) != identity:
                continue
            if entry.counters.get("disabled"):
                # Reactivate: same call, reset failures, keep the snapshot
                # blob so the first check still diffs against the old baseline.
                # The fresh counters are written to disk HERE, before the
                # update: a reactivation that only reset memory would be
                # undone by the next process's load(), which reads the
                # disabled counters back and keeps the monitor dark.
                entry.counters = fresh_counters(entry.spec.id, now + self._first_check_delay_ms())
                entry.generation += 1
                self._write_counters(entry)
                await self.update(list(self.monitors))
                return {
                    "reactivated": True,
                    "spec": entry.spec,
                    # §4.5: the instant rides the outcome as structured
                    # details — the receipt's next_due_at is never null
                    # (round-1 review F4).
                    "next_due_at": entry.counters.get("next_due_at"),
                }
            # The due instant rides the outcome like the created/reactivated
            # branches' (§4.5): the watch this arm found IS armed, and a
            # receipt that rendered it as due-less would say otherwise.
            return {
                "duplicate": True,
                "spec": entry.spec,
                "next_due_at": entry.counters.get("next_due_at"),
            }

        if len(entries) >= self._settings.max_monitors:
            return {
                "error": (
                    f"monitor limit reached ({self._settings.max_monitors} per session) — "
                    "cancel one first (monitor list)."
                ),
                "malformed": False,
            }
        same_name = sum(1 for entry in entries if entry.spec.name == spec.name)
        if same_name >= 2:
            return {
                "error": (
                    f"three monitors named '{spec.name}' is a storm — cancel one or "
                    "use a distinct name."
                ),
                "malformed": False,
            }

        entry = _Entry(
            spec=spec, counters=fresh_counters(spec.id, now + self._first_check_delay_ms())
        )
        # D6 — VALIDATE ON ARM: run the watched call ONCE, here, before the index
        # write makes the watch live.
        #
        # WHY at arm and not only on the first tick: the incident armed a watch
        # whose call could never run in the directory ticks use, and nothing said
        # so until the operator noticed 90 minutes of silence. The probe turns
        # "18 identical failing ticks" into one sentence in the ARM RECEIPT.
        #
        # WHY it does not REFUSE on failure: a watch on a resource that does not
        # exist yet (a PR not opened, a file not written, a queue still empty) is
        # a legitimate watch, and refusing it would delete the feature's most
        # common use. Only what can NEVER work is refused — an invalid cwd or a
        # tool that is gone are refused by the two gates above, as sentences.
        #
        # WHY it is opt-in per request: the probe is a real call, so it runs only
        # when the caller asked for it — the ``monitor`` tool, whose call the
        # operator has just approved. A programmatic arm keeps its pre-probe
        # behaviour exactly (one scheduled check, no call at arm).
        probe: dict[str, Any] | None = None
        if request.get("probe"):
            probe = await self._probe_baseline(spec)
        self._entries[spec.id] = entry
        self._next_seq += 1
        baseline_adopted = probe is not None and "error" not in probe
        if baseline_adopted and probe is not None:
            # ONE call, not two: the probe's result IS the baseline, adopted
            # through the same state machine the first scheduled check would have
            # used — the first-ever result is baseline establishment, never a
            # delivery, so nothing is reported for it either way.
            self._apply_success(
                entry, str(probe.get("text") or ""), now, handle=probe.get("handle")
            )
        await self.update(list(self.monitors))
        result: dict[str, Any] = {
            "created": True,
            "spec": spec,
            # The first check's instant, for the receipt's structured
            # details (§4.5; round-1 review F4).
            "next_due_at": entry.counters.get("next_due_at"),
            "baseline_adopted": baseline_adopted,
        }
        if probe is not None and "error" in probe:
            # The receipt names the failure LOUDLY (``_monitor_create`` leads its
            # text with it); the watch is armed anyway.
            result["baseline_error"] = str(probe["error"])
            # ...and the PUBLISHED ROW says so too, or `lop monitor status` (and
            # every other cold reader) shows a healthy row for up to a full
            # interval after an arm that already failed (QA round 1, Q3).
            # ``unavailable``/``blind``/``fatal`` probes are NOT opened here: their
            # policies need a scheduled tick's accounting to stay honest, and a
            # probe is not one.
            probe_kind = str(probe.get("kind") or "")
            if probe_kind not in ("unavailable", "blind", "fatal"):
                self._record_arm_probe_failure(
                    entry, str(probe["error"]), now, kind=probe_kind or None
                )
        return result

    def _record_arm_probe_failure(
        self, entry: _Entry, error: str, now: int, *, kind: str | None = None
    ) -> None:
        """Open a failure episode from the arm-time probe, without announcing it.

        The probe is a REAL call and it failed, so the row must not read healthy
        until the next scheduled check. What it must NOT do is announce: the arm
        receipt already carries the failure, and announcing here would say the
        same thing twice in one turn.

        BOTH failure classes are recorded, because both are real failed calls and
        the row owes the reader the truth in the gap before tick 1 (QA round 1,
        Q3):

        - ``failing``: opens the episode as its FIRST tick and leaves the latch
          unset, so the first scheduled check is tick 2, reaches the threshold (1
          with no baseline) and announces exactly once — as it would have without
          the probe;
        - anything else: ONE strike, because that class really is heading for the
          disable, and the count is published so the row says "1 failed check,
          below the threshold" instead of "nothing has changed".

        ``next_due_at`` is deliberately NOT touched: the probe is not a scheduled
        check, and the receipt has already promised when the first one runs.
        """
        counters = entry.counters
        counters["checks"] = int(counters.get("checks") or 0) + 1
        counters["last_error"] = str(error)[:500]
        if kind == "failing":
            counters["failing"] = True
            counters["failing_since"] = now
            counters["failing_ticks"] = 1
            counters["failing_notice_checks"] = FAILING_CHECKS_WITHOUT_BASELINE
            counters["last_note"] = "the watch's call already failed at arm"
            return
        counters["consecutive_failures"] = int(counters.get("consecutive_failures") or 0) + 1
        counters["failure_limit"] = self._settings.max_consecutive_failures
        counters["last_note"] = "a check already failed at arm"

    async def _probe_baseline(self, spec: MonitorSpec) -> dict[str, Any]:
        """Run one arm-time check; ``{"error": …}`` or ``{"text": …, "handle": …}``.

        Bounded by the same timeout the scheduled checks use, so an arm can never
        hang on a slow call: the probe is a convenience at the moment of arming,
        not a gate the operator has to wait behind. Every failure shape the check
        runner can produce is turned into the ``error`` form — the caller needs
        only "did it work" and "what did it say".
        """
        timeout_s = self._settings.run_timeout_ms / 1000.0
        try:
            outcome = await asyncio.wait_for(self._check_runner(spec), timeout=timeout_s)
        except asyncio.TimeoutError:
            return {"error": f"its first check timed out after {int(timeout_s)}s"}
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — an arm must not fail on its probe
            return {"error": f"{type(exc).__name__}: {exc}"}
        error = outcome.get("error")
        if error:
            # THE CLASSIFICATION RIDES ALONG (QA round 1, Q3). The caller records a
            # ``failing`` probe as the episode's first tick so the published row
            # does not read healthy until the next scheduled check — and must not
            # do that for an ``unavailable``/``fatal``/``blind`` probe, whose own
            # policies already say the right thing in the counters they set.
            classified: dict[str, Any] = {"error": str(error)}
            if outcome.get("kind"):
                # ``.get``, not ``[]``: CheckOutcome is ``total=False``, so the key
                # is optional by contract and pyright is right to refuse the item
                # access.
                classified["kind"] = str(outcome.get("kind"))
            return classified
        return {"text": outcome.get("text") or "", "handle": outcome.get("handle") or None}

    async def cancel(self, monitor_id: str) -> dict[str, Any]:
        """Remove one monitor; the shared update persists the shrunken list."""
        entry = self._entries.get(monitor_id)
        if entry is None:
            return {
                "error": (
                    f"No monitor with id '{monitor_id}' "
                    f"(known: {', '.join(sorted(self._entries)) or 'none'})"
                ),
                "malformed": False,
            }
        remaining = [spec for spec in self.monitors if spec.id != monitor_id]
        await self.update(remaining)
        return {"cancelled": monitor_id, "spec": entry.spec}

    async def retry_unavailable(self, now_ms: int | None = None) -> int:
        """Pull every UNAVAILABLE monitor's next check forward; returns how many.

        WHY: an MCP-backed monitor whose server is still connecting at boot
        fails its first check as "unavailable" and backs off — up to
        ``RETRY_CAP_MS`` (15 min). Without this, a cold resume that loses the
        race to its own MCP handshake leaves the watch dark for that whole
        backoff although the tool arrived seconds later. The session's MCP
        ``on_tools_changed`` hook (``Session.refresh_tools``) calls this, so the
        next check runs within ``FIRST_CHECK_MAX_MS`` of the tools appearing.

        CONSTRAINTS: only ACTIVE rows with ``unavailable_since`` set and no
        check in flight move; healthy rows are untouched, and the episode
        counters (``unavailable_since``/``unavailable_ticks``) are NOT reset —
        a check that succeeds clears them itself (``_restored_notice``), and
        one that fails again must keep counting toward the stall/gone notices.
        Under ``_write_lock`` like every other entry-table mutation; a no-op
        moves nothing and does not pump.
        """
        if self._disposed:
            return 0
        moved = 0
        async with self._write_lock:
            now = now_ms if now_ms is not None else self._now()
            for entry in self._entries.values():
                if not self._is_active(entry, now) or entry.spec.id in self._inflight:
                    continue
                if not entry.counters.get("unavailable_since"):
                    continue
                entry.counters["next_due_at"] = now + self._first_check_delay_ms()
                self._write_counters(entry)
                moved += 1
        if moved:
            await self.pump(now_ms)
        return moved

    # -- the tick -----------------------------------------------------------

    async def pump(self, now_ms: int | None = None) -> int:
        """Start every due check the semaphore admits; returns how many started.

        Called by the armed timer and by the session's async init (the
        ``needs_rearm`` path). Each started check is a task; the pump itself
        never awaits a check, so a slow call cannot stall an arm landing
        behind it.
        """
        if self._disposed:
            return 0
        started = 0
        async with self._write_lock:
            now = now_ms if now_ms is not None else self._now()
            due = [entry for entry in self._entries.values() if self._is_due(entry, now)]
            due.sort(key=lambda entry: entry.counters.get("next_due_at") or 0)
            loop = asyncio.get_running_loop()
            for entry in due:
                if entry.spec.id in self._inflight:
                    # No overlap, ever: the tick is skipped and counted; the
                    # backoff keeps a slow check from making this a spin.
                    entry.counters["skipped_overlap"] = (
                        int(entry.counters.get("skipped_overlap") or 0) + 1
                    )
                    entry.counters["next_due_at"] = now + min(entry.spec.every_ms, 15_000)
                    self._write_counters(entry)
                    logger.debug("monitor %s skipped: previous check still running", entry.spec.id)
                    continue
                if self._sem.locked():
                    # Deferred, NOT skipped: the check stays due and runs on
                    # the next pass with a free slot.
                    continue
                self._inflight.add(entry.spec.id)
                task = loop.create_task(self._run_check(entry.spec.id, entry.generation))
                self._check_tasks.add(task)
                task.add_done_callback(self._check_tasks.discard)
                started += 1
            self._arm()
        return started

    async def _run_check(self, monitor_id: str, generation: int) -> None:
        async with self._sem:
            try:
                entry = self._entries.get(monitor_id)
                if entry is None or entry.generation != generation:
                    return
                outcome = await self._check_runner(entry.spec)
                deliveries: list[MonitorDelivery] = []
                # Notices are collected under the lock and SENT outside it, the
                # same shape as a delivery: the sink opens a turn (or queues
                # onto the steering queue), and doing that while holding the
                # write lock would stall an arm landing behind it.
                notices: list[MonitorNotice] = []
                change: _Change | None = None
                async with self._write_lock:
                    entry = self._entries.get(monitor_id)
                    if self._disposed or entry is None or entry.generation != generation:
                        # A disposed scheduler must never write: dispose()
                        # cancelled this task, and the guard is what makes the
                        # "cannot apply its result after teardown" claim true
                        # even for a tool that swallowed the cancel and
                        # returned a result anyway — the shape measured live
                        # (a cancelled check's steering-detach NOTICE was
                        # adopted as m28's baseline).
                        return
                    now = self._now()
                    error = outcome.get("error")
                    kind = outcome.get("kind")
                    # Read BEFORE the outcome applies, compared after: see the
                    # edge publish below.
                    was_unavailable = bool(entry.counters.get("unavailable_since"))
                    if kind == "blind":
                        # The call RAN; only the comparison is impossible. It is
                        # neither a success (nothing may be adopted as the new
                        # baseline) nor a failure (the call did not fail), so it
                        # gets its own branch: no strike, no baseline advance,
                        # one notice per episode.
                        notice = self._apply_blind(entry, str(error or ""), now)
                        if notice is not None:
                            notices.append(notice)
                        restored = self._restored_notice(entry, now)
                        if restored is not None:
                            notices.append(restored)
                    elif error:
                        notice = self._apply_failure(entry, error, now, kind=kind)
                        if notice is not None:
                            notices.append(notice)
                    else:
                        change = self._apply_success(
                            entry,
                            outcome.get("text") or "",
                            now,
                            handle=outcome.get("handle") or None,
                        )
                        # AFTER the success, so the empty count reflects THIS check
                        # and a call that starts producing output clears it.
                        silent = self._apply_empty(entry, now, bool(outcome.get("empty")))
                        if silent is not None:
                            notices.append(silent)
                        seeing = self._clear_blind(entry, now)
                        if seeing is not None:
                            notices.append(seeing)
                        restored = self._restored_notice(entry, now)
                        if restored is not None:
                            notices.append(restored)
                    self._write_counters(entry)
                    # AN UNAVAILABLE EPISODE PUBLISHES ITS TWO EDGES, NEVER ITS
                    # TICKS. Every reader outside this process — `lop monitor
                    # status`, the desktop route, the TUI band through the
                    # frontend state — renders the published row, and a tick
                    # publishes nothing (§10.2), so before this an unreachable
                    # watch rendered as a normal one everywhere but here until
                    # the 24 h episode ended in a disable (agent review round 1,
                    # MAJOR-1). One site, AFTER the write, so the published row
                    # and the counters file agree on every key — the retry's
                    # ``next_due_at`` included — and one call even when the
                    # episode's end IS the disable.
                    stall_edge = was_unavailable != bool(entry.counters.get("unavailable_since"))
                    if entry.counters.get("disabled") or stall_edge:
                        self._notify_change()
                if change is not None:
                    # The gate runs OUTSIDE the write lock: the classifier
                    # call is a network call bounded by ``timeoutMs``, and a
                    # locked network call would stall the pump and every arm
                    # queued behind it (§5.2's "a slow call cannot stall an
                    # arm landing behind it").
                    choice = await self._classify_change(change)
                    suppressed = suppressed_counter(choice)
                    async with self._write_lock:
                        entry = self._entries.get(monitor_id)
                        if self._disposed or entry is None or entry.generation != generation:
                            return  # cancelled during the call: drop the deliver
                        delivery = self._settle_change(entry, change, suppressed)
                        if delivery is not None:
                            deliveries.append(delivery)
                for delivery in deliveries:
                    try:
                        await self._maybe_await(self._deliver(delivery))
                    except Exception:
                        # A delivery that throws still advances — one broken
                        # monitor must not become a hot loop.
                        logger.warning("monitor delivery failed for %s", monitor_id, exc_info=True)
                for notice in notices:
                    await self._send_notice(notice, generation=generation)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("monitor check failed for %s", monitor_id, exc_info=True)
            finally:
                self._inflight.discard(monitor_id)
                # A deferred monitor may now have a free slot, and the timer
                # must reflect the new earliest due.
                self._arm()

    # -- lifecycle notices -------------------------------------------------

    async def _send_notice(self, notice: MonitorNotice, *, generation: int) -> None:
        """Hand one notice to the sink and settle its exactly-once latch.

        The latch is set only after the sink returned, so a notice that never
        reached a session is re-announced rather than lost — for the disable it
        happens on the next open, for a stall on the next tick (the episode is
        still open, so the notice is earned again). The settle is re-checked
        under the lock against the entry's GENERATION, so a notice for a monitor
        cancelled while the sink was working cannot resurrect a latch on a
        replacement.
        """
        if self._announce is None:
            return
        try:
            await self._maybe_await(self._announce(notice))
        except Exception:  # noqa: BLE001 — a notice must not break the ladder
            logger.warning("monitor notice failed for %s", notice.monitor_id, exc_info=True)
            return
        latch = _NOTICE_LATCH.get(notice.kind)
        if latch is None:
            return
        async with self._write_lock:
            entry = self._entries.get(notice.monitor_id)
            if entry is None or entry.generation != generation:
                return
            entry.counters[latch] = True
            self._write_counters(entry)

    async def announce_unannounced_disables(self) -> int:
        """Announce every disable whose notice never went out; returns the count.

        Runs at session open, which is what makes the latch work across a
        crash and across an upgrade: a counters file written before
        ``disable_notified`` existed has the key absent, so a disabled monitor
        already sitting in the store is told to the operator exactly once on
        the next hosted open — instead of staying invisible, which is how the
        live store ended up with monitors that had quietly stopped.
        """
        if self._announce is None:
            return 0
        now = self._now()
        pending: list[tuple[MonitorNotice, int]] = []
        async with self._write_lock:
            for entry in self._ordered_entries():
                counters = entry.counters
                if not counters.get("disabled") or counters.get("disable_notified"):
                    continue
                notice = self._disable_notice(
                    entry, now, failures=int(counters.get("consecutive_failures") or 0)
                )
                if notice is not None:
                    pending.append((notice, entry.generation))
        for notice, generation in pending:
            await self._send_notice(notice, generation=generation)
        return len(pending)

    # -- the tick's decision core ------------------------------------------

    def _previous_result(
        self, spec: MonitorSpec, counters: dict[str, Any], old_hash: str
    ) -> str | None:
        """The FULL previous result (normalized), or ``None`` when unusable.

        Same normalization the commit used, then the committed hash is
        re-derived from the body: a handle that does not reproduce ``old_hash``
        is not this monitor's baseline (a torn counters file, or an ignore/
        sort_lines edit since), and the caller falls back to the snapshot.
        """
        handle = str(counters.get("last_handle") or "")
        if not handle:
            return None
        body = _read_spill_body(handle)
        if body is None:
            return None
        previous = normalize(
            body,
            sort_lines=spec.sort_lines,
            ignore=spec.ignore,
            normalize_timestamps=self._settings.normalize_timestamps,
        )
        if content_hash(previous) != old_hash:
            logger.info(
                "monitor %s previous spill does not reproduce its baseline; using the snapshot",
                spec.id,
            )
            return None
        return previous

    def _apply_success(
        self, entry: _Entry, text: str, now: int, *, handle: str | None = None
    ) -> _Change | None:
        """One successful check: the diff state machine of §7.2 in full.

        Returns the detected change awaiting the §8 gate, or ``None`` for
        every quiet/absorbed path (unchanged tick, baseline establishment,
        loss-recovery re-adopt, a failed snapshot write). The DELIVERY
        decision is deliberately not made here: the gate's call is a network
        call and may not run under the write lock, so the diff commits its
        state (blob first, counters second — the §7.2 order) and
        :meth:`_settle_change` applies the verdict afterwards.
        """
        spec = entry.spec
        counters = entry.counters
        old_last_check = int(counters.get("last_check_at") or 0)
        skipped = _skipped_checks_since(old_last_check, spec.every_ms, now)
        # Whether THIS RESULT is a record log decides the gate, not a setting and
        # not the source's provenance: the record view has already separated
        # records from bookkeeping, which is the separation the gate would
        # otherwise be asked to make from opaque text (§8, and see
        # ``_classify_change``). A source that changes shape mid-stream is
        # handled by the basis check below, not by this flag.
        record_view = is_record_log(text)
        normalized = normalize(
            text,
            sort_lines=spec.sort_lines,
            ignore=spec.ignore,
            normalize_timestamps=self._settings.normalize_timestamps,
        )
        new_hash = content_hash(normalized)
        old_hash = str(counters.get("content_hash") or "")
        pure_addition = False
        # Set only by the rendered-diff branch below; every other path (loss
        # semantics, re-adopt) has no previews at all and never had a hiding
        # preview to report.
        delta_hidden = False

        counters["checks"] = int(counters.get("checks") or 0) + 1
        counters["last_check_at"] = now
        counters["consecutive_failures"] = 0
        counters["last_error"] = ""

        if new_hash == old_hash:
            # Quiet tick. One ≲1 KiB write is the whole cost; the blob is only
            # probed so a blob-only loss heals for free (§7.2). A spilled result
            # is recorded as the next diff's full baseline even here: the
            # content is identical, so the handle addresses the same text.
            if handle:
                counters["last_handle"] = handle
            if old_hash and not monitor_state.snapshot_exists(
                self._config_dir, self._session_id, spec.id
            ):
                self._write_snapshot(spec, normalized)
            counters["next_due_at"] = self._advance(spec, now)
            return None

        blob = monitor_state.read_snapshot(self._config_dir, self._session_id, spec.id)
        blob_text = ""
        truncated = False
        if blob is not None:
            blob_text = str(blob.get("snapshot") or "")
            truncated = bool(blob.get("snapshot_truncated"))

        # The baseline's comparison BASIS is part of the baseline. A blob the
        # CURRENT pipeline would restate differently was captured under an
        # earlier normalizer, so diffing the stored text directly against
        # today's output reports the change of SHAPE as a change in the watched
        # source: on the 2026-11-05 record split one envelope line became one
        # line per message, and every live MCP watch would have fired that
        # rewrite as a delta (or had it suppressed as metadata).
        #
        # So a legacy blob is RESTATED by today's pipeline and used as the diff
        # base: the stored text is the old shape of the same content, which
        # keeps a change that landed during the upgrade deliverable instead of
        # adopting a fresh baseline and dropping it (review MIN-3). A plain-text
        # blob restates to itself, so nothing changes for it at all. Only a
        # CLIPPED legacy blob has no comparable old side — it is re-established
        # silently, which is the one upgrade case that reports nothing.
        restated: str | None = None
        unreadable_basis = False
        if (
            blob is not None
            and monitor_state.snapshot_normalizer(blob) != monitor_state.NORMALIZER_TAG
        ):
            if truncated:
                unreadable_basis = True
            else:
                restated = normalize(
                    blob_text,
                    sort_lines=spec.sort_lines,
                    ignore=spec.ignore,
                    normalize_timestamps=self._settings.normalize_timestamps,
                )

        # The right-hand side of the diff. It equals ``normalized`` except on a
        # basis-change tick, where BOTH sides are compared in the legacy mask
        # space (see below): the commit still stores the unmasked rendering.
        diff_text = normalized

        diff_base: str | None
        if not old_hash:
            if blob is None or truncated:
                # No baseline at all (first check, or both files lost), or a
                # clipped prefix with no hash to rebuild it from: the baseline
                # is established SILENTLY — never a full-dump delivery.
                self._establish_baseline(
                    spec,
                    counters,
                    normalized,
                    now,
                    handle=handle,
                    note=(
                        "baseline re-established (record-view)"
                        if unreadable_basis
                        else (
                            "baseline re-established"
                            if old_last_check or truncated
                            else "baseline captured"
                        )
                    ),
                )
                return None
            # A full stored snapshot with a lost hash is still complete
            # content: diff against it, exactly as before — restated first when
            # an earlier normalizer wrote it.
            if restated is not None:
                diff_base, diff_text = legacy_masked(restated), legacy_masked(normalized)
            else:
                diff_base = blob_text
        else:
            # The diff's old side is the FULL previous result. The stored
            # snapshot is clipped at ``snapshotMaxChars``; diffing against that
            # prefix renders the boundary continuation for any change past the
            # cut, which reads as real changed lines to the gate (review round
            # 1, F1). The spill handle recorded on the last commit addresses the
            # whole body, so it is preferred; the snapshot is the fallback only
            # when there is no usable handle.
            diff_base = self._previous_result(spec, counters, old_hash)
            if diff_base is None:
                if restated is not None:
                    # The stored text was written by an earlier normalizer, so it
                    # carries the OLD mask (``at <ts> EST``, ``Message TS:
                    # <ts>.147170``) while today's rendering unmasks both. No
                    # restatement can undo a mask, so the two are only comparable
                    # in the mask space: mask BOTH sides and diff that, and only a
                    # real content change survives (review round 2, R2-MAJ-A —
                    # measured as a phantom ``+2/-2`` re-render on an unchanged
                    # source, ``+25/-25`` on a 20-record window).
                    diff_base, diff_text = legacy_masked(restated), legacy_masked(normalized)
                elif unreadable_basis:
                    # A clipped legacy blob cannot be restated and no handle
                    # reproduced the hash: there is no comparable old side, so
                    # the baseline is re-captured rather than guessed at.
                    self._establish_baseline(
                        spec,
                        counters,
                        normalized,
                        now,
                        handle=handle,
                        note="baseline re-established (record-view)",
                    )
                    return None
                elif blob is not None and not truncated:
                    diff_base = blob_text

        # The bound applies to EVERY baseline source (review round 2, F1). It
        # used to sit inside the ``old_hash`` branch, so the recovery path —
        # counters lost with a large untruncated snapshot — fed the quadratic
        # matcher unbounded (measured: an 8.13 s synchronous stall under the
        # write lock). A bound that only covers one of two paths is not a bound.
        if diff_base is not None and (
            len(diff_base) > MAX_DIFF_CHARS
            or len(diff_text) > MAX_DIFF_CHARS
            or diff_base.count("\n") > MAX_DIFF_LINES
            or diff_text.count("\n") > MAX_DIFF_LINES
        ):
            logger.info(
                "monitor %s result is too large to diff; delivering a checksum delta",
                spec.id,
            )
            diff_base = None

        if diff_base is None:
            # No usable diff baseline (handle evicted/incomplete, a truncated
            # snapshot, or a result past the matcher's bound): a checksum/
            # boundary delta is not evidence, so it is delivered without
            # asking the gate.
            delta_text, changes = beyond_window_text(new_hash)
            delta_hidden = True
        else:
            if not has_line_difference(diff_base, diff_text):
                # The hash disagreed while the content did not: the stored
                # hash was TORN (a crash between the two writes), or a handle
                # reproduced the same text under a different stored hash.
                # Re-adopt and stay quiet.
                counters["content_hash"] = new_hash
                counters["last_handle"] = handle or str(counters.get("last_handle") or "")
                counters["next_due_at"] = self._advance(spec, now)
                return None
            pure_addition = is_pure_addition(diff_base, diff_text)
            rendered = render_delta(
                diff_base,
                diff_text,
                max_delta_lines=self._settings.max_delta_lines,
                delta_max_chars=self._settings.delta_max_chars,
            )
            delta_text, changes = rendered.text, rendered.changed_lines
            delta_hidden = rendered.hidden

        # Order matters (§7.2): blob first, counters second — the counters
        # write is the commit point, so the baseline advances only when its
        # hash lands.
        if not self._write_snapshot(spec, normalized):
            counters["last_note"] = "snapshot write failed"
            counters["next_due_at"] = self._advance(spec, now)
            return None
        counters["content_hash"] = new_hash
        # Only a COMMITTED comparable check records its handle as the next
        # baseline; a failed snapshot write above leaves the old one in place.
        counters["last_handle"] = handle or ""
        counters["last_change_at"] = now
        counters["next_due_at"] = self._advance(spec, now)
        return _Change(
            spec=spec,
            changes=changes,
            delta_text=delta_text,
            skipped=skipped,
            # A snapshot of the just-incremented counter, taken while the
            # write lock is held: the delivery envelope names the check count
            # of THIS check even if a later settle lands after edits.
            checks=int(counters["checks"]),
            at_ms=now,
            pure_addition=pure_addition,
            hidden=delta_hidden,
            record_view=record_view,
        )

    async def _classify_change(self, change: _Change) -> str | None:
        """One gate call for one change (§8.1); the class, or ``None`` → deliver.

        Serialised across the session's monitors (``_classify_lock``), bounded
        by ``classifyMaxChars`` through
        :func:`~local_operator.monitors.classify.gate_state` (whose marker
        floor is the one place the bound is not literal; it also carries the
        monitor's name and purpose so the verdict has something to judge the
        delta AGAINST), and fail-OPEN on
        every fault — including a gate callback that raises. The deadline is NOT
        duplicated here: the call's own bound is
        ``values.classification.timeoutMs`` inside
        ``ClassificationService.decide`` (§8.2), and a second timer would be a
        second policy for one call (the classification layer's own rule).
        """
        if self._classify is None:
            return None
        if change.hidden:
            # Fail OPEN: a model asked to judge a delta whose previews hide the
            # difference can only answer from the counts and the spill footer,
            # and the live gate answered "non-material-metadata" for a Slack
            # thread whose seven new replies were behind a 200-char prefix. The
            # evidence is not shown, so it is delivered rather than judged.
            logger.info(
                "monitor %s delta preview hides its change; delivering without the gate",
                change.spec.id,
            )
            return None
        if change.record_view:
            # A record log's delta is already sorted: one line per record, with
            # the envelope's paging bookkeeping removed and timestamps scrubbed.
            # The gate exists to separate signal from bookkeeping in OPAQUE
            # text, and that separation has happened structurally, so a record
            # appearing (or its thread count changing) is material by the
            # gate's own rubric. Asking anyway is not a tuning problem: on real
            # Slack traffic the live gate answered non-material-metadata 5
            # times out of 6 and swallowed a thread reply entirely (QA round 1
            # Q-1/Q-2, 2026-11-05). Noise control for a record watch is
            # ``ignore``, ``limit`` and the hourly delivery cap — not a model.
            return None
        if change.pure_addition:
            # Appended content is new information, never "bookkeeping that
            # changed": deliver without a model call (and without a model's
            # chance to swallow it). Measured 2026-10-09: even with the
            # monitor's purpose in the state, the live gate swallowed one of
            # three appended build-log lines.
            return None
        state = gate_state(
            change.spec.name,
            change.spec.description,
            change.delta_text,
            self._settings.classify_max_chars,
        )
        async with self._classify_lock:
            try:
                return await self._classify(state)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a monitor may not silently stop reporting
                logger.warning(
                    "monitor classify failed for %s; delivering (fail open)",
                    change.spec.id,
                    exc_info=True,
                )
                return None

    def _settle_change(
        self, entry: _Entry, change: _Change, suppressed: str | None
    ) -> MonitorDelivery | None:
        """Apply the gate's verdict to one change: count a suppression, or deliver.

        The rate cap runs only on the DELIVER path, and that order is the
        contract's (§5.2 step 5: the gate, "then either deliver (§9) or
        suppress + count") — a suppressed change was never a delivery
        candidate, and letting noise consume the hourly window would spend the
        budget the operator's material messages are owed. A suppression still
        calls ``_notify_change``: the index row is what every cold reader (the
        CLI, the desktop route, the TUI band) derives its health hint from, and
        a row that keeps saying "nothing has changed" while the gate suppresses
        every change is the blind spot this counters key now closes. The rewrite
        is cheap and suppressions are rare — a monitor that suppresses often is
        exactly the row an operator needs to see.
        """
        counters = entry.counters
        if suppressed is not None:
            counted = counters["suppressed"]
            counted[suppressed] = int(counted.get(suppressed) or 0) + 1
            self._write_counters(entry)
            # INFO, not debug: a suppression is a change nobody was told about,
            # and at debug level the 2026-10-09 gate regression left no trace
            # outside the counters file.
            logger.info("monitor %s change suppressed (%s)", change.spec.id, suppressed)
            self._notify_change()
            return None
        if not self._rate_allows(counters, change.at_ms):
            counters["suppressed"]["rate_cap"] = (
                int(counters["suppressed"].get("rate_cap") or 0) + 1
            )
            counters["rate_cap_held"] = int(counters.get("rate_cap_held") or 0) + 1
            self._write_counters(entry)
            # Same reason as the gate branch: the index row carries the
            # suppression counts, and a rate-capped change would otherwise be
            # invisible to every cold reader (review round 1, F3).
            self._notify_change()
            return None
        counters["rate_window_count"] = int(counters.get("rate_window_count") or 0) + 1
        counters["deliveries"] = int(counters.get("deliveries") or 0) + 1
        held = int(counters.get("rate_cap_held") or 0)
        counters["rate_cap_held"] = 0
        self._write_counters(entry)
        self._notify_change()
        return MonitorDelivery(
            monitor_id=change.spec.id,
            name=change.spec.name,
            tool=change.spec.tool,
            changes=change.changes,
            checks=change.checks,
            skipped=change.skipped,
            delta_text=change.delta_text,
            at_ms=change.at_ms,
            held_by_cap=held,
            final=self._is_final(change.spec, change.at_ms),
            description=change.spec.description,
            notify=change.spec.notify,
        )

    def _apply_failure(
        self, entry: _Entry, error: str, now: int, *, kind: str | None = None
    ) -> MonitorNotice | None:
        """Apply one failed check; returns the notice it produced, if any.

        Four policies, chosen by ``kind``:

        - ``unavailable``: the call never ran, so nothing is known about the
          watched thing. NO strike is charged and the diff accounting
          (``checks``, ``last_check_at``, the baseline hash) is left untouched —
          a reconnect window must not disable a monitor, and it must not make
          the resume delta look bigger than it is either. The retry rides the
          same capped ladder, keyed on ``unavailable_ticks``. A stall notice
          goes out once per episode, after ``UNAVAILABLE_STALL_MS``, and the
          episode ends by disabling at ``UNAVAILABLE_GONE_MS``;
        - ``failing``: the call RAN and the tool said it could not do the job.
          It is not a strike either — see ``_apply_failing``, which owns the
          whole episode and returns before this ladder is touched;
        - ``fatal``: deterministic and never self-healing, so it disables on
          the FIRST occurrence rather than after five identical failures;
        - absent/anything else: the ordinary ladder, unchanged.
        """
        counters = entry.counters
        counters["last_error"] = str(error)[:500]
        if kind == "unavailable":
            if not counters.get("unavailable_since"):
                counters["unavailable_since"] = now
                # Recorded at the ENTER edge for the same reason as the blind and
                # failing episodes record theirs: ``_restored_notice`` runs AFTER
                # the successful check adopted its baseline, so by then the answer
                # to "was there one?" is always yes (QA round 2, Q-R2-5).
                counters["restored_no_baseline"] = not bool(counters.get("content_hash"))
            counters["unavailable_ticks"] = int(counters.get("unavailable_ticks") or 0) + 1
            counters["last_note"] = "tool unavailable — retrying"
            ticks = int(counters["unavailable_ticks"])
            if now - int(counters["unavailable_since"]) >= UNAVAILABLE_GONE_MS:
                self._disable(entry, "tool unavailable for 24h", now, kind="unreachable")
                return self._disable_notice(entry, now, failures=ticks)
            counters["next_due_at"] = now + min(
                RETRY_CAP_MS, entry.spec.every_ms * (2 ** max(0, ticks - 1))
            )
            if (
                not counters.get("unavailable_notified")
                and now - int(counters["unavailable_since"]) >= UNAVAILABLE_STALL_MS
            ):
                # The LATCH IS NOT SET HERE: ``_send_notice`` settles it after
                # the sink returned, exactly as the disable latch is settled, so
                # a stall notice that never reached a session is retried on the
                # next tick instead of leaving a silent episode and a phantom
                # "running again" (review round 1, R2).
                return MonitorNotice(
                    monitor_id=entry.spec.id,
                    name=entry.spec.name,
                    tool=entry.spec.tool,
                    kind="stalled",
                    at_ms=now,
                    checks=int(counters.get("checks") or 0),
                    deliveries=int(counters.get("deliveries") or 0),
                    detail=counters["last_error"],
                )
            logger.debug(
                "monitor %s unavailable for %d ticks: %s",
                entry.spec.id,
                ticks,
                counters["last_error"],
            )
            return None

        if kind == "failing":
            # THE FAILING EPISODE IS NOT A STRIKE (D3; review round 1,
            # BLOCKER-1). It owns its whole episode in ``_apply_failing`` and
            # returns BEFORE this ladder is touched, so a watch on a resource
            # that does not exist yet survives its fifth failing tick.
            return self._apply_failing(entry, now)
        counters["checks"] = int(counters.get("checks") or 0) + 1
        counters["last_check_at"] = now
        counters["consecutive_failures"] = int(counters.get("consecutive_failures") or 0) + 1
        failures = counters["consecutive_failures"]
        if kind == "fatal":
            self._disable(entry, counters["last_error"], now, kind="fatal")
            logger.warning(
                "monitor %s disabled on a fatal failure: %s", entry.spec.id, counters["last_error"]
            )
            return self._disable_notice(entry, now, failures=failures)
        if failures >= self._settings.max_consecutive_failures:
            self._disable(entry, counters["last_error"], now)
            logger.warning(
                "monitor %s disabled after %d consecutive failures: %s",
                entry.spec.id,
                failures,
                counters["last_error"],
            )
            return self._disable_notice(entry, now, failures=failures)
        # THE STRIKE CLASS SPEAKS BEFORE IT DIES (round-2 manager decision; review
        # round 1, MINOR-5). Everything reaching here charges a strike and is three
        # away from the disable, so the operator gets ONE notice at the second
        # consecutive failure instead of waiting ~35 minutes for a silent death at
        # five. The latch is settled by ``_send_notice`` after the sink returned, as
        # every latch in this file is, and ``_restored_notice`` clears it on the
        # next success so the notice is never the last word.
        if failures == 1:
            # The ENTER edge is published, exactly like the failing episode's: the
            # row must be able to say "1 failed check, below the threshold" instead
            # of "nothing has changed" while the strikes are accruing.
            counters["failure_limit"] = self._settings.max_consecutive_failures
            # And it is the edge that remembers whether there was anything to miss
            # while this run of failures lasts (QA round 2, Q-R2-5).
            counters["restored_no_baseline"] = not bool(counters.get("content_hash"))
            self._notify_change()
        if failures >= STRIKING_CHECKS_BEFORE_NOTICE and not counters.get("striking_notified"):
            counters["failure_limit"] = self._settings.max_consecutive_failures
            striking_notice = MonitorNotice(
                monitor_id=entry.spec.id,
                name=entry.spec.name,
                tool=entry.spec.tool,
                kind="striking",
                at_ms=now,
                checks=int(counters.get("checks") or 0),
                deliveries=int(counters.get("deliveries") or 0),
                failures=failures,
                detail=counters["last_error"],
                failure_limit=self._settings.max_consecutive_failures,
            )
        else:
            striking_notice = None
        backoff = min(RETRY_CAP_MS, entry.spec.every_ms * (2 ** (failures - 1)))
        counters["next_due_at"] = now + backoff
        return striking_notice

    def _apply_failing(self, entry: _Entry, now: int) -> MonitorNotice | None:
        """Apply one failed CALL: keep watching, tell the operator once, no strike.

        THE RULE THIS METHOD EXISTS FOR (D3; agent review round 1, BLOCKER-1). A
        ``kind="failing"`` check ran and the tool said it could not do the job.
        That is NOT the same fact as a check that came back wrong, and the
        strike ladder is the wrong instrument for it: it exists to stop a watch
        whose call never answers, and spending it here disabled a legitimate
        watch after five ticks. A file that does not exist yet, a PR not opened
        yet, a rate-limited source that recovers after ten minutes — each is a
        state the arm receipt calls legitimate, and each was killed by the
        ladder in ~25 minutes at a 5-minute interval.

        So it follows the ``unavailable`` precedent in this file rather than the
        strike ladder: NO strike (``consecutive_failures`` is left untouched, so
        an unrelated ordinary error still gets its full ladder), the retry rides
        the same capped backoff keyed on its OWN counter (``failing_ticks``), ONE
        latched notice per episode, and the episode ends either by a success
        (``_restored_notice``) or by disabling after ``FAILING_GONE_MS`` of
        CONTINUOUS failure — the same 24 h horizon, and the same reason, as
        ``UNAVAILABLE_GONE_MS``.

        ``checks``/``last_check_at`` DO advance, unlike an unavailable tick: the
        call ran and answered (an error is an answer), so counting it is honest
        and keeps the "never checked" hint off a watch that has failed from its
        first tick. The baseline (``content_hash``/snapshot) is NOT advanced, so
        a recovery still reports everything missed as one delta.
        """
        counters = entry.counters
        if not counters.get("failing_since"):
            counters["failing_since"] = now
        counters["failing"] = True
        counters["failing_ticks"] = int(counters.get("failing_ticks") or 0) + 1
        ticks = int(counters["failing_ticks"])
        counters["checks"] = int(counters.get("checks") or 0) + 1
        counters["last_check_at"] = now
        counters["last_note"] = "the watch's call is failing — retrying"
        threshold = (
            FAILING_CHECKS_AFTER_BASELINE
            if counters.get("content_hash")
            else FAILING_CHECKS_WITHOUT_BASELINE
        )
        # PUBLISHED, not just used here: every reader outside this process
        # renders the index row, and the row has to be able to tell a single
        # failed check (deliberately not paged) from an episode that has reached
        # the threshold (QA round 1, Q3) without importing this module's policy.
        counters["failing_notice_checks"] = threshold
        if ticks == 1:
            # The ENTER edge is published (not every failing tick): the list row
            # and the TUI band render the published row, and a row in a failing
            # episode must say so instead of "nothing has changed" (D8). Same
            # rule and same reason as ``_apply_blind``. It also remembers whether
            # a baseline existed when the episode began, so the ``restored`` notice
            # does not promise a delta over a gap in which nothing was missed (QA
            # round 2, Q-R2-5): the successful check that ends the episode adopts
            # its own baseline before ``_restored_notice`` reads anything.
            counters["restored_no_baseline"] = not bool(counters.get("content_hash"))
            self._notify_change()
        if now - int(counters["failing_since"]) >= FAILING_GONE_MS:
            # A DAY of continuous failure is not a resource that is about to
            # appear. The notice names the failure, and ``disabled_kind``
            # carries WHY so the retro-announce (which reads the counters file
            # alone) cannot claim a strike count this episode never charged
            # (review round 1, R3).
            self._disable(
                entry,
                str(counters.get("last_error") or "its call kept failing"),
                now,
                kind="failing",
            )
            logger.warning(
                "monitor %s disabled after 24h of failing calls: %s",
                entry.spec.id,
                counters.get("last_error"),
            )
            return self._disable_notice(entry, now, failures=ticks)
        counters["next_due_at"] = now + min(
            RETRY_CAP_MS, entry.spec.every_ms * (2 ** max(0, ticks - 1))
        )
        if ticks < threshold or counters.get("failing_notified"):
            return None
        # The latch is settled by ``_send_notice`` AFTER the sink returned,
        # exactly like the stall/disable latches, so a notice that never reached
        # a session is re-earned on the next tick instead of leaving a silent
        # episode. ``_restored_notice`` clears the episode on the next success.
        return MonitorNotice(
            monitor_id=entry.spec.id,
            name=entry.spec.name,
            tool=entry.spec.tool,
            kind="failing",
            at_ms=now,
            checks=int(counters.get("checks") or 0),
            deliveries=int(counters.get("deliveries") or 0),
            failures=ticks,
            detail=counters["last_error"],
            no_baseline=not bool(counters.get("content_hash")),
        )

    def _disable(self, entry: _Entry, reason: str, now: int, *, kind: str = "checks") -> None:
        """Mark one monitor disabled; the counters file is the durable record.

        ``kind`` is stored BESIDE the reason because the notice's wording has to
        outlive the process that wrote it: a retro-announce on the next hosted
        open reads the counters file alone, and telling the operator "after 5
        consecutive failed checks" for a 24-hour unreachable episode — which
        charged no strikes at all — is a false claim in text they are meant to
        trust (review round 1, R3; QA round 1, Q2).
        """
        counters = entry.counters
        counters["disabled"] = True
        counters["disabled_kind"] = kind
        counters["disabled_reason"] = str(reason)[:500]
        # No next check is due for a disabled monitor (§10.3): the counters
        # file states that instead of keeping a stale instant that every tick
        # gate would have to second-guess (§11.3; QA round-1 observation 2).
        counters["next_due_at"] = None
        counters["unavailable_since"] = 0
        counters["unavailable_ticks"] = 0
        # The failing episode's own bookkeeping ends with the disable for the same
        # reason: a reactivated monitor must not inherit an episode clock, which
        # would disable it again on its very first failing tick.
        counters["failing"] = False
        counters["failing_since"] = 0
        counters["failing_ticks"] = 0
        # The episode's "was there a baseline" answer dies with it, for the same
        # reason the episode clock does (QA round 2, Q-R2-5).
        counters.pop("restored_no_baseline", None)

    def _disable_notice(self, entry: _Entry, now: int, *, failures: int) -> MonitorNotice | None:
        """The disable notice, unless this monitor's has already gone out.

        The latch is what makes it exactly-once across the two paths that can
        disable a monitor and across a crash: it is cleared only by a
        successful announce (``announce_disable`` below), so a disable whose
        notice never reached a session is re-announced on the next open.
        """
        counters = entry.counters
        if counters.get("disable_notified"):
            return None
        failure_kind = str(counters.get("disabled_kind") or "checks")
        return MonitorNotice(
            monitor_id=entry.spec.id,
            name=entry.spec.name,
            tool=entry.spec.tool,
            kind="disabled",
            # The instant the monitor last RAN is the closest honest answer to
            # "when did this stop", and it is what a retro-announce has; a
            # counter with no check yet falls back to the caller's clock.
            at_ms=int(counters.get("last_check_at") or 0) or now,
            failure_kind=failure_kind,
            checks=int(counters.get("checks") or 0),
            deliveries=int(counters.get("deliveries") or 0),
            failures=failures,
            detail=str(counters.get("disabled_reason") or counters.get("last_error") or ""),
            # §14.4: a disabled watch is the one notice that asks for attention.
            notify=bool(entry.spec.notify),
        )

    def _restored_notice(self, entry: _Entry, now: int) -> MonitorNotice | None:
        """Clear a struck, unavailable OR failing episode on the first success.

        A silent episode resumes silently — there is nothing to tell anyone.
        One the operator was TOLD about gets a matching "running again", so the
        stall/failing/striking notice is never the last word.

        All three are cleared here because all three are "this watch cannot report
        right now": each is told once per episode, and each owes the operator the
        matching recovery. The published index is refreshed on the exit edge so no
        surface keeps saying "watch is failing" about a watch that has just worked.

        The STRIKE class is identified by its LATCH, not by
        ``consecutive_failures``: ``_apply_success`` has already reset that counter
        by the time this runs, and a latch is what survives the recovery tick.
        """
        counters = entry.counters
        available = bool(counters.get("unavailable_since") or counters.get("unavailable_ticks"))
        was_failing = bool(counters.get("failing"))
        was_striking = bool(counters.get("striking_notified"))
        if not available and not was_failing and not was_striking:
            return None
        notified = (
            bool(counters.get("unavailable_notified"))
            or bool(counters.get("failing_notified"))
            or was_striking
        )
        counters["unavailable_since"] = 0
        counters["unavailable_ticks"] = 0
        counters["unavailable_notified"] = False
        counters["failing"] = False
        counters["failing_notified"] = False
        counters["striking_notified"] = False
        # The EPISODE CLOCK goes with them: the 24 h horizon measures CONTINUOUS
        # failure, so a success resets it exactly as it resets the ticks
        # (``UNAVAILABLE_GONE_MS``'s counterpart; without this a watch that fails
        # for an hour a day would be disabled on its third day).
        counters["failing_since"] = 0
        counters["failing_ticks"] = 0
        # CONSUMED ONCE, whether or not a notice goes out: a reactivated or
        # later-episode watch must not inherit the previous episode's answer to
        # "was there a baseline to miss?" (QA round 2, Q-R2-5).
        no_baseline = bool(counters.pop("restored_no_baseline", False))
        if was_failing or was_striking:
            # Same edge-publish rule as ``_clear_blind``: a cold reader renders
            # the published row, and a stale "watch is failing" would outlive
            # the episode it describes.
            self._notify_change()
        if not notified:
            return None
        return MonitorNotice(
            monitor_id=entry.spec.id,
            name=entry.spec.name,
            tool=entry.spec.tool,
            kind="restored",
            at_ms=now,
            checks=int(counters.get("checks") or 0),
            deliveries=int(counters.get("deliveries") or 0),
            no_baseline=no_baseline,
        )

    def _apply_blind(self, entry: _Entry, reason: str, now: int) -> MonitorNotice | None:
        """Apply one uncomparable check: mark blind, advance nothing comparable.

        The check itself ran, so ``checks``/``last_check_at``/``next_due_at``
        move (a blind monitor must not hot-loop on a due instant it never
        advances), but the two pieces of state that decide what the operator is
        told stay put: ``content_hash`` and the snapshot are NOT rewritten, so
        the next comparable check diffs against the last good baseline and
        reports everything missed as one delta. That is exactly what the live
        failure lost — the truncated display projection was adopted as the
        baseline, and the seven new replies became unrecoverable.

        No strike is charged: ``consecutive_failures`` resets, because the call
        succeeded and the failure ladder exists for calls that did not.
        """
        counters = entry.counters
        was_blind = bool(counters.get("blind"))
        previous_reason = str(counters.get("blind_reason") or "")
        counters["checks"] = int(counters.get("checks") or 0) + 1
        counters["last_check_at"] = now
        counters["consecutive_failures"] = 0
        counters["last_error"] = ""
        if not was_blind:
            # Recorded BEFORE anything can establish a baseline: it is what
            # lets the blind/seeing notices say whether the gap has anything to
            # diff against (review round 1, F4).
            counters["blind_no_baseline"] = not bool(counters.get("content_hash"))
        counters["blind"] = True
        counters["blind_reason"] = str(reason)[:500]
        counters["last_note"] = "result not comparable — blind"
        counters["next_due_at"] = self._advance(entry.spec, now)
        if not was_blind or counters["blind_reason"] != previous_reason:
            # The INDEX is what every cold reader renders its hint from, and it
            # is only rewritten on a change event (§10.2). Without this the
            # in-session row would say BLIND while `lop monitor status` kept
            # saying "nothing has changed"; a reason that changes mid-episode
            # is stale on the wire without the second half.
            self._notify_change()
        logger.info("monitor %s is blind: %s", entry.spec.id, counters["blind_reason"])
        if counters.get("blind_notified"):
            return None
        # The latch is settled by ``_send_notice`` after the sink returned, so
        # a notice that never reached a session is re-announced on the next
        # blind tick rather than leaving a silent episode.
        return MonitorNotice(
            monitor_id=entry.spec.id,
            name=entry.spec.name,
            tool=entry.spec.tool,
            kind="blind",
            at_ms=now,
            checks=int(counters.get("checks") or 0),
            deliveries=int(counters.get("deliveries") or 0),
            detail=counters["blind_reason"],
            no_baseline=bool(counters.get("blind_no_baseline")),
        )

    def _clear_blind(self, entry: _Entry, now: int) -> MonitorNotice | None:
        """End a blind episode on the first comparable check; announce if told.

        Same rule as ``_restored_notice``: an episode nobody was told about
        resumes silently, and one the operator SAW gets the matching "seeing
        again" so the blind notice is never the last word.
        """
        counters = entry.counters
        if not counters.get("blind"):
            return None
        notified = bool(counters.get("blind_notified"))
        no_baseline = bool(counters.pop("blind_no_baseline", False))
        counters["blind"] = False
        counters["blind_reason"] = ""
        counters["blind_notified"] = False
        # Same reason as the enter side: the derived index must stop saying
        # BLIND the moment the watch can see again.
        self._notify_change()
        if not notified:
            return None
        return MonitorNotice(
            monitor_id=entry.spec.id,
            name=entry.spec.name,
            tool=entry.spec.tool,
            kind="seeing",
            at_ms=now,
            checks=int(counters.get("checks") or 0),
            deliveries=int(counters.get("deliveries") or 0),
            no_baseline=no_baseline,
        )

    def _apply_empty(self, entry: _Entry, now: int, empty: bool) -> MonitorNotice | None:
        """Count consecutive empty results; one notice when it never saw any.

        D4: EMPTY OUTPUT IS NOT A FAILURE. ``grep`` with no match, a file still
        being written, a queue still quiet — those are the states a watch
        legitimately waits in, and calling them failures would make every working
        quiet watch noisy (it would also page on the ``grep``/``diff`` exit-1
        shapes, which are ANSWERS, not errors).

        But a call that has produced NOTHING for ``EMPTY_CHECKS_BEFORE_NOTICE``
        consecutive checks is a different fact: it has never returned anything to
        compare, so no change could ever be detected. One notice per episode says
        so, and a single non-empty result clears it.
        """
        counters = entry.counters
        if not empty:
            was_empty = bool(counters.get("empty_checks")) or bool(counters.get("empty_notified"))
            counters["empty_checks"] = 0
            counters["empty_notified"] = False
            if was_empty:
                # THE EXIT EDGE IS PUBLISHED for the same reason as the blind
                # clear: the row is what every cold reader renders, and it must
                # stop saying "returned nothing" the moment the call returns
                # something (review round 1, MAJOR-2).
                self._notify_change()
            return None
        counters["empty_checks"] = int(counters.get("empty_checks") or 0) + 1
        count = counters["empty_checks"]
        if count == 1 or count == EMPTY_CHECKS_BEFORE_NOTICE:
            # The two edges the published row has to carry: the first empty
            # (so the count is on the wire) and the one the footer starts
            # speaking at. Publishing every empty tick would be noise.
            self._notify_change()
        if count < EMPTY_CHECKS_BEFORE_NOTICE or counters.get("empty_notified"):
            return None
        return MonitorNotice(
            monitor_id=entry.spec.id,
            name=entry.spec.name,
            tool=entry.spec.tool,
            kind="empty",
            at_ms=now,
            checks=int(counters.get("checks") or 0),
            deliveries=int(counters.get("deliveries") or 0),
            failures=count,
        )

    def _rate_allows(self, counters: dict[str, Any], now: int) -> bool:
        start = int(counters.get("rate_window_start") or 0)
        if start == 0 or now - start >= DELIVERY_WINDOW_MS:
            counters["rate_window_start"] = now
            counters["rate_window_count"] = 0
            return True
        return int(counters.get("rate_window_count") or 0) < self._settings.max_deliveries_per_hour

    def _is_final(self, spec: MonitorSpec, now: int) -> bool:
        """Whether the cancel hint is obsolete (the monitor stops soon).

        The last check before ``until`` — within one interval of it — is the
        final delivery: the monitor will not run again, so "cancel once its
        goal is met" no longer describes anything the reader can act on.
        """
        return spec.until_at is not None and spec.until_at - now <= spec.every_ms

    def _advance(self, spec: MonitorSpec, now: int) -> int | None:
        """The next due instant: now + every + positive-only jitter (§11.2).

        Clamped at ``until_at`` (a monitor never checks past its stop time);
        the due scan treats "next_due ≥ until" as expired.
        """
        jitter_cap = float(min(5_000, spec.every_ms // 10))
        candidate = now + spec.every_ms + int(self._uniform(0.0, jitter_cap))
        if spec.until_at is not None:
            return min(candidate, spec.until_at)
        return candidate

    # -- internals ----------------------------------------------------------

    def _ordered_entries(self) -> list[_Entry]:
        return sorted(self._entries.values(), key=lambda entry: entry.spec.created_at)

    def _is_active(self, entry: _Entry, now: int) -> bool:
        if entry.counters.get("disabled"):
            return False
        spec = entry.spec
        return not (spec.until_at is not None and now >= spec.until_at)

    def _is_due(self, entry: _Entry, now: int) -> bool:
        if not self._is_active(entry, now):
            return False
        due = entry.counters.get("next_due_at")
        return isinstance(due, int) and not isinstance(due, bool) and due <= now

    def _first_check_delay_ms(self) -> int:
        return int(self._uniform(float(FIRST_CHECK_MIN_MS), float(FIRST_CHECK_MAX_MS)))

    def _write_counters(self, entry: _Entry) -> None:
        try:
            monitor_state.write_counters(
                self._config_dir, self._session_id, entry.spec.id, entry.counters
            )
        except Exception:
            logger.warning("monitor counters write failed for %s", entry.spec.id, exc_info=True)

    def _establish_baseline(
        self,
        spec: MonitorSpec,
        counters: dict[str, Any],
        normalized: str,
        now: int,
        *,
        handle: str | None,
        note: str,
    ) -> None:
        """Adopt ``normalized`` as the baseline with NO delivery (§7.2).

        The first check, the recovery from lost state and the upgrade that
        changes the comparison basis (a snapshot written by an earlier
        normalizer) all end in the same place: the pair is written blob-first
        and the hash advances only when the blob landed. The caller returns
        ``None`` afterwards, i.e. a silent tick.
        """
        if self._write_snapshot(spec, normalized):
            counters["content_hash"] = content_hash(normalized)
            counters["last_handle"] = handle or ""
            counters["last_note"] = note
        else:
            counters["last_note"] = "baseline write failed"
        counters["next_due_at"] = self._advance(spec, now)

    def _write_snapshot(self, spec: MonitorSpec, normalized: str) -> bool:
        """Write the snapshot blob; ``False`` when the write did not land.

        A failed blob write must NOT advance the baseline hash — the counters
        file's hash and the blob are a pair (§7.2), and advancing the hash
        while the blob stayed stale would make every later diff compare
        against the wrong window. A quiet-tick heal treats failure as
        "nothing happened"; the change path records the failure and keeps the
        old hash so the next check retries the whole diff.
        """
        cap = self._settings.snapshot_max_chars
        truncated = len(normalized) > cap
        snapshot = normalized[:cap] if truncated else normalized
        try:
            monitor_state.write_snapshot(
                self._config_dir,
                self._session_id,
                spec.id,
                snapshot,
                truncated=truncated,
            )
        except Exception:
            logger.warning("monitor snapshot write failed for %s", spec.id, exc_info=True)
            return False
        return True

    def _notify_change(self) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change()
        except Exception:  # noqa: BLE001 — observation cannot break scheduling
            logger.warning("monitor on_change failed", exc_info=True)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _arm(self) -> None:
        self._cancel_timer()
        if self._disposed:
            return
        now = self._now()
        due_values: list[int] = [
            due
            for entry in self._entries.values()
            if self._is_active(entry, now)
            for due in (entry.counters.get("next_due_at"),)
            if isinstance(due, int) and not isinstance(due, bool)
        ]
        if not due_values:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (constructed outside async): the session's async
            # init re-arms with one pump().
            logger.debug("monitor scheduler armed without a running event loop")
            self.needs_rearm = True
            return
        self.needs_rearm = False
        next_due = min(due_values)
        delay_ms = max(0, next_due - now)
        delay_ms = min(delay_ms, MAX_ARM_MS)
        delay_ms = max(delay_ms, MIN_ARM_MS)
        self._timer = loop.call_later(delay_ms / 1000.0, self._on_timer)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _on_timer(self) -> None:
        self._timer = None
        if self._disposed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.pump())
        self._tick_tasks.add(task)
        task.add_done_callback(self._tick_tasks.discard)

    def dispose(self) -> None:
        """Cancel the armed timer and every in-flight tick/check task.

        asyncio has no ``unref``: this is what stops a pending monitor from
        keeping the event loop alive, and what ensures a check in flight
        cannot apply its result after teardown.
        """
        self._disposed = True
        self._cancel_timer()
        for task in [*self._tick_tasks, *self._check_tasks]:
            if not task.done():
                task.cancel()
        self._tick_tasks.clear()
        self._check_tasks.clear()
