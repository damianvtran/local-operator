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
}

#: The delivery rate window (§9.4).
DELIVERY_WINDOW_MS = 3_600_000
#: How long a monitor's tool may stay out of reach before the operator is told
#: (§D3). One notice per episode, not one per tick: the point is that a stalled
#: watch stops being silent, not that it becomes noisy.
UNAVAILABLE_STALL_MS = 30 * 60_000
#: How long "out of reach" may last before the monitor is disabled (§D3). This
#: is how "genuinely gone" is settled without a strike count: a server that is
#: reconnecting comes back, one that is uninstalled does not.
UNAVAILABLE_GONE_MS = 24 * 3_600_000


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


class CheckOutcome(TypedDict, total=False):
    """What one check produced: model-visible text, or a failure reason.

    ``kind`` is the runner's classification of a failure, and it exists so the
    ladder can tell "the tool is not reachable right now" from "the tool
    returned an error". Absent means an ordinary error and keeps the strike
    policy unchanged: five consecutive failures disable the monitor.

    - ``unavailable``: the call never ran. No strike is charged (a reconnect
      window must not disable a monitor), it is retried on the same backoff
      ladder, and a stall notice goes out after ``UNAVAILABLE_STALL_MS``;
    - ``fatal``: the failure is deterministic and cannot self-heal (the tool's
      own schema rejected the arguments). Disabled on the first occurrence.
    """

    text: str | None
    error: str | None
    kind: str | None


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
        self._entries[spec.id] = entry
        self._next_seq += 1
        await self.update(list(self.monitors))
        return {
            "created": True,
            "spec": spec,
            # The first check's instant, for the receipt's structured
            # details (§4.5; round-1 review F4).
            "next_due_at": entry.counters.get("next_due_at"),
        }

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
                    if entry is None or entry.generation != generation:
                        return  # cancelled/replaced mid-flight: drop the result
                    now = self._now()
                    error = outcome.get("error")
                    if error:
                        notice = self._apply_failure(entry, error, now, kind=outcome.get("kind"))
                        if notice is not None:
                            notices.append(notice)
                    else:
                        change = self._apply_success(entry, outcome.get("text") or "", now)
                        restored = self._restored_notice(entry, now)
                        if restored is not None:
                            notices.append(restored)
                    self._write_counters(entry)
                    if entry.counters.get("disabled"):
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
                        if entry is None or entry.generation != generation:
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

    def _apply_success(self, entry: _Entry, text: str, now: int) -> _Change | None:
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
        normalized = normalize(
            text,
            sort_lines=spec.sort_lines,
            ignore=spec.ignore,
            normalize_timestamps=self._settings.normalize_timestamps,
        )
        new_hash = content_hash(normalized)
        old_hash = str(counters.get("content_hash") or "")
        pure_addition = False

        counters["checks"] = int(counters.get("checks") or 0) + 1
        counters["last_check_at"] = now
        counters["consecutive_failures"] = 0
        counters["last_error"] = ""

        if new_hash == old_hash:
            # Quiet tick. One ≲1 KiB write is the whole cost; the blob is only
            # probed so a blob-only loss heals for free (§7.2).
            if old_hash and not monitor_state.snapshot_exists(
                self._config_dir, self._session_id, spec.id
            ):
                self._write_snapshot(spec, normalized)
            counters["next_due_at"] = self._advance(spec, now)
            return None

        blob = monitor_state.read_snapshot(self._config_dir, self._session_id, spec.id)
        if blob is None:
            if not old_hash:
                # No baseline at all (first check, or both files lost): the
                # baseline is established SILENTLY — never a full-dump
                # delivery.
                if self._write_snapshot(spec, normalized):
                    counters["content_hash"] = new_hash
                    counters["last_note"] = (
                        "baseline re-established" if old_last_check else "baseline captured"
                    )
                else:
                    counters["last_note"] = "baseline write failed"
                counters["next_due_at"] = self._advance(spec, now)
                return None
            delta_text, changes = beyond_window_text(new_hash)
        else:
            blob_text = str(blob.get("snapshot") or "")
            truncated = bool(blob.get("snapshot_truncated"))
            if not old_hash and truncated:
                # Counters lost AND the blob is truncated: the hash cannot be
                # rebuilt (§7.2 "a truncated blob cannot yield the hash") and a
                # clipped prefix cannot diff honestly, so the baseline is
                # re-established silently rather than diffed against a window
                # that may hide the change.
                if self._write_snapshot(spec, normalized):
                    counters["content_hash"] = new_hash
                    counters["last_note"] = "baseline re-established"
                else:
                    counters["last_note"] = "baseline write failed"
                counters["next_due_at"] = self._advance(spec, now)
                return None
            if not has_line_difference(blob_text, normalized):
                if not truncated:
                    # The hash disagreed while the content did not: the stored
                    # hash was TORN (a crash between the two writes). Re-adopt
                    # the blob ((recompute) and stay quiet.
                    counters["content_hash"] = new_hash
                    counters["next_due_at"] = self._advance(spec, now)
                    return None
                delta_text, changes = beyond_window_text(new_hash)
            else:
                pure_addition = not truncated and is_pure_addition(blob_text, normalized)
                delta_text, changes = render_delta(
                    blob_text,
                    normalized,
                    max_delta_lines=self._settings.max_delta_lines,
                    delta_max_chars=self._settings.delta_max_chars,
                )

        # Order matters (§7.2): blob first, counters second — the counters
        # write is the commit point, so the baseline advances only when its
        # hash lands.
        if not self._write_snapshot(spec, normalized):
            counters["last_note"] = "snapshot write failed"
            counters["next_due_at"] = self._advance(spec, now)
            return None
        counters["content_hash"] = new_hash
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
        budget the operator's material messages are owed. A suppression does
        not ``_notify_change``: the counters file is the durable record, the
        index row (what the front ends read today) carries no suppression
        figures, and §10.2 names delivery/arm/cancel/disable as the triggers —
        slice 4's surfaces are where a counter repaint gets its reader.
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
            return None
        if not self._rate_allows(counters, change.at_ms):
            counters["suppressed"]["rate_cap"] = (
                int(counters["suppressed"].get("rate_cap") or 0) + 1
            )
            counters["rate_cap_held"] = int(counters.get("rate_cap_held") or 0) + 1
            self._write_counters(entry)
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

        Three policies, chosen by ``kind``:

        - ``unavailable``: the call never ran, so nothing is known about the
          watched thing. NO strike is charged and the diff accounting
          (``checks``, ``last_check_at``, the baseline hash) is left untouched —
          a reconnect window must not disable a monitor, and it must not make
          the resume delta look bigger than it is either. The retry rides the
          same capped ladder, keyed on ``unavailable_ticks``. A stall notice
          goes out once per episode, after ``UNAVAILABLE_STALL_MS``, and the
          episode ends by disabling at ``UNAVAILABLE_GONE_MS``;
        - ``fatal``: deterministic and never self-healing, so it disables on
          the FIRST occurrence rather than after five identical failures;
        - absent/anything else: the ordinary ladder, unchanged.
        """
        counters = entry.counters
        counters["last_error"] = str(error)[:500]
        if kind == "unavailable":
            if not counters.get("unavailable_since"):
                counters["unavailable_since"] = now
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
        backoff = min(RETRY_CAP_MS, entry.spec.every_ms * (2 ** (failures - 1)))
        counters["next_due_at"] = now + backoff
        return None

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
        """Clear an unavailable episode on the first success; notice if stalled.

        A silent episode resumes silently — there is nothing to tell anyone.
        One the operator was TOLD about gets a matching "running again", so the
        stall notice is never the last word.
        """
        counters = entry.counters
        if not counters.get("unavailable_since") and not counters.get("unavailable_ticks"):
            return None
        notified = bool(counters.get("unavailable_notified"))
        counters["unavailable_since"] = 0
        counters["unavailable_ticks"] = 0
        counters["unavailable_notified"] = False
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
