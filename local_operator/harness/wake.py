"""Scheduled wakes — a near-verbatim port of omp ``wake/schedule.ts`` +
``wake/scheduler.ts``.

Splits cleanly into a pure layer (shape, parsing, recurrence math — zero
timers) and a live layer (:class:`WakeScheduler`, which owns the schedules and
a single armed timer). Persistence lives in the session transcript as a
``wake_schedules`` custom entry, handled by the caller via the ``persist``
callback; this module never touches disk.

Key semantics carried over from omp:

- ``parse_wake_duration`` REJECTS bare numbers (``60`` reads as both seconds
  and milliseconds; guessing wrong is a runaway loop).
- Missed occurrences are SKIPPED, not replayed: a laptop asleep six hours owes
  one hourly check, not six.
- ``build_wake_schedule`` returns the error as a STRING rather than raising, so
  the tool's failure path is a sentence the model can act on.
- ``MAX_ARM_MS`` caps the armed timer at one minute; long-dated wakes re-check
  the wall clock on a tick so sleep/clock-skew/timezone changes are absorbed.
- asyncio has no ``timer.unref()`` — the scheduler MUST be :meth:`disposed
  <WakeScheduler.dispose>` explicitly so a pending wake never keeps the event
  loop alive.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from datetime import datetime, time, timedelta
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_WAKE_INTERVAL_MS = 60_000  # a wake starts a full turn; sub-minute starves the user
MAX_WAKE_SCHEDULES = 16
MAX_WAKE_MESSAGE_CHARS = 2_000
PAST_AT_GRACE_MS = 5_000

MAX_ARM_MS = 60_000  # never arm further out than this; re-check on a tick
MIN_ARM_MS = 25  # no zero-delay re-entry loop
LOAD_GRACE_MS = 2_000  # an overdue wake adopted at resume fires shortly AFTER load

WAKE_SCHEDULES_CUSTOM_TYPE = "wake_schedules"
WAKE_PROMPT_MESSAGE_TYPE = "wake_prompt"

_DURATION_UNITS_MS = {
    "s": 1_000,
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}
_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdwSMHDW])\s*$")
_CLOCK_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class WakeSchedule(BaseModel):
    """One scheduled wake. ``id`` is a stable per-session handle (``w1``…)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    message: str  # the self-prompt delivered on fire
    next_due_at: int  # epoch ms
    every_ms: int | None = None  # absent => one-shot
    until_at: int | None = None  # hard stop
    limit: int | None = None  # retire after N deliveries
    fired_count: int = 0
    created_at: int = 0


class DueWake(BaseModel):
    """A wake that is due right now, handed to the ``deliver`` callback."""

    model_config = ConfigDict(extra="forbid")

    schedule: WakeSchedule
    occurrence: int  # 1-based = fired_count + 1 at fire time
    planned_total: int | None = None
    final: bool = False


WakeRetireReason = str  # "limit" | "until" | "one-shot" | "cancelled"


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------


def parse_wake_duration(text: str) -> int | None:
    """Parse ``45s``/``30m``/``2h``/``7d``/``1w`` into milliseconds.

    A bare number is REJECTED on purpose (returns ``None``): ``60`` reads as
    both seconds and milliseconds, and guessing wrong is a runaway loop.
    """
    if not isinstance(text, str):
        return None
    match = _DURATION_RE.match(text)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    return value * _DURATION_UNITS_MS[unit]


def parse_wake_at(text: str, now_ms: int) -> int | None:
    """Parse a wake time into epoch ms. Tries, in order: ``+duration``,
    ``HH:MM`` (next local occurrence), then ISO-8601. Returns ``None`` on no
    match.

    ``HH:MM`` uses date arithmetic (``date + 1 day``) rather than ``+24h`` so
    a DST transition keeps the requested wall-clock time.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("+"):
        duration = parse_wake_duration(stripped[1:])
        if duration is None:
            return None
        return now_ms + duration

    clock = _CLOCK_RE.match(stripped)
    if clock:
        hour = int(clock.group(1))
        minute = int(clock.group(2))
        if hour > 23 or minute > 59:
            return None
        now_dt = datetime.fromtimestamp(now_ms / 1000.0)
        today_target = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if today_target > now_dt:
            target = today_target
        else:
            # Advance the calendar day (not +24h) so the wall-clock time holds
            # across a DST boundary.
            tomorrow = now_dt.date() + timedelta(days=1)
            target = datetime.combine(tomorrow, time(hour=hour, minute=minute))
        return int(target.timestamp() * 1000)

    # ISO-8601 (Python 3.11+ fromisoformat accepts offsets and a trailing Z).
    try:
        parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _format_duration(ms: int) -> str:
    """Render a duration as the shortest exact unit (``1h``, ``45s``, …)."""
    for unit in ("w", "d", "h", "m", "s"):
        step = _DURATION_UNITS_MS[unit]
        if ms % step == 0:
            return f"{ms // step}{unit}"
    return f"{ms}ms"


# ---------------------------------------------------------------------------
# Build + advance (pure)
# ---------------------------------------------------------------------------


def build_wake_schedule(
    request: dict[str, Any], existing: list[WakeSchedule], now_ms: int
) -> dict[str, Any]:
    """Validate a wake-create request. Returns ``{"schedule": WakeSchedule}``
    or ``{"error": str}`` — it returns the error text rather than raising, so
    the tool's failure path is a sentence the model can act on.

    Recognized request keys: ``message`` (required), ``in`` or ``at`` (one
    required), plus optional ``every``, ``until``, ``limit``. Ids ``w1``..``w16``
    are assigned automatically to the first free slot.
    """
    message = request.get("message")
    if not isinstance(message, str) or not message.strip():
        return {"error": "wake requires a non-empty 'message'."}
    message = message.strip()
    if len(message) > MAX_WAKE_MESSAGE_CHARS:
        return {"error": f"wake message must be at most {MAX_WAKE_MESSAGE_CHARS} characters."}
    if len(existing) >= MAX_WAKE_SCHEDULES:
        return {"error": f"at most {MAX_WAKE_SCHEDULES} wake schedules are allowed."}

    in_val = request.get("in")
    at_val = request.get("at")
    if in_val is not None:
        duration = parse_wake_duration(str(in_val))
        if duration is None:
            return {
                "error": f"invalid duration '{in_val}'; use e.g. 45s, 30m, 2h, 7d, 1w."
            }
        next_due_at = now_ms + duration
    elif at_val is not None:
        parsed = parse_wake_at(str(at_val), now_ms)
        if parsed is None:
            return {
                "error": f"invalid time '{at_val}'; use +duration, HH:MM, or an ISO-8601 timestamp."
            }
        next_due_at = parsed
    else:
        return {"error": "wake requires 'in' (e.g. '30m') or 'at' (e.g. '09:00')."}

    # Past-at grace: up to PAST_AT_GRACE_MS in the past is accepted and fires
    # immediately; anything older is a user mistake worth surfacing.
    if next_due_at < now_ms - PAST_AT_GRACE_MS:
        return {"error": "wake time is in the past."}
    if next_due_at < now_ms:
        next_due_at = now_ms

    every_ms: int | None = None
    every_val = request.get("every")
    if every_val is not None:
        every_ms = parse_wake_duration(str(every_val))
        if every_ms is None:
            return {"error": f"invalid 'every' duration '{every_val}'."}
        if every_ms < MIN_WAKE_INTERVAL_MS:
            return {
                "error": f"wake interval must be at least {MIN_WAKE_INTERVAL_MS // 1000}s."
            }

    until_at: int | None = None
    until_val = request.get("until")
    if until_val is not None:
        until_at = parse_wake_at(str(until_val), now_ms)
        if until_at is None:
            return {"error": f"invalid 'until' time '{until_val}'."}

    limit: int | None = None
    limit_val = request.get("limit")
    if limit_val is not None:
        try:
            limit = int(limit_val)
        except (TypeError, ValueError):
            return {"error": f"invalid 'limit' '{limit_val}'."}
        if limit < 1:
            return {"error": "'limit' must be a positive integer."}

    used = {schedule.id for schedule in existing}
    wake_id: str | None = None
    for i in range(1, MAX_WAKE_SCHEDULES + 1):
        candidate = f"w{i}"
        if candidate not in used:
            wake_id = candidate
            break
    if wake_id is None:
        return {"error": "no free wake id."}

    schedule = WakeSchedule(
        id=wake_id,
        message=message,
        next_due_at=next_due_at,
        every_ms=every_ms,
        until_at=until_at,
        limit=limit,
        fired_count=0,
        created_at=now_ms,
    )
    return {"schedule": schedule}


def advance_wake_schedule(schedule: WakeSchedule, now_ms: int) -> dict[str, Any]:
    """Advance a schedule after one fire. Returns ``{"next": WakeSchedule}`` or
    ``{"retired": reason}``.

    Missed occurrences are SKIPPED, not replayed: the next due time jumps to
    the first occurrence strictly after ``now_ms``, so a machine asleep six
    hours owes one hourly fire, not six.
    """
    fired = schedule.fired_count + 1

    if schedule.every_ms is None:
        return {"retired": "one-shot"}
    if schedule.limit is not None and fired >= schedule.limit:
        return {"retired": "limit"}

    every = schedule.every_ms
    base = schedule.next_due_at
    if base > now_ms:
        next_due = base
    else:
        # Skip every missed occurrence and land on the first one after now.
        missed = (now_ms - base) // every + 1
        next_due = base + missed * every

    if schedule.until_at is not None and next_due > schedule.until_at:
        return {"retired": "until"}

    next_schedule = schedule.model_copy(update={"fired_count": fired, "next_due_at": next_due})
    return {"next": next_schedule}


# ---------------------------------------------------------------------------
# Delivery formatting (pure; used by the session to build the self-prompt)
# ---------------------------------------------------------------------------


def format_wake_delivery_text(due: DueWake) -> str:
    """One envelope line then the verbatim message. The envelope always carries
    the handle, because an agent that has to guess its own wake id cannot honour
    "stop when the goal is met". A final delivery drops the cancel hint."""
    schedule = due.schedule
    bits: list[str] = []
    if due.planned_total is not None:
        bits.append(f"{due.occurrence}/{due.planned_total}")
    else:
        bits.append(str(due.occurrence))
    if schedule.every_ms is not None:
        bits.append(f"every {_format_duration(schedule.every_ms)}")
    meta = ", ".join(bits)

    if due.final:
        envelope = f"(alarm) Scheduled wake {schedule.id} ({meta})."
    else:
        envelope = (
            f'(alarm) Scheduled wake {schedule.id} ({meta}) — '
            f'cancel with wake({{op:"cancel",id:"{schedule.id}"}}) once its goal is met.'
        )
    return f"{envelope}\n\n{schedule.message}"


# ---------------------------------------------------------------------------
# Live scheduler
# ---------------------------------------------------------------------------


class WakeScheduler:
    """Owns the wake schedules plus a single armed asyncio timer.

    Three load-bearing properties from omp are preserved:

    1. ``MAX_ARM_MS`` — a wake a week out arms a one-minute re-check tick
       rather than a 604,800,000 ms timeout, so sleep/clock-skew/timezone
       changes are absorbed by re-reading the wall clock.
    2. asyncio has no ``timer.unref()``; :meth:`dispose` cancels the armed
       handle so a pending wake NEVER keeps the event loop alive.
    3. ``LOAD_GRACE_MS`` — an overdue wake adopted at :meth:`load` fires
       shortly AFTER load, not inside it, so the UI has attached and the wake
       appears live in the conversation.

    A delivery that throws still advances the schedule (otherwise one broken
    wake becomes a hot loop).
    """

    def __init__(
        self,
        *,
        now: Callable[[], int],
        deliver: Callable[[DueWake], Awaitable[None] | None],
        persist: Callable[[list[WakeSchedule]], Awaitable[None] | None],
        on_retire: Callable[[WakeSchedule, WakeRetireReason], Awaitable[None] | None] | None = None,
    ) -> None:
        self._now = now
        self._deliver = deliver
        self._persist = persist
        self._on_retire = on_retire
        self._schedules: list[WakeSchedule] = []
        self._timer: asyncio.TimerHandle | None = None
        self._disposed = False

    @property
    def schedules(self) -> tuple[WakeSchedule, ...]:
        return tuple(self._schedules)

    @property
    def disposed(self) -> bool:
        return self._disposed

    def load(self, schedules: list[WakeSchedule] | tuple[WakeSchedule, ...]) -> None:
        """Adopt persisted schedules. NO persist (would duplicate per resume).
        Overdue schedules are pushed to ``now + LOAD_GRACE_MS`` so they fire
        shortly after load rather than inside it."""
        now = self._now()
        adopted: list[WakeSchedule] = []
        for schedule in schedules:
            copy = schedule.model_copy(deep=True)
            if copy.next_due_at <= now:
                copy = copy.model_copy(update={"next_due_at": now + LOAD_GRACE_MS})
            adopted.append(copy)
        adopted.sort(key=lambda s: s.created_at)
        self._schedules = adopted
        self._arm()

    async def update(self, schedules: list[WakeSchedule] | tuple[WakeSchedule, ...]) -> None:
        """Caller-driven change: persist the full list then re-arm."""
        copies = [schedule.model_copy(deep=True) for schedule in schedules]
        copies.sort(key=lambda s: s.created_at)
        self._schedules = copies
        await self._maybe_await(self._persist(list(self._schedules)))
        self._arm()

    async def pump(self, now_ms: int | None = None) -> int:
        """Fire every due wake (delivering each and advancing it), persist if
        anything changed, and re-arm. Returns the number of wakes fired."""
        if self._disposed:
            return 0
        now = now_ms if now_ms is not None else self._now()

        due = [s for s in self._schedules if s.next_due_at <= now]
        kept = [s for s in self._schedules if s.next_due_at > now]
        fired = 0

        for schedule in due:
            occurrence = schedule.fired_count + 1
            if schedule.every_ms is None:
                planned_total: int | None = 1
            else:
                planned_total = schedule.limit
            advanced = advance_wake_schedule(schedule, now)
            final = "retired" in advanced
            due_wake = DueWake(
                schedule=schedule,
                occurrence=occurrence,
                planned_total=planned_total,
                final=final,
            )
            try:
                await self._maybe_await(self._deliver(due_wake))
            except Exception:
                # A delivery that throws still advances the schedule, otherwise
                # one broken wake becomes a hot loop.
                logger.warning("wake delivery failed for %s", schedule.id, exc_info=True)
            fired += 1
            if "next" in advanced:
                kept.append(advanced["next"])
            else:
                if self._on_retire is not None:
                    try:
                        await self._maybe_await(self._on_retire(schedule, advanced["retired"]))
                    except Exception:
                        logger.warning("wake on_retire failed for %s", schedule.id, exc_info=True)

        if fired:
            kept.sort(key=lambda s: s.created_at)
            self._schedules = kept
            await self._maybe_await(self._persist(list(self._schedules)))

        self._arm()
        return fired

    def dispose(self) -> None:
        """Cancel the armed timer. asyncio has no ``unref``, so this is the only
        thing that stops a pending wake from keeping the loop alive."""
        self._disposed = True
        self._cancel_timer()

    # -- internals ----------------------------------------------------------

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _arm(self) -> None:
        self._cancel_timer()
        if self._disposed or not self._schedules:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop; callers drive pump() manually
        now = self._now()
        next_due = min(schedule.next_due_at for schedule in self._schedules)
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
        loop.create_task(self._tick())

    async def _tick(self) -> None:
        await self.pump()
