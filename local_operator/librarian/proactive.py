"""The Librarian's proactive cadence engine — the ONE re-armer of her rows.

WHAT THIS IS. Aida's engine (``local_operator/aida/proactive.py``) trimmed to
the job the Library needs: one daily pass over the catalog. Exactly one
internal cadence row, id ``librarian-cadence``, armed as a one-shot due at the
next local wall-clock occurrence of ``librarian.cadence.at`` (default 03:30);
plus the bounded ``librarian-extra-N`` one-shots requested through her tray.
Every row the engine owns has an id under the ``librarian-`` prefix, and that
prefix is the ownership boundary: pause, disable and re-arm all work by
dropping or rewriting those ids and never touch an ordinary ``wN`` row the
operator or an agent's own ``wake`` tool armed.

WHY NOT A TIMER. A one-shot retires when it fires, so the next day's row has
to be created during (or right after) the fire, and the only writer that may
do that for a LIVE session is the session. :func:`reconcile` is therefore
called from ``Session._persist_wake_schedules`` (the one editor of the final
list, which the scheduler persists right after a fire) and
``Session._aida_reconcile_now`` (the config watcher's live seam, so a pause or
resume issued in another process reaches a live session within its 2 s tick).
With no runtime, :func:`ensure_armed` writes through
:mod:`local_operator.wakes.arm` (the documented external writer). The two
never run against the same session at once: ``arm.py`` refuses a session a
live runtime owns, and the in-session path only ever sees sessions that have
one.

ONE CATCH-UP AFTER A MISSED DAY. The row is a one-shot, so a day the machine
was closed leaves it overdue and the generic wake catch-up delivers that ONE
occurrence when a runtime opens — the engine never arms a second row while the
owed one is outstanding. On top of that, ``state.json`` carries
``cadence_due_ms`` (the last due the engine armed): when the row itself is
gone (a hand edit, a filtered drop) and that due is more than a day stale,
:func:`cadence_schedule` re-arms at ``now`` once instead of waiting for the
next 03:30. Because the catch-up writes its own ``cadence_due_ms``, a second
reconcile in the same window sees a fresh due and does not stack another —
"one catch-up" is a property of the ledger, not of a flag someone can lose.

PAUSE SEMANTICS. ``librarian.cadence.paused`` (a config key, so ``/settings``
shows it and ``lop config`` can set it) is the authority. Pause = write the
key + stamp ``held_at`` on her wake-index entry (the supervisor skips held
entries exactly as it skips stopped ones) + cancel ``librarian-*`` rows through
whichever writer owns them. Two guards close the windows in between: the
load-time filter (:func:`filter_on_load` — a session opened while held does
not arm ``librarian-*`` rows at all) and the delivery-time guard
(:func:`delivery_allowed` — a fire that comes due while held is dropped rather
than delivered). Resume clears ``held_at`` and re-arms at the next cadence
time. The delivery guard fails OPEN on its own errors: the wake is already
durably due by then, pause is additionally enforced at load, in reconcile and
by the supervisor's skip, and eating a scheduled turn over a transient read
error is the worse trade.

DISABLE. ``librarian.enabled`` false, or ``LOCAL_OPERATOR_NO_LIBRARIAN``
truthy, is harder than pause: creation never happens
(:func:`local_operator.librarian.bootstrap.ensure_session` gates first) and
any existing ``librarian-*`` rows are dropped at the next reconcile/load.
Nothing is remembered, because a disabled install never wanted her.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_operator.harness.wake_types import MAX_WAKE_SCHEDULES, WakeSchedule
from local_operator.librarian import state
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable
from local_operator.wakes.store import is_patience_row

logger = logging.getLogger(__name__)

#: The cadence row's id. Stable across arming writers: the engine re-arms by
#: rebuilding this row, so an id that drifted between writers would leave two
#: cadence rows (or none) depending on which side armed last.
CADENCE_ID = "librarian-cadence"

#: The id prefix of engine-armed extra one-shots. ``librarian-extra-<n>`` with
#: the lowest free ``n``, chosen from the ids present at reconcile time.
EXTRA_ID_PREFIX = "librarian-extra-"

#: The ownership boundary. Every row the engine creates or drops starts here;
#: anything else on the list belongs to the user or another engine.
ROW_PREFIX = "librarian-"

#: Journal type for the refusals the bounds produce (budget exhausted, spacing
#: floor, list full). Context-invisible, like Aida's.
CUSTOM_ENTRY_TYPE = "librarian_proactive"

DEFAULT_CADENCE_AT = "03:30"
DEFAULT_MAX_EXTRA_PER_DAY = 1
DEFAULT_MIN_GAP_MINUTES = 720
DEFAULT_ENABLED = True
DEFAULT_PAUSED = False

_HOUR_MS = 3_600_000

#: How stale an owed due has to be before a re-arm counts as "a missed day".
#: A day's cadence fires once; if the row is gone and the last due is at least
#: this old, one catch-up run is owed. Anything shorter is still today's pass.
CATCH_UP_AFTER_MS = 24 * _HOUR_MS

#: The delivered message. The seed is the policy; this is the trigger line the
#: model reads when the cadence fires, so it names the job and the quiet-day
#: contract in one sentence.
CADENCE_MESSAGE = (
    "The Librarian's daily library pass is due. Read the library stats and the "
    "unorganized assets, then organise a bounded batch. Reply with "
    f"{'Nothing needs organizing today.'!r} when there is nothing to do."
)

DEFAULT_EXTRA_MESSAGE = "The Librarian has an extra library pass to run."

#: Display-label TEMPLATES for the engine-armed rows on the human receipt
#: surfaces, keyed by the id the engine writes. Templates, not literals: the
#: receipt must follow ``librarian.name`` exactly as Aida's does.
WAKE_DISPLAY_LABEL_TEMPLATES: dict[str, str] = {
    CADENCE_ID: "{name}'s library pass",
}

EXTRA_DISPLAY_LABEL_TEMPLATE = "{name}'s extra library pass"


@dataclass(frozen=True)
class CadencePolicy:
    """The resolved cadence configuration — defaults applied, one reader.

    ``enabled`` folds the config key and the environment kill switch; every
    consumer asks THIS object rather than re-reading either.
    """

    enabled: bool
    paused: bool
    at: str
    max_extra_per_day: int
    min_gap_minutes: int


def _read_librarian_section(config_dir: Path | str) -> Mapping[str, Any]:
    """The ``librarian`` config mapping, best-effort.

    A fresh install has no section at all — absent means defaults, and a
    config file that cannot be read means defaults too: the cadence's whole
    job is to try again tomorrow, so a broken read must never be *louder* than
    the settings it could not find. A hand-edited section (a string where a
    mapping belongs) degrades the same way.
    """
    try:
        from local_operator.config import ConfigManager

        raw = ConfigManager(config_dir=Path(config_dir)).get_config_value("librarian", None)
    except Exception:  # noqa: BLE001 — defaults are the answer to an unreadable file
        logger.debug("librarian: could not read the config section", exc_info=True)
        return {}
    return raw if isinstance(raw, Mapping) else {}


def policy(config_dir: Path | str) -> CadencePolicy:
    """Resolve the cadence policy: config keys + the environment kill switch."""
    section = _read_librarian_section(config_dir)
    cadence = section.get("cadence")
    cadence = cadence if isinstance(cadence, Mapping) else {}
    enabled = section.get("enabled")
    enabled = DEFAULT_ENABLED if enabled is None else bool(enabled)
    if state.env_disabled():
        enabled = False
    at = cadence.get("at")
    at = at.strip() if isinstance(at, str) and at.strip() else DEFAULT_CADENCE_AT
    max_extra = cadence.get("max_extra_per_day")
    try:
        max_extra = int(max_extra) if max_extra is not None else DEFAULT_MAX_EXTRA_PER_DAY
    except (TypeError, ValueError):
        max_extra = DEFAULT_MAX_EXTRA_PER_DAY
    gap = cadence.get("min_gap_minutes")
    try:
        gap = int(gap) if gap is not None else DEFAULT_MIN_GAP_MINUTES
    except (TypeError, ValueError):
        gap = DEFAULT_MIN_GAP_MINUTES
    paused = bool(cadence.get("paused", DEFAULT_PAUSED))
    return CadencePolicy(
        enabled=enabled,
        paused=paused,
        at=at,
        max_extra_per_day=max(0, max_extra),
        min_gap_minutes=max(0, gap),
    )


def parse_cadence_at(at: str) -> tuple[int, int] | None:
    """``"HH:MM"`` → ``(hour, minute)``; ``None`` when it is not a clock time.

    The one validator for ``librarian.cadence.at``. Invalid values fall back
    to the default at :func:`next_cadence_ms` rather than raising: the key is
    free text in the settings registry, and a typo must not cost the daily
    pass (it falls back visibly — the fallback is logged).
    """
    parts = at.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def next_cadence_ms(now_ms: int, at: str = DEFAULT_CADENCE_AT) -> int:
    """The next local occurrence of ``at`` after ``now_ms``, in epoch ms.

    Local wall clock, on the calendar — ``datetime`` arithmetic over today's
    date, so a DST transition keeps the requested clock time. An unparseable
    ``at`` falls back to :data:`DEFAULT_CADENCE_AT`, logged at DEBUG because
    it is read per arm.
    """
    clock = parse_cadence_at(at)
    if clock is None:
        logger.debug("librarian: invalid cadence time %r; using %s", at, DEFAULT_CADENCE_AT)
        clock = parse_cadence_at(DEFAULT_CADENCE_AT)
        assert clock is not None
    hour, minute = clock
    now = datetime.fromtimestamp(now_ms / 1000.0)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = datetime.combine(now.date() + timedelta(days=1), target.time())
    return int(target.timestamp() * 1000)


def is_librarian_row(wake_id: str) -> bool:
    """Whether ``wake_id`` is a row this engine owns."""
    return isinstance(wake_id, str) and wake_id.startswith(ROW_PREFIX)


def is_cadence_family_row(wake_id: str) -> bool:
    """Whether ``wake_id`` is a PASS row: the cadence or an extra.

    The quiet-day sentinel applies to these rows and only these; a user-armed
    wake is the user's own intent and must never be silenced by it.
    """
    return isinstance(wake_id, str) and (
        wake_id == CADENCE_ID or wake_id.startswith(EXTRA_ID_PREFIX)
    )


def cadence_message(config_dir: Path | str, now_ms: int) -> str:
    """The cadence row's message.

    Constant by design: the per-day taxonomy lives in the Library metadata
    store, which the seed reads with the ``library`` tool, not in a wake
    message that would go stale the moment a convention changes. The
    signature keeps Aida's shape so the two engines stay comparable.
    """
    return CADENCE_MESSAGE


def hold_active(config_dir: Path | str) -> bool:
    """Paused or disabled — the state in which ``librarian-*`` rows are not armed.

    The load-time filter and the delivery guard both ask this ONE question, so
    "held" cannot mean two things on two seams.
    """
    pol = policy(config_dir)
    return (not pol.enabled) or pol.paused


def delivery_allowed(config_dir: Path | str) -> bool:
    """Whether a ``librarian-*`` fire may deliver right now (fail-open, see module)."""
    try:
        return not hold_active(config_dir)
    except Exception:  # noqa: BLE001 — documented fail-open
        logger.warning("librarian: could not resolve the hold state; delivering", exc_info=True)
        return True


def filter_on_load(schedules: Sequence[WakeSchedule], config_dir: Path | str) -> list[WakeSchedule]:
    """The load-time hold: drop ``librarian-*`` rows while held. Never raises.

    Called from ``Session._load_wake_schedules`` for her session only (the
    caller gate is :func:`local_operator.librarian.state.is_librarian_session`,
    one stat). Rows the filter drops are dropped from the in-memory list, so
    the session's next persist writes the cancellation into the transcript —
    the "pause cancels ``librarian-*`` rows" contract reached through the ONE
    writer rather than a file edit behind the session's back.
    """
    if not any(is_librarian_row(s.id) for s in schedules):
        return list(schedules)
    if not hold_active(config_dir):
        return list(schedules)
    kept = [s for s in schedules if not is_librarian_row(s.id)]
    if len(kept) != len(schedules):
        logger.info(
            "librarian: holding %d pass row(s) while paused/disabled",
            len(schedules) - len(kept),
        )
    return kept


@dataclass
class ReconcileResult:
    """The full-list reconcile's new list, plus notes to journal.

    ``notes`` are human-readable refusals (budget, spacing, capacity) that the
    caller appends to the transcript as :data:`CUSTOM_ENTRY_TYPE` entries — the
    design's "observable, not silent" bound. ``changed`` is True when the list
    differs from the input, so a caller can skip a write (and the transcript
    append it would cost) when nothing moved.
    """

    schedules: list[WakeSchedule]
    notes: list[str]
    changed: bool


def _ledger_today(config_dir: Path | str, now_ms: int) -> tuple[dict[str, Any], int]:
    """The extra-run ledger for TODAY, resetting across local days.

    Returns ``(ledger, armed_today)`` where ``ledger`` is the mapping to write
    back (``{"day": "YYYY-MM-DD", "armed": n}``), never mutating state itself —
    the caller holds the lock.
    """
    raw = state.read_state(config_dir) or {}
    ledger = raw.get("extras")
    today = datetime.fromtimestamp(now_ms / 1000.0).date().isoformat()
    if not isinstance(ledger, Mapping) or ledger.get("day") != today:
        ledger = {"day": today, "armed": 0}
    try:
        armed = int(ledger.get("armed", 0))
    except (TypeError, ValueError):
        armed = 0
    return dict(ledger, armed=armed), armed


def _next_extra_id(existing: set[str]) -> str:
    """The lowest free ``librarian-extra-<n>`` id."""
    n = 1
    while f"{EXTRA_ID_PREFIX}{n}" in existing:
        n += 1
    return f"{EXTRA_ID_PREFIX}{n}"


def _parse_extra_due(request: Mapping[str, Any], now_ms: int) -> tuple[int | None, str, str]:
    """``(due_ms, message, refusal)`` for one tray request.

    The request grammar is the wake tool's own subset — ``in`` or ``at``, plus
    an optional ``message`` — resolved through the SAME parsers the tool uses,
    so a request the seed writes is validated exactly like one the tool would
    have taken. A bare string is accepted as a timing-only shorthand
    (``"in 4h"``), because that is the shape the design's examples use;
    anything else is refused with a note, never raised.
    """
    from local_operator.harness.wake import parse_wake_at, parse_wake_duration

    if isinstance(request, str):
        text = request.strip()
        lowered = text.lower()
        if lowered.startswith("at "):
            request = {"at": text[3:].strip()}
        elif lowered.startswith("in "):
            request = {"in": text[3:].strip()}
        else:
            request = {"in": text}
    if not isinstance(request, Mapping):
        return None, "", "ignored a malformed extra-pass request (not an object)."
    message = request.get("message")
    message = message.strip() if isinstance(message, str) and message.strip() else ""
    in_val, at_val = request.get("in"), request.get("at")
    due: int | None = None
    if in_val is not None:
        duration = parse_wake_duration(str(in_val))
        if duration is None:
            return (
                None,
                message,
                f"ignored an extra-pass request with an invalid 'in' ({in_val!r}).",
            )
        due = now_ms + duration
    elif at_val is not None:
        due = parse_wake_at(str(at_val), now_ms)
        if due is None:
            return (
                None,
                message,
                f"ignored an extra-pass request with an invalid 'at' ({at_val!r}).",
            )
    if due is None:
        return None, message, "ignored an extra-pass request with no time."
    if due < now_ms - 5_000:
        return None, message, "ignored an extra-pass request whose time is in the past."
    return max(due, now_ms + 1_000), message, ""


def cadence_schedule(
    now_ms: int,
    at: str = DEFAULT_CADENCE_AT,
    *,
    config_dir: Path | str,
    existing: WakeSchedule | None = None,
) -> WakeSchedule:
    """One-shot cadence, calendar-based, with the one-catch-up rule (module).

    ``created_at`` keeps the scheduler's stable ordering, and an existing row
    keeps its own: rebuilding would mint a new ``created_at`` on every
    reconcile, which makes ``changed`` permanently true (so a watcher tick or
    an after-turn drain would rewrite the transcript entry for nothing). An
    overdue existing row is brought to ``now`` ONLY when it is more than a day
    stale; a shorter lateness is left for the generic catch-up exactly as Aida
    leaves it.
    """
    if existing is not None:
        due = existing.next_due_at
        if due + CATCH_UP_AFTER_MS <= now_ms:
            due = now_ms + 1_000
        created_at = existing.created_at
    else:
        raw = state.read_state(config_dir) or {}
        saved_due = raw.get("cadence_due_ms")
        if type(saved_due) is int and saved_due + CATCH_UP_AFTER_MS <= now_ms:
            # THE ONE CATCH-UP: the row vanished and the last due is a day or
            # more stale, so the day it named was missed. Run one now.
            due = now_ms + 1_000
        else:
            due = next_cadence_ms(now_ms, at)
        created_at = now_ms
    if due <= now_ms:
        due = now_ms + 1_000
    # Prepare BEFORE row persistence so a crash between the two leaves the
    # catch-up owed rather than spent. The mirror is written under the lock
    # because several attended writers reach this path.
    try:
        with state.locked(config_dir):
            raw = state.read_state(config_dir) or {}
            # Only when the mirror MOVED: every persist reaches this path, and
            # an unconditional write would make an idle reconcile touch
            # state.json on every turn.
            if raw.get("cadence_due_ms") != due:
                state.update_state(config_dir, cadence_due_ms=due)
    except Exception:  # noqa: BLE001 — a mirror failure must not cost the arm
        logger.debug("librarian: could not mirror the cadence due", exc_info=True)
    return WakeSchedule(
        id=CADENCE_ID,
        message=cadence_message(config_dir, now_ms),
        next_due_at=due,
        every_ms=None,
        created_at=created_at,
        # NOTIFY IS THE POINT OF THIS ROW: with no surface attached, a pass
        # that finds work must be able to reach the operator. Whether it
        # ACTUALLY banners is decided at settle time by the reply-aware veto,
        # which reads the sentinel: a quiet day stays silent.
        notify=True,
    )


def _cadence_due(rows: Sequence[WakeSchedule], pol: CadencePolicy, now_ms: int) -> int:
    """The cadence row's due time, or the time it *would* be armed for.

    Used only by the spacing check: the cadence row may not exist yet in the
    in-memory list (it is appended last), and its future occurrence is exactly
    the due time the spacing floor must consider.
    """
    for row in rows:
        if row.id == CADENCE_ID:
            return row.next_due_at
    return next_cadence_ms(now_ms, pol.at)


def reconcile(
    schedules: Sequence[WakeSchedule],
    *,
    config_dir: Path | str,
    session_id: str,
    now_ms: int | None = None,
    cadence_fired: WakeSchedule | None = None,
) -> ReconcileResult:
    """The in-session full-list reconcile. Pure file side effects: the ledger.

    Order and content, in one place so the persist seam and the config-watch
    seam cannot diverge:

    1. Not her session ⇒ untouched (identity is read once, here, so a caller
       cannot forget the gate).
    2. Disabled or paused ⇒ drop every ``librarian-*`` row; nothing is
       re-armed and the tray is left where it is (a paused agent must not
       queue work for her own resume).
    3. Active ⇒ keep every row — hers included, so an extra in flight is never
       silently cancelled — ensure exactly one ``librarian-cadence`` row when
       none is present, consume the tray within ``max_extra_per_day`` /
       ``min_gap_minutes`` / the schedule cap, and record the accepted fire's
       ``last_run_at``.

    The ledger mutation happens under :func:`state.locked` — the only slow-ish
    work here — and everything else is in-memory. The caller journals
    ``notes``; this function never touches the transcript, so it stays
    callable from the scheduler's write lock without a second writer.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if cadence_fired is not None:
        # THE RUN RECEIPT: the accepted fire is the only fact that says a pass
        # actually STARTED (arming a row is not running one). Counts are
        # derived at read time from the metadata store's edit log; the state
        # keeps only the instant, so a failed read cannot lose it.
        try:
            with state.locked(config_dir):
                state.update_state(config_dir, last_run_at=now)
        except Exception:  # noqa: BLE001 — a receipt must not cost the re-arm
            logger.debug("librarian: could not record the run instant", exc_info=True)
    original = list(schedules)
    if not state.is_librarian_session(config_dir, session_id):
        return ReconcileResult(schedules=original, notes=[], changed=False)

    pol = policy(config_dir)
    notes: list[str] = []
    if not pol.enabled or pol.paused:
        kept = [s for s in original if not is_librarian_row(s.id)]
        return ReconcileResult(schedules=kept, notes=notes, changed=kept != original)

    # ACTIVE: keep everything, including her existing rows. An extra asked for
    # earlier is a request in flight — dropping it here would silently cancel
    # it at the next persist.
    kept = list(original)

    # -- extras tray --------------------------------------------------------
    # The tray is CLEARED only after the rows are built into ``kept``, so a
    # crash between the two duplicates a pass rather than losing one.
    tray = state.read_extras(config_dir)
    if tray and pol.max_extra_per_day > 0:
        ledger, armed_today = _ledger_today(config_dir, now)
        taken = 0
        due_requests: list[tuple[int, str]] = []
        for request in tray:
            if armed_today + taken >= pol.max_extra_per_day:
                notes.append(
                    f"extra-pass request refused: budget exhausted "
                    f"({pol.max_extra_per_day}/day)."
                )
                continue
            due, message, refusal = _parse_extra_due(request, now)
            if refusal:
                notes.append(refusal)
                continue
            assert due is not None
            gap_ms = pol.min_gap_minutes * 60_000
            cadence_due = _cadence_due(kept, pol, now)
            too_close = (
                any(abs(row.next_due_at - due) < gap_ms for row in kept if is_librarian_row(row.id))
                or abs(cadence_due - due) < gap_ms
            )
            if too_close:
                notes.append(
                    "extra-pass request refused: within the "
                    f"{pol.min_gap_minutes}-minute spacing floor of her other wakes."
                )
                continue
            due_requests.append((due, message))
            taken += 1
        for due, message in due_requests:
            kept.append(
                WakeSchedule(
                    id=_next_extra_id({row.id for row in kept}),
                    message=message or DEFAULT_EXTRA_MESSAGE,
                    next_due_at=due,
                    every_ms=None,
                    created_at=now,
                    notify=True,
                )
            )
        if taken:
            try:
                with state.locked(config_dir):
                    state.update_state(config_dir, extras=dict(ledger, armed=armed_today + taken))
                    state.clear_extras(config_dir)
            except Exception:  # noqa: BLE001 — a contended lock must not eat the turn
                logger.warning("librarian: extra ledger write failed", exc_info=True)
                notes.append("extra-pass tray left unread (another operation is in flight).")
        else:
            # PROCESSED AND REFUSED is still consumed: a request the bounds
            # rejected would otherwise be re-noted on every later persist.
            state.clear_extras(config_dir)
    elif tray:
        notes.append("extra-pass request refused: extra passes are disabled (0/day).")
        state.clear_extras(config_dir)

    # -- cadence ------------------------------------------------------------
    existing = next((row for row in kept if row.id == CADENCE_ID), None)
    candidate = cadence_schedule(now, pol.at, config_dir=config_dir, existing=existing)
    kept = [candidate if row.id == CADENCE_ID else row for row in kept]
    if existing is None:
        kept.append(candidate)

    if cadence_fired is not None:
        try:
            with state.locked(config_dir):
                state.update_state(config_dir, last_run_at=now)
        except Exception:  # noqa: BLE001 — a receipt must not cost the reconcile
            logger.debug("librarian: could not record last_run_at", exc_info=True)

    if len(kept) > MAX_WAKE_SCHEDULES:
        logger.warning("librarian: schedule list over the cap; dropping her extras")
        extras = [row for row in kept if row.id.startswith(EXTRA_ID_PREFIX)]
        for row in extras[: len(kept) - MAX_WAKE_SCHEDULES]:
            kept.remove(row)
        notes.append("dropped an extra pass: the schedule list is full.")

    return ReconcileResult(schedules=kept, notes=notes, changed=kept != original)


async def append_notes(transcript: Any, notes: Sequence[str]) -> None:
    """Append engine notes to a transcript, best-effort.

    ``notes`` are refusals the bounds produced; journaling them as
    :data:`CUSTOM_ENTRY_TYPE` custom entries is what makes the bound
    "observable, not silent". Custom entries never enter the model's context,
    so the note is for whoever reads the transcript next, human or agent.
    Every failure is swallowed: a note about a refusal must never become a
    second failure.
    """
    if not notes:
        return
    for note in notes:
        try:
            await transcript.append_custom(CUSTOM_ENTRY_TYPE, {"note": note})
        except Exception:  # noqa: BLE001 — best-effort by contract
            logger.warning("librarian: could not journal an engine note", exc_info=True)
            return


def _row_ids(entry: Mapping[str, Any] | None) -> set[str]:
    ids: set[str] = set()
    for raw in (entry or {}).get("schedules", []) or []:
        if isinstance(raw, Mapping) and isinstance(raw.get("id"), str):
            ids.add(raw["id"])
    return ids


def _librarian_ids(entry: Mapping[str, Any] | None) -> list[str]:
    return sorted(wake_id for wake_id in _row_ids(entry) if is_librarian_row(wake_id))


async def _cancel_ids(
    root: Path, session_id: str, ids: Sequence[str], *, now_ms: int
) -> tuple[list[str], bool]:
    """Cancel each id through the external writer. Returns (cancelled, owner_blocked).

    ``owner_blocked`` is True when ``arm.py`` refused because a live runtime
    holds the session — the one refusal that is not an error: the live session
    owns its rows, and the pause/resume config write reaches it through the
    config watcher. A 404 (row already gone) is a success for this purpose.
    """
    from local_operator.wakes.arm import WakeWriteError, cancel_wake

    cancelled: list[str] = []
    for wake_id in ids:
        try:
            await cancel_wake(root, session_id, wake_id, now_ms=now_ms)
            cancelled.append(wake_id)
        except WakeWriteError as exc:
            if exc.status == 503:
                return cancelled, True
            logger.debug("librarian: cancel of %s refused: %s", wake_id, exc)
        except Exception:  # noqa: BLE001 — best-effort: the config is the authority
            logger.warning("librarian: cancel of %s failed", wake_id, exc_info=True)
    return cancelled, False


def _iso_due(due_ms: int) -> str:
    """A due instant in the wake grammar's ISO form (local, offset-aware)."""
    return datetime.fromtimestamp(due_ms / 1000).isoformat()


def _transcript_cadence(config_dir: Path | str, session_id: str) -> WakeSchedule | None:
    """The authoritative snapshot, not a potentially stale/missing wake index."""
    from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
    from local_operator.session.transcript import read_latest_custom_entry

    entry = read_latest_custom_entry(
        Path(config_dir) / "sessions" / session_id, WAKE_SCHEDULES_CUSTOM_TYPE
    )
    if entry is None:
        return None
    for raw in entry.payload.get("details", {}).get("schedules", []):
        if isinstance(raw, Mapping) and raw.get("id") == CADENCE_ID:
            return WakeSchedule.model_validate(raw)
    return None


def reap_orphan_rows(config_dir: Path | str, *, keep: str = "") -> tuple[str, ...]:
    """Drop the engine's rows from sessions that no longer exist on disk.

    Her session id is minted by ``bootstrap.ensure_session`` and re-minted
    whenever the directory ``state.json`` names is gone, so every re-creation
    leaves the PREVIOUS incarnation's armed cadence behind in the wake index —
    a row keyed to an id with no transcript, which no runtime can ever be
    started for. The supervisor refuses to engage one, but nothing removed
    them, so they accumulate as what ``lop wake status`` shows and what a
    reader has to rule out before believing her schedule.

    SCOPE, deliberately narrow. Only rows this engine owns (``librarian-*``)
    are removed, only from entries whose session TRANSCRIPT is absent
    (``keep``'s own entry is never touched), and an entry left with nothing is
    removed through ``store.write_entry``'s own empty-list rule rather than an
    unlink here. The entry's other keys ride through ``preserve``, so a ghost
    entry carrying someone else's rows keeps them.
    """
    from local_operator.resume import TRANSCRIPT_NAME
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    try:
        index = wake_store.read_index(root)
    except Exception:  # noqa: BLE001 — an unreadable index reaps nothing
        logger.warning("librarian: could not read the wake index to reap orphans", exc_info=True)
        return ()

    reaped: list[str] = []
    for session_id, entry in index.items():
        if not session_id or session_id == keep or not isinstance(entry, Mapping):
            continue
        if not _librarian_ids(entry):
            continue
        if (root / "sessions" / session_id / TRANSCRIPT_NAME).is_file():
            continue
        kept = [
            row
            for row in entry.get("schedules", []) or []
            if not is_librarian_row(str(row.get("id", "")) if isinstance(row, Mapping) else "")
        ]
        try:
            wake_store.write_entry(
                root,
                session_id,
                cwd=str(entry.get("cwd") or Path.home()),
                schedules=kept,
                preserve=entry,
            )
        except Exception:  # noqa: BLE001 — best-effort, like every store writer
            logger.warning(
                "librarian: could not reap the orphan rows for %s", session_id, exc_info=True
            )
            continue
        reaped.append(session_id)
    if reaped:
        logger.info("librarian: reaped orphaned engine rows for %s", ", ".join(sorted(reaped)))
    return tuple(reaped)


async def _drain_tray_external(root: Path, session_id: str, pol: CadencePolicy, now: int) -> None:
    """Arm the tray's requests for a session with NO live owner, bounded.

    The external half of the reconcile's tray consume. Rows are armed one at a
    time through ``arm_wake`` and the tray is cleared only after the last one
    landed; a refusal mid-drain leaves the file for the next boot, and the
    duplicate that can come out of a crash is bounded to one pass.
    """
    from local_operator.wakes.arm import WakeWriteError, arm_wake

    tray = state.read_extras(root)
    if not tray:
        return
    if pol.max_extra_per_day <= 0:
        logger.info("librarian: extra passes are disabled; leaving the tray")
        return
    ledger, armed_today = _ledger_today(root, now)
    taken = 0
    for request in tray:
        if armed_today + taken >= pol.max_extra_per_day:
            break
        due, message, refusal = _parse_extra_due(request, now)
        if refusal:
            logger.info("librarian: %s", refusal)
            continue
        assert due is not None
        try:
            await arm_wake(
                root,
                session_id,
                {"message": message or DEFAULT_EXTRA_MESSAGE, "at": _iso_due(due), "notify": True},
                now_ms=now,
            )
        except WakeWriteError as exc:
            if exc.status in (409, 503):
                return
            logger.warning("librarian: could not arm an extra pass: %s", exc)
            return
        except Exception:  # noqa: BLE001 — the tray is retried next boot
            logger.warning("librarian: could not arm an extra pass", exc_info=True)
            return
        taken += 1
    if taken:
        try:
            with state.locked(root):
                state.update_state(root, extras=dict(ledger, armed=armed_today + taken))
        except Exception:  # noqa: BLE001 — the rows exist; the receipt can wait
            logger.warning("librarian: extra ledger write failed", exc_info=True)
    # CLEARED EVEN WHEN NOTHING WAS ARMED: every request above was either armed
    # or judged invalid, and leaving it would re-log the same refusal on every
    # boot. An early return (an owner, a write error) skipped this on purpose.
    state.clear_extras(root)


async def ensure_armed(
    config_dir: Path | str, session_id: str, *, now_ms: int | None = None
) -> str:
    """Arm the cadence for a session with NO live owner. Returns one word.

    The boot/desktop writer: reads the derived index (the only schedule read
    that does not open the session) and writes through
    :mod:`local_operator.wakes.arm` — transcript first, index second, install
    hook third, exactly the discipline every external wake writer shares.

    Returns ``"armed"`` (a cadence row was written), ``"present"`` (one was
    already there), ``"paused"``/``"disabled"`` (nothing armed; any existing
    rows were best-effort cancelled), ``"owner"`` (a live runtime holds the
    session — it reconciles on its own watcher tick), ``"busy"`` (a peer held
    the STORE lock for the whole wait; benign and retried on the next tick or
    boot), ``"no-session"`` (nothing on disk to arm against) or ``"failed"``
    (a refusal/logged error). Never raises: the callers are boot paths whose
    failures must not fail them.
    """
    from local_operator.wakes import store as wake_store
    from local_operator.wakes.arm import (
        STATUS_CONFLICT,
        WakeWriteError,
        arm_wake,
        edit_wake,
        repair_index,
    )

    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not session_id:
        return "no-session"
    try:
        pol = policy(root)
        entry = wake_store.read_entry(root, session_id)
        if entry is None and not (root / "sessions" / session_id).is_dir():
            return "no-session"
        ids = _row_ids(entry)
        own_ids = [i for i in ids if is_librarian_row(i)]
        if not pol.enabled or pol.paused:
            if own_ids:
                await _cancel_ids(root, session_id, own_ids, now_ms=now)
            return "disabled" if not pol.enabled else "paused"
        # An explicit stop is a gate on the whole pass, exactly as in Aida:
        # a stopped session must not come back with its daily row, and the
        # tray must not be drained for a session the platform declared inert.
        if wake_store.is_held(entry):
            return "held"
        result = "present"
        current = _transcript_cadence(root, session_id)
        if current is not None:
            # The transcript wins even if a crash left its derived index behind.
            if CADENCE_ID not in ids:
                await repair_index(root, session_id)
                ids.add(CADENCE_ID)
            candidate = cadence_schedule(now, pol.at, config_dir=root, existing=current)
            if candidate != current:
                await edit_wake(
                    root,
                    session_id,
                    CADENCE_ID,
                    {"message": candidate.message, "at": _iso_due(candidate.next_due_at)},
                    now_ms=now,
                )
                result = "armed"
        elif CADENCE_ID not in ids:
            try:
                candidate = cadence_schedule(now, pol.at, config_dir=root)
                await arm_wake(
                    root,
                    session_id,
                    {
                        "message": candidate.message,
                        "at": _iso_due(candidate.next_due_at),
                        "notify": True,
                    },
                    wake_id=CADENCE_ID,
                    now_ms=now,
                )
                result = "armed"
            except WakeWriteError as exc:
                if exc.status == 503:
                    return "owner"
                if exc.status != STATUS_CONFLICT:
                    return "failed"
                # The row is there; only the derived index is behind (every
                # external arm writes transcript first and index second).
                # Re-project instead of reporting a defect on a working row.
                await repair_index(root, session_id)
                result = "present"
        await _drain_tray_external(root, session_id, pol, now)
        return result
    except WakeLockBusy:
        # A peer holds the store lock; normal on a seam with several attended
        # writers, and the next boot retries.
        return "busy"
    except WakeLockUnavailable:
        logger.info("librarian: the wake store is unavailable; skipping this arm")
        return "failed"
    except Exception:  # noqa: BLE001 — a boot path's failure is a log line
        logger.warning("librarian: ensure_armed failed", exc_info=True)
        return "failed"


async def run_now(config_dir: Path | str, session_id: str, *, now_ms: int | None = None) -> str:
    """Arm one extra pass due now, for the desktop's "Organize now".

    An EXPLICIT operator request, so it is deliberately NOT charged against
    ``max_extra_per_day``: that budget bounds what the seed can ask for by
    itself, not what a person can press. It still refuses while paused or
    disabled, and still refuses a session a live runtime owns (that runtime's
    own reconcile is the writer).
    """
    from local_operator.wakes import store as wake_store
    from local_operator.wakes.arm import WakeWriteError, arm_wake

    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not session_id:
        return "no-session"
    pol = policy(root)
    if not pol.enabled:
        return "disabled"
    if pol.paused:
        return "paused"
    try:
        entry = wake_store.read_entry(root, session_id)
        if wake_store.is_held(entry):
            return "held"
        wake_id = _next_extra_id(_row_ids(entry))
        await arm_wake(
            root,
            session_id,
            {
                "message": DEFAULT_EXTRA_MESSAGE,
                "at": _iso_due(now + 1_000),
                "notify": True,
            },
            wake_id=wake_id,
            now_ms=now,
        )
    except WakeWriteError as exc:
        if exc.status == 503:
            return "owner"
        logger.warning("librarian: could not arm a one-shot pass: %s", exc)
        return "failed"
    except Exception:  # noqa: BLE001 — an op reports a word, never a traceback
        logger.warning("librarian: could not arm a one-shot pass", exc_info=True)
        return "failed"
    return "armed"


def mark_held(config_dir: Path | str, session_id: str, *, now_ms: int | None = None) -> bool:
    """Stamp ``held_at`` on her wake-index entry. Returns whether an entry was found.

    The same derived-index marker discipline as ``stopped_at``: schedules are
    never deleted by a pause, the entry gains a key the supervisor skips, and
    the session's own open keeps it. Best-effort by contract — the config key
    is the authority and the guards above do not consult this file.

    A ROWLESS entry cannot be stamped: ``wake_store.write_entry`` treats an
    empty schedule list as "remove the entry", and a pause must never delete
    one. That shape is covered by the published ``librarian.cadence.paused``
    bit, so the supervisor's record plumbing stops all the same.
    """
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    try:
        entry = wake_store.read_entry(root, session_id)
        if entry is None:
            return False
        schedules = entry.get("schedules") or []
        if not schedules:
            return False
        held_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        wake_store.write_entry(
            root,
            session_id,
            cwd=entry.get("cwd") or "",
            schedules=schedules,
            preserve=dict(entry, held_at=held_ms),
        )
        return True
    except Exception:  # noqa: BLE001 — a derived marker must not cost the pause
        logger.warning("librarian: could not stamp held_at", exc_info=True)
        return False


def clear_held(config_dir: Path | str, session_id: str) -> bool:
    """Drop ``held_at`` from her wake-index entry. Best-effort, like :func:`mark_held`."""
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    try:
        entry = wake_store.read_entry(root, session_id)
        if entry is None:
            return False
        schedules = entry.get("schedules") or []
        if not schedules:
            return False
        wake_store.write_entry(
            root,
            session_id,
            cwd=entry.get("cwd") or "",
            schedules=schedules,
            preserve=dict(entry),
            clear=("held_at",),
        )
        return True
    except Exception:  # noqa: BLE001 — a derived marker must not cost the resume
        logger.warning("librarian: could not clear held_at", exc_info=True)
        return False


def _set_paused_config(config_dir: Path | str, paused: bool) -> None:
    """Write ``librarian.cadence.paused`` through the settings facade.

    Deliberate rather than a raw ``yml`` edit: the config watcher diffs
    REGISTRY keys, so this exact path is what delivers a pause/resume to a
    live session in another process within its 2 s tick, and what makes
    ``/settings`` and ``lop config`` show and set the same key. Raises when the
    registry entry is missing or the write is refused — the caller renders a
    refusal.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    setting = settings_io.BY_KEY.get("librarian.cadence.paused")
    if setting is None:  # pragma: no cover - registry guarantee, pinned by tests
        raise RuntimeError("librarian.cadence.paused is not registered in settings_io")
    settings_io.write_setting(ConfigManager(config_dir=Path(config_dir)), setting, bool(paused))


@dataclass(frozen=True)
class PauseOutcome:
    """What a pause did, for the receipt and the tests."""

    cancelled: tuple[str, ...]
    owner_blocked: bool
    held: bool


async def pause(
    config_dir: Path | str, session_id: str, *, now_ms: int | None = None
) -> PauseOutcome:
    """Pause her proactive output. Config first — it is the authority.

    Order is load-bearing: the config write (a) reaches every live session
    within the watcher tick and (b) makes the load-time filter and the
    delivery guard suppress at once. Only then are her rows cancelled through
    whichever writer can (refused, not failed, when a runtime owns them — the
    watcher reconcile drops them there) and ``held_at`` stamped for the
    supervisor.
    """
    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    with state.locked(root):
        state.update_state(root, paused_at=now)
        _set_paused_config(root, True)
    from local_operator.wakes import store as wake_store

    entry = wake_store.read_entry(root, session_id)
    own_ids = _librarian_ids(entry)
    # A paused agent must not keep a hidden patience timer nagging on her
    # behalf, and a cycle that cannot re-arm is dead weight. Same best-effort
    # writer as her own rows.
    patience_ids = [
        str(row.get("id") or "")
        for row in (entry or {}).get("schedules") or ()
        if isinstance(row, Mapping) and is_patience_row(row)
    ]
    cancelled: list[str] = []
    owner_blocked = False
    if own_ids or patience_ids:
        cancelled, owner_blocked = await _cancel_ids(
            root, session_id, [*own_ids, *patience_ids], now_ms=now
        )
    held = mark_held(root, session_id, now_ms=now)
    return PauseOutcome(cancelled=tuple(cancelled), owner_blocked=owner_blocked, held=held)


async def resume(config_dir: Path | str, session_id: str, *, now_ms: int | None = None) -> str:
    """Resume her proactive output and re-arm at the next cadence time.

    Returns the :func:`ensure_armed` word for the arm attempt (``"owner"``
    means a live session will pick the re-arm up on its watcher tick, which is
    correct and expected, not a failure).
    """
    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    with state.locked(root):
        state.update_state(root, paused_at=None)
        _set_paused_config(root, False)
    clear_held(root, session_id)
    return await ensure_armed(root, session_id, now_ms=now)


def status(config_dir: Path | str, session_id: str = "") -> dict[str, Any]:
    """The read-only status behind the ``status`` op and the route's GET."""
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    pol = policy(root)
    sid = session_id or state.session_id_of(root) or ""
    cadence_due: int | None = None
    if sid:
        entry = wake_store.read_entry(root, sid)
        for raw in (entry or {}).get("schedules") or ():
            if isinstance(raw, Mapping) and raw.get("id") == CADENCE_ID:
                due = raw.get("next_due_at")
                cadence_due = int(due) if isinstance(due, int) else None
                break
    raw = state.read_state(root) or {}
    last_run = raw.get("last_run_at")
    return {
        "enabled": pol.enabled,
        "paused": pol.paused,
        "at": pol.at,
        "session_id": sid or None,
        "cadence_due_at": cadence_due,
        "last_run_at": last_run if type(last_run) is int else None,
        "max_extra_per_day": pol.max_extra_per_day,
        "min_gap_minutes": pol.min_gap_minutes,
    }


def wake_display_label(wake_id: str) -> str:
    """The human label for an engine-armed row; unknown ids name themselves.

    Identity for everything else — a user-created wake's own words ARE its
    identity — so a caller can substitute unconditionally. The label is
    FORMATTED at lookup, never stored: it must read the name she is actually
    called or the receipt contradicts the body it reports.
    """
    if wake_id in WAKE_DISPLAY_LABEL_TEMPLATES:
        template = WAKE_DISPLAY_LABEL_TEMPLATES[wake_id]
    elif wake_id.startswith(EXTRA_ID_PREFIX):
        template = EXTRA_DISPLAY_LABEL_TEMPLATE
    else:
        return wake_id
    # Lazy, like this module's other cross-imports: a label must never be the
    # reason the engine cannot load.
    from local_operator.librarian.bootstrap import display_name

    return template.format(name=display_name())


#: The receipt sentences for a refused :func:`run_now`. Kept beside the op so
#: the route and any later surface cannot invent a second account of it.
STATUS_SENTENCES = {
    "busy": "another operation is holding the wake store; try again.",
    "failed": "the wake store refused the write; see the logs.",
}
