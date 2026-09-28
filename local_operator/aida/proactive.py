"""Aida's proactive cadence engine — the ONE re-armer of her ``aida-*`` wakes.

WHAT THE ENGINE OWNS. Exactly one internal cadence row, id ``aida-cadence``,
armed as a one-shot due at the next local wall-clock occurrence of
``aida.cadence.at`` (default 09:00); plus the bounded ``aida-extra-N`` one-shots
she requests through her escalation tray. Every row the engine owns has an id
under the ``aida-`` prefix, and that prefix is the ownership boundary: pause,
disable and re-arm all work by dropping or rewriting those ids and never touch
an ordinary ``wN`` row the operator or the agent's own ``wake`` tool armed.

TWO WRITERS, ONE INVARIANT. While a runtime owns her session, schedule state is
written by the session itself — :func:`reconcile` produces the new full list and
the caller hands it to ``Session.set_wake_schedules`` (transcript first, index
second, exactly one writer). With no runtime, :func:`ensure_armed` writes through
:mod:`local_operator.wakes.arm` (the documented external writer). The two never
run against the same session at the same time: ``arm.py`` refuses a session with
a live owner, and the in-session path only ever sees sessions that have one. The
single invariant is: **the engine is the one RE-ARMER of ``aida-*`` rows and
the only code that DROPS them.** Exactly one second armer exists and is
documented rather than hidden: ``onboarding.greet`` adds the one-shot
``aida-greeting`` at most once per install and never touches another row; the
engine itself re-arms that row when it is owed (the ensure in :func:`reconcile`,
and :func:`resume`), and it is the only code that DROPS rows — the earlier
wording ("the only code that creates or drops") was falsified by the onboarding
line and is corrected here (review round 1, m1).

RE-ARM, AND WHY THE SESSION HOOKS RATHER THAN A TIMER. A one-shot retires when
it fires. The next day's row therefore has to be created during (or after) the
fire, and the only writer that may do that for a live session is the session —
so :func:`reconcile` is called from ``Session._persist_wake_schedules`` (the one
editor of the final list, which the scheduler persists right after a fire), from
the session's after-turn seam (so an escalation tray write is consumed at the
end of the turn that made it), and from ``Session._apply_config_change`` (so a
pause or resume issued from OUTSIDE the process — another terminal, the desktop
app — reaches a live session within the config watcher's 2 s tick; that is the
platform's own LIVE-key seam, not a bespoke RPC). ``WakeScheduler.on_retire``
was considered and rejected: it is called under the scheduler's write lock, and
re-arming there would need a lock re-entry that deadlocks.

PAUSE SEMANTICS. ``aida.cadence.paused`` (a config key, so ``/settings`` shows
it and ``lop config`` can set it) is the authority. Pause = write the key +
stamp ``held_at`` on her wake-index entry (the supervisor skips held entries
exactly as it skips stopped ones) + cancel ``aida-*`` rows through whichever
writer owns them. A live session honours the change at its next reconcile (the
config watcher delivers it), and two guards close the windows in between: the
load-time filter (:func:`filter_on_load` — a session opened while held does not
arm ``aida-*`` rows at all) and the delivery-time guard
(:func:`delivery_allowed` — an ``aida-*`` fire that comes due while held is
dropped rather than delivered). Resume clears ``held_at`` and re-arms at the
next cadence time; the live session picks it up on the same watcher tick.

The delivery guard fails OPEN on its own errors (a read failure delivers): the
wake is already durably due by then, the pause is additionally enforced at
load, in reconcile and by the supervisor's skip, and eating a scheduled turn
over a transient read error is the worse trade. It is fail-CLOSED for the one
case it exists to close — a readable "paused" answer always suppresses.

DISABLE. ``aida.enabled`` false, or ``LOCAL_OPERATOR_NO_AIDA`` truthy, is a
harder switch than pause: creation never happens (:func:`local_operator.aida.
bootstrap.ensure_session` gates first) and any existing ``aida-*`` rows are
dropped at the next reconcile/load. It is not a pause — nothing is remembered,
because "disabled installs" are installs that never wanted her.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_operator.aida import state
from local_operator.harness.wake_types import MAX_WAKE_SCHEDULES, WakeSchedule

logger = logging.getLogger(__name__)

#: The cadence row's id. Stable across arming writers: the engine re-arms by
#: rebuilding this row, so an id that drifted between writers would leave two
#: cadence rows (or none) depending on which side armed last.
CADENCE_ID = "aida-cadence"

#: The id prefix of engine-armed escalation one-shots. ``aida-extra-<n>`` with
#: the lowest free ``n``, chosen from the ids present at reconcile time.
EXTRA_ID_PREFIX = "aida-extra-"

#: The ownership prefix, in one place. ``is_aida_row`` is the ONE reader, so a
#: future internal row (a third kind) joins the family by naming, not by
#: touching every filter.
ROW_PREFIX = "aida-"

#: Transcript custom-entry type for engine notes — budget refusals and similar
#: bounds made observable to the agent (and to a human reading the transcript).
#: Custom entries never enter LLM context, so a note is for the reader, exactly
#: like the refusal it records.
CUSTOM_ENTRY_TYPE = "aida_proactive"

DEFAULT_CADENCE_AT = "09:00"
DEFAULT_MAX_EXTRA_PER_DAY = 2
DEFAULT_MIN_GAP_MINUTES = 90
#: The two boolean defaults this engine reads. ``DEFAULT_ENABLED`` is also the
#: boot gate's fallback (``bootstrap.config_enabled`` re-exports it, so the
#: gate and the engine cannot disagree about "absent means enabled"), and both
#: are pinned against their registry rows by the settings anti-drift test.
DEFAULT_ENABLED = True
DEFAULT_PAUSED = False

#: The cadence self-prompt. It carries the two rules the engine cannot enforce
#: on her behalf: report only what needs action (the anti-spam half of R16),
#: and route escalation requests through the tray rather than arming wakes
#: directly (so the engine keeps its single-writer property).
CADENCE_MESSAGE = (
    "Daily proactive check-in. Review the operator's current state — active and "
    "stale sessions, projects and workstreams, scheduled wakes, usage signals, and "
    "anything you started earlier that is still in flight — and report ONLY what "
    "needs the operator's action, in a few short lines. If there is nothing "
    'actionable, reply with exactly "(no action needed)" and nothing else. To '
    "schedule a follow-up check, write it to the escalation tray described in your "
    "instructions instead of arming wakes directly."
)

#: The self-prompt an escalation request without its own message gets. The tray
#: format lets her give a message; when she does not, this is a useful default
#: rather than an empty turn.
DEFAULT_EXTRA_MESSAGE = (
    "Proactive follow-up: revisit what you flagged earlier and report only if "
    "something still needs the operator's action."
)


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


def _read_aida_section(config_dir: Path | str) -> Mapping[str, Any]:
    """The ``aida`` config mapping, best-effort.

    A fresh install has no ``aida`` section at all — absent means defaults, and
    a config file that cannot be read means defaults too: the cadence's whole
    job is to try again tomorrow, so a broken read must never be *louder* than
    the settings it could not find. A hand-edited section (a string where a
    mapping belongs) degrades the same way.
    """
    try:
        from local_operator.config import ConfigManager

        raw = ConfigManager(config_dir=Path(config_dir)).get_config_value("aida", None)
    except Exception:  # noqa: BLE001 — defaults are the answer to an unreadable file
        logger.debug("aida: could not read the config section", exc_info=True)
        return {}
    return raw if isinstance(raw, Mapping) else {}


def policy(config_dir: Path | str) -> CadencePolicy:
    """Resolve the cadence policy: config keys + the environment kill switch."""
    section = _read_aida_section(config_dir)
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

    The one validator for ``aida.cadence.at``. Invalid values fall back to the
    default at :func:`next_cadence_ms` rather than raising: the key is free
    text in the settings registry, and a typo must not cost her the cadence
    (it falls back visibly — the fallback is logged).
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
    date, so a DST transition keeps the requested clock time (the same rule
    ``parse_wake_at``'s ``HH:MM`` branch documents). An unparseable ``at``
    falls back to :data:`DEFAULT_CADENCE_AT`, logged once per call site's
    patience — it is read per arm, so the log is throttled to DEBUG.
    """
    clock = parse_cadence_at(at)
    if clock is None:
        logger.debug("aida: invalid cadence time %r; using %s", at, DEFAULT_CADENCE_AT)
        clock = parse_cadence_at(DEFAULT_CADENCE_AT)
        assert clock is not None
    hour, minute = clock
    now = datetime.fromtimestamp(now_ms / 1000.0)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        # Calendar-day advance, not +24h: see the docstring.
        target = datetime.combine(now.date() + timedelta(days=1), target.time())
    return int(target.timestamp() * 1000)


def is_aida_row(wake_id: str) -> bool:
    """Whether ``wake_id`` is a row this engine owns."""
    return isinstance(wake_id, str) and wake_id.startswith(ROW_PREFIX)


def cadence_schedule(now_ms: int, at: str = DEFAULT_CADENCE_AT) -> WakeSchedule:
    """The cadence row: one-shot, due at the next ``at``.

    ``created_at`` carries ``now`` so the scheduler's stable ordering keeps it
    where it was planted; ``next_due_at`` is the next occurrence, never "now"
    (a due-now cadence would fire inside ``LOAD_GRACE_MS`` of an ensure, which
    is a boot-time turn nobody asked for).
    """
    return WakeSchedule(
        id=CADENCE_ID,
        message=CADENCE_MESSAGE,
        next_due_at=next_cadence_ms(now_ms, at),
        every_ms=None,
        created_at=now_ms,
    )


def hold_active(config_dir: Path | str) -> bool:
    """Paused or disabled — the state in which ``aida-*`` rows are not armed.

    The load-time filter and the delivery guard both ask this ONE question, so
    "held" cannot mean two things on two seams.
    """
    pol = policy(config_dir)
    return (not pol.enabled) or pol.paused


def delivery_allowed(config_dir: Path | str) -> bool:
    """Whether an ``aida-*`` fire may deliver right now (fail-open, see module).

    Returns True when the policy cannot be read; see the module docstring for
    why a transient read failure delivers rather than eats the wake.
    """
    try:
        return not hold_active(config_dir)
    except Exception:  # noqa: BLE001 — documented fail-open
        logger.warning("aida: could not resolve the hold state; delivering", exc_info=True)
        return True


def filter_on_load(schedules: Sequence[WakeSchedule], config_dir: Path | str) -> list[WakeSchedule]:
    """The load-time hold: drop ``aida-*`` rows while held. Never raises.

    Called from ``Session._load_wake_schedules`` for her session only (the
    caller gate is :func:`local_operator.aida.state.is_aida_session`, one
    stat). Rows the filter drops are dropped from the in-memory list, so the
    session's next persist writes the cancellation into the transcript — that
    is the "pause cancels ``aida-*`` rows" contract reached through the ONE
    writer rather than a file edit behind the session's back.
    """
    if not any(is_aida_row(s.id) for s in schedules):
        return list(schedules)
    if not hold_active(config_dir):
        return list(schedules)
    kept = [s for s in schedules if not is_aida_row(s.id)]
    logger.info("aida: holding %d cadence row(s) while paused/disabled", len(schedules) - len(kept))
    return kept


@dataclass
class ReconcileResult:
    """The full-list reconcile's new list, plus notes to journal.

    ``notes`` are human-readable refusals (budget, spacing, capacity) that the
    caller appends to the transcript as :data:`CUSTOM_ENTRY_TYPE` entries —
    the design's "observable, not silent" bound. ``changed`` is True when the
    list differs from the input, so a caller can skip a write (and the
    transcript append it would cost) when nothing moved.
    """

    schedules: list[WakeSchedule]
    notes: list[str]
    changed: bool


def _extras_today(config_dir: Path | str, now_ms: int) -> tuple[dict[str, Any], int]:
    """The escalation ledger for TODAY, resetting across local days.

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


def _parse_extra_due(request: Mapping[str, Any], now_ms: int) -> tuple[int | None, str, str]:
    """``(due_ms, message, refusal)`` for one escalation request.

    The request grammar is the wake tool's own subset — ``in`` or ``at``, plus
    an optional ``message`` — resolved through the SAME parsers the tool uses
    (``harness.wake.parse_wake_at`` / ``parse_wake_duration``), so a request she
    writes in the tray is validated exactly like one the tool would have taken.
    A bare string is accepted as a timing-only shorthand (``"in 4h"``), because
    that is the shape the design's example uses; anything else is refused with
    a note, never raised.
    """
    from local_operator.harness.wake import parse_wake_at, parse_wake_duration

    if isinstance(request, str):
        # The design's shorthand: a bare ``"in 4h"`` / ``"at 14:00"`` string.
        # Split on the leading word rather than passing the whole string on —
        # ``parse_wake_duration("in 4h")`` is None, so the naive form would
        # refuse exactly the example the tray format documents.
        text = request.strip()
        lowered = text.lower()
        if lowered.startswith("at "):
            request = {"at": text[3:].strip()}
        elif lowered.startswith("in "):
            request = {"in": text[3:].strip()}
        else:
            request = {"in": text}
    if not isinstance(request, Mapping):
        return None, "", "ignored a malformed escalation request (not an object)."
    message = request.get("message")
    message = message.strip() if isinstance(message, str) and message.strip() else ""
    in_val, at_val = request.get("in"), request.get("at")
    due: int | None = None
    if in_val is not None:
        duration = parse_wake_duration(str(in_val))
        if duration is None:
            refusal = f"ignored an escalation request with an invalid 'in' ({in_val!r})."
            return None, message, refusal
        due = now_ms + duration
    elif at_val is not None:
        due = parse_wake_at(str(at_val), now_ms)
        if due is None:
            refusal = f"ignored an escalation request with an invalid 'at' ({at_val!r})."
            return None, message, refusal
    if due is None:
        return None, message, "ignored an escalation request with neither 'in' nor 'at'."
    # The same past-time floor build_wake_schedule applies: up to 5 s in the
    # past fires immediately, anything older is a mistake worth a note.
    if due < now_ms - 5_000:
        return None, message, "ignored an escalation request whose time is in the past."
    return max(due, now_ms + 1_000), message, ""


def _next_extra_id(existing: set[str]) -> str:
    """The lowest free ``aida-extra-<n>`` id."""
    n = 1
    while f"{EXTRA_ID_PREFIX}{n}" in existing:
        n += 1
    return f"{EXTRA_ID_PREFIX}{n}"


def reconcile(
    schedules: Sequence[WakeSchedule],
    *,
    config_dir: Path | str,
    session_id: str,
    now_ms: int | None = None,
) -> ReconcileResult:
    """The in-session full-list reconcile. Pure file side effects: the ledger.

    Order and content, in one place so the persist seam and the after-turn seam
    cannot diverge:

    1. Not her session ⇒ untouched (identity is read once, here, so a caller
       cannot forget the gate).
    2. Disabled or paused ⇒ drop every ``aida-*`` row; nothing is re-armed and
       the tray is left where it is (a paused assistant must not queue work for
       her own resume).
    3. Active ⇒ keep every row — hers included, so an escalation in flight is
       never silently cancelled — ensure exactly one ``aida-cadence`` row when
       none is present, and consume the escalation tray within
       ``max_extra_per_day`` / ``min_gap_minutes`` / the 16-row cap.

    The ledger mutation happens under :func:`state.locked` — the only slow-ish
    work here (a file read + write) — and everything else is in-memory. The
    caller journals ``notes``; this function never touches the transcript, so it
    stays callable from the scheduler's write lock without a second writer.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    original = list(schedules)
    if not state.is_aida_session(config_dir, session_id):
        return ReconcileResult(schedules=original, notes=[], changed=False)

    pol = policy(config_dir)
    notes: list[str] = []
    if not pol.enabled or pol.paused:
        kept = [s for s in original if not is_aida_row(s.id)]
        if pol.paused and any(s.id == GREETING_WAKE_ID for s in original):
            # THE ONE-TIME GREETING DIES WITH THE PAUSE (review round 1, m1).
            # It is a one-shot with no retry of its own, so an armed-but-
            # unfired row dropped here would be lost for good while
            # ``greeted_at`` kept claiming it was delivered — and the paused
            # greet receipt promises a resume delivers it. Un-stamp, and the
            # resume (or the active branch below) arms it again.
            _clear_greeted(config_dir)
        return ReconcileResult(schedules=kept, notes=notes, changed=kept != original)
    # ACTIVE: keep everything, including her existing rows. An extra she asked
    # for earlier is a request in flight — dropping it here would silently
    # cancel it at the next persist (a turn's end, a fire, any wake-tool
    # mutation), which is precisely the "silently dead reminder" failure the
    # wake subsystem keeps removing. Missed-while-down occurrences are NOT
    # special-cased either: they stay and the generic catch-up path delivers
    # them ("one wake implementation").
    kept = list(original)

    # -- escalation tray ----------------------------------------------------
    # THE TRAY IS CONSUMED INSIDE THE LOCK (review round 1, M1b). Consuming
    # first and locking second destroyed the whole batch whenever the lock was
    # contended, and the note that came out of it claimed the opposite —
    # "left unread" — about a file that had already been unlinked. Consuming
    # under the lock means a contended acquisition leaves the tray exactly
    # where it was, and the note below is now true.
    try:
        with state.locked(config_dir):
            requests = state.consume_escalations(config_dir)
            if requests:
                ledger, armed = _extras_today(config_dir, now)
                taken = 0
                for request in requests:
                    if pol.max_extra_per_day <= 0:
                        notes.append("escalation request refused: escalation is disabled (0/day).")
                        continue
                    if armed + taken >= pol.max_extra_per_day:
                        notes.append(
                            f"escalation request refused: budget exhausted "
                            f"({pol.max_extra_per_day}/day)."
                        )
                        continue
                    if len(kept) + taken >= MAX_WAKE_SCHEDULES:
                        notes.append("escalation request refused: the schedule list is full.")
                        continue
                    due, message, refusal = _parse_extra_due(request, now)
                    if refusal:
                        notes.append(refusal)
                        continue
                    assert due is not None
                    gap_ms = pol.min_gap_minutes * 60_000
                    # The spacing floor is measured against the OTHER Aida rows
                    # and against the cadence's own next occurrence — the
                    # cadence row may not be in ``kept`` yet (it is appended
                    # last), so its would-be due time is computed rather than
                    # looked up.
                    cadence_due = _cadence_due(kept, pol, now)
                    too_close = any(
                        abs(row.next_due_at - due) < gap_ms for row in kept if is_aida_row(row.id)
                    ) or (cadence_due is not None and abs(cadence_due - due) < gap_ms)
                    if too_close:
                        notes.append(
                            "escalation request refused: within the "
                            f"{pol.min_gap_minutes}-minute spacing floor of another Aida wake."
                        )
                        continue
                    kept.append(
                        WakeSchedule(
                            id=_next_extra_id({row.id for row in kept}),
                            message=message or DEFAULT_EXTRA_MESSAGE,
                            next_due_at=due,
                            every_ms=None,
                            created_at=now,
                        )
                    )
                    taken += 1
                if taken:
                    state.update_state(config_dir, extras=dict(ledger, armed=armed + taken))
    except Exception:  # noqa: BLE001 — a contended lock must not eat the turn
        logger.warning("aida: escalation consume failed", exc_info=True)
        # TRUE NOW, and it was not before (M1b): the tray is consumed inside
        # the lock, so a contended acquisition leaves the file exactly where
        # it was. The sentence is what makes the failure observable.
        notes.append("escalation tray left unread (another Aida operation is in flight).")

    # -- cadence ------------------------------------------------------------
    if not any(row.id == CADENCE_ID for row in kept):
        # REUSE the existing cadence row whenever one exists — rebuilding it
        # would mint a new ``created_at`` on every reconcile, which makes
        # ``changed`` permanently true (so a watcher tick or an after-turn
        # drain would rewrite the transcript entry for nothing) and would move
        # an instant the user may have just read in ``/aida status``. An
        # OVERDUE one is kept on purpose rather than re-armed forward: the
        # generic wake machinery owns missed-while-down delivery, and this
        # engine deliberately has no second implementation of it.
        existing = next((row for row in original if row.id == CADENCE_ID), None)
        kept.append(existing if existing is not None else cadence_schedule(now, pol.at))

    # -- the one-time greeting, on the same ensure rule as the cadence -------
    # A live session is the ONLY writer that can arm it when the owner holds
    # the rows (`arm_wake` refuses with a 503), so an owed greeting is
    # re-armed here rather than left to a caller that cannot write (review
    # round 1, m1). Owed means ``greeted_at is None`` — never armed, or
    # un-stamped by the pause above — and the provider gate is the same one
    # ``onboarding.greet`` applies, because the turn cannot run without one.
    if not any(row.id == GREETING_WAKE_ID for row in kept):
        from local_operator.aida import onboarding as _onboarding

        if _onboarding.greeted_at(config_dir) is None and _onboarding.provider_configured(
            config_dir
        ):
            kept.append(
                WakeSchedule(
                    id=GREETING_WAKE_ID,
                    message=_onboarding.GREETING_MESSAGE,
                    next_due_at=now,
                    every_ms=None,
                    created_at=now,
                )
            )
            # And the ledger moves WITH the arm on this path too: without the
            # stamp a later reconcile (the row retired after firing) would find
            # no greeting row and arm a second one.
            _onboarding.mark_greeted(config_dir, now)
    return ReconcileResult(schedules=kept, notes=notes, changed=kept != original)


def _cadence_due(rows: Sequence[WakeSchedule], pol: CadencePolicy, now_ms: int) -> int | None:
    """The cadence row's due time, or the time it *would* be armed for.

    Used only by the spacing check: the cadence row may not exist yet in the
    in-memory list (it is appended last), and its future occurrence is exactly
    the due time the spacing floor must consider.
    """
    for row in rows:
        if row.id == CADENCE_ID:
            return row.next_due_at
    return next_cadence_ms(now_ms, pol.at)


async def append_notes(transcript: Any, notes: Sequence[str]) -> None:
    """Append engine notes to a transcript, best-effort.

    ``notes`` are refusals the bounds produced (budget exhausted, spacing
    floor, list full); journaling them as :data:`CUSTOM_ENTRY_TYPE` custom
    entries is what makes the bound "observable, not silent" — custom entries
    never enter the model's context, so the note is for whoever reads the
    transcript next, human or agent. Every failure is swallowed: a note about
    a bound is observation, and observation may never cost the write it
    observes.
    """
    for note in notes:
        text = str(note).strip()
        if not text:
            continue
        try:
            await transcript.append_custom(CUSTOM_ENTRY_TYPE, {"note": text})
        except Exception:  # noqa: BLE001 — observation never costs the write
            logger.warning("aida: could not journal a proactive note", exc_info=True)


# ---------------------------------------------------------------------------
# External (no-runtime) operations — the arm.py side of the engine
# ---------------------------------------------------------------------------


def _row_ids(entry: Mapping[str, Any] | None) -> set[str]:
    ids: set[str] = set()
    for raw in (entry or {}).get("schedules") or ():
        if isinstance(raw, Mapping) and isinstance(raw.get("id"), str):
            ids.add(raw["id"])
    return ids


#: The one-shot the onboarding greeting arms. It lives with `onboarding`
#: (which owns its message and stamp) and is imported lazily so this module
#: and that one do not form a load-time cycle.
GREETING_WAKE_ID = "aida-greeting"


def _clear_greeted(config_dir: Path | str) -> None:
    """Un-stamp the greeting via `onboarding`, best-effort."""
    try:
        from local_operator.aida import onboarding as _onboarding

        _onboarding.clear_greeted(config_dir)
    except Exception:  # noqa: BLE001 — a stamp must not cost a pause/reconcile
        logger.warning("aida: could not clear the greeting stamp", exc_info=True)


def _aida_ids(entry: Mapping[str, Any] | None) -> list[str]:
    return sorted(wake_id for wake_id in _row_ids(entry) if is_aida_row(wake_id))


async def _cancel_ids(
    root: Path,
    session_id: str,
    ids: Sequence[str],
    *,
    now_ms: int,
) -> tuple[list[str], bool]:
    """Cancel each id through the external writer. Returns (cancelled, owner_blocked).

    ``owner_blocked`` is True when ``arm.py`` refused because a live runtime
    holds the session — the one refusal that is not an error: the live session
    owns its rows, the pause/resume config write reaches it through the config
    watcher, and its own reconcile drops or re-arms them there. A 404 (row
    already gone) is a success for this purpose.
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
            logger.debug("aida: cancel of %s refused: %s", wake_id, exc)
        except Exception:  # noqa: BLE001 — best-effort: the config is the authority
            logger.warning("aida: cancel of %s failed", wake_id, exc_info=True)
    return cancelled, False


def _iso_due(due_ms: int) -> str:
    """A due instant in the wake grammar's ISO form (local, offset-aware).

    ``parse_wake_at`` accepts ISO-8601 and resolves a naive string against the
    LOCAL clock, so the round trip is symmetric; building it from the same
    ``next_cadence_ms`` the in-session path uses keeps the two arm paths
    agreeing on the instant they mean.
    """
    return datetime.fromtimestamp(due_ms / 1000).isoformat()


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
    Aida rows were best-effort cancelled), ``"owner"`` (a live runtime holds
    the session — it will reconcile on its own watcher tick), ``"no-session"``
    (nothing on disk to arm against) or ``"failed"`` (a refusal/logged error).
    Never raises: the callers are boot paths whose failures must not fail them.
    """
    from local_operator.wakes import store as wake_store
    from local_operator.wakes.arm import WakeWriteError, arm_wake

    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not session_id:
        # No session minted yet (a pause before her first open, or a resume
        # back to that state): nothing on disk to arm against, and the next
        # ``ensure_session`` arms the cadence itself.
        return "no-session"
    try:
        pol = policy(root)
        entry = wake_store.read_entry(root, session_id)
        if entry is None and not (root / "sessions" / session_id).is_dir():
            return "no-session"
        ids = _row_ids(entry)
        aida_ids = [i for i in ids if is_aida_row(i)]
        if not pol.enabled or pol.paused:
            if aida_ids:
                await _cancel_ids(root, session_id, aida_ids, now_ms=now)
            return "disabled" if not pol.enabled else "paused"
        result = "present"
        if CADENCE_ID not in ids:
            try:
                await arm_wake(
                    root,
                    session_id,
                    {"message": CADENCE_MESSAGE, "at": _iso_due(next_cadence_ms(now, pol.at))},
                    wake_id=CADENCE_ID,
                    now_ms=now,
                )
                result = "armed"
            except WakeWriteError as exc:
                return "owner" if exc.status == 503 else "failed"
        await _drain_tray_external(root, session_id, pol, now)
        return result
    except Exception:  # noqa: BLE001 — boot paths must not fail on her account
        logger.warning("aida: ensure_armed failed", exc_info=True)
        return "failed"


async def _drain_tray_external(
    root: Path, session_id: str, pol: CadencePolicy, now: int
) -> list[str]:
    """Consume the escalation tray with no runtime — the same bounds, one writer.

    Mirrors :func:`reconcile`'s tray branch (budget, spacing floor, 16-row
    cap) over ``arm.py`` calls instead of an in-memory list. Arms with the
    engine's own ``aida-extra-N`` id via ``arm_wake``'s ``wake_id`` parameter,
    which is what lets a later pause find and cancel exactly these rows.

    Notes (refusals) are logged here rather than journaled: journaling takes a
    transcript writer, and the transcript's owner is the session — which does
    not exist on this path. The in-session reconcile journals the same notes
    to her transcript; this path is the recovery sweep for a tray left behind
    by a process that died mid-turn.
    """
    from local_operator.wakes import store as wake_store
    from local_operator.wakes.arm import WakeWriteError, arm_wake

    notes: list[str] = []
    entry = wake_store.read_entry(root, session_id)
    ids = _row_ids(entry)
    taken = 0
    with state.locked(root):
        # CONSUMED INSIDE THE LOCK, like the in-session branch (review round
        # 1, M1b): a contended lock must leave the tray untouched rather than
        # eat it on the way to writing a note about a file that is gone.
        requests = state.consume_escalations(root)
        if not requests:
            return []
        ledger, armed = _extras_today(root, now)
        due_rows: list[int] = [
            int(raw["next_due_at"])
            for raw in (entry or {}).get("schedules") or ()
            if isinstance(raw, Mapping)
            and isinstance(raw.get("id"), str)
            and is_aida_row(str(raw["id"]))
            and isinstance(raw.get("next_due_at"), int)
        ]
        cadence_due = next_cadence_ms(now, pol.at)
        for request in requests:
            if pol.max_extra_per_day <= 0:
                notes.append("escalation request refused: escalation is disabled (0/day).")
                continue
            if armed + taken >= pol.max_extra_per_day:
                notes.append(
                    f"escalation request refused: budget exhausted "
                    f"({pol.max_extra_per_day}/day)."
                )
                continue
            if len(ids) + taken >= MAX_WAKE_SCHEDULES:
                notes.append("escalation request refused: the schedule list is full.")
                continue
            due, message, refusal = _parse_extra_due(request, now)
            if refusal or due is None:
                notes.append(refusal or "escalation request unreadable.")
                continue
            gap_ms = pol.min_gap_minutes * 60_000
            too_close = (abs(cadence_due - due) < gap_ms) or any(
                abs(existing - due) < gap_ms for existing in due_rows
            )
            if too_close:
                notes.append(
                    "escalation request refused: within the "
                    f"{pol.min_gap_minutes}-minute spacing floor of another Aida wake."
                )
                continue
            extra_id = _next_extra_id(ids)
            try:
                await arm_wake(
                    root,
                    session_id,
                    {"message": message or DEFAULT_EXTRA_MESSAGE, "at": _iso_due(due)},
                    wake_id=extra_id,
                    now_ms=now,
                )
            except WakeWriteError as exc:
                # A 503 means an owner appeared mid-drain. "Leave the rest for
                # that owner" is only implementable by WRITING IT BACK (review
                # round 1, M1a): the owner's reconcile reads the FILE, and this
                # sweep unlinked it two dozen lines up — the first cut broke
                # here and the consumed requests simply vanished, which is the
                # silent loss the tray exists to make visible. The failed
                # request goes back with the ones after it, and the handover is
                # named in the notes so the outcome is observable rather than
                # inferred from a later row count.
                logger.debug("aida: external extra arm refused: %s", exc)
                remaining = list(requests[requests.index(request) :])
                state.restore_escalations(root, remaining)
                notes.append(f"escalation request(s) left for her live session: {len(remaining)}")
                break
            except Exception:  # noqa: BLE001 — one refusal must not stop the sweep
                logger.warning("aida: external extra arm failed", exc_info=True)
                continue
            ids.add(extra_id)
            # THE SPACING SET GROWS WITH EACH ARM (QA round 1, Q1): this drain
            # is the second writer of the same bound the in-session reconcile
            # enforces against its growing `kept` list, and a set frozen at
            # entry accepted `in 1m` + `in 2m` with a 90-minute floor between
            # them — the two paths disagreed about the rule the README states.
            due_rows.append(due)
            taken += 1
        if taken:
            state.update_state(root, extras=dict(ledger, armed=armed + taken))
    for note in notes:
        logger.info("aida: %s", note)
    return notes


def mark_held(config_dir: Path | str, session_id: str, *, now_ms: int | None = None) -> bool:
    """Stamp ``held_at`` on her wake-index entry. Returns whether an entry was found.

    The same derived-index marker discipline as ``stopped_at``
    (``control._mark_wakes_dormant``): schedules are never deleted by a pause,
    the entry gains a key the supervisor skips, and the session's own open
    keeps it (the rebuild clears only ``stopped_at``). Best-effort by
    contract — the config key is the authority and the guards above do not
    consult this file — so every failure is logged and answered ``False``.
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
        logger.warning("aida: could not stamp held_at", exc_info=True)
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
        logger.warning("aida: could not clear held_at", exc_info=True)
        return False


def _set_paused_config(config_dir: Path | str, paused: bool) -> None:
    """Write ``aida.cadence.paused`` through the settings facade.

    The settings write is deliberate rather than a raw ``yml`` edit: the
    config watcher diffs REGISTRY keys, so this exact path is what delivers a
    pause/resume to a live session in another process within its 2 s tick
    (``Session._apply_config_change``), and what makes ``/settings`` and
    ``lop config`` show and set the same key. Raises when the registry entry
    is missing or the write is refused — the caller renders a refusal.
    """
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    setting = settings_io.BY_KEY.get("aida.cadence.paused")
    if setting is None:  # pragma: no cover - registry guarantee, pinned by tests
        raise RuntimeError("aida.cadence.paused is not registered in settings_io")
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

    aida_ids = _aida_ids(wake_store.read_entry(root, session_id))
    cancelled: list[str] = []
    owner_blocked = False
    if aida_ids:
        cancelled, owner_blocked = await _cancel_ids(root, session_id, aida_ids, now_ms=now)
    if GREETING_WAKE_ID in cancelled:
        # Cancelled before it could fire: the greeting is OWED again, not
        # delivered, and the resume re-arms it (review round 1, m1).
        _clear_greeted(root)
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
    # The greeting first, and best-effort: the cadence arm below is the
    # load-bearing half of a resume. When a runtime owns the rows this refuses
    # with "owner" and the owner's own reconcile arms it (the ensure in
    # `reconcile`), which is what makes the paused greet receipt's promise —
    # "/aida resume delivers it" — true on both paths (review round 1, m1).
    try:
        from local_operator.aida import onboarding as _onboarding

        await _onboarding.greet(root, session_id, now_ms=now)
    except Exception:  # noqa: BLE001 — a greeting must not fail the resume
        logger.warning("aida: could not re-arm the greeting on resume", exc_info=True)
    return await ensure_armed(root, session_id, now_ms=now)


def status(config_dir: Path | str, session_id: str = "") -> dict[str, Any]:
    """The read-only status behind the ``status`` op and the receipts."""
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
    now = int(time.time() * 1000)
    ledger, armed_today = _extras_today(root, now) if sid else ({}, 0)
    return {
        "enabled": pol.enabled,
        "paused": pol.paused,
        "at": pol.at,
        "session_id": sid or None,
        "cadence_due_at": cadence_due,
        "extras_today": armed_today,
        "max_extra_per_day": pol.max_extra_per_day,
        "min_gap_minutes": pol.min_gap_minutes,
    }
