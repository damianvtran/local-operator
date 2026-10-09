"""Aida's proactive cadence engine — the ONE re-armer of her ``aida-*`` wakes.

WHAT THE ENGINE OWNS. Exactly one internal cadence row, id ``aida-cadence``,
armed as a one-shot due at the next local wall-clock occurrence of
``aida.cadence.at`` (default 08:30); plus the bounded ``aida-extra-N`` one-shots
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

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_operator.aida import state
from local_operator.harness.wake_types import MAX_WAKE_SCHEDULES, WakeSchedule
from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable
from local_operator.wakes.store import is_patience_row

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

DEFAULT_CADENCE_AT = "08:30"
DEFAULT_MAX_EXTRA_PER_DAY = 2
DEFAULT_MIN_GAP_MINUTES = 90
#: The two boolean defaults this engine reads. ``DEFAULT_ENABLED`` is also the
#: boot gate's fallback (``bootstrap.config_enabled`` re-exports it, so the
#: gate and the engine cannot disagree about "absent means enabled"), and both
#: are pinned against their registry rows by the settings anti-drift test.
DEFAULT_ENABLED = True
DEFAULT_PAUSED = False

#: The ONE spelling of "nothing needs you today". Three readers share it so they
#: cannot drift apart: the cadence prompt and the escalation extras ASK for it,
#: and the settle-time banner veto (``Session._aida_cadence_banner_veto``)
#: RECOGNISES it through :func:`reply_is_quiet`. A reply that normalises to this
#: is a quiet day and must never put a banner on the operator's screen.
QUIET_REPLY = "(no action needed)"

#: The sentence that asks for :data:`QUIET_REPLY`. Appended to a check-in message
#: that does not already carry the sentinel (see :func:`with_quiet_clause`) —
#: otherwise an escalation extra whose model-written message never mentions it
#: would be answered in free prose, and the veto would then let a "nothing to
#: report" banner through.
QUIET_CLAUSE = (
    f'If nothing needs the operator\'s action, reply with exactly "{QUIET_REPLY}" and '
    "nothing else."
)

#: The cadence self-prompt. It carries the two rules the engine cannot enforce
#: on her behalf: report only what needs action (the anti-spam half of R16),
#: and route escalation requests through the tray rather than arming wakes
#: directly (so the engine keeps its single-writer property).
CADENCE_MESSAGE = (
    "Daily proactive check-in. Review the operator's current state — active and "
    "stale sessions, projects and workstreams, scheduled wakes, usage signals, and "
    "anything you started earlier that is still in flight — and report ONLY what "
    "needs the operator's action, in a few short lines. If there is nothing "
    "actionable, give the one quiet-day tip below in a sentence when this message "
    f'carries one, otherwise reply with exactly "{QUIET_REPLY}" and nothing '
    "else. To schedule a follow-up check, write it to the escalation tray described in your "
    "instructions instead of arming wakes directly."
)

#: The self-prompt an escalation request without its own message gets. The tray
#: format lets her give a message; when she does not, this is a useful default
#: rather than an empty turn.
DEFAULT_EXTRA_MESSAGE = (
    "Proactive follow-up: revisit what you flagged earlier and report only if "
    f"something still needs the operator's action. {QUIET_CLAUSE}"
)

#: How many banners Aida may raise in a rolling 24 hours, across the cadence and
#: every escalation extra. A CONSTANT, not a config key: a key would drag in the
#: ``/settings`` registry (AGENTS.md "Adding a configuration key") for a safety
#: net that should never be the thing a user tunes. The arithmetic is why it is
#: 3: one cadence + ``DEFAULT_MAX_EXTRA_PER_DAY`` extras is the engine's own
#: ceiling, so the cap only bites when a catch-up or a loop has produced MORE
#: check-ins than the engine meant to arm. A chief of staff that pings more than
#: this gets her notifications switched off, which silences the ones that matter.
MAX_BANNERS_PER_DAY = 3

#: The window :data:`MAX_BANNERS_PER_DAY` is counted over.
BANNER_WINDOW_MS = 24 * 3_600_000

#: THE ONE-LINE SWITCH FOR TIPS. A quiet-day tip (``onboarding.tip_offer``) is a
#: SILENT row by decision: it lands in her conversation and keeps the unseen
#: mark, so any surface shows it unread, but it does not raise an OS banner —
#: a tip is non-actionable by definition ("use only if nothing needs action") and
#: at most one per 20 h, so a banner for it is the operator being interrupted to
#: be told something optional. The operator may overrule this later; flipping the
#: constant to ``True`` is the whole change (the veto then lets a tip through, and
#: it still counts against :data:`MAX_BANNERS_PER_DAY`).
TIP_REPLY_NOTIFIES = False


def _peel_reply(text: str) -> str:
    """``text`` with symmetric wrappers, padding and a trailing period removed.

    The normalisation :func:`reply_is_quiet` documents: repeatedly strip
    whitespace, one trailing period and ONE symmetric wrapper pair (quotes,
    backticks, parens/brackets, emphasis) until nothing moves — so
    ``(No action needed).``, `` `no action needed` `` and ``"NO ACTION
    NEEDED."`` all land on the bare phrase. Deliberately NO synonym folding:
    a paraphrase ("No action needed today.") stays a banner, and the count of
    those is watched via the ``banners`` ledger (design section 9) before any
    broader matching would be considered.
    """
    pairs = (("`", "`"), ('"', '"'), ("'", "'"), ("(", ")"), ("[", "]"), ("*", "*"), ("_", "_"))
    current = str(text or "")
    for _ in range(8):  # bounded: each round must peel or strip something to continue
        before = current
        current = current.strip().rstrip(".").rstrip()
        for opener, closer in pairs:
            if len(current) >= 2 and current[0] == opener and current[-1] == closer:
                current = current[1:-1].strip()
                break
        if current == before:
            break
    return current


def reply_is_quiet(text: str | None) -> bool:
    """Whether an assistant reply IS the quiet sentinel (:data:`QUIET_REPLY`).

    THE VETO's reply test, and the reason it is not a raw ``==``: the model
    writes ``(No action needed).``, backticked it, pads it, or capitalises it —
    and a raw match would then banner every quiet day, which is how a chief of
    staff gets her notifications switched off (taking the actionable ones with
    them). Case, whitespace, one symmetric wrapper pair and a trailing period
    are folded; a paraphrase is NOT, deliberately (a daily paraphrase banner is
    visible in the ``banners`` ledger, and design option B is the upgrade if
    the count shows drift).
    """
    if not text:
        return False
    folded = _peel_reply(str(text)).casefold()
    return folded in {QUIET_REPLY.casefold(), QUIET_REPLY.strip("()").casefold()}


def reply_is_tip(text: str | None) -> bool:
    """Whether an assistant reply is the quiet-day tip the row asked for.

    Keyed on the ``Tip:`` prefix ``onboarding.TIP_CLAUSE_PREFIX`` asks for
    (plus the handful of close shapes a model writes when it half-forgets).
    A tip is a SILENT row by decision — see :data:`TIP_REPLY_NOTIFIES` for the
    one-line switch that reverses it.
    """
    if TIP_REPLY_NOTIFIES or not text:
        return False
    # The markdown a model may wrap the prefix in is stripped by the peel + lstrip;
    # after that the one shape asked for is a literal "Tip:" opening.
    folded = _peel_reply(str(text)).lstrip("*_ `").casefold()
    return folded.startswith("tip:")


def with_quiet_clause(message: str) -> str:
    """``message`` guaranteed to ASK for :data:`QUIET_REPLY` when it is quiet.

    Applied at the two places an escalation extra's message is finalised (the
    reconcile and the external drain): the cadence prompt carries the sentinel
    itself, but a custom extra message the model wrote is free prose — without
    this clause the engine would never be told the one spelling the veto keys
    on, so a "nothing to report" extra would be reported in paraphrase and
    banner. Idempotent: a message already carrying the sentinel is unchanged.
    """
    text = message or ""
    if QUIET_REPLY in text:
        return text
    return f"{text} {QUIET_CLAUSE}" if text else QUIET_CLAUSE


#: The id prefix of trigger check-in rows: ``aida-trigger-<8 hex>``, the hex
#: deterministic in the consumed record's sorted fingerprints so a re-consume
#: re-arms the SAME row (idempotent) and the engine can diff it against the
#: resident list. Owned by this engine like every other ``aida-*`` row: the
#: wake-trigger layer records a PENDING record
#: (``local_operator.wakes.triggers``); arming and dropping the row stays here.
TRIGGER_ID_PREFIX = "aida-trigger-"

#: Transcript custom-entry type for a consumed trigger receipt: the record's
#: target + instance fingerprints, journaled after a successful persist — the
#: durable receipt the trigger design asks for (§1.7). Context-invisible like
#: :data:`CUSTOM_ENTRY_TYPE`, so it is for whoever reads the transcript next.
TRIGGER_CUSTOM_ENTRY_TYPE = "aida_trigger"

#: Display label for trigger rows on the human receipt surfaces, on the same
#: lazy-map rule as the other engine rows.
TRIGGER_DISPLAY_LABEL_TEMPLATE = "{name}'s check-in"

#: How many instances render into one wake message before "and N more"
#: summarises the rest, and the per-line clip. The record itself caps at 20
#: (``wakes.triggers``'s INSTANCE_CAP); the wake engine's own message bound is
#: the hard stop above this.
_TRIGGER_MESSAGE_MAX_INSTANCES = 20
_TRIGGER_LINE_MAX_CHARS = 200


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


def is_cadence_family_row(wake_id: str) -> bool:
    """Whether ``wake_id`` is a CHECK-IN row: the cadence or an escalation extra.

    The banner veto (``Session._aida_cadence_banner_veto``) applies to these
    rows and only these: trigger check-ins (``aida-trigger-*``) ask her to ACT
    rather than to report, so the sentinel rule does not describe them and their
    rows stay quiet; a user-armed ``wN`` wake is the user's own intent and a
    sentinel reply must never silence it. Kept beside :func:`is_aida_row` so the
    two ownership questions cannot drift apart one call site at a time.
    """
    return isinstance(wake_id, str) and (
        wake_id == CADENCE_ID or wake_id.startswith(EXTRA_ID_PREFIX)
    )


def banner_budget_spent(config_dir: Path | str, *, now_ms: int | None = None) -> bool:
    """Whether her rolling-24h banner budget (:data:`MAX_BANNERS_PER_DAY`) is used up.

    LOCK-FREE BY DESIGN, and that is the load-bearing property: the decision
    runs at settle time on the event loop (``Session._emit`` ->
    ``_finalize_attention_notify``), and ``state.locked`` can park the caller
    for up to its 5 s wait — a check-in settling must never freeze every other
    surface in the process. ``state.read_json`` is an atomic-replace read, so an
    unlocked reader sees either the previous or the next complete file, which
    is all a rolling count needs. Unreadable/absent ledger answers "not spent":
    the budget is a safety net, and a transient read failure must not eat a
    banner the operator asked for (the greeting ledger separately fails
    CLOSED — that one guards the never-contacted rule).
    """
    data = state.read_json(state.onboarding_path(config_dir), what="onboarding") or {}
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return len(_banner_stamps(data, now=now)) >= MAX_BANNERS_PER_DAY


def note_banner_sent(config_dir: Path | str, *, now_ms: int | None = None) -> None:
    """Record one banner against the rolling budget, under the aida lock.

    Called (via ``asyncio.to_thread``) after a publish that carried
    ``notify=True`` — the row is durable first, so a crash between the two errs
    toward one banner counted rather than one forgotten. The write prunes
    stamps older than :data:`BANNER_WINDOW_MS` in the same locked pass, so the
    list cannot grow without bound. Best-effort by contract for its caller: a
    stamp failure must never touch the outcome it describes.
    """
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    path = state.onboarding_path(config_dir)
    with state.locked(config_dir):
        data = state.read_json(path, what="onboarding") or {}
        stamps = _banner_stamps(data, now=now)
        stamps.append(now)
        data["banners"] = stamps
        state.write_json(path, data)


def _banner_stamps(data: Mapping[str, Any], *, now: int) -> list[int]:
    """The budget timestamps still inside the window, as ints.

    Shape-tolerant on read (ints and floats, bools rejected as the house rule
    everywhere in this package) because the file is shared state: an older
    build or a hand-edit must not crash a settle. ``now`` comes from the
    caller so a single evaluation counts against one instant.
    """
    raw = data.get("banners")
    if not isinstance(raw, (list, tuple)):
        return []
    kept: list[int] = []
    for stamp in raw:
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            continue
        value = int(stamp)
        if 0 <= now - value < BANNER_WINDOW_MS:
            kept.append(value)
    return kept


def cadence_message(config_dir: Path | str, now_ms: int) -> str:
    """The cadence row's message: the standing prompt plus today's nudge window.

    R25's bound lives here, at the moment a cadence row is BUILT: the nudge
    permission is part of the row (its message is what the fire carries), and
    ``onboarding.nudge_offer`` both decides whether the window is open and
    stamps it — so a fire can never carry a nudge whose window was not spent.
    The import is function-local, like the engine's other onboarding reads
    (the two modules are import-light and must not form a load-time cycle).
    A failure answers the bare standing prompt: the cadence outranks the
    ledger, and a closed window is the safe direction.
    """
    try:
        from local_operator.aida import onboarding

        clause = onboarding.nudge_offer(config_dir, now_ms=now_ms)
    except Exception:  # noqa: BLE001 — a ledger miss must not cost the check-in
        logger.warning("aida: could not resolve the nudge window", exc_info=True)
        clause = None
    # ONE quiet-day tip (audit A10/U15): the "(no action needed)" reply was the
    # only thing a fresh install ever heard from its check-in, and the tips
    # that would have helped (phone access via the Radient relay first) were
    # nowhere. Same ledger discipline as the nudge — offered = stamped.
    try:
        from local_operator.aida import onboarding

        tip = onboarding.tip_offer(config_dir, now_ms=now_ms)
    except Exception:  # noqa: BLE001
        logger.warning("aida: could not resolve the tip", exc_info=True)
        tip = None
    clauses = [c for c in (clause, tip) if c]
    return "\n\n".join([CADENCE_MESSAGE, *clauses]) if clauses else CADENCE_MESSAGE


def session_class_reactive(config_dir: Path | str, session_id: str) -> bool:
    """Whether her attached profile is currently OUTSIDE the proactive class.

    Resolved at the moment it is asked (attachment sidecar + registry, else the
    packaged seed), so a class switch reaches every gate on its next
    evaluation — the R36 delivery-time rule. FAIL-CLOSED like the class module
    itself: an unreadable class reads reactive (a stop that keeps messaging is
    worse than a check-in that misses a day; the cadence simply tries again at
    the next reconcile). This is the EXTERNAL half (boot/ensure paths); the
    live session resolves the same question its own way so it pays no registry
    construction, but both end in ``action_class.session_action_class``.
    """
    from local_operator.action_class import PROACTIVE, session_action_class

    root = Path(config_dir)
    try:
        from local_operator.agents import AgentRegistry

        registry = AgentRegistry(root)
    except Exception:  # noqa: BLE001 — seeds-only resolution without one
        registry = None
    return session_action_class(root / "sessions" / session_id, registry=registry) != PROACTIVE


def cadence_schedule(
    now_ms: int, at: str = DEFAULT_CADENCE_AT, *, config_dir: Path | str
) -> WakeSchedule:
    """The cadence row: one-shot, due at the next ``at``.

    ``created_at`` carries ``now`` so the scheduler's stable ordering keeps it
    where it was planted; ``next_due_at`` is the next occurrence, never "now"
    (a due-now cadence would fire inside ``LOAD_GRACE_MS`` of an ensure, which
    is a boot-time turn nobody asked for). ``config_dir`` is REQUIRED — the
    message composes the R25 nudge clause through :func:`cadence_message`, and
    a caller that could skip it would be a second path that arms a row without
    the ledger moving.
    """
    return WakeSchedule(
        id=CADENCE_ID,
        message=cadence_message(config_dir, now_ms),
        next_due_at=next_cadence_ms(now_ms, at),
        every_ms=None,
        created_at=now_ms,
        # NOTIFY IS THE POINT OF THIS ROW: with no surface attached, a
        # check-in that finds something actionable must be able to reach the
        # operator (rung 4 of the completion ladder raises the OS banner).
        # Whether it ACTUALLY banners is decided at settle time by the veto
        # (``Session._aida_cadence_banner_veto``), which reads the reply: a
        # quiet day — the sentinel or a tip — stays silent. Before this flag
        # no Aida row set it, so her shipped cadence was silent no matter
        # what the reply said.
        notify=True,
    )


def hold_active(config_dir: Path | str) -> bool:
    """Paused or disabled — the state in which ``aida-*`` rows are not armed.

    The load-time filter and the delivery guard both ask this ONE question, so
    "held" cannot mean two things on two seams.
    """
    pol = policy(config_dir)
    return (not pol.enabled) or pol.paused


def delivery_allowed(config_dir: Path | str, *, class_reactive: bool = False) -> bool:
    """Whether an ``aida-*`` fire may deliver right now (fail-open, see module).

    Returns True when the policy cannot be read; see the module docstring for
    why a transient read failure delivers rather than eats the wake. The class
    gate is carried IN by the caller (the live session resolves it from its own
    registry at delivery time — one resolution per fire, R36), so this module
    never grows a second attachment reader; ``class_reactive=True`` suppresses
    exactly as a pause would.
    """
    try:
        return not (hold_active(config_dir) or class_reactive)
    except Exception:  # noqa: BLE001 — documented fail-open
        logger.warning("aida: could not resolve the hold state; delivering", exc_info=True)
        return not class_reactive


def filter_on_load(
    schedules: Sequence[WakeSchedule],
    config_dir: Path | str,
    *,
    class_reactive: bool = False,
) -> list[WakeSchedule]:
    """The load-time hold: drop ``aida-*`` rows while held (or reactive). Never raises.

    Called from ``Session._load_wake_schedules`` for her session only (the
    caller gate is :func:`local_operator.aida.state.is_aida_session`, one
    stat). Rows the filter drops are dropped from the in-memory list, so the
    session's next persist writes the cancellation into the transcript — that
    is the "pause cancels ``aida-*`` rows" contract reached through the ONE
    writer rather than a file edit behind the session's back. The class is the
    same lever one size up (R36: reactive = cadence + patience off; the
    cadence row is dropped, and a later switch back re-arms it at the next
    reconcile — the same restore-on-active path an un-pause uses).
    """
    if not any(is_aida_row(s.id) for s in schedules):
        return list(schedules)
    if not hold_active(config_dir) and not class_reactive:
        return list(schedules)
    kept = [s for s in schedules if not is_aida_row(s.id)]
    if len(kept) != len(schedules):
        logger.info(
            "aida: holding %d cadence row(s) while paused/disabled/reactive",
            len(schedules) - len(kept),
        )
    return kept


@dataclass
class ReconcileResult:
    """The full-list reconcile's new list, plus notes to journal.

    ``notes`` are human-readable refusals (budget, spacing, capacity) that the
    caller appends to the transcript as :data:`CUSTOM_ENTRY_TYPE` entries —
    the design's "observable, not silent" bound. ``changed`` is True when the
    list differs from the input, so a caller can skip a write (and the
    transcript append it would cost) when nothing moved.

    ``settle`` is the consumed trigger record's compare-and-delete token
    (``(target, ((source, key), ...), expected_updated_at_ms)``), or ``None``
    when no trigger record was consumed. It leaves the engine ONLY so the
    caller can settle after persisting the returned list — a record deleted
    before its row is durable is a check-in nobody will ever deliver (design
    §3.2's crash windows all hang off that ordering).
    """

    schedules: list[WakeSchedule]
    notes: list[str]
    changed: bool
    settle: tuple[str, tuple[tuple[str, str], ...], int] | None = None


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
    class_reactive: bool = False,
    attended: bool = True,
) -> ReconcileResult:
    """The in-session full-list reconcile. Pure file side effects: the ledger.

    ``attended`` is whether a local human surface holds this session right now
    (``Session._aida_greeting_may_land``). It gates ONE thing — arming the
    greeting — and defaults ``True`` so every caller that predates it keeps its
    behaviour; the fire-time check stays the authority either way.

    Order and content, in one place so the persist seam and the after-turn seam
    cannot diverge:

    1. Not her session ⇒ untouched (identity is read once, here, so a caller
       cannot forget the gate).
    2. Disabled, paused or class-reactive ⇒ drop every ``aida-*`` row; nothing
       is re-armed and the tray is left where it is (a paused assistant must
       not queue work for her own resume). ``class_reactive`` is the general
       switch (R36): the same drop, without touching the pause bookkeeping.
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

    # RE-PUBLISH THE TRIGGER SNAPSHOT on every one of her reconciles: boot,
    # the daily fire, after-turn and every config-watch tick pass through
    # here, and the published settings must not lag a hand-edited config.yml
    # beyond that (the design's stated residual). Best-effort, and
    # ``publish_settings`` skips the write when the values did not move.
    publish_trigger_settings(config_dir)

    pol = policy(config_dir)
    notes: list[str] = []
    if not pol.enabled or pol.paused or class_reactive:
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
                            f"{pol.min_gap_minutes}-minute spacing floor of her other wakes."
                        )
                        continue
                    kept.append(
                        WakeSchedule(
                            id=_next_extra_id({row.id for row in kept}),
                            # The model-written message gets the sentinel
                            # clause appended when it lacks one, so the veto's
                            # reply test applies to extras exactly as it does
                            # to the cadence (one constant, three writers).
                            message=with_quiet_clause(message or DEFAULT_EXTRA_MESSAGE),
                            next_due_at=due,
                            every_ms=None,
                            created_at=now,
                            # An extra is a check-in like the cadence: it may
                            # banner when actionable (settle-time veto reads
                            # the reply).
                            notify=True,
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
        notes.append("escalation tray left unread (another operation is in flight).")

    # -- cadence ------------------------------------------------------------
    # NOT BEFORE SHE HAS MET THE USER (audit A9): on a fresh install the first
    # thing she says must be her greeting in a window the user is looking at,
    # never an 08:30 check-in fired by the supervisor into a conversation
    # nobody has opened. ``cadence_allowed`` answers True for delivered,
    # skipped (an existing install) and legacy ledgers, so only a genuinely
    # first-run install waits — and an existing cadence row is never dropped
    # by this gate, only not created.
    if not any(row.id == CADENCE_ID for row in kept) and _cadence_allowed(config_dir):
        # REUSE the existing cadence row whenever one exists — rebuilding it
        # would mint a new ``created_at`` on every reconcile, which makes
        # ``changed`` permanently true (so a watcher tick or an after-turn
        # drain would rewrite the transcript entry for nothing) and would move
        # an instant the user may have just read in ``/aida status``. An
        # OVERDUE one is kept on purpose rather than re-armed forward: the
        # generic wake machinery owns missed-while-down delivery, and this
        # engine deliberately has no second implementation of it.
        existing = next((row for row in original if row.id == CADENCE_ID), None)
        if existing is not None and not existing.notify:
            # A LEGACY ROW: armed before her check-ins were allowed to banner
            # (``WakeSchedule.notify`` defaulted False and no Aida row set it).
            # Upgraded IN PLACE rather than rebuilt — a rebuild would mint a
            # new ``created_at`` on every reconcile, which the REUSE note above
            # exists to prevent — and the upgrade is what makes this fix
            # effective on an established install at its next persist instead
            # of only after the current row's next fire.
            existing = existing.model_copy(update={"notify": True})
        kept.append(
            existing
            if existing is not None
            else cadence_schedule(now, pol.at, config_dir=config_dir)
        )

    # -- the one-time greeting, armed only once a PERSON asked for it --------
    # A live session is the ONLY writer that can arm it when the owner holds
    # the rows (`arm_wake` refuses with a 503), so a REQUESTED greeting is
    # armed here rather than left to a caller that cannot write (review round
    # 1, m1). ``greeting_armable`` is "requested + a provider": the old gate
    # was ``greeted_at is None``, which let any headless runtime of her
    # session (a supervisor fire, `lop exec`, the mobile daemon) greet an
    # install nobody was looking at (audit A1). Only an attended surface moves
    # the ledger to requested (``onboarding.request_greeting``).
    if not any(row.id == GREETING_WAKE_ID for row in kept):
        from local_operator.aida import onboarding as _onboarding

        # ``attended`` as well as armable (review round 1, R-3): a fire withheld
        # in a headless runtime goes back to ``requested``, and without this the
        # SAME runtime's next persist would arm it again, due now, and withhold
        # it again — a loop bounded only by the runtime's lifetime. The arm waits
        # for a person exactly as the fire does.
        if attended and _onboarding.greeting_armable(config_dir):
            kept.append(
                WakeSchedule(
                    id=GREETING_WAKE_ID,
                    message=_onboarding.greeting_message(config_dir),
                    next_due_at=now,
                    every_ms=None,
                    created_at=now,
                    # Hidden: the trigger is a model-facing fact line, and her
                    # reply must be the first VISIBLE row (audit A3/A4).
                    hidden=True,
                )
            )
            # The ledger moves WITH the arm on this path too: without it a
            # later reconcile (the row retired after firing, before the
            # delivery stamp landed) would arm a second greeting.
            _onboarding.mark_greeted(config_dir, now)

    # -- wake triggers -------------------------------------------------------
    # Consumed LAST, so an appended check-in row is part of the same list the
    # caller persists (the engine stays the ONE writer of her schedule list).
    # The settle token rides the result: the CALLER deletes the record only
    # after the list is durable. Unreachable on a hold — the disabled/paused/
    # reactive branch above returned first, leaving the record pending for a
    # resume or the next active pass (the engine's half of the suppression
    # matrix; creation-side gates are the supervisor's half).
    consume = consume_triggers(kept, config_dir=config_dir, session_id=session_id, now_ms=now)
    kept = consume.schedules
    notes.extend(consume.notes)
    return ReconcileResult(
        schedules=kept, notes=notes, changed=kept != original, settle=consume.settle
    )


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
# Wake triggers — consuming a pending check-in into ONE row
# ---------------------------------------------------------------------------


def publish_trigger_settings(config_dir: Path | str) -> None:
    """Re-publish the wake-trigger settings snapshot. Best-effort, sync.

    Called from her reconcile (so boot, the daily fire, after-turn and every
    config-watch tick re-publish), from :func:`ensure_armed` and from the
    bootstrap's ensure — the design's writers (b)/(c), which keep the snapshot
    within a day of a hand-edited ``config.yml`` even with no settings write
    anywhere. The publish itself skips the file write when the values did not
    move.
    """
    try:
        from local_operator.wakes import triggers as wake_triggers

        wake_triggers.publish_settings(config_dir)
    except Exception:  # noqa: BLE001 — a snapshot never costs a reconcile
        logger.warning("aida: could not publish the trigger settings snapshot", exc_info=True)


@dataclass
class TriggerConsume:
    """What a consume produced: the list, notes, and the settle token.

    ``settle`` is ``(target, ((source, key), ...), expected_updated_at_ms)`` —
    the compare-and-delete token for the consumed record. The caller settles
    ONLY after the returned list is durable: a record deleted before its row
    is persisted is a check-in nobody will ever deliver.
    """

    schedules: list[WakeSchedule]
    notes: list[str]
    changed: bool
    settle: tuple[str, tuple[tuple[str, str], ...], int] | None = None


def has_pending_triggers(config_dir: Path | str, session_id: str) -> bool:
    """Whether ``session_id`` has a pending trigger record. One stat.

    The cheap gate for the after-turn drain: without it the drain would run a
    full reconcile on every turn of hers, instead of exactly when a check-in
    (or the escalation tray) actually wants the engine.
    """
    try:
        from local_operator.wakes import triggers as wake_triggers

        return wake_triggers.has_pending(Path(config_dir), session_id)
    except Exception:  # noqa: BLE001 — a gate must answer, not raise
        return False


def consume_triggers(
    schedules: Sequence[WakeSchedule],
    *,
    config_dir: Path | str,
    session_id: str,
    now_ms: int | None = None,
) -> TriggerConsume:
    """Turn a pending trigger record into AT MOST ONE ``aida-trigger-*`` row.

    THE CONSUME PROTOCOL (design §3.2), in order:

    0. The shared decline gate (``triggers.declines``) — master switch, env,
       and the belt-and-braces aida states — must be clear; a gated record is
       LEFT, never settled (review round 1, R2).
    1. Read ``<config>/wakes/triggers/pending/<session_id>.json``; none ⇒
       no-op. This is only reachable from the engine's ACTIVE branch, so the
       hold/reactive re-check has already happened authoritatively.
    2. Re-verify every KNOWN instance against the authoritative rule
       (``projects.progress_is_stale`` over the live row): a refresh that
       landed between record and consume drops the instance; a source this
       engine cannot verify is left in the record for its own consumer.
    3. If anything survives and the deterministic row id (sha1 over the sorted
       fingerprints) is absent from the list, build ONE one-shot row due NOW.
       If the id IS present, skip and settle — a re-consume can never arm a
       second row (the crash windows of §3.2). If the schedule list is at the
       engine's cap, skip with a note and do NOT settle: the record is not
       lost, and a later consume with room arms it.
    """
    from local_operator.wakes import triggers as wake_triggers

    current = list(schedules)
    root = Path(config_dir)
    record = wake_triggers.read_pending_record(root, session_id)
    if record is None:
        return TriggerConsume(schedules=current, notes=[], changed=False)

    # THE SAME GATE THE PASS AND THE SUPERVISOR USE (review round 1, R2): the
    # master switch is creation-AND-consumption, so a record that PREDATES a
    # switch-off must not arm a check-in either. The record is LEFT — nothing
    # settles — so re-enabling within its TTL reconsiders it and the TTL drops
    # it otherwise; the engine's own declines (disable/pause/reactive/env) are
    # already authoritative above (the ACTIVE branch returned first), and
    # sharing the gate keeps every lever on ONE reader.
    if wake_triggers.declines(root, session_id) is not None:
        return TriggerConsume(schedules=current, notes=[], changed=False)

    moment = int(time.time() * 1000) if now_ms is None else int(now_ms)
    updated = record.get("updated_at_ms")
    expected = int(updated) if isinstance(updated, int) and not isinstance(updated, bool) else 0

    kept: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    for raw in record.get("instances") or []:
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("source") or "")
        key = str(raw.get("key") or "")
        if not source or not key:
            continue
        if source != "project_staleness":
            # Not a source this engine can verify: leave it untouched, so a
            # future consumer (or a newer engine) still finds it.
            continue
        keys.append((source, key))
        if _trigger_project_still_stale(root, key):
            kept.append(dict(raw))
    token: tuple[str, tuple[tuple[str, str], ...], int] = (session_id, tuple(keys), expected)

    if not kept:
        # Everything settled since the record was written (or nothing was
        # verifiable): nothing to say; settle the record, leave the list alone.
        return TriggerConsume(schedules=current, notes=[], changed=False, settle=token)

    row_id = wake_triggers.trigger_row_id(kept)
    if any(row.id == row_id for row in current):
        # ALREADY ARMED — a re-consume after a crash between persist and
        # settle. Skip and settle: never a second row, never a duplicate
        # check-in.
        return TriggerConsume(schedules=current, notes=[], changed=False, settle=token)

    if len(current) >= MAX_WAKE_SCHEDULES:
        # Observable, not silent (design §3.2): the note is journaled and the
        # record stays pending — a later consume with room arms it.
        return TriggerConsume(
            schedules=current,
            notes=["trigger check-in skipped: the schedule list is full."],
            changed=False,
        )

    row = WakeSchedule(
        id=row_id,
        message=compose_trigger_message(kept, overflow=int(record.get("overflow") or 0)),
        next_due_at=moment,
        every_ms=None,
        created_at=moment,
    )
    return TriggerConsume(schedules=[*current, row], notes=[], changed=True, settle=token)


def settle_triggers(
    config_dir: Path | str,
    session_id: str,
    settle: tuple[str, tuple[tuple[str, str], ...], int] | None,
) -> bool:
    """Compare-and-delete the consumed trigger record. Called AFTER the persist.

    The token rides :class:`ReconcileResult` out of :func:`reconcile` precisely
    so this cannot run before the armed row is durable — the crash window in
    which a record is deleted and its row never persisted is the one this
    ordering exists to close (design §3.2). Best-effort: a failure leaves the
    record, and the next consume (row present ⇒ skip and settle) retries.
    """
    if not settle:
        return False
    target, keys, expected = settle
    if target != session_id or not keys:
        return False
    try:
        from local_operator.wakes import triggers as wake_triggers

        return wake_triggers.settle(
            Path(config_dir), target, list(keys), expected_updated_at_ms=expected
        )
    except Exception:  # noqa: BLE001 — a settle failure just retries later
        logger.warning("aida: could not settle the trigger record", exc_info=True)
        return False


def _trigger_project_still_stale(root: Path, project_id: str) -> bool:
    """The AUTHORITATIVE staleness re-check at consume (design §2.2's belt).

    ``projects.progress_is_stale`` is the rule; a missing/unreadable row
    answers False — the instance is dropped, because the record is about a
    project that no longer needs the check-in and an unreadable row is the
    store's problem to repair, never a reason to wake. Never raises.
    """
    try:
        from local_operator.projects import ProjectRegistry, progress_is_stale

        return progress_is_stale(ProjectRegistry(root).get_project(project_id))
    except Exception:  # noqa: BLE001 — a verify failure drops the instance
        logger.debug("aida: could not re-verify project %s", project_id, exc_info=True)
        return False


def compose_trigger_message(instances: Sequence[Mapping[str, Any]], *, overflow: int = 0) -> str:
    """The wake message for one armed check-in, from structured fields only.

    Every rendered field is clipped (project names/titles are user data and a
    pathological one must not push the instruction out of the message), the
    sessions are summarised, and instances beyond the record's own cap are
    one line — "and N more". The wake engine's ``MAX_WAKE_MESSAGE_CHARS``
    bound is the hard stop; this stays well under it for the common case.

    The contract text is the design's (§2.4 and §3): message, don't do — she
    never writes a progress line on a session's behalf; one bounded resume is
    allowed for a dead/stalled session; a project with no live session is
    surfaced, not spawned.
    """
    lines = [
        "Project check-in. These tracked projects have stale records (no progress "
        "beyond the configured staleness window):",
        "",
    ]
    for entry in list(instances)[:_TRIGGER_MESSAGE_MAX_INSTANCES]:
        payload = entry.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        name = _clip(str(payload.get("display_name") or entry.get("key") or "unknown"))
        status = _clip(str(payload.get("status") or "unknown"), 40)
        age = _age_label(payload.get("progress_age_s"))
        sessions = payload.get("sessions")
        sessions = sessions if isinstance(sessions, list) else []
        parts: list[str] = []
        for session in sessions[:4]:
            if not isinstance(session, Mapping):
                continue
            sid = _clip(str(session.get("id") or ""), 20)
            live = _clip(str(session.get("live") or "cold"), 12)
            idle = _age_label(session.get("last_activity_age_s"))
            parts.append(f"{sid} ({live}, idle {idle})" if idle != "unknown" else f"{sid} ({live})")
        if len(sessions) > 4:
            parts.append(f"+{len(sessions) - 4} more")
        where = ", ".join(parts) if parts else "none"
        lines.append(
            f"- {_clip(str(entry.get('key') or 'unknown'), 60)} — \"{name}\", "
            f"status {status}, last progress {age}; linked sessions: {where}"
        )
    if overflow > 0:
        lines.append(f"- and {overflow} more stale project(s).")
    lines.extend(
        [
            "",
            "Wake to check in on them: message each linked session (or the project's "
            "manager) asking for a status update and a `project` progress refresh — "
            "do not update the records yourself. For a session that looks dead or "
            "stalled, one bounded resume/restart attempt is allowed; if a project "
            "has no live session at all, surface it to the operator with a "
            "recommendation. Then reply briefly with what needs the operator's action.",
        ]
    )
    return "\n".join(lines)


def _clip(text: str, limit: int = _TRIGGER_LINE_MAX_CHARS) -> str:
    """Whitespace-collapsed and bounded — user data rendered into a message."""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _age_label(age_s: Any) -> str:
    """``2h``/``45m``/``30s`` — or ``unknown`` for anything not a number."""
    if isinstance(age_s, bool) or not isinstance(age_s, (int, float)) or age_s < 0:
        return "unknown"
    seconds = int(age_s)
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


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

#: Display-label TEMPLATES for the engine-armed rows on the human receipt
#: surfaces, keyed by the id the engine writes. Without a label the first
#: frame of a fresh install's first conversation is `wake  aida-greeting
#: (1/1).` — a raw schedule id where a greeting belongs (design review round
#: 1, D2). Kept HERE, beside the ids, so a renamed row cannot leave a stale
#: label behind on a surface that spelled the label itself; `harness/rows.py`
#: reads the map lazily, the same way it reads `WAKE_SCRATCH_CLAUSE`.
#:
#: TEMPLATES, not literals. A frozen literal said "Aida" no matter what the
#: operator renamed her to: the receipt read "Aida's introduction" above a
#: body that signs off "Introduce yourself as Sovereign" — one card, two
#: names (design review round 3, D4). `{name}` is filled at LOOKUP from
#: `naming.display_name()`, the reader every other surface uses, so the
#: receipt and the delivered body agree by construction — and a rename is
#: live on the next paint, with no cached label to invalidate.
WAKE_DISPLAY_LABEL_TEMPLATES: dict[str, str] = {
    GREETING_WAKE_ID: "{name}'s introduction",
    CADENCE_ID: "{name}'s check-in",
}

#: The label template for the bounded escalation one-shots
#: (``aida-extra-<n>``): their ids carry a counter, so they are matched by
#: prefix.
EXTRA_DISPLAY_LABEL_TEMPLATE = "{name}'s follow-up"


def wake_display_label(wake_id: str) -> str:
    """The human label for an engine-armed row; unknown ids name themselves.

    Identity for everything else — a user-created wake's own words ARE its
    identity — so a caller can substitute unconditionally. The label is
    FORMATTED at lookup, never stored: it must read the name she is actually
    called or the receipt contradicts the greeting it reports (design review
    round 3, D4). ``display_name`` never raises, so this lookup adds no new
    failure to the map read it replaces; a pathological failure still lands
    on the caller's guard, which falls back to the raw id.
    """
    if wake_id in WAKE_DISPLAY_LABEL_TEMPLATES:
        template = WAKE_DISPLAY_LABEL_TEMPLATES[wake_id]
    elif wake_id.startswith(EXTRA_ID_PREFIX):
        template = EXTRA_DISPLAY_LABEL_TEMPLATE
    elif wake_id.startswith(TRIGGER_ID_PREFIX):
        template = TRIGGER_DISPLAY_LABEL_TEMPLATE
    else:
        return wake_id
    # Lazy, like this module's other cross-imports: a label must never be the
    # reason the engine cannot load.
    from local_operator.aida import naming

    return template.format(name=naming.display_name())


def _cadence_allowed(config_dir: Path | str) -> bool:
    """Whether the cadence may be CREATED now (see ``onboarding.cadence_allowed``).

    Lazy import for the module-cycle reason the other onboarding reads give;
    fail-OPEN, because an existing user losing their check-in to a ledger read
    error is the worse failure than a first-run user getting one early.
    """
    try:
        from local_operator.aida import onboarding as _onboarding

        return _onboarding.cadence_allowed(config_dir)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        state.note_lock_refusal("the cadence gate's ledger read", exc)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("aida: cadence gate failed; allowing", exc_info=True)
        return True


def _clear_greeted(config_dir: Path | str) -> None:
    """Un-stamp the greeting via `onboarding`, best-effort."""
    try:
        from local_operator.aida import onboarding as _onboarding

        _onboarding.clear_greeted(config_dir)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        state.note_lock_refusal("the greeting un-stamp", exc)
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


def reap_orphan_rows(config_dir: Path | str, *, keep: str = "") -> tuple[str, ...]:
    """Drop the engine's rows from sessions that no longer exist on disk.

    THE CHURN CASE, which is what this exists for. Her session id is minted by
    ``bootstrap.ensure_session`` and re-minted whenever the directory ``state.json``
    names is gone, so every re-creation leaves the PREVIOUS incarnation's armed
    cadence behind in the wake index — a row keyed to an id with no transcript,
    which no runtime can ever be started for. The supervisor already refuses to
    engage one (``_session_exists``, added for exactly these ids: two of them
    were eating a 30 s deadline per pass, forever), but nothing removed them:
    the operator's own install had accumulated three (``0827221cce46``,
    ``cad094a6f02b``, ``ce526344aa86``), each carrying a full check-in message,
    and they were cleared by hand. Rows that can never fire are not neutral —
    they are what ``lop wake status`` shows, what the picker counts, and what a
    reader has to rule out before believing anything about her schedule.

    SCOPE, deliberately narrow. Only rows this engine owns (``aida-*``, which
    includes the trigger rows) are removed, only from entries whose session
    TRANSCRIPT is absent (``keep``'s own entry is never touched, whether or not it
    exists yet), and an entry left with nothing is removed through
    ``store.write_entry``'s own empty-list rule rather than an unlink here —
    ``test_no_session_deletion`` holds every remover outside ``session/`` to an
    allow-list with a reason, and the store is the writer that owns that shape.
    The entry's other keys ride through ``preserve``, so a ghost entry carrying
    someone else's rows keeps them (a different session's wakes are not this
    engine's to delete) and keeps its lateness stamps.

    Synchronous and cheap: one directory listing plus one ``is_file`` per entry
    carrying engine rows, and the whole scan is skipped for entries with none —
    the only wake-carrying sessions on a typical machine are a handful. Never
    raises; a failure to read the index answers ``()``. Returns the ids reaped,
    so the caller and the tests can assert on what actually moved.
    """
    from local_operator.resume import TRANSCRIPT_NAME
    from local_operator.wakes import store as wake_store

    root = Path(config_dir)
    try:
        index = wake_store.read_index(root)
    except Exception:  # noqa: BLE001 — an unreadable index reaps nothing
        logger.warning("aida: could not read the wake index to reap orphans", exc_info=True)
        return ()

    reaped: list[str] = []
    for session_id, entry in index.items():
        if not session_id or session_id == keep or not isinstance(entry, Mapping):
            continue
        if not _aida_ids(entry):
            continue
        if (root / "sessions" / session_id / TRANSCRIPT_NAME).is_file():
            continue
        kept = [
            row
            for row in entry.get("schedules", []) or []
            if not is_aida_row(str(row.get("id", "")) if isinstance(row, Mapping) else "")
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
            logger.warning("aida: could not reap the orphan rows for %s", session_id, exc_info=True)
            continue
        reaped.append(session_id)
    if reaped:
        logger.info("aida: reaped orphaned engine rows for %s", ", ".join(sorted(reaped)))
    return tuple(reaped)


async def ensure_armed(
    config_dir: Path | str,
    session_id: str,
    *,
    now_ms: int | None = None,
    class_reactive: bool | None = None,
) -> str:
    """Arm the cadence for a session with NO live owner. Returns one word.

    The boot/desktop writer: reads the derived index (the only schedule read
    that does not open the session) and writes through
    :mod:`local_operator.wakes.arm` — transcript first, index second, install
    hook third, exactly the discipline every external wake writer shares.

    ``class_reactive`` is resolved FROM THE ATTACHMENT when the caller does not
    know it (the boot paths, which have no registry loaded): a class switch is
    as hard a hold as a pause here, and the resolution fails CLOSED — an
    unreadable class reads reactive, because a stop that keeps messaging is the
    nagging bug the switch exists to end. The live-session path passes its own
    fresh value instead.

    Returns ``"armed"`` (a cadence row was written), ``"present"`` (one was
    already there), ``"waiting"`` (a first-run install whose greeting has not
    been delivered yet — the cadence follows the greeting),
    ``"paused"``/``"disabled"``/``"reactive"``/``"held"``
    (nothing armed; for the first three any existing Aida rows were best-effort
    cancelled, and ``"held"`` leaves a stopped session dormant), ``"owner"`` (a
    live runtime holds the session — it will reconcile on its own watcher tick),
    ``"busy"`` (a peer held the STORE lock for the whole wait; benign and
    retried on the next tick or boot — see that handler), ``"no-session"``
    (nothing on disk to arm against) or ``"failed"`` (a refusal/logged error).
    Never raises: the callers are boot paths whose failures must not fail them.
    """
    from local_operator.wakes import store as wake_store
    from local_operator.wakes.arm import (
        STATUS_CONFLICT,
        WakeWriteError,
        arm_wake,
        repair_index,
    )

    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not session_id:
        # No session minted yet (a pause before her first open, or a resume
        # back to that state): nothing on disk to arm against, and the next
        # ``ensure_session`` arms the cadence itself.
        return "no-session"
    # Boot is one of the trigger snapshot's publish points (writers (b)): the
    # supervisor may evaluate long before her first reconcile, and the
    # snapshot must describe THIS config, not a stale one.
    publish_trigger_settings(root)
    if class_reactive is None:
        class_reactive = session_class_reactive(root, session_id)
    try:
        pol = policy(root)
        entry = wake_store.read_entry(root, session_id)
        if entry is None and not (root / "sessions" / session_id).is_dir():
            return "no-session"
        ids = _row_ids(entry)
        aida_ids = [i for i in ids if is_aida_row(i)]
        if not pol.enabled or pol.paused or class_reactive:
            if aida_ids:
                await _cancel_ids(root, session_id, aida_ids, now_ms=now)
            if not pol.enabled:
                return "disabled"
            return "paused" if pol.paused else "reactive"
        # AN EXPLICIT STOP IS A GATE ON THE WHOLE PASS, which is why it sits
        # ABOVE the id check rather than inside it. ``is_held`` covers both
        # markers a user lever writes — ``stopped_at`` (``/stop``, stamped by
        # the TUI on the same entry) and ``held_at`` (pause) — and the
        # supervisor and the trigger layer's own gate already read it that way
        # (``triggers.declines`` answers "held"). Two things hang off it: the
        # ARM (a stopped session must not come back with its check-in, or the
        # boot arm this change adds would walk straight past a stop the
        # operator asked for) and the TRAY DRAIN below, which writes new rows
        # — consuming a request for a session the platform has declared inert
        # is the same mistake in the other direction (QA round 1, Q1: an entry
        # parked by ``/stop`` WITH its rows kept still drained). Nothing is
        # cancelled here either — a dormant entry's rows are the ones that come
        # back when the session is reopened (``_rebuild_wake_index_entry``
        # clears ``stopped_at``).
        if wake_store.is_held(entry):
            return "held"
        result = "present"
        if CADENCE_ID not in ids and not _cadence_allowed(root):
            # First run, greeting not yet delivered: the boot paths call this
            # on every launch, and arming here would schedule a headless 08:30
            # turn before she has met the user (audit A9). The delivery's own
            # reconcile arms it right after the greeting fires.
            result = "waiting"
        elif CADENCE_ID not in ids:
            try:
                await arm_wake(
                    root,
                    session_id,
                    {
                        "message": cadence_message(root, now),
                        "at": _iso_due(next_cadence_ms(now, pol.at)),
                        # Same contract as ``cadence_schedule``: the row may
                        # banner; the settle-time veto reads her actual reply
                        # before one is raised.
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
                # THE ROW IS THERE; ONLY THE DERIVED INDEX IS BEHIND. Every
                # external arm writes transcript first and index second, so a
                # crash, a kill or a full disk in that window leaves a row the
                # index has never seen — and this function reads the index (the
                # one schedule read that does not open the session), so it is
                # looking at the wrong side of the truth. Reporting "failed"
                # here was the silent half of the reported defect: the cadence
                # existed and nothing could fire it until the operator happened
                # to open the conversation. Re-project the transcript instead.
                await repair_index(root, session_id)
                result = "present"
        await _drain_tray_external(root, session_id, pol, now)
        return result
    except WakeLockBusy as exc:
        # A PEER HOLDS THE STORE LOCK, and that is a normal answer here rather
        # than a failure. Why it needs saying: this seam has several attended
        # writers (the TUI launch hook's ensure task, the app's first-run
        # route, the runtime's reconcile, the wake supervisor), and the two
        # boot hooks used to RUN CONCURRENTLY — the launch hook is an asyncio
        # task and the route armed the same row from its own coroutine. Both
        # take the store lock for the row write and again for the tray, so one
        # waited out the peer and was refused; the app's first-run boot then
        # printed a traceback for a boot that was working correctly (CI
        # `tui-e2e (ubuntu-latest, 1)`, run 37886200214). Two things prevent
        # that now: ``App._route_first_run_boot`` awaits the launch hook's task
        # before its own ensure (order, not race), and this refusal — the
        # fallback for a peer OUTSIDE this process — is answered quietly
        # through :func:`state.note_lock_refusal`, sharing one shape with every
        # other site on the seam. "busy" is a miss for this tick: whoever loses
        # arms nothing, the row it would have written is the row the winner is
        # writing, and the next tick or boot arms it anyway.
        state.note_lock_refusal("the cadence arm", exc)
        return "busy"
    except WakeLockUnavailable as exc:
        # Not retryable, unlike the busy lock: the lock FILE could not be
        # created at all, so the next attempt is refused too (the lock module's
        # own distinction) — the word stays ``"failed"`` so callers keep
        # treating it as one, and the line is quiet for the same reason.
        state.note_lock_refusal("the cadence arm", exc)
        return "failed"
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
    # ACQUIRED OFF THE LOOP — `wakes.lock`'s own documented pattern for a lock
    # held across awaits (``WakeWriteLock``'s docstring; ``state.wake_lock``
    # exists for exactly this). ``state.locked`` acquires on the CALLER's
    # thread, so a contended take parks whoever runs it for the whole
    # ``LOCK_WAIT_S`` — measured 5.09 s on this seam, with the app unable to
    # paint for the duration and no way for the peer to release early, since
    # the peer is blocked on a lock this wait holds open. On a worker thread
    # the wait costs a thread rather than the frame, and the refusal still
    # arrives as the same ``WakeLockBusy`` — answered quietly by
    # :func:`state.note_lock_refusal` where the caller asked for the arm.
    lock = state.wake_lock(root)
    await asyncio.to_thread(lock.acquire)
    try:
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
        # ``enumerate``, not ``requests.index(request)`` (review round 2,
        # NIT-1): ``list.index`` resolves to the FIRST EQUAL item, so a tray
        # holding duplicate entries would re-restore an already-processed one
        # when a later duplicate was refused mid-drain. The position in this
        # loop is the exact remainder boundary; a value-search only guesses at
        # it.
        for index, request in enumerate(requests):
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
                    f"{pol.min_gap_minutes}-minute spacing floor of her other wakes."
                )
                continue
            extra_id = _next_extra_id(ids)
            try:
                await arm_wake(
                    root,
                    session_id,
                    {
                        "message": with_quiet_clause(message or DEFAULT_EXTRA_MESSAGE),
                        "at": _iso_due(due),
                        # Same contract as the reconcile arm above: an extra
                        # may banner when actionable; a quiet reply is vetoed
                        # at settle time.
                        "notify": True,
                    },
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
                remaining = list(requests[index:])
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
    finally:
        await asyncio.to_thread(lock.release)
    # WARNING, not info (QA round 2, Q3): every note here is an operation the
    # operator asked for that did NOT take effect (a refusal) or was handed to
    # another writer — and `lop serve` configures its console logging at the
    # platform default (WARNING), so info-level lines were exactly the lines
    # nobody could see on a default server. The README's "observable rather
    # than silent" is about these events; they ride the level the default
    # surfaces.
    for note in notes:
        logger.warning("aida: %s", note)
    return notes


def mark_held(config_dir: Path | str, session_id: str, *, now_ms: int | None = None) -> bool:
    """Stamp ``held_at`` on her wake-index entry. Returns whether an entry was found.

    The same derived-index marker discipline as ``stopped_at``
    (``control._mark_wakes_dormant``): schedules are never deleted by a pause,
    the entry gains a key the supervisor skips, and the session's own open
    keeps it (the rebuild clears only ``stopped_at``). Best-effort by
    contract — the config key is the authority and the guards above do not
    consult this file — so every failure is logged and answered ``False``.

    A ROWLESS entry cannot be stamped: ``wake_store.write_entry`` treats an
    empty schedule list as "remove the entry", and a pause must never delete
    one. That shape is covered by the published ``aida.cadence.paused`` bit
    the pause writes through the settings facade (``triggers.declines``), so
    the supervisor's record plumbing stops all the same (review round 1, R1).
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
    """Drop ``held_at`` from her wake-index entry. Best-effort, like :func:`mark_held`.

    Symmetric with :func:`mark_held`, including the rowless no-op: an entry
    with no schedules carries no marker to clear (and rewriting it would
    delete it), and the config-published pause bit is what gate paths read
    for that shape.
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

    entry = wake_store.read_entry(root, session_id)
    aida_ids = _aida_ids(entry)
    # PAUSE CANCELS PENDING PATIENCE CYCLES (§8.2.4): a paused assistant must
    # not keep a hidden timer nagging on her behalf, and a cycle that cannot
    # re-arm is dead weight. Same best-effort writer as the Aida rows; the
    # arming side is blocked by the session's proactive hold.
    patience_ids = [
        str(row.get("id") or "")
        for row in (entry or {}).get("schedules") or ()
        if isinstance(row, Mapping) and is_patience_row(row)
    ]
    cancelled: list[str] = []
    owner_blocked = False
    if aida_ids or patience_ids:
        cancelled, owner_blocked = await _cancel_ids(
            root, session_id, [*aida_ids, *patience_ids], now_ms=now
        )
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
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        state.note_lock_refusal("the resume's greeting arm", exc)
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
