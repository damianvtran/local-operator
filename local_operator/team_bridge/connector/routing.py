"""Which local session a DM reply should wake: the one that asked.

WHY THIS EXISTS. Several local sessions share one Team Bridge identity, but the
connector hands every inbound message to ONE of them, the designated
chief-of-staff session. When a task session DMs a colleague's agent, the answer
used to land in the designated session, which had asked nothing; the asker
never saw it. This module decides, per inbound DM, whether the answer should go
to the asker instead. It is a PURE decision over facts the caller gathers (the
outbound ledger, the session registry), so the whole rule table is unit
tested without a store, a registry or a clock.

THE RULE, strictest first. Anything not named below goes to the designated
session, which is exactly what happened before this module existed:

1. Only a ``dm``. A group or channel has many members and no "the asker"; an
   older hub that does not say the kind (``None``) is treated the same.
2. Only a human or an agent sender that is NOT this account. A message from our
   own account is one of our sessions talking to another, and an app/integration
   post is never an answer to a question.
3. A thread root counts ONLY when it is a message one of our sessions posted in
   this same conversation (a ledger hit on the hub's message id). It can at most
   pick among sessions that already messaged this peer here; no inbound value
   can name a session the ledger does not hold for this conversation.
4. Otherwise: exactly ONE session posted in this DM within
   :data:`POSTER_WINDOW_S`. Two or more is ambiguous, so the message goes to the
   designated session with the candidates listed on its card.
5. The chosen session must be live and advertise the routed-inbox capability
   (an older build would never find a row routed to it). Anything else falls
   back to the designated session.

The text of a message is never read here: a body that spells a ref, a session id
or "reply to Bill" selects nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

#: How long after posting into a DM a session still counts as "the one that is
#: waiting on the answer". An ESTIMATE, not a measurement: the only data is two
#: replies from the 2026-11 incident, 31 s and 8 min 9 s, and 30 min is the
#: guard's own ``TRIGGER_TTL_S`` rounded to something a person recognises. Too
#: short degrades to today's behaviour (the designated session gets it). TO
#: MEASURE IT: for each DM, take the gap from a post by this account to the
#: peer's next inbound message (``channel_history`` on both sides of the
#: conversations), then set this above the largest gap that still was an answer.
POSTER_WINDOW_S = 30 * 60.0

#: Closed reasons, for the card and the logs. Never a session id or a body.
ROUTED_THREAD = "thread_hit"
ROUTED_SOLE_POSTER = "sole_poster"
DESIGNATED_DEFAULT = "designated"
DESIGNATED_NOT_DM = "not_dm"
DESIGNATED_SENDER = "sender_not_eligible"
DESIGNATED_NO_LEDGER = "ledger_unavailable"
DESIGNATED_AMBIGUOUS = "ambiguous"
DESIGNATED_NOT_LIVE = "not_live"
DESIGNATED_NO_POSTER = "no_poster"
DESIGNATED_IS_ASKER = "designated_asked"


@dataclass(frozen=True)
class RouteDecision:
    """Where one inbound message goes. ``target`` None = the designated session."""

    target: str | None
    reason: str
    #: ``(session_id, latest_post_at)`` for the sessions that could be the asker,
    #: set only when the answer to "who asked" was ambiguous: the designated
    #: session's card names them so it can tell the owner which to look at.
    candidates: tuple[tuple[str, float], ...] = ()


def decide_route(
    *,
    kind: str | None,
    sender_class: str | None,
    sender_account_id: str,
    own_account_id: str,
    designated_session: str,
    thread_root_id: str | None,
    ledger: tuple[str | None, dict[str, float]] | None,
    live_capable: frozenset[str],
) -> RouteDecision:
    """The ONE routing decision. Pure: no clock, no I/O, no message text.

    ``ledger`` is :func:`reply_guard.routing_facts` already narrowed to the
    window (``None`` = the ledger could not be read). ``live_capable`` is the set
    of sessions that are live AND advertise the routed-inbox capability.
    """

    if kind != "dm":
        return RouteDecision(None, DESIGNATED_NOT_DM)
    if (
        sender_class not in ("human", "agent")
        or not sender_account_id
        or sender_account_id == own_account_id
    ):
        return RouteDecision(None, DESIGNATED_SENDER)
    if ledger is None:
        return RouteDecision(None, DESIGNATED_NO_LEDGER)
    thread_session, posters = ledger

    if thread_root_id and thread_session:
        # A ledger hit decides on its own and has no window: the thread is the
        # sender saying which of our messages this answers.
        return _pick(thread_session, ROUTED_THREAD, designated_session, live_capable)

    if not posters:
        return RouteDecision(None, DESIGNATED_NO_POSTER)
    if len(posters) > 1:
        ordered = tuple(sorted(posters.items(), key=lambda item: item[1], reverse=True))
        return RouteDecision(None, DESIGNATED_AMBIGUOUS, candidates=ordered)
    (only,) = posters
    return _pick(only, ROUTED_SOLE_POSTER, designated_session, live_capable)


def _pick(
    session_id: str, reason: str, designated_session: str, live_capable: frozenset[str]
) -> RouteDecision:
    if session_id == designated_session:
        return RouteDecision(None, DESIGNATED_IS_ASKER)
    if session_id not in live_capable:
        return RouteDecision(None, DESIGNATED_NOT_LIVE)
    return RouteDecision(session_id, reason)


# -- budgets -------------------------------------------------------------------

#: A routed destination is woken at most once per this many seconds: the
#: coalescer's per-destination gate for routed groups (the ordinary window is
#: 3 s, which is too short for a session a stranger can keep waking). An
#: ESTIMATE: screening already bounds inbound volume (settings.py: peak 40
#: screening units an hour, ~16 distinct messages), and 10 s is comfortably
#: above one agent turn. Measure with ``wake_diagnostics`` per-destination
#: counts once those exist.
ROUTED_WAKE_MIN_INTERVAL_S = 10.0

#: Routed wakes a NON-designated session may take per hour. An ESTIMATE for the
#: same reason; the designated session is not budgeted here (it always was the
#: destination). Over budget the message goes to the designated session: it is
#: kept, never held or dropped. In memory only: a restart resets it, which at
#: worst grants one more hour's allowance.
ROUTED_WAKES_PER_HOUR = 20

#: A routed row nobody picked up is handed back to the designated session after
#: this long. An ESTIMATE: it only has to exceed a busy session's turn, because
#: a row a live turn has CLAIMED is never reclaimed.
ROUTED_RECLAIM_AFTER_S = 15 * 60.0

#: How often the loop looks for rows to reclaim (the sweep is an indexed read
#: first and a write only when something is due).
ROUTED_SWEEP_EVERY_S = 60.0


class RoutedWakeBudget:
    """Sliding one-hour count of routed wakes per destination session."""

    def __init__(self, per_hour: int = ROUTED_WAKES_PER_HOUR) -> None:
        self.per_hour = per_hour
        self._wakes: dict[str, list[float]] = {}

    def _recent(self, session_id: str, now: float) -> list[float]:
        kept = [t for t in self._wakes.get(session_id, []) if t > now - 3600.0]
        self._wakes[session_id] = kept
        return kept

    def has_room(self, session_id: str, now: float | None = None) -> bool:
        moment = time.time() if now is None else now
        return len(self._recent(session_id, moment)) < self.per_hour

    def charge(self, session_id: str, now: float | None = None) -> None:
        moment = time.time() if now is None else now
        self._recent(session_id, moment).append(moment)


#: The connector process's budget (one connector per config dir holds the lock).
ROUTED_BUDGET = RoutedWakeBudget()
DESIGNATED_OVER_BUDGET = "over_budget"


# -- the plan the wire-in acts on -------------------------------------------------


@dataclass(frozen=True)
class RoutePlan:
    """A :class:`RouteDecision` with what the cards need to say about it."""

    target: str | None
    reason: str
    #: Routed: the ref of the target's newest ask in this DM (display only).
    ask_ref: str | None = None
    #: Designated + ambiguous: ``(session_id, minutes_ago, ref or None)``, newest first.
    candidates: tuple[tuple[str, int, str | None], ...] = ()


def plan_route(
    *,
    config_dir: Path,
    endpoint_id: str,
    own_account_id: str,
    designated_session: str,
    conversation_id: str,
    kind: str | None,
    sender_class: str | None,
    sender_account_id: str,
    thread_root_id: str | None,
    now: float | None = None,
) -> RoutePlan:
    """Gather the facts and decide. Every failure is the designated session.

    The ledger is read only when the cheap checks pass (a group, channel or own
    account never touches it), and liveness only when a session was actually
    picked.
    """

    from .. import reply_guard

    moment = time.time() if now is None else now
    ledger: tuple[str | None, dict[str, float]] | None = None
    if kind == "dm" and sender_class in ("human", "agent"):
        ledger = reply_guard.routing_facts(
            config_dir,
            endpoint_id=endpoint_id,
            conversation_id=conversation_id,
            thread_root_id=thread_root_id,
            since=window_start(moment),
        )
    candidates_pool: frozenset[str] = frozenset()
    if ledger is not None and (ledger[0] or ledger[1]):
        candidates_pool = live_capable_sessions(None)
    decision = decide_route(
        kind=kind,
        sender_class=sender_class,
        sender_account_id=sender_account_id,
        own_account_id=own_account_id,
        designated_session=designated_session,
        thread_root_id=thread_root_id,
        ledger=ledger,
        live_capable=candidates_pool,
    )
    if decision.target is not None and not ROUTED_BUDGET.has_room(decision.target, moment):
        return RoutePlan(None, DESIGNATED_OVER_BUDGET)
    if decision.target is not None:
        return RoutePlan(
            decision.target,
            decision.reason,
            ask_ref=reply_guard.latest_ask_ref(
                config_dir,
                endpoint_id=endpoint_id,
                conversation_id=conversation_id,
                session_id=decision.target,
            ),
        )
    listed = tuple(
        (
            session_id,
            max(0, int((moment - at) // 60)),
            reply_guard.latest_ask_ref(
                config_dir,
                endpoint_id=endpoint_id,
                conversation_id=conversation_id,
                session_id=session_id,
            ),
        )
        for session_id, at in decision.candidates
    )
    return RoutePlan(None, decision.reason, candidates=listed)


# -- liveness ----------------------------------------------------------------

#: One scan per connector iteration. A scan is ~0.3-0.7 ms with a handful of
#: records (measured 2026-11-18), but a burst of messages in one iteration would
#: repeat it per message for no new information.
_ITERATION_CACHE: contextvars.ContextVar[dict[str, frozenset[str]] | None] = contextvars.ContextVar(
    "team_bridge_routing_liveness", default=None
)


@contextlib.contextmanager
def iteration_scope() -> Iterator[None]:
    """Cache liveness for the messages one connector iteration handles."""

    token = _ITERATION_CACHE.set({})
    try:
        yield
    finally:
        _ITERATION_CACHE.reset(token)


def live_capable_sessions(root: Path | None, *, fresh: bool = False) -> frozenset[str]:
    """Session ids that are LIVE and advertise ``team-bridge-routed-inbox-v1``.

    NOT ``bulk_resume.live_session_ids``: that one reaps stale records as it
    scans and counts a "wedged" session (pid alive, loop not answering) as live,
    and a wedged session cannot read a routed row. ``reap=False`` keeps this a
    pure read. Any failure is an EMPTY set, which routes nothing.

    ``fresh`` skips the per-iteration cache: the pre-delivery recheck must see
    the registry as it is now, not as it was when the iteration began.
    """

    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import TEAM_BRIDGE_ROUTED_INBOX_CAPABILITY

    cache = _ITERATION_CACHE.get()
    key = str(root)
    if cache is not None and not fresh and key in cache:
        return cache[key]
    try:
        scanned = registry.scan(root, reap=False)
        found = frozenset(
            str(record.session_id)
            for record, state in scanned
            if state == "live"
            and getattr(record, "session_id", None)
            and TEAM_BRIDGE_ROUTED_INBOX_CAPABILITY in (getattr(record, "capabilities", None) or ())
        )
    except Exception:  # noqa: BLE001 — a liveness read is a filter, never a gate
        found = frozenset()
    if cache is not None:
        cache[key] = found
    return found


def window_start(now: float | None = None) -> float:
    """The earliest post time that still counts as "recent" for the poster rule."""

    return (time.time() if now is None else now) - POSTER_WINDOW_S


__all__ = [
    "DESIGNATED_OVER_BUDGET",
    "ROUTED_BUDGET",
    "ROUTED_RECLAIM_AFTER_S",
    "ROUTED_SWEEP_EVERY_S",
    "ROUTED_WAKES_PER_HOUR",
    "ROUTED_WAKE_MIN_INTERVAL_S",
    "RoutePlan",
    "RoutedWakeBudget",
    "plan_route",
    "DESIGNATED_AMBIGUOUS",
    "DESIGNATED_DEFAULT",
    "DESIGNATED_IS_ASKER",
    "DESIGNATED_NOT_DM",
    "DESIGNATED_NOT_LIVE",
    "DESIGNATED_NO_LEDGER",
    "DESIGNATED_NO_POSTER",
    "DESIGNATED_SENDER",
    "POSTER_WINDOW_S",
    "ROUTED_SOLE_POSTER",
    "ROUTED_THREAD",
    "RouteDecision",
    "decide_route",
    "iteration_scope",
    "live_capable_sessions",
    "window_start",
]
