"""Patience waits — a HIDDEN internal timer an agent attaches to a sent message.

WHAT IT IS (R30–R35). A proactive-class agent that asks the user (or a peer) a
question may attach a patience wait: an internal timer that wakes it if no reply
arrives in ~5 minutes, so it can decide whether to follow up. The timer is not
rendered anywhere (no wake line, no card, no badge, no notification); its fire
text reaches only the model's context. The cycle is bounded: waits back off
(×3 per attempt), an episode allows at most ``max_attempts`` outbound waits, and
the whole episode dies at its TTL regardless of per-message overrides — past the
bound the agent sends ONE terminal acknowledgement and stops until the user (or
a new cadence/scheduled wake) returns.

WHY IT RIDES THE WAKE ENGINE, AND WHERE IT DOES NOT. Rows are ordinary
``WakeSchedule`` entries tagged ``kind="patience"`` (plus ``hidden`` and the
episode fields): one writer (``Session._persist_wake_schedules``), one index,
one supervisor, and persist/advance/retire/dormancy come free. What is NOT
shared is every semantic that makes patience patience — attach, cancel-on-reply,
backoff, attempts, TTL — and those live HERE, pure and testable. A separate
store would have duplicated the whole wake substrate and created a second
answer to "what happens on resume".

THE TWO RULES THAT ARE EASY TO GET WRONG, both pinned by tests:

1. **Cancel-on-reply is a WATERMARK, not a flag.** A fire is stale when its
   ``armed_at`` predates the session's last REAL user/peer inbound, evaluated at
   delivery time from the transcript (the shared truth, so it holds across
   runtimes and restarts). The trap: wake and patience deliveries are themselves
   ``attribution="user"`` custom messages — "role user" is NOT a reply. The
   watermark counts only (a) real user turns (typed prompts, including steered
   ones) and (b) peer messages, and it EXCLUDES harness-injected user rows (the
   goal judge's continuation prompt is not a person). Best-effort explicit
   cancels at ``prompt()``/``receive_peer_message`` entry are the tidy half;
   this watermark is the half that survives a crash, a restart, or another
   runtime having written the reply.

2. **The episode lives in the transcript, because the row does not.** A fire
   retires its one-shot row (the pump advances it), so continuation after a
   fire — attempt N+1, the backoff wait, the TTL clock — is reconstructed from
   the fire facts carried in the ``wake_prompt`` details plus the pending row
   when one exists. Deriving it from a row's lifetime alone would lose the
   episode at exactly the moment it matters (no reply ⇒ re-arm in the same
   cycle), and a sidecar file would be a third writer of wake state.

OVER-CANCEL IS DELIBERATE (design §8.2.3): a reply retires ALL of the session's
pending patience rows, and an explicit re-arm costs nothing (the agent attaches
a fresh wait when it responds). A missed cancel is a nagging bug; an
over-cancel is one re-arm. Per-target precision is deferred with it.

CROSS-CLASS SAFETY. Every entry point here checks the session's action class at
the moment it acts (``local_operator.action_class.session_action_class``); a
reactive session cannot arm (its ``patience`` tool does not exist either), and a
fire that comes due after a switch to reactive is retired silently by the
session's delivery guard — see ``Session._deliver_patience_wake``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from local_operator.harness.wake_types import MAX_WAKE_MESSAGE_CHARS, MAX_WAKE_SCHEDULES

#: The ONE patience predicate and the ONE human-surface filter, imported from
#: the store (which stays stdlib-only so every index reader can carry them).
#: Re-exported through this module's public names for its internal callers;
#: there is no second implementation to drift (agent review round 1, R3).
from local_operator.wakes.store import (
    is_internal_wake_row as _store_is_internal_wake_row,
)
from local_operator.wakes.store import is_patience_row as _store_is_patience_row
from local_operator.wakes.store import scheduled_rows as _store_scheduled_rows

logger = logging.getLogger(__name__)

#: The id prefix of every patience row. ``patience-<n>`` with the lowest free
#: ``n``, chosen from the ids present at arm time (the ``aida-extra-`` pattern).
#: A namespace of its OWN so ``w1..w16`` allocation in ``build_wake_schedule``
#: can never collide with one.
ID_PREFIX = "patience-"

#: The user-facing kind value that marks a row as a patience wait.
KIND = "patience"

# ---- config defaults (read via :func:`policy`; pinned by the settings tests) --

DEFAULT_WAIT_MS = 300_000  # 5 min — R30's "~5-10 minutes, typically"
DEFAULT_BACKOFF = 3
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TTL_MS = 7_200_000  # 2 h hard stop, regardless of per-message overrides
DEFAULT_MAX_PENDING = 4

#: The clamp on any single wait. 60 s is the wake engine's own floor
#: (``MIN_WAKE_INTERVAL_MS``) — a sub-minute timer would starve a turn — and
#: 24 h is the design's ceiling for an agent-chosen timeout.
MIN_TIMEOUT_MS = 60_000
MAX_TIMEOUT_MS = 86_400_000

#: Slack added to the episode TTL when bounding the transcript scan: an episode
#: cannot hold a fire older than its TTL, so anything above that is provably
#: irrelevant to the current classification. 30 min covers re-arm latency
#: (fires happen inside turns, and a turn may sit queued for a while).
SCAN_SLACK_MS = 1_800_000

#: Page size for the file-backed scan (the tool path). The search stops at the
#: first page whose oldest row predates the cutoff, so a quiet transcript costs
#: one page.
_SCAN_PAGE = 200
#: A hard page cap so a pathological journal (clock skew, a forest of rows
#: inside the window) cannot turn an arm into an unbounded read. 20 pages =
#: 4,000 rows; an episode's artifacts inside a 2.5 h window that exceed this
#: are already operating far outside the mechanism's bounds.
_SCAN_MAX_PAGES = 20


@dataclass(frozen=True)
class PatiencePolicy:
    """The resolved bounds — one reader, defaults applied (``proactive.*``)."""

    default_ms: int
    backoff: int
    max_attempts: int
    ttl_ms: int
    max_pending: int


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_proactive_section(config_dir: Path | str) -> Mapping[str, Any]:
    """The ``proactive`` config mapping, best-effort (the aida reader's twin).

    Absent means defaults; an unreadable file means defaults too — these bounds
    guard a HIDDEN mechanism, and a broken read must never be louder than the
    behaviour it could not find.
    """
    try:
        from local_operator.config import ConfigManager

        raw = ConfigManager(config_dir=Path(config_dir)).get_config_value("proactive", None)
    except Exception:  # noqa: BLE001 — defaults are the answer to an unreadable file
        logger.debug("patience: could not read the proactive section", exc_info=True)
        return {}
    return raw if isinstance(raw, Mapping) else {}


def policy(config_dir: Path | str) -> PatiencePolicy:
    """Resolve the patience bounds from ``proactive.patience.*``."""
    section = _read_proactive_section(config_dir)
    patience = section.get("patience")
    patience = patience if isinstance(patience, Mapping) else {}
    return PatiencePolicy(
        default_ms=max(_int_or(patience.get("default_ms"), DEFAULT_WAIT_MS), MIN_TIMEOUT_MS),
        backoff=max(_int_or(patience.get("backoff"), DEFAULT_BACKOFF), 1),
        max_attempts=max(_int_or(patience.get("max_attempts"), DEFAULT_MAX_ATTEMPTS), 1),
        ttl_ms=max(_int_or(patience.get("episode_ttl_ms"), DEFAULT_TTL_MS), MIN_TIMEOUT_MS),
        max_pending=max(_int_or(patience.get("max_pending"), DEFAULT_MAX_PENDING), 0),
    )


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def is_patience_row(row: Any) -> bool:
    """Whether a schedule (model or dumped dict) is a patience wait.

    ONE callable, re-exported from :mod:`local_operator.wakes.store` (which
    stays stdlib-only so the index readers can carry it): the session holds
    models, the index and supervisor hold dicts, and BOTH must be filtered by
    the same predicate — the design's "count scheduled rows only" rule reaches
    the CLI, the TUI panel, the desktop feed and the dormant receipt, and a
    second spelling of this check is how one of them starts counting patience
    rows (agent review round 1, R3).
    """
    return _store_is_patience_row(row)


def is_internal_wake_row(row: Any) -> bool:
    """Whether a schedule is one of the hidden internal timers (R3's union).

    Re-exported from :mod:`local_operator.wakes.store`, beside
    :func:`is_patience_row` and for the same reason: every HUMAN-SURFACE
    subtraction (a queue-timer row joined patience in that set) must ask the
    same question, and a second spelling is how one of them starts counting an
    internal row again.
    """
    return _store_is_internal_wake_row(row)


def patience_rows(rows: Iterable[Any]) -> list[Any]:
    return [row for row in rows if is_patience_row(row)]


def scheduled_rows(rows: Iterable[Any]) -> list[Any]:
    """The rows a human surface may show: the store's ONE filter (R3)."""
    return _store_scheduled_rows(list(rows))


def pending_rows(rows: Sequence[Any]) -> list[Any]:
    """Patience rows that are still waiting to fire (they always are — a fired
    one-shot retires — so this is every patience row in the list; the helper
    exists so the intent reads at call sites)."""
    return patience_rows(rows)


def next_patience_id(rows: Sequence[Any]) -> str:
    """The lowest free ``patience-<n>`` among the existing ids."""
    used = {
        str(getattr(row, "id", "") or (row.get("id") if isinstance(row, Mapping) else ""))
        for row in rows
    }
    n = 1
    while f"{ID_PREFIX}{n}" in used:
        n += 1
    return f"{ID_PREFIX}{n}"


def row_field(row: Any, name: str) -> Any:
    """One field off a model-or-dict row, for the mixed call sites."""
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


# ---------------------------------------------------------------------------
# Cycle math (pure — the "waits, backoff, attempts, TTL" tests)
# ---------------------------------------------------------------------------


def clamp_timeout_ms(requested_ms: int | None, pol: PatiencePolicy) -> int:
    """A per-message timeout clamped to the design's 60 s..24 h window."""
    value = pol.default_ms if requested_ms is None else _int_or(requested_ms, pol.default_ms)
    return max(MIN_TIMEOUT_MS, min(int(value), MAX_TIMEOUT_MS))


def patience_wait_ms(attempt: int, requested_ms: int | None, pol: PatiencePolicy) -> int:
    """The wait for ``attempt`` of an episode.

    Attempt 1 is the agent's own choice (its ``timeout``, clamped; the default
    when omitted — R38). Later attempts carry the BACKOFF floor
    ``default × backoff^(attempt-1)``: omitted timeouts advance exactly
    5m → 15m → 45m with the shipped defaults, and an explicit timeout may only
    EXTEND a later wait, never shorten it — a bound the agent can talk itself
    out of is not a bound. The result is clamped to 60 s..24 h; the episode TTL
    caps it again at arm time.
    """
    if attempt <= 1:
        return clamp_timeout_ms(requested_ms, pol)
    floor = pol.default_ms * (pol.backoff ** (attempt - 1))
    if requested_ms is None:
        wait = floor
    else:
        wait = max(_int_or(requested_ms, 0), floor)
    return max(MIN_TIMEOUT_MS, min(int(wait), MAX_TIMEOUT_MS))


# ---------------------------------------------------------------------------
# Transcript facts (the watermark and the episode, from the shared truth)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FireFact:
    """One patience fire, as read back from a ``wake_prompt`` delivery."""

    ts_ms: int
    episode_id: str
    attempt: int
    episode_started_at_ms: int


@dataclass(frozen=True)
class EpisodeFacts:
    """What the transcript says about replies, fires and markers.

    ``last_inbound_ms`` is the WATERMARK (rule 1 in the module docstring);
    ``last_fire`` is the newest patience fire still relevant; ``last_marker_ms``
    is the newest event that reopens the question after a terminal fire — a real
    user/peer inbound or a non-patience wake delivery ("until the user returns
    or a new cadence/scheduled wake brings it back", R35).
    """

    last_inbound_ms: int | None = None
    last_fire: FireFact | None = None
    last_marker_ms: int | None = None

    def marker_after(self, ts_ms: int) -> bool:
        return self.last_marker_ms is not None and self.last_marker_ms > ts_ms


def _payload(message: Any) -> Mapping[str, Any]:
    payload = getattr(message, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def _payload_details(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    details = payload.get("details")
    return details if isinstance(details, Mapping) else {}


def _is_real_user_row(payload: Mapping[str, Any]) -> bool:
    """A TYPED user turn — not a harness injection, not a custom message.

    The wake/patience delivery trap from the module docstring lives here: those
    deliveries are ``attribution="user"`` CustomMessages, and a naive
    ``role == "user"`` check would let a wake delivery cancel a patience wait.
    Harness-injected rows (the goal judge's continuation prompt) are excluded
    too: they are chrome, not a person, and a patience wait must not be closed
    by the harness talking to itself.
    """
    if payload.get("custom_type"):
        return False
    if str(payload.get("role") or "") != "user":
        return False
    provider_payload = payload.get("provider_payload")
    if isinstance(provider_payload, Mapping):
        from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY

        if provider_payload.get(RENDERED_INJECTION_KEY):
            return False
    return True


def _is_peer_row(payload: Mapping[str, Any]) -> bool:
    from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE

    return str(payload.get("custom_type") or "") == PEER_MESSAGE_MESSAGE_TYPE


def _wake_kind(payload: Mapping[str, Any]) -> str | None:
    """``None`` when the row is not a wake delivery; else its kind string."""
    from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE

    if str(payload.get("custom_type") or "") != WAKE_PROMPT_MESSAGE_TYPE:
        return None
    details = _payload_details(payload)
    return str(details.get("kind") or "scheduled")


def _fire_fact(ts_ms: int, payload: Mapping[str, Any]) -> FireFact:
    details = _payload_details(payload)
    return FireFact(
        ts_ms=ts_ms,
        episode_id=str(details.get("episode_id") or details.get("wake_id") or ""),
        attempt=_int_or(details.get("attempt"), 0),
        episode_started_at_ms=_int_or(details.get("episode_started_at"), ts_ms),
    )


def scan_entries(entries: Iterable[Any], *, cutoff_ms: int | None = None) -> EpisodeFacts:
    """Derive :class:`EpisodeFacts` from transcript entries.

    A single backward walk: entries are append-ordered, so the walk stops as
    soon as one predates ``cutoff_ms`` and the cost is proportional to the
    window, never the journal. Accepts ``TranscriptEntry``-like rows (the live
    session passes ``transcript.entries()``) or plain mappings.
    """
    last_inbound: int | None = None
    last_fire: FireFact | None = None
    last_marker: int | None = None
    ordered = list(entries)
    for entry in reversed(ordered):
        ts_ms = int(
            float(
                getattr(entry, "ts", None)
                or (entry.get("ts") if isinstance(entry, Mapping) else 0)
                or 0
            )
            * 1000
        )
        # Everything older than the cutoff is provably irrelevant: a live
        # episode's artifacts sit inside [now - TTL, now], and this scan is
        # only asked about live episodes (older ones are closed by age before
        # any comparison happens). Stopping UNCONDITIONALLY is what keeps a
        # long-quiet session's arm from walking its whole journal.
        if cutoff_ms is not None and ts_ms and ts_ms <= cutoff_ms:
            break
        payload = _payload(entry)
        if not payload and isinstance(entry, Mapping):
            # Mappings come in two shapes: a full entry (``ts``/``type``/
            # ``payload``) or the payload itself. Prefer the envelope's inner
            # payload so a hand-built full entry classifies the same as a
            # ``TranscriptEntry``; a bare payload passes through untouched.
            inner = entry.get("payload")
            payload = inner if isinstance(inner, Mapping) else entry
        if not payload:
            continue
        if _is_real_user_row(payload) or _is_peer_row(payload):
            if last_inbound is None:
                last_inbound = ts_ms
            if last_marker is None:
                last_marker = ts_ms
            continue
        kind = _wake_kind(payload)
        if kind is None:
            # Not a wake delivery: bookkeeping, assistant rows, tool rows —
            # none of them says anything about a reply or a reopen.
            continue
        if kind == KIND:
            if last_fire is None:
                last_fire = _fire_fact(ts_ms, payload)
            # A patience fire is not a marker (rule: an inbound reply or a NEW
            # cadence/scheduled wake reopens; another patience fire does not).
        elif last_marker is None:
            last_marker = ts_ms
    return EpisodeFacts(
        last_inbound_ms=last_inbound, last_fire=last_fire, last_marker_ms=last_marker
    )


def scan_transcript(session_dir: Path | str, *, now_ms: int, window_ms: int) -> EpisodeFacts:
    """The file-backed scan for callers with no live transcript object (tools).

    Bounded by ``window_ms`` and a page cap; a missing transcript answers empty
    facts (a session that never wrote one has no replies to announce).
    """
    from local_operator.session.transcript import read_transcript_page

    cutoff = now_ms - max(window_ms, 0)
    collected: list[Any] = []
    before: str | None = None
    for _ in range(_SCAN_MAX_PAGES):
        try:
            page = read_transcript_page(session_dir, before_id=before, limit=_SCAN_PAGE)
        except FileNotFoundError:
            return EpisodeFacts()
        if not page.entries:
            break
        collected.extend(page.entries)
        oldest = min((float(getattr(e, "ts", 0.0) or 0.0) for e in page.entries), default=0.0)
        if oldest * 1000 <= cutoff or not page.has_more:
            break
        before = page.entries[0].id
    return scan_entries(collected, cutoff_ms=cutoff)


# ---------------------------------------------------------------------------
# Episode derivation and the arm plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeState:
    """The session's open patience episode, or the reason there is none."""

    episode_id: str
    attempt: int
    started_at_ms: int
    terminal: bool
    pending_row: Any = None


@dataclass(frozen=True)
class ArmOutcome:
    """``arm``'s result: the new row (None on refusal) and a refusal sentence."""

    row: Any = None
    replaced_id: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.row is not None


def open_episode(
    rows: Sequence[Any], facts: EpisodeFacts, pol: PatiencePolicy, *, now_ms: int
) -> EpisodeState | None:
    """Derive the open episode from the pending row + the transcript facts.

    Rules, in order:

    - A pending patience row counts only while no inbound has landed since it
      was armed (``cancel-on-reply`` normally retires it first; a race or a
      cross-runtime write can leave one behind, and the watermark check at
      delivery will retire it — it must not, however, be treated as the
      episode to continue).
    - The newest of {pending row, last fire} governs. A fire counts while no
      inbound landed after it.
    - A fire at ``max_attempts`` with NO marker after it is TERMINAL: the cycle
      is closed and further arms are refused (R35) until an inbound or a new
      scheduled/cadence wake reopens the question.
    - Anything older than the episode TTL is closed by age.
    """
    pending = None
    for row in pending_rows(rows):
        armed_at = _int_or(row_field(row, "armed_at"), 0)
        if facts.last_inbound_ms is not None and armed_at and armed_at < facts.last_inbound_ms:
            continue  # dead: a reply landed after it was armed
        if pending is None or _int_or(row_field(pending, "armed_at"), 0) <= armed_at:
            pending = row

    fire = facts.last_fire
    if (
        fire is not None
        and facts.last_inbound_ms is not None
        and fire.ts_ms <= facts.last_inbound_ms
    ):
        fire = None  # a reply closed it

    if fire is not None and fire.attempt >= pol.max_attempts:
        if not facts.marker_after(fire.ts_ms):
            started = fire.episode_started_at_ms or fire.ts_ms
            if now_ms - started <= pol.ttl_ms:
                return EpisodeState(
                    episode_id=fire.episode_id,
                    attempt=fire.attempt,
                    started_at_ms=started,
                    terminal=True,
                    pending_row=None,
                )
            return None
        fire = None  # a marker reopened the question: treat as no open episode

    if pending is None and fire is None:
        return None

    pending_armed = _int_or(row_field(pending, "armed_at"), 0) if pending is not None else -1
    if pending is not None and (fire is None or pending_armed >= fire.ts_ms):
        started = _int_or(row_field(pending, "created_at"), 0) or pending_armed or now_ms
        if now_ms - started > pol.ttl_ms:
            return None  # closed by age; its fire (if any) retires at delivery
        return EpisodeState(
            episode_id=str(row_field(pending, "episode_id") or row_field(pending, "id") or ""),
            attempt=_int_or(row_field(pending, "attempt"), 0),
            started_at_ms=started,
            terminal=False,
            pending_row=pending,
        )

    # ``fire`` is non-None here: the pending-row branch above returned, and
    # the guard that chose this branch asserts it.
    assert fire is not None
    started = fire.episode_started_at_ms or fire.ts_ms
    if now_ms - started > pol.ttl_ms:
        return None
    return EpisodeState(
        episode_id=fire.episode_id,
        attempt=fire.attempt,
        started_at_ms=started,
        terminal=False,
        pending_row=None,
    )


def plan_arm(
    rows: Sequence[Any],
    facts: EpisodeFacts,
    pol: PatiencePolicy,
    *,
    now_ms: int,
    requested_ms: int | None = None,
    note: str = "",
    after: str = "",
) -> ArmOutcome:
    """Decide what an ``arm`` should write, or why it must refuse.

    Refusals are sentences the model can act on (the wake builder's contract):
    the attempt bound, the pending cap, the schedule-list cap, and "no room
    left before the TTL".
    """
    if len(note or "") > MAX_WAKE_MESSAGE_CHARS:
        return ArmOutcome(error=f"note must be at most {MAX_WAKE_MESSAGE_CHARS} characters.")
    state = open_episode(rows, facts, pol, now_ms=now_ms)
    if state is not None and state.terminal:
        return ArmOutcome(
            error=(
                "this patience cycle is closed (the maximum number of attempts was reached). "
                "Send one final acknowledgement that you will wait for their return; a new "
                "wait may be armed once they reply, or after a scheduled wake brings you back."
            )
        )

    pending = list(pending_rows(rows))
    if state is None:
        attempt = 1
        episode_id = next_patience_id(rows)
        started = now_ms
        replaced_id = ""
        existing = None
    else:
        attempt = state.attempt + 1
        if attempt > pol.max_attempts:
            return ArmOutcome(
                error="this patience cycle has reached its attempt bound; let it close."
            )
        episode_id = state.episode_id or next_patience_id(rows)
        # A fire-continuation reuses the retired row's id. Guard the one case
        # that could silently drop somebody else's row: the id is taken (an
        # old-format or hand-edited snapshot). Mint a fresh id instead — the
        # episode id is bookkeeping here, and the TTL anchor (`started`) is
        # what the next classification actually reads.
        if state.pending_row is None and any(
            str(row_field(r, "id") or "") == episode_id for r in rows
        ):
            episode_id = next_patience_id(rows)
        started = state.started_at_ms or now_ms
        existing = state.pending_row
        replaced_id = str(row_field(existing, "id") or "") if existing is not None else ""

    deadline = started + pol.ttl_ms
    if deadline - now_ms < MIN_TIMEOUT_MS:
        return ArmOutcome(
            error=(
                "the patience episode's time bound is reached; let the cycle close and "
                "wait for the user's return."
            )
        )
    wait = patience_wait_ms(attempt, requested_ms, pol)
    due = min(now_ms + wait, deadline)

    if existing is None:
        if len(rows) >= MAX_WAKE_SCHEDULES:
            return ArmOutcome(error=f"at most {MAX_WAKE_SCHEDULES} wake schedules are allowed.")
        if len(pending) >= pol.max_pending:
            return ArmOutcome(
                error=(
                    f"at most {pol.max_pending} pending patience waits; cancel one first "
                    "(patience(op='cancel', id=...))."
                )
            )

    from local_operator.harness.wake_types import WakeSchedule

    merged_note = (note or "").strip()
    if not merged_note and existing is not None:
        merged_note = str(row_field(existing, "note") or "")
    row = WakeSchedule(
        id=episode_id,
        # The message field is not delivered verbatim for patience rows (the
        # fire text is built at delivery time); it mirrors the note so a raw
        # transcript/index reader sees what the wait was about.
        message=merged_note,
        next_due_at=due,
        every_ms=None,
        fired_count=0,
        # EPISODE START, preserved across re-arms: the TTL clock and the fire
        # facts both read it (see wake_types.WakeSchedule.created_at's note).
        created_at=started,
        kind=KIND,
        hidden=True,
        episode_id=episode_id,
        attempt=attempt,
        armed_at=now_ms,
        armed_after=(after or "").strip(),
        note=merged_note,
    )
    return ArmOutcome(row=row, replaced_id=replaced_id)


async def arm_patience(
    scheduler: Any,
    *,
    session_dir: Path | str,
    config_dir: Path | str,
    action_class: str,
    now_ms: int,
    requested_ms: int | None = None,
    note: str = "",
    after: str = "",
    facts: EpisodeFacts | None = None,
    suppressed: bool = False,
) -> ArmOutcome:
    """The ONE arm path: class gate, engine hold, bounds, plan, persist-and-re-arm.

    ``createIf`` on the tool only keeps the SURFACE honest; the class is
    re-checked here because a switch must take effect on a running session at
    its next decision point (R36) — the inventory it was advertised in may be
    a turn old. ``suppressed`` is the ENGINE hold (Aida's pause/disabled
    switch), which blocks NEW arms while it is on (§8.2.4: pause cancels
    pending cycles and blocks new ones); the caller resolves it because only
    the host knows whether an engine holds this session.
    """
    from local_operator.action_class import PROACTIVE

    if action_class != PROACTIVE:
        return ArmOutcome(
            error=(
                "patience requires the proactive class; this session is reactive. "
                "Switch the agent's class to proactive first."
            )
        )
    if suppressed:
        return ArmOutcome(
            error=(
                "proactive output is paused for this session; patience waits are not "
                "armed while the pause is on. Resume it first (/aida resume)."
            )
        )
    pol = policy(config_dir)
    if facts is None:
        facts = scan_transcript(session_dir, now_ms=now_ms, window_ms=pol.ttl_ms + SCAN_SLACK_MS)
    rows = list(scheduler.schedules)
    outcome = plan_arm(
        rows, facts, pol, now_ms=now_ms, requested_ms=requested_ms, note=note, after=after
    )
    if not outcome.ok:
        return outcome
    updated = [
        r for r in rows if str(row_field(r, "id") or "") != str(row_field(outcome.row, "id"))
    ]
    updated.append(outcome.row)
    await scheduler.update(updated)
    return outcome


async def cancel_patience(scheduler: Any, *, row_id: str = "") -> tuple[list[str], str]:
    """Retire one or all pending patience rows. Returns ``(cancelled_ids, error)``.

    ``row_id`` empty means ALL of the session's pending waits — the deliberate
    over-cancel the design names, and what the switch-to-reactive cleanup uses.
    """
    rows = list(scheduler.schedules)
    target = [r for r in rows if is_patience_row(r)]
    if row_id:
        target = [r for r in target if str(row_field(r, "id") or "") == row_id]
        if not target:
            known = ", ".join(str(row_field(r, "id") or "") for r in pending_rows(rows)) or "none"
            return [], f"no pending patience wait with id {row_id!r} (known: {known})"
    if not target:
        return [], ""
    cancelled = {str(row_field(r, "id") or "") for r in target}
    updated = [r for r in rows if str(row_field(r, "id") or "") not in cancelled]
    await scheduler.update(updated)
    return sorted(cancelled), ""


def retire_all(rows: Sequence[Any]) -> tuple[list[Any], list[str]]:
    """``(kept, cancelled_ids)`` — the pure half of the reply-time cancel.

    Used by ``Session.cancel_pending_patience`` (through the scheduler's
    update, the ONE writer) and by the class-switch cleanup.
    """
    cancelled = [str(row_field(r, "id") or "") for r in rows if is_patience_row(r)]
    kept = [r for r in rows if not is_patience_row(r)]
    return kept, cancelled


# ---------------------------------------------------------------------------
# Delivery-time helpers (the session's fire path)
# ---------------------------------------------------------------------------


def fire_is_stale(row: Any, facts: EpisodeFacts) -> bool:
    """The watermark rule: ``armed_at`` predates the last real inbound.

    Strictly earlier: an inbound at the exact millisecond of the arm is
    ambiguous, and the safe direction differs per side — over-delivering one
    hidden note costs a turn the agent can ignore, while retiring a fire that
    should deliver loses the follow-up the agent asked for. Keep ``<`` and say
    so here.
    """
    armed_at = _int_or(row_field(row, "armed_at"), 0)
    last = facts.last_inbound_ms
    if not armed_at or last is None:
        return False
    return armed_at < last


def fire_past_ttl(row: Any, pol: PatiencePolicy, *, now_ms: int) -> bool:
    """Whether the fire is past its EPISODE TTL — retired silently, no turn."""
    started = _int_or(row_field(row, "created_at"), 0)
    if not started:
        return False
    return now_ms - started > pol.ttl_ms


def _target_label(row: Any) -> str:
    ref = str(row_field(row, "armed_after") or "")
    if ref.startswith("peer:"):
        name = ref[len("peer:") :].strip() or "the peer"
        return f"a reply from {name}"
    if ref:
        return "a reply to your message"
    return "a reply"


def fire_note(row: Any, pol: PatiencePolicy, *, now_ms: int) -> str:
    """The hidden fire text: target, elapsed, attempt, and the next step.

    Rendered for the MODEL only — the delivery path that uses it emits no
    receipt event and every human surface filters the row (see the design's
    §8.2.2 item list). The final-attempt branch is the terminal state's one
    instruction (R35): one acknowledgement, then stop.
    """
    from local_operator.harness.wake import format_duration

    armed_at = _int_or(row_field(row, "armed_at"), 0)
    elapsed = max(now_ms - armed_at, 0) if armed_at else 0
    attempt = max(_int_or(row_field(row, "attempt"), 1), 1)
    elapsed_text = f" after {format_duration(elapsed)}" if elapsed else ""
    lines = [
        f"(patience) Still no {_target_label(row)}{elapsed_text} — "
        f"attempt {attempt} of {pol.max_attempts} in this patience cycle. "
        "This is an internal timer note; it was not shown to anyone."
    ]
    note = str(row_field(row, "note") or "").strip()
    if note:
        lines.append(note)
    if attempt >= pol.max_attempts:
        lines.append(
            "This was the FINAL attempt of this cycle: send ONE terminal acknowledgement "
            "that you will wait for their return, then stop. Do not attach another "
            "patience wait; the cycle reopens only when they reply."
        )
    else:
        lines.append(
            "If you still need an answer, send your message and arm another patience "
            "wait (the next wait backs off); otherwise let the cycle close."
        )
    return "\n\n".join(lines)


def fire_details(row: Any) -> dict[str, Any]:
    """The extra ``wake_prompt`` details a patience fire carries.

    These are what make continuation-after-a-fire reconstructible (see the
    module docstring): the row retires with the fire, and the NEXT arm reads
    these back to learn the episode, the attempt and the TTL anchor.
    """
    return {
        "kind": KIND,
        "hidden": True,
        "episode_id": str(row_field(row, "episode_id") or row_field(row, "id") or ""),
        "attempt": _int_or(row_field(row, "attempt"), 0),
        "episode_started_at": _int_or(row_field(row, "created_at"), 0),
    }
