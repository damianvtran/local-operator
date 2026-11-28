"""Prompt-cache keep-alive: re-read an always-on session's prefix across idle gaps.

WHY. An always-on session (the chief of staff, a Team Bridge session) idle-exits
90 s after its last turn and is relaunched per message. Its Anthropic prefix is
cached with the 1-hour TTL, so any gap longer than an hour re-writes the whole
prefix at cache-WRITE price on the next message. Measured on the chief-of-staff
session that motivated this (14 days of ``analytics.db``, ~200k context, Opus):
62 gaps over an hour, 38 of which ended in ``cache_read=0``; a cold request cost
a median $1.98 and a warm re-read $0.079. A request that re-reads the prefix
every 55 minutes, at most 4 times per gap, saved a simulated $77.3 per 14 days
(N=1: $52.5, N=3: $74.9, N=5: $77.0 — the saving plateaus at 4). The busiest
day needed 13 pings, which is what the daily cap of 24 sits above.

MECHANISM. One hidden wake row (``kind="cache_keepalive"``) per session, armed
at every main-thread turn end for ``anchor + ttl - LEAD``. The wake supervisor
relaunches a runtime for it exactly as it does for an ``ask_timeout`` row, and
that runtime sends ONE read-only request over the current prefix (the compaction
advisor's shape: same system blocks, tools and history, ``tool_choice="none"``)
and re-arms the row from the ping's own start. Nothing joins the conversation.

This module is PURE: no clock, no session, no I/O. ``Session`` gathers the
inputs and calls :func:`arm_decision` / :func:`fire_decision`; the unit tests
drive the same functions with tables.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

__all__ = [
    "DEFAULT_KEEPALIVE_DAILY_MAX",
    "DEFAULT_KEEPALIVE_MAX_CONSECUTIVE",
    "DEFAULT_KEEPALIVE_MODE",
    "KEEPALIVE_KIND",
    "KEEPALIVE_LEAD_S",
    "KEEPALIVE_MAX_TOKENS",
    "KEEPALIVE_PING_INTERVAL_S",
    "KEEPALIVE_PURPOSE",
    "KEEPALIVE_ROW_PREFIX",
    "KEEPALIVE_TTL_S",
    "KEEPALIVE_TURN_TEXT",
    "KeepaliveDecision",
    "KeepaliveMode",
    "arm_decision",
    "fire_decision",
    "prefix_digest",
    "row_id",
]

KeepaliveMode = Literal["off", "chief_of_staff", "always_on"]

#: ``chief_of_staff``: only the session ``aida/state.json`` names. The one
#: measured case; ``always_on`` widens it to the Team Bridge session too.
DEFAULT_KEEPALIVE_MODE: KeepaliveMode = "chief_of_staff"

#: Pings per idle gap before the chain stops and the cold start happens anyway.
#: The backtest's saving plateaus here (module docstring); a dead-quiet night
#: costs at most 4 x ~$0.08 before that.
DEFAULT_KEEPALIVE_MAX_CONSECUTIVE = 4

#: Pings per session per trailing 24 h, counted from the analytics ledger. Above
#: the measured busiest day (13 at N=4); a ledger that cannot be read refuses.
DEFAULT_KEEPALIVE_DAILY_MAX = 24

#: The wake kind and the request purpose (one literal, two namespaces): the
#: ledger's ``by_purpose`` reports the spend under this name. The row predicate
#: lives in ``wakes/store.py`` (``is_cache_keepalive_row``) beside the other
#: internal kinds, because that module must stay stdlib-only.
KEEPALIVE_KIND = "cache_keepalive"
KEEPALIVE_PURPOSE = "cache_keepalive"

#: Row ids are ``cache-keepalive:<anchor_ms>`` — a FRESH id per write, so the
#: pump retiring the row that just fired can never remove the re-armed one.
KEEPALIVE_ROW_PREFIX = "cache-keepalive:"

#: Only the 1-hour cache is kept warm. On the 5-minute cache a ping every
#: ~4 minutes would cost more than the cold start it prevents.
KEEPALIVE_TTL_S = 3600.0

#: Fire this long before expiry: the supervisor's scan slice (10 s) plus its
#: engage deadline (180 s, ``wakes/supervisor.py``) plus a relaunch's MCP wait
#: (``DECLARED_MCP_READY_TIMEOUT_S`` = 30 s, ``session/declared_mcp_ready.py``)
#: is 220 s, so a ping that starts late still reads before the entry lapses.
KEEPALIVE_LEAD_S = 300.0

#: Spacing of a chain's pings (``ttl - lead``); the economic pass's cold-next
#: estimate extends the expiry by this much per remaining ping.
KEEPALIVE_PING_INTERVAL_S = KEEPALIVE_TTL_S - KEEPALIVE_LEAD_S

#: The appended user turn. Fixed text, so every ping writes the same few tail
#: tokens; the answer is discarded.
KEEPALIVE_TURN_TEXT = "Reply OK."

#: The generation bound. ``max_tokens`` is not part of the cache key, so this
#: only bounds what the ping can spend on output; small but not 1, so adaptive
#: thinking can stop on its own rather than every ping ending on ``length``.
KEEPALIVE_MAX_TOKENS = 64


def row_id(anchor_ms: int) -> str:
    return f"{KEEPALIVE_ROW_PREFIX}{int(anchor_ms)}"


def prefix_digest(
    system_blocks: Sequence[str], tools: Iterable[Any], message_ids: Sequence[str]
) -> str:
    """sha256 over what the cached prefix is made of: system, tools, history ids.

    ``message_ids`` is the id of every message the arm saw, in order (``""`` for
    one without an id, so positions still count). The row stores the digest AND
    that count; the fire hashes only its context's FIRST ``count`` ids. A
    message appended since (a quiet peer message, which never runs a turn) is
    therefore not part of the comparison: the ping sends it as uncached tail
    and still reads the prefix. Hashing the arm-time LAST id instead refused
    every such fire, which on the chief of staff (a peer message every ~30
    min) killed every overnight chain before its first ping. A message removed
    or replaced at or before the recorded span (compaction rewrote history)
    shifts an id and still refuses.

    Computed by the SAME call at turn end (stored on the row) and at fire (after
    the relaunched runtime's tools connected). A mismatch means the ping would
    write a new prefix rather than read the old one, so it is not sent. Tools
    are reduced to their wire fields; ``execute`` and other callables never
    reach the provider and are not stable across processes.
    """
    wire_tools = [
        {
            "name": str(getattr(tool, "name", "")),
            "description": str(getattr(tool, "description", "")),
            "parameters": getattr(tool, "parameters", {}) or {},
        }
        for tool in tools
    ]
    payload = json.dumps(
        {
            "system": list(system_blocks),
            "tools": wire_tools,
            # A key the pre-split digest never used, so a row armed by that
            # build can never compare equal here: it refuses once and clears.
            "message_ids": list(message_ids),
        },
        sort_keys=True,
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KeepaliveDecision:
    """``ok`` plus the reason, logged at INFO either way (the runtime log is INFO-only)."""

    ok: bool
    reason: str


def arm_decision(
    *,
    mode: str,
    covered: bool,
    is_subagent: bool,
    ttl_s: float | None,
    anchor_ms: int,
    cache_read_tokens: int,
    cache_write_1h_tokens: int,
    route_matches: bool,
    other_wake_due_ms: int | None,
    remaining_pings: int,
    saver_active: bool,
) -> KeepaliveDecision:
    """Whether a turn end (or a ping) should arm the next keep-alive row.

    ``remaining_pings`` is ``keepalive_max_consecutive`` minus the pings this
    chain already sent: a turn end passes the whole budget, a ping passes what
    is left after it, so a spent chain writes no row it would only refuse.

    ``covered`` is the identity half of ``mode`` (the session is the chief of
    staff, or the Team Bridge session under ``always_on``), resolved by the
    caller because it needs config reads.

    The cache-activity rule is deliberately NOT "skip when nothing was read": a
    cold request (read 0, the whole prefix written at 1h) is exactly the one
    worth keeping warm. Only a request that neither read nor wrote a 1h entry
    has nothing to keep.

    ``saver_active``: Radient Saver reduces (or may reduce) this session's
    turns. The splice rewrites a turn's leading messages into epochs, and the
    ping, not being a turn, is never spliced, so it would pay a full write of
    a prefix the next turn does not send. Routing the ping through the splice
    instead was rejected: a splice can upload, bill and re-cut epochs, which a
    request that must be a pure re-read cannot risk.
    """
    if mode == "off":
        return KeepaliveDecision(False, "mode_off")
    if is_subagent:
        return KeepaliveDecision(False, "subagent")
    if not covered:
        return KeepaliveDecision(False, "not_covered")
    if saver_active:
        return KeepaliveDecision(False, "saver_active")
    if remaining_pings <= 0:
        return KeepaliveDecision(False, "consecutive_cap")
    if ttl_s != KEEPALIVE_TTL_S:
        return KeepaliveDecision(False, "not_1h_cache")
    if anchor_ms <= 0:
        return KeepaliveDecision(False, "no_cache_anchor")
    if cache_read_tokens <= 0 and cache_write_1h_tokens <= 0:
        return KeepaliveDecision(False, "no_cache_activity")
    if not route_matches:
        return KeepaliveDecision(False, "route_mismatch")
    expiry = int(anchor_ms + KEEPALIVE_TTL_S * 1000)
    if other_wake_due_ms is not None and other_wake_due_ms < expiry:
        # That fire's own request re-reads the prefix in time.
        return KeepaliveDecision(False, "wake_keeps_cache_warm")
    return KeepaliveDecision(True, "armed")


def fire_decision(
    *,
    now_ms: int,
    anchor_ms: int,
    attempt: int,
    max_consecutive: int,
    daily_count: int | None,
    daily_max: int,
    stored_digest: str,
    current_digest: str | None,
    route_matches: bool,
    busy: bool,
    saver_active: bool,
) -> KeepaliveDecision:
    """Whether a fired keep-alive row may send its ping now.

    Ordered cheapest first; every refusal retires the chain (the next real turn
    end arms a fresh one).
    """
    if now_ms >= int(anchor_ms + KEEPALIVE_TTL_S * 1000):
        # A late fire (the Mac slept through the due time): the entry is gone
        # and a ping would only pay the cold write the next message pays anyway.
        return KeepaliveDecision(False, "expired")
    if attempt >= max_consecutive:
        return KeepaliveDecision(False, "consecutive_cap")
    if saver_active:
        # Checked again at fire: Saver can be switched on between the arm and
        # a relaunched runtime's fire, and the digest does not see the splice.
        return KeepaliveDecision(False, "saver_active")
    if busy:
        # A turn, compaction or advisor owns the conversation; its own end
        # re-arms with a fresher anchor.
        return KeepaliveDecision(False, "busy")
    if not route_matches:
        return KeepaliveDecision(False, "route_mismatch")
    if daily_count is None:
        return KeepaliveDecision(False, "ledger_unreadable")
    if daily_count >= daily_max:
        return KeepaliveDecision(False, "daily_cap")
    if not stored_digest or current_digest != stored_digest:
        return KeepaliveDecision(False, "prefix_changed")
    return KeepaliveDecision(True, "ping")
