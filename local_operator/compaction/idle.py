"""Idle / pre-wake compaction: fold a parked session while its cache is warm.

WHY. A session that ends a turn on a large context and then sits idle — a
wake scheduled for later, or an operator who stepped away — pays a full read
of that context on its next request. Inside the provider's prompt-cache TTL
that read is cheap (a cache READ, ~0.1x input price); after it, the next
request re-writes the whole prefix at cache-WRITE price (1.25x on Anthropic's
5-minute cache, 2x on its 1-hour cache) and every later wake pays to re-read
it. A compaction run just BEFORE the TTL lapses spends one cache-read advisor
call plus one summarization, and every subsequent wake then starts on the
reduced context. This is the pattern omp uses (compact before the cache TTL
expires once a conversation has gone quiet), adapted to this harness's
advisor-gated, anti-thrash posture.

MATERIAL INFORMATION. Nothing here weakens what a pass preserves. An idle
pass is the ordinary pass (same summarizer, same verbatim preserved user
turns, same carried asks/decisions, same snapcompact archive); the only
difference is WHO picks the verbatim window. A parked advisor reads the whole
conversation and anchors ``preserve_from`` at the start of still-open work,
and the pass runs only on its explicit ``compact_now`` with high confidence.
Any other answer — no, unsure, malformed, timed out — is a durable DECLINE
that suppresses the advisor until enough new material has accumulated.

ANTI-THRASH, derived from the transcript (so it survives a restart):

- a pass is considered only once the context has grown by at least
  ``G = min(idle_min_growth_tokens, 0.25 * T_eff)`` since the larger of the
  last pass's after-size and the last decline's size;
- the pass refuses when fewer than ``G / 2`` tokens would be summarized;
- therefore at most one idle pass (or one advisor call) per ``G`` of growth,
  and two consecutive wakes with small growth can never both compact.

This module is PURE: no clock, no session, no I/O. ``Session`` gathers the
inputs (``compaction_context_tokens``, the transcript markers, the wake
scheduler's next due time, the last usage's cache-write split) and calls
:func:`idle_compaction_due`; tests drive the same function with tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "CACHE_TTL_1H_S",
    "CACHE_TTL_5M_S",
    "COMPACTION_IDLE_DECLINED_TYPE",
    "COMPACTION_IDLE_PASS_TYPE",
    "FALLBACK_CACHE_TTL_S",
    "FIRE_LEAD_MARGIN_S",
    "IDLE_GROWTH_FRACTION",
    "IDLE_THRESHOLD_FRACTION",
    "IdleDecision",
    "PARKED_ADVISOR_NOTE",
    "PARKED_MIN_CONFIDENCE",
    "cache_ttl_from_usage",
    "fallback_cache_ttl_s",
    "idle_compaction_due",
    "idle_fire_at_ms",
    "idle_growth_for",
    "idle_threshold_for",
]

#: Idle threshold as a fraction of the effective trigger ``T_eff``. Scales
#: the 200k default down on small windows: on a 200k window (T_eff 160k) the
#: idle line is 96k, on a 64k window (T_eff 51.2k) it is ~30.7k. Above 60%
#: the idle pass would race the ordinary size trigger it is meant to precede.
IDLE_THRESHOLD_FRACTION = 0.6

#: Growth gate as a fraction of ``T_eff``: 25% so a small window still needs
#: a quarter of its trigger in NEW material between idle passes.
IDLE_GROWTH_FRACTION = 0.25

#: Floor on the parked advisor's self-reported confidence. Higher than the
#: beta advisor's 0.6 default because a parked hint is trusted with more: it
#: may NARROW the verbatim window below the task-boundary floor (nobody is
#: mid-task, so "since the last genuine user turn" overstates what is live),
#: and the operator is not there to notice a bad cut before the next wake
#: builds on it.
PARKED_MIN_CONFIDENCE = 0.75

#: Slack between the advisor's worst-case finish and the cache expiry: the
#: fire time is ``E - (advisor_timeout_s + FIRE_LEAD_MARGIN_S)``. 30 s covers
#: the summarizer's first request (which must also read the warm prefix) and
#: event-loop scheduling jitter of ``call_later``.
FIRE_LEAD_MARGIN_S = 30.0

#: Anthropic's two explicit cache lifetimes (the ``cache_write_5m_tokens`` /
#: ``cache_write_1h_tokens`` split on ``Usage``).
CACHE_TTL_5M_S = 300.0
CACHE_TTL_1H_S = 3600.0

#: TTL assumed when the last usage carries no cache-write split (OpenAI-family
#: wires, or a turn that only READ the cache). Mirrors the lifetimes in
#: ``providers/account_affinity._LIFETIMES``: anthropic:messages 300 s / 3600 s
#: explicit, openai:responses 1800 s documented (300 s heuristic), openai:chat
#: and codex:responses 300 s. The SHORTEST lifetime a provider family may use
#: is taken: firing early on a cache that is still warm only costs an earlier
#: pass, while firing late on an expired one turns the advisor's cheap read
#: into a full cache write — the one outcome this feature must never produce.
FALLBACK_CACHE_TTL_S: Mapping[str, float] = {
    "anthropic": CACHE_TTL_5M_S,
    "openai": CACHE_TTL_5M_S,
    "codex": CACHE_TTL_5M_S,
}
_UNKNOWN_PROVIDER_TTL_S = CACHE_TTL_5M_S

#: Transcript custom types (``append_custom``; custom entries never enter LLM
#: context, see ``Transcript.append_custom``). Both are the durable half of the
#: anti-thrash rule: a restarted runtime re-derives the growth baseline from
#: them rather than from process memory.
#:
#: ``compaction_idle_pass`` records the after-size of an idle pass against its
#: marker id. It is a separate row rather than a field on the marker because
#: the marker is committed BEFORE the pass measures ``tokens_after`` (the
#: rebuilt context is priced after ``append_compaction`` returns) and the
#: transcript is append-only.
COMPACTION_IDLE_PASS_TYPE = "compaction_idle_pass"
#: ``compaction_idle_declined`` records the context size at which the parked
#: advisor said no (or failed); the next consideration needs ``G`` growth past it.
COMPACTION_IDLE_DECLINED_TYPE = "compaction_idle_declined"

#: One model-facing paragraph (English by design: prompts are not product
#: copy) appended to the advisor's question when the session is parked. It is
#: APPENDED to the final user turn so the conversation prefix stays
#: byte-identical and the call remains a cache read.
PARKED_ADVISOR_NOTE = (
    "The agent is now IDLE: its turn has ended and nothing will run until the "
    "next scheduled wake or until the user returns. This is a parked "
    "compaction check, run so the next turn starts on a smaller context. "
    "Nobody is mid-step, so you may anchor `preserve_from` later than the last "
    "user message when the work before your anchor is finished — but anchor it "
    "at the START of any work that is still open (an unfinished task, a pending "
    "wake's purpose, an unanswered question), because everything before the "
    "anchor survives only as a summary. Answer `compact_now: true` only when "
    "that summary would lose nothing the next turn needs; when in doubt answer "
    "false with an honest, low confidence."
)


@dataclass(frozen=True, slots=True)
class IdleDecision:
    """The predicate's verdict plus the numbers it judged, for logs and tests."""

    due: bool
    #: Machine token naming the first failed gate, or ``"due"``.
    reason: str
    context_tokens: int = 0
    threshold_tokens: int = 0
    growth_tokens: int = 0
    growth_needed_tokens: int = 0
    #: Whether an armed wake exists at all (due after the cache expiry when the
    #: verdict is ``due``) — recorded in the advisor posture.
    wake_pending: bool = False


def idle_threshold_for(t_eff: int, idle_threshold_tokens: int) -> int:
    """``min(idle_threshold_tokens, 0.6 * T_eff)``; non-positive config disables."""
    if idle_threshold_tokens <= 0 or t_eff <= 0:
        return 0
    return min(int(idle_threshold_tokens), int(IDLE_THRESHOLD_FRACTION * t_eff))


def idle_growth_for(t_eff: int, idle_min_growth_tokens: int) -> int:
    """``G = min(idle_min_growth_tokens, 0.25 * T_eff)``, floored at 1 token.

    The floor matters: ``G = 0`` would make "growth since the last pass" true
    of every context and remove the anti-thrash rule entirely.
    """
    configured = max(1, int(idle_min_growth_tokens))
    if t_eff <= 0:
        return configured
    return max(1, min(configured, int(IDLE_GROWTH_FRACTION * t_eff)))


def cache_ttl_from_usage(usage: Any) -> float | None:
    """TTL the last request WROTE its prefix with, or ``None`` when unknown.

    A request that only read the cache reports zero writes; the caller keeps
    the previous answer sticky in that case (a read refreshes the lifetime the
    prefix was written with, it does not change it).
    """
    if usage is None:
        return None
    if int(getattr(usage, "cache_write_1h_tokens", 0) or 0) > 0:
        return CACHE_TTL_1H_S
    if int(getattr(usage, "cache_write_5m_tokens", 0) or 0) > 0:
        return CACHE_TTL_5M_S
    return None


def fallback_cache_ttl_s(provider: str | None) -> float:
    """Conservative TTL for a provider with no observed cache-write split."""
    return FALLBACK_CACHE_TTL_S.get(str(provider or "").lower(), _UNKNOWN_PROVIDER_TTL_S)


def idle_fire_at_ms(cache_expiry_ms: int, advisor_timeout_s: float) -> int:
    """``F = E - (advisor_timeout_s + FIRE_LEAD_MARGIN_S)``.

    Early enough that the advisor's worst case still reads a warm prefix.
    """
    lead_ms = int((max(0.0, float(advisor_timeout_s)) + FIRE_LEAD_MARGIN_S) * 1000)
    return int(cache_expiry_ms) - lead_ms


def idle_compaction_due(
    *,
    enabled: bool,
    advisor_enabled: bool,
    strategy: str,
    is_subagent: bool,
    context_tokens: int,
    t_eff: int,
    idle_threshold_tokens: int,
    idle_min_growth_tokens: int,
    baseline_after_tokens: int | None,
    declined_at_tokens: int | None,
    next_wake_due_ms: int | None,
    cache_expiry_ms: int,
    now_ms: int,
    advisor_timeout_s: float,
) -> IdleDecision:
    """Whether an idle pass should be attempted now. Gates, in order:

    1. ``enabled`` (compaction on AND idle on), ``advisor_enabled``,
       ``strategy != "off"``, not a subagent — a child's context dies with its
       job, and its parent's turn is what is waiting on it, so it is never
       idle in the sense that matters.
    2. ``C >= min(idle_threshold_tokens, 0.6 * T_eff)``.
    3. ``C - max(A_M, D) >= G``: growth since the last pass's after-size
       (``baseline_after_tokens``; ``0`` when the session never compacted,
       ``None`` when a pass exists but its after-size is not yet observable)
       and since the last decline (``declined_at_tokens``).
    4. No armed wake due at or before the cache expiry ``E``: that wake will
       find the cache warm and its own turn end re-arms this check, so
       compacting now would only spend the advisor call early.
    5. The cache must still be warm for the advisor's worst case
       (``now + advisor_timeout_s <= E``); a cold advisor read costs a full
       cache write of ``C``, which is the outcome this feature must not buy.
    """
    wake_pending = next_wake_due_ms is not None

    def verdict(reason: str, **extra: int) -> IdleDecision:
        return IdleDecision(
            due=reason == "due",
            reason=reason,
            context_tokens=int(context_tokens),
            wake_pending=wake_pending,
            **extra,
        )

    if not enabled or strategy == "off":
        return verdict("disabled")
    if not advisor_enabled:
        return verdict("advisor_off")
    if is_subagent:
        return verdict("subagent")
    threshold = idle_threshold_for(t_eff, idle_threshold_tokens)
    if threshold <= 0 or context_tokens < threshold:
        return verdict("below_idle_threshold", threshold_tokens=threshold)
    growth_needed = idle_growth_for(t_eff, idle_min_growth_tokens)
    if baseline_after_tokens is None:
        # A pass exists but nothing has measured what it left: the context
        # IS that after-size (no request has grown it yet), so growth is 0.
        return verdict(
            "no_growth_baseline", threshold_tokens=threshold, growth_needed_tokens=growth_needed
        )
    baseline = max(int(baseline_after_tokens), int(declined_at_tokens or 0))
    growth = int(context_tokens) - baseline
    if growth < growth_needed:
        return verdict(
            "insufficient_growth",
            threshold_tokens=threshold,
            growth_tokens=growth,
            growth_needed_tokens=growth_needed,
        )
    if next_wake_due_ms is not None and next_wake_due_ms <= cache_expiry_ms:
        return verdict(
            "wake_keeps_cache_warm",
            threshold_tokens=threshold,
            growth_tokens=growth,
            growth_needed_tokens=growth_needed,
        )
    # Judged against the advisor's budget alone, not the full fire lead: the
    # ``FIRE_LEAD_MARGIN_S`` slack is what absorbs ``call_later`` lateness, so
    # a timer that wakes a few ms after ``F`` must still pass this gate.
    if now_ms + int(max(0.0, float(advisor_timeout_s)) * 1000) > cache_expiry_ms:
        return verdict(
            "cache_cold",
            threshold_tokens=threshold,
            growth_tokens=growth,
            growth_needed_tokens=growth_needed,
        )
    return verdict(
        "due",
        threshold_tokens=threshold,
        growth_tokens=growth,
        growth_needed_tokens=growth_needed,
    )
