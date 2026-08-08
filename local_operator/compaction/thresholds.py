"""Compaction settings and trigger-threshold math.

The numbers here are cache-economics, not cosmetics:
- The reserve keeps room for the model's next output + tool calls after a
  compaction; it is floored at 15% of the window so small windows do not
  compact down to nothing. A DEFAULTED reserve that is impossible for a small
  window is recovered to the proportional 15% reserve
  (``resolveBudgetReserveTokens``); an explicit reserve is always honored.
- With no explicit threshold/percent/reserve, the trigger is the
  ``docs/REWRITE.md`` §C default: the lesser of 80% of the window and
  600 000 tokens.
- ``should_compact`` fires strictly *above* the resolved threshold so a
  context sitting exactly on the threshold does not thrash.
- :data:`RECOVERY_BAND` is the anti-thrash hysteresis added after a live
  production dead-loop bug (a pass that shaves just under the line every turn
  sustained an auto-continue dead loop): a compaction pass only counts as
  having created headroom when the residual lands at or below
  ``0.8 x threshold``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "DEFAULT_RESERVE_TOKENS",
    "DEFAULT_THRESHOLD_CAP_TOKENS",
    "CompactionSettings",
    "RECOVERY_BAND",
    "cleared_headroom",
    "compaction_context_tokens",
    "effective_reserve_tokens",
    "resolve_budget_reserve_tokens",
    "resolve_strategy",
    "resolve_threshold_tokens",
    "should_compact",
]

#: Reserve floor applied when ``reserve_tokens`` is unset
#: (``DEFAULT_RESERVE_TOKENS``). ``reserve_tokens is None`` — not comparing
#: values against this default — is what marks the reserve as *defaulted*,
#: which :func:`resolve_budget_reserve_tokens` needs to recover impossible
#: defaults.
DEFAULT_RESERVE_TOKENS = 16384

#: docs/REWRITE.md §C default threshold: the lesser of 80% of the model
#: context window and this absolute ceiling.
DEFAULT_THRESHOLD_CAP_TOKENS = 600_000


class CompactionSettings(BaseModel):
    """Compaction knobs, mirrored from ``config.yml`` ``values.compaction.*``.

    ``threshold_tokens`` and ``threshold_percent`` are negative by default,
    meaning "unset" — resolution then falls back to the reserve/default-based
    threshold. ``reserve_tokens`` is ``None`` when unset: that provenance (not
    the value) marks the reserve as defaulted, and a defaulted reserve that is
    impossible for the window falls back to the 15% proportional reserve.
    """

    enabled: bool = True
    strategy: Literal["auto", "context-full", "snapcompact", "off"] = Field(
        default="auto",
        description=(
            "Compaction mechanism. 'auto' picks snapcompact for vision models,"
            " else context-full (see resolve_strategy); 'off' disables."
        ),
    )
    reserve_tokens: int | None = Field(
        default=None,
        description=(
            "Min headroom kept after compaction. None = defaulted: an impossible"
            " default recovers to the 15% proportional reserve; an explicit"
            " reserve is always honored."
        ),
    )
    keep_recent_tokens: int = Field(
        default=20000, description="Tokens of recent history kept verbatim across a compaction."
    )
    threshold_percent: float = Field(
        default=-1.0, description="Percent-of-window trigger; <= 0 means reserve-based."
    )
    threshold_tokens: int = Field(
        default=-1, description="Explicit token trigger; > 0 wins over percent."
    )
    max_threshold_tokens: int = Field(
        default=600_000,
        description=(
            "Ceiling on the RESOLVED default threshold when no explicit trigger"
            " is set. Providers routinely advertise a context window far larger"
            " than the aggregate serving path can actually sustain (a 1.05M"
            " advertisement whose requests start aborting around 250k is the"
            " case that motivated this knob), and the default threshold of"
            " min(window*0.8, 600k) inherits the advertisement's optimism. This"
            " caps the threshold before that optimism reaches the trigger math."
        ),
    )
    auto_continue: bool = Field(
        default=True,
        description="Schedule a continuation prompt after a successful post-turn pass.",
    )
    mid_turn_enabled: bool = Field(
        default=True, description="Allow threshold compaction at safe tool-loop boundaries."
    )


def effective_reserve_tokens(window_tokens: int, settings: CompactionSettings) -> int:
    """Reserve that actually applies for a given context window.

    ``max(floor(window * 0.15), settings.reserve_tokens ?? DEFAULT_RESERVE_TOKENS)``
    — the configured reserve is a floor, never a ceiling: small windows get at
    least 15% so the post-compaction context is not squeezed to nothing.
    """
    reserve = settings.reserve_tokens
    if reserve is None:
        reserve = DEFAULT_RESERVE_TOKENS
    return max(int(window_tokens * 0.15), reserve)


def resolve_budget_reserve_tokens(window_tokens: int, settings: CompactionSettings) -> int:
    """Reserve used to derive the trigger threshold
    (``resolveBudgetReserveTokens``).

    The default absolute reserve predates small bundled windows and can leave
    no practical budget there; recover a DEFAULTED reserve that is impossible
    for the window (>= window, or would push the threshold below the 15%
    floor) to the proportional ``max(1, floor(w * 0.15))`` reserve so small
    windows stay usable. Explicit reserves — including one that happens to
    equal the default — always win, because they intentionally shrink the
    usable prompt budget; provenance is ``reserve_tokens is None``, never a
    value comparison against the default. Only a reserve >= window falls back
    for explicit reserves too (the threshold must stay strictly below the
    window).
    """
    reserve = effective_reserve_tokens(window_tokens, settings)
    proportional = max(1, int(window_tokens * 0.15))
    reserve_was_defaulted = settings.reserve_tokens is None
    default_reserve_impossible = reserve_was_defaulted and reserve >= window_tokens - proportional
    reserve_exceeds_window = reserve >= window_tokens
    if default_reserve_impossible or reserve_exceeds_window:
        return proportional
    return reserve


def resolve_threshold_tokens(window_tokens: int, settings: CompactionSettings) -> int:
    """Context size at which compaction triggers for this window.

    Precedence (first match wins):

    1. ``threshold_tokens > 0`` — explicit override, clamped to ``[1, w-1]``.
    2. ``threshold_percent > 0`` — ``max(1, floor(w * clamp(pct, 1, 99) / 100))``.
    3. Otherwise:
       - explicit ``reserve_tokens`` — ``clamp(w - resolve_budget_reserve_tokens(w, s), 1, w-1)``;
         an explicit reserve always defines the usable prompt budget and
         bypasses the §C default;
       - defaulted reserve that is impossible for this window — the 15%
         proportional recovery (same clamp);
       - defaulted reserve, feasible window — the docs/REWRITE.md §C default:
         ``clamp(min(int(w * 0.8), 600_000, max_threshold_tokens), 1, w - 1)``;
         the ``max_threshold_tokens`` tail caps a provider-advertised window
         whose practical serving ceiling is far lower than the raw
         advertisement (the LO-on-LO session that stalled at ~250k on a
         1.05M-advertised model motivated it).
    """
    if window_tokens <= 0:
        return 0
    if settings.threshold_tokens > 0:
        return max(1, min(settings.threshold_tokens, window_tokens - 1))
    if settings.threshold_percent > 0:
        pct = min(max(settings.threshold_percent, 1.0), 99.0)
        return max(1, int(window_tokens * pct / 100.0))
    if settings.reserve_tokens is not None:
        reserve = resolve_budget_reserve_tokens(window_tokens, settings)
        return max(1, min(window_tokens - reserve, window_tokens - 1))
    effective = effective_reserve_tokens(window_tokens, settings)
    proportional = max(1, int(window_tokens * 0.15))
    if effective >= window_tokens - proportional or effective >= window_tokens:
        # Defaulted reserve is impossible for this window: 15% recovery.
        return max(1, min(window_tokens - proportional, window_tokens - 1))
    default_threshold = min(
        int(window_tokens * 0.8), DEFAULT_THRESHOLD_CAP_TOKENS, settings.max_threshold_tokens
    )
    return max(1, min(default_threshold, window_tokens - 1))


def should_compact(context_tokens: int, window_tokens: int, settings: CompactionSettings) -> bool:
    """Whether the current context exceeds the compaction threshold.

    Strictly greater-than so a context exactly on the threshold is stable;
    ``window_tokens <= 0`` (unknown window) never triggers.
    """
    if not settings.enabled or settings.strategy == "off" or window_tokens <= 0:
        return False
    return context_tokens > resolve_threshold_tokens(window_tokens, settings)


def compaction_context_tokens(provider_reported: int | None, local_estimate: int) -> int:
    """Context size used for trigger checks.

    ``max(provider_reported, local_estimate)``: provider usage is ground
    truth, but it is *floored by the local estimate* because a compression
    hook can deflate what the provider sees while real stored history grows
    unbounded — trusting the provider alone would then never trigger.
    """
    if provider_reported is None:
        return local_estimate
    return max(provider_reported, local_estimate)


#: A compaction pass counts as having created headroom only when the residual
#: context lands at or below ``RECOVERY_BAND * threshold``. Below the band the
#: pass barely helped and scheduling an auto-continue (or re-firing) would
#: thrash the prompt cache; see the live dead-loop bug it guards against.
RECOVERY_BAND = 0.8


def cleared_headroom(residual_tokens: int, threshold_tokens: int) -> int:
    """Headroom a compaction pass created, negative when it failed to clear.

    ``threshold - residual``; callers pair this with :data:`RECOVERY_BAND`
    (residual must be ``<= RECOVERY_BAND * threshold``) before scheduling an
    auto-continue, so marginal passes never sustain a dead loop.
    """
    return threshold_tokens - residual_tokens


def resolve_strategy(
    settings: CompactionSettings, model: Any
) -> Literal["context-full", "snapcompact"]:
    """Concrete strategy for one model: ``snapcompact`` iff the setting says
    so, or ``auto`` and the model reads images; else ``context-full``.

    ``model`` is duck-typed (only ``supports_images`` is read) so this module
    never imports harness types; ``strategy == "off"`` is gated by the caller
    (:func:`should_compact`) and resolves to ``context-full`` here.
    """
    if settings.strategy == "snapcompact":
        return "snapcompact"
    if settings.strategy == "auto" and getattr(model, "supports_images", False):
        return "snapcompact"
    return "context-full"
