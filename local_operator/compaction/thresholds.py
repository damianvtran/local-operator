"""Compaction settings and trigger-threshold math.

The numbers here are cache-economics, not cosmetics:
- :func:`resolve_threshold_tokens` is the ONE place a compaction trigger is
  derived: ``min(threshold_percent x window, threshold_tokens)``. Two knobs,
  one resolved number, no caller re-deriving it.
- The reserve keeps room for the model's next output + tool calls after a
  compaction; it is floored at 15% of the window so small windows do not
  compact down to nothing. As a trigger input it can only pull the trigger
  EARLIER (see :func:`resolve_threshold_tokens`).
- ``should_compact`` fires strictly *above* the resolved threshold so a
  context sitting exactly on the threshold does not thrash.
- :data:`RECOVERY_BAND` is the anti-thrash hysteresis added after a live
  production dead-loop bug (a pass that shaves just under the line every turn
  sustained an auto-continue dead loop): a compaction pass only counts as
  having created headroom when the residual lands at or below
  ``0.8 x threshold``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_RESERVE_TOKENS",
    "DEFAULT_THRESHOLD_PERCENT",
    "DEFAULT_THRESHOLD_TOKENS",
    "CompactionSettings",
    "RECOVERY_BAND",
    "cleared_headroom",
    "compaction_context_tokens",
    "effective_reserve_tokens",
    "resolve_strategy",
    "resolve_threshold_cap_tokens",
    "resolve_threshold_percent",
    "resolve_threshold_tokens",
    "should_compact",
]

#: Reserve floor applied when ``reserve_tokens`` is unset
#: (``DEFAULT_RESERVE_TOKENS``). ``reserve_tokens is None`` — not comparing
#: values against this default — is what marks the reserve as *defaulted*.
DEFAULT_RESERVE_TOKENS = 16384

#: Percentage trigger default: compact once the context passes 80% of the
#: model's context window. Expressed as a FRACTION; ``0.80`` and ``80`` are
#: both accepted in config and mean the same thing (see
#: :func:`resolve_threshold_percent`).
DEFAULT_THRESHOLD_PERCENT = 0.80

#: Absolute trigger default: compact once the context passes 600k tokens even
#: when that is a small fraction of a very large window.
DEFAULT_THRESHOLD_TOKENS = 600_000


class CompactionSettings(BaseModel):
    """Compaction knobs, mirrored from ``config.yml`` ``values.compaction.*``.

    Two knobs decide WHEN a pass fires — ``threshold_percent`` (fraction of
    the model's context window) and ``threshold_tokens`` (absolute ceiling) —
    and :func:`resolve_threshold_tokens` is the only thing that combines them.
    ``reserve_tokens`` is ``None`` when unset: that provenance (not the value)
    marks the reserve as defaulted, and only an explicit reserve constrains
    the trigger.
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
            "Min headroom kept after compaction. None = defaulted (the 15%"
            " proportional floor). An EXPLICIT reserve also caps the trigger at"
            " window - reserve, so it can only make compaction fire earlier."
        ),
    )
    keep_recent_tokens: int = Field(
        default=20000, description="Tokens of recent history kept verbatim across a compaction."
    )
    threshold_percent: float = Field(
        default=DEFAULT_THRESHOLD_PERCENT,
        description=(
            "Percentage trigger, as a fraction of the model context window."
            " 0.80 and 80 both mean 80%. Out of range (<= 0 or > 100) falls back"
            " to 0.80 with a warning."
        ),
    )
    threshold_tokens: int = Field(
        default=DEFAULT_THRESHOLD_TOKENS,
        description=(
            "Absolute token trigger. The resolved trigger is the SMALLER of"
            " this and the percentage trigger. Non-positive falls back to"
            " 600000 with a warning."
        ),
    )
    auto_continue: bool = Field(
        default=True,
        description="Schedule a continuation prompt after a successful post-turn pass.",
    )
    mid_turn_enabled: bool = Field(
        default=True, description="Allow threshold compaction at safe tool-loop boundaries."
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: Any) -> Any:
        """Read the superseded ``max_threshold_tokens`` key as ``threshold_tokens``.

        ``max_threshold_tokens`` was a ceiling applied on top of a separate
        absolute trigger. Under the single ``min(percent x window, tokens)``
        rule the absolute knob IS the ceiling, so two keys meant one thing —
        the exact "two competing notions of when to compact" that let a
        resolved trigger disagree with itself. A config carrying the old key
        keeps working (silently dropping a user's explicit ceiling is how a
        session sails past a provider's real serving limit), and the warning
        names the rename so it can be fixed once.
        """
        if not isinstance(data, dict) or "max_threshold_tokens" not in data:
            return data
        data = dict(data)  # never mutate the caller's config dict
        legacy = data.pop("max_threshold_tokens")
        if "threshold_tokens" in data:
            logger.warning(
                "values.compaction.max_threshold_tokens is superseded by threshold_tokens; "
                "ignoring the legacy key because threshold_tokens is set as well"
            )
        else:
            data["threshold_tokens"] = legacy
            logger.warning(
                "values.compaction.max_threshold_tokens is superseded by "
                "threshold_tokens (same meaning: the absolute trigger ceiling); "
                "reading it as threshold_tokens: %s — rename it in config.yml",
                legacy,
            )
        return data


#: Settings whose out-of-range values have already been reported, so a
#: misconfigured session warns once instead of on every turn (the trigger is
#: resolved at every tool-loop boundary).
_WARNED: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    """``logger.warning`` de-duplicated per (setting, value) for the process."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


def resolve_threshold_percent(settings: CompactionSettings) -> float:
    """Validated percentage trigger as a fraction in ``(0, 1]``.

    Two spellings are accepted on purpose: ``0.80`` (the field default, a
    fraction) and ``80`` (what a key named ``*_percent`` invites a user to
    write). Values above 1 are read as percentages, values at or below 1 as
    fractions. Without this, ``threshold_percent: 80`` would resolve to 8000%
    of the window (a trigger that never fires) or ``threshold_percent: 0.8``
    to 0.8% (a session that compacts every turn) depending on which spelling
    the code picked.

    Anything outside ``(0, 100]`` — zero, negative, 250 — falls back to the
    documented default with a warning rather than breaking the session, the
    same posture the config coercion takes for an invalid block.
    """
    raw = float(settings.threshold_percent)
    if raw <= 0.0 or raw > 100.0:
        _warn_once(
            f"threshold_percent={raw}",
            "values.compaction.threshold_percent=%s is out of range (0 < p <= 100); "
            "using the default %s",
            raw,
            DEFAULT_THRESHOLD_PERCENT,
        )
        return DEFAULT_THRESHOLD_PERCENT
    return raw / 100.0 if raw > 1.0 else raw


def resolve_threshold_cap_tokens(settings: CompactionSettings) -> int:
    """Validated absolute trigger in tokens (positive)."""
    raw = int(settings.threshold_tokens)
    if raw <= 0:
        _warn_once(
            f"threshold_tokens={raw}",
            "values.compaction.threshold_tokens=%s is not a positive token count; "
            "using the default %s",
            raw,
            DEFAULT_THRESHOLD_TOKENS,
        )
        return DEFAULT_THRESHOLD_TOKENS
    return raw


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


def resolve_threshold_tokens(window_tokens: int, settings: CompactionSettings) -> int:
    """Context size at which compaction triggers for this window — THE rule.

    ``min(threshold_percent x window, threshold_tokens)``, clamped to
    ``[1, window - 1]``.

    ``min`` is deliberate: the EARLIER of the two thresholds wins, because the
    two knobs guard different failure modes and each is only meaningful on one
    side of the window-size range.

    - The percentage keeps a small-context model compacting in proportion to
      what it can actually hold: 80% of a 200k window is 160k, and an absolute
      600k trigger there could never fire at all.
    - The absolute ceiling stops a very large window from letting one session
      grow to a size that is slow and expensive on every single request even
      though it technically still fits: at 600k of a 1M window every turn is
      re-sending 600k tokens.

    Which is why a resolved trigger must never be re-derived by a caller. A
    session on a 1M-context model was observed compacting at ~235k — a
    defensive ceiling on top of a second absolute trigger had collapsed to
    23% of the window — throwing away three quarters of its usable context
    per pass, and every one of those unnecessary passes was also an
    opportunity for a compaction bug (the snapcompact image-replay path) to
    destroy the session's history outright.

    An EXPLICIT ``reserve_tokens`` (never the 16384 default, whose provenance
    is ``reserve_tokens is None``) additionally caps the trigger at
    ``window - effective_reserve_tokens``, so a user who demands more
    post-compaction headroom than the percentage leaves gets it. It can only
    pull the trigger earlier; a reserve at or above the window is impossible
    to honour and is ignored rather than resolving to "never compact".
    """
    if window_tokens <= 0:
        return 0
    percent = resolve_threshold_percent(settings)
    trigger = min(int(window_tokens * percent), resolve_threshold_cap_tokens(settings))
    if settings.reserve_tokens is not None:
        reserve = effective_reserve_tokens(window_tokens, settings)
        if 0 < reserve < window_tokens:
            trigger = min(trigger, window_tokens - reserve)
    return max(1, min(trigger, window_tokens - 1))


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
