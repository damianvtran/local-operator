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
    "DEFAULT_WIRE_BYTES_BUDGET",
    "DEFAULT_WIRE_BYTES_TRIGGER",
    "CompactionSettings",
    "RECOVERY_BAND",
    "WIRE_RECOVERY_BAND",
    "cleared_headroom",
    "cleared_wire_headroom",
    "over_wire_budget",
    "resolve_wire_bytes_budget",
    "resolve_wire_bytes_trigger",
    "compaction_context_tokens",
    "effective_reserve_tokens",
    "resolve_advisor_floor_tokens",
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

#: HARD ceiling on the serialized request, in bytes — the number the render
#: seam sheds frames to satisfy. NOT a token figure; see
#: :func:`~.tokens.estimate_wire_bytes` for why the two rulers stay apart.
#:
#: 24 MB is Anthropic's 32 MB Messages-API cap minus 25% headroom for the
#: system prompt, the tool schemas, the ~2.2% JSON envelope, and the growth
#: the NEXT request adds before another guard runs.
#:
#: Calibrated against real traffic rather than guessed: a scan of all **4,738
#: sessions** in a production session store found exactly **one** above 24 MB
#: (the session that wedged on HTTP 413), two above 16 MB, ten above 8 MB, and
#: none above 32 MB. So this budget sheds nothing that works today — it fires
#: only on the shape that was already failing outright.
DEFAULT_WIRE_BYTES_BUDGET = 24_000_000

#: SOFT trigger: fires a real compaction pass (with a summary) well before the
#: hard budget forces amputation. ~2/3 of the hard budget, which on the
#: measured store is above every session but the three largest — the point is
#: that a screenshot-heavy session compacts *properly and early* instead of
#: surviving by shedding frames at the wall.
DEFAULT_WIRE_BYTES_TRIGGER = 16_000_000


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
    # ``None`` — not 0 — is the ordinary-session default, and the distinction
    # is the whole feature. A chat session's images are attachments the user
    # pasted: each one is distinct evidence and there is no "newer view of the
    # same thing" to supersede it, so pruning them would throw away content.
    # A screen-driving surface (the evaluation runner, a browser-driving
    # session) sees the SAME surface again every turn, and there the old
    # frames are dead weight: the old pilot measured 3 visible frames as the
    # only configuration that scored above zero, and every-frame-kept as 0.0
    # at $262. So the knob is opt-in per surface, and ``None`` means the frame
    # prune is not even consulted — ``run_compaction_pass`` with the defaults
    # is byte-identical to a pass without it, which a test pins.
    #
    # EXTENDED, not reversed: ``None`` now means "never prune frames for
    # size-of-CONTEXT reasons", and a byte shed engages underneath it when the
    # request would otherwise be REFUSED outright (``wire_bytes_budget``). The
    # argument above assumes pruning frames costs the user content for no
    # forced reason, which is exactly right up to the wall; at the wall the
    # alternative is not "keep every frame", it is losing the whole session —
    # a session that 413s loses all 42 frames, while the shed keeps 28. The
    # default and the distinct-attachments reasoning are both untouched, and a
    # session under budget still performs no work here at all.
    keep_recent_frames: int | None = Field(
        default=None,
        description=(
            "Screenshots kept verbatim across a compaction pass; older ones are"
            " replaced by a short notice. None = never prune frames (ordinary"
            " sessions, where images are distinct attachments). Screen-driving"
            " surfaces opt in with a small count."
        ),
    )

    # --- Transport size budget (BYTES, not tokens) --------------------------
    #
    # These two are the only knobs in this model measured in bytes, and the
    # separation is deliberate: the token thresholds above answer "how much of
    # the context WINDOW is occupied", these answer "will the HTTP request be
    # accepted". A screenshot-heavy session can sit at 15% of a 1M window and
    # still be over a provider's request cap, because a flat per-image token
    # charge (correct for billing) is blind to base64 length. Never compare a
    # value here against a token figure.
    #
    # Registered in ``settings_io.SETTINGS`` so an operator on a provider with
    # a different cap can move them without editing code.
    wire_bytes_budget: int = Field(
        default=DEFAULT_WIRE_BYTES_BUDGET,
        description=(
            "Hard ceiling on the serialized request in BYTES. Older screenshots"
            " are shed from the rendered history (never from the transcript) to"
            " stay under it. Non-positive disables the shed."
        ),
    )
    wire_bytes_trigger: int = Field(
        default=DEFAULT_WIRE_BYTES_TRIGGER,
        description=(
            "Soft trigger in BYTES: a compaction pass fires once the serialized"
            " request passes this, so a screenshot-heavy session summarises"
            " early instead of shedding frames at the hard budget."
            " Non-positive disables the byte trigger."
        ),
    )

    # --- Speculative compaction advisor (BETA, off by default) -------------
    #
    # Every field below is optional with an inert default, so a config written
    # before this feature validates unchanged AND resolves to byte-identical
    # behaviour: with ``advisor_enabled`` false nothing reads the rest.
    #
    # The advisor exists because the shipped trigger is a SIZE trigger and the
    # thing that actually hurts is a cut landing inside a live task. The
    # operator has explicitly asked to retain capacity up to the 600k ceiling
    # when genuinely needed, so the fix is NOT a lower default threshold; it is
    # a semantic second opinion that may only pull the trigger earlier, and
    # only down to ``advisor_floor_tokens``.
    advisor_enabled: bool = Field(
        default=False,
        description=(
            "BETA. Ask the model, off the turn's critical path, whether the"
            " context is at a natural task boundary and worth compacting early."
            " The advice can only make a pass fire EARLIER, never later, and"
            " never below advisor_floor_tokens."
        ),
    )
    advisor_every_n_turns: int = Field(
        default=20,
        description=(
            "Turns between advisor calls. The advisor reads the whole context,"
            " so it is cheap only as a prompt-cache READ; asking every turn"
            " would multiply that by the turn count for advice that changes"
            " slowly."
        ),
    )
    advisor_floor_tokens: int = Field(
        default=200_000,
        description=(
            "Hard floor on the advisor-lowered trigger. The advisor may pull"
            " the trigger down to this and no further, so a confidently wrong"
            " hint costs an early pass, never a compaction treadmill."
        ),
    )
    advisor_trigger_tokens: int = Field(
        default=300_000,
        description=(
            "Context size below which the advisor is not consulted at all."
            " Under it there is no problem to solve and the call would be pure"
            " cost."
        ),
    )
    advisor_min_confidence: float = Field(
        default=0.6,
        description="Hints below this self-reported confidence are discarded, not repaired.",
    )
    advisor_timeout_s: float = Field(
        default=30.0,
        description=(
            "Total budget for one advisor call. Nothing awaits the call, so"
            " this bounds the background task rather than a turn."
        ),
    )
    advisor_max_calls: int = Field(
        default=200,
        description=(
            "Ceiling on advisor calls per session, so a very long run cannot"
            " drift. 0 means NO CALLS (the advisor is off), not 'unlimited' —"
            " deliberately unlike the sibling knobs, where 0 disables a"
            " restriction. This one IS the restriction, so the fail-closed"
            " reading is the safe one: a config that zeroes a spend ceiling"
            " must not thereby remove it."
        ),
    )
    advisor_cooldown_turns: int = Field(
        default=60,
        description=(
            "Turns the advisor is suppressed for after an advisor-triggered"
            " pass. Anti-thrash: the pass just moved the boundary it would be"
            " asked to judge."
        ),
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


def resolve_advisor_floor_tokens(window_tokens: int, settings: CompactionSettings) -> int:
    """Lowest trigger the compaction advisor is allowed to ask for.

    Lives beside :func:`resolve_threshold_tokens` because it is the same kind
    of number and must be derived in the same one place. It is a FLOOR, not a
    trigger: :func:`should_compact` takes ``min(threshold, this)``, so the
    advisor moves the trigger down to it and can never move it up.

    Clamped into ``[1, resolve_threshold_tokens(...)]``. The upper clamp is
    what guarantees the "earlier only" posture even when a config sets
    ``advisor_floor_tokens`` above the ordinary trigger (which would otherwise
    read as "compact later when the advisor is on" — a second, competing
    trigger). A non-positive floor is meaningless and falls back to the
    ordinary trigger, i.e. the advisor changes nothing.
    """
    threshold = resolve_threshold_tokens(window_tokens, settings)
    floor = int(getattr(settings, "advisor_floor_tokens", 0) or 0)
    if floor <= 0:
        return threshold
    return max(1, min(floor, threshold))


def should_compact(
    context_tokens: int,
    window_tokens: int,
    settings: CompactionSettings,
    *,
    advisory_ok: bool = False,
    wire_bytes: int = 0,
) -> bool:
    """Whether the current context exceeds the compaction threshold.

    Strictly greater-than so a context exactly on the threshold is stable;
    ``window_tokens <= 0`` (unknown window) never triggers.

    ``advisory_ok`` is the compaction advisor's ONLY entry point into the
    trigger, and it is deliberately a parameter of THIS function rather than a
    second predicate beside it. The advisor answers "is now a good moment?";
    it does not answer "should we compact?", and a separate
    ``advisor_should_compact()`` would be exactly the trigger drift this
    module's docstring forbids — two functions free to disagree about when a
    pass is due, which is how a 1M-context session ended up firing at 23% of
    its window.

    So an accepted hint lowers the threshold to
    ``min(threshold, resolve_advisor_floor_tokens(...))`` and nothing else.
    That is the same posture an explicit ``reserve_tokens`` already has in
    :func:`resolve_threshold_tokens`: an additional input that can only pull
    the trigger EARLIER, never later, and never past a floor. Every other
    property of the trigger — monotonicity in ``context_tokens`` (which the
    session's cheap upper-bound pre-gate depends on), the strict inequality,
    the disabled/off short-circuit — is untouched, so a future reader citing
    this as precedent should carry those constraints too.

    ``wire_bytes`` is the second input with that posture, and it is here for
    the same reason ``advisory_ok`` is: it is a different QUESTION about the
    same decision ("will the request be accepted?" beside "does the context
    fit the window?"), and a separate ``should_compact_for_size()`` beside
    this one would be exactly the trigger drift the module docstring forbids.
    So it is OR-ed into the one resolved answer, where it can only pull the
    trigger EARLIER.

    The byte term is what makes an image-heavy session compact at all. Its
    token estimate is honest and small — 42 screenshots read as 154,690
    tokens, 15.5% of a 1M window — while the serialized request is 34 MB,
    past the provider's cap. No token threshold can see that, because the
    per-image token charge is flat by design (see
    :func:`~.tokens.estimate_wire_bytes`).

    Monotonicity is preserved in BOTH numeric arguments, which the session's
    cheap upper-bound pre-gate depends on: raising either can only turn a
    ``False`` into a ``True``. Bytes are exact and cheap, so the pre-gate can
    pass the real figure rather than a bound.

    With ``advisory_ok`` false and ``wire_bytes`` 0 (their defaults, and what
    every caller but the plan gate passes) the function is byte-identical to
    its previous form.
    """
    if not settings.enabled or settings.strategy == "off" or window_tokens <= 0:
        return False
    threshold = resolve_threshold_tokens(window_tokens, settings)
    if advisory_ok and getattr(settings, "advisor_enabled", False):
        threshold = min(threshold, resolve_advisor_floor_tokens(window_tokens, settings))
    if context_tokens > threshold:
        return True
    byte_trigger = resolve_wire_bytes_trigger(settings)
    return byte_trigger > 0 and wire_bytes > byte_trigger


def resolve_wire_bytes_budget(settings: CompactionSettings) -> int:
    """THE hard byte ceiling, resolved in one place.

    Same discipline as :func:`resolve_threshold_tokens`: one resolver, no
    caller re-deriving the number from the settings model. A non-positive
    value means "no byte ceiling" and is returned as ``0`` so every caller can
    test it the same way; a garbage value falls back to the default rather
    than disabling the guard silently, because a typo here is the difference
    between a session that recovers and one that 413s forever.
    """
    raw = getattr(settings, "wire_bytes_budget", DEFAULT_WIRE_BYTES_BUDGET)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("compaction.wire_bytes_budget is not an integer; using the default")
        return DEFAULT_WIRE_BYTES_BUDGET
    return value if value > 0 else 0


def resolve_wire_bytes_trigger(settings: CompactionSettings) -> int:
    """THE soft byte trigger, resolved in one place.

    Clamped to the hard budget: a trigger ABOVE the ceiling would mean the
    render seam amputates frames before a proper compaction pass ever fires,
    which inverts the whole design — the soft trigger exists so the session
    summarises with a model call instead of surviving by amputation. Returns
    ``0`` when the byte trigger is disabled.
    """
    raw = getattr(settings, "wire_bytes_trigger", DEFAULT_WIRE_BYTES_TRIGGER)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("compaction.wire_bytes_trigger is not an integer; using the default")
        value = DEFAULT_WIRE_BYTES_TRIGGER
    if value <= 0:
        return 0
    budget = resolve_wire_bytes_budget(settings)
    return min(value, budget) if budget > 0 else value


def over_wire_budget(wire_bytes: int, settings: CompactionSettings) -> bool:
    """Whether ``wire_bytes`` is past the HARD ceiling.

    The render seam's question ("must I shed before sending?"), kept beside
    the trigger's question so the two read the same resolved number.
    """
    budget = resolve_wire_bytes_budget(settings)
    return budget > 0 and wire_bytes > budget


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


#: The byte-side twin of :data:`RECOVERY_BAND`, and it exists because that
#: constant CANNOT cover the byte trigger. ``RECOVERY_BAND`` is defined on
#: tokens against the token threshold; a pass fired by byte pressure can leave
#: the token residual comfortably inside the token band while the request is
#: still over the byte budget — the token side would then say "headroom
#: created", schedule an auto-continue, and the next turn would fire the byte
#: trigger again on a context nothing had shrunk. That is precisely the live
#: dead-loop ``RECOVERY_BAND`` was added for, arrived at through the new
#: trigger, so the guard has to be restated in the new trigger's own units.
#:
#: Same 0.8 ratio, same meaning: a pass counts as having created byte headroom
#: only when the residual request lands at or below ``0.8 x`` the soft byte
#: trigger.
WIRE_RECOVERY_BAND = 0.8


def cleared_wire_headroom(residual_bytes: int, settings: CompactionSettings) -> bool:
    """Whether a pass created real BYTE headroom, for the auto-continue gate.

    Returns ``True`` when the byte trigger is disabled, so a session that
    never opted into byte pressure keeps exactly the token-only behaviour it
    has today — this must not become a second veto on ordinary text sessions.

    Otherwise the residual must sit at or below
    ``WIRE_RECOVERY_BAND * trigger``. A pass that shaved a screenshot or two
    and is still near the trigger has NOT recovered, and scheduling a
    continuation on it re-enters the same pass next turn.
    """
    trigger = resolve_wire_bytes_trigger(settings)
    if trigger <= 0:
        return True
    return residual_bytes <= WIRE_RECOVERY_BAND * trigger


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
