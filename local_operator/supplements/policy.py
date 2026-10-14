"""Turn-supplement policy: the kill switch, the settings snapshot and the reserved purposes.

Design authority: ``docs/design/turn-supplements.md`` §2.12 (configuration) and §6 (the
lane plan). This module is a LEAF on purpose -- standard library at import, the settings
reader imported lazily -- because the runtime handle reads :func:`enabled` on EVERY turn end
and must not pay for anything heavier than a module-level boolean.

THE KILL SWITCH. ``LOP_SUPPLEMENTS`` follows the ask-gate pattern
(``asks/policy.py::ASK_GATE``): only ``0``/``false``/``no``/``off`` (case- and
whitespace-insensitive) turn the feature off; an ABSENT variable or a typo leaves the
shipped default -- ON -- because the feature's every failure path is "show nothing", so a
typo must not silently unbuild it. It is read from the environment ONCE, at import; a test
that needs the other mode monkeypatches :data:`SUPPLEMENTS`.

THE DEFAULTS LIVE HERE. ``settings_io`` rows carry the same numbers and
``tests/unit/test_settings_io.py::_consumer_defaults`` pins each to the constant below, so a
registry default that drifts from what the code does is a red test and not a painted lie.
"""

import os
from dataclasses import dataclass
from typing import Any, Final, Mapping

#: The environment kill switch (read once at import; see the module docstring).
SUPPLEMENTS_ENV: Final = "LOP_SUPPLEMENTS"
SUPPLEMENTS: bool = os.environ.get(SUPPLEMENTS_ENV, "").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def enabled() -> bool:
    """Whether the feature runs in this process.

    A function and not the attribute, for the ``asks.policy.gate_enabled`` reason:
    monkeypatching :data:`SUPPLEMENTS` must take effect on every path at once.
    """
    return SUPPLEMENTS


# ---------------------------------------------------------------------------
# Ledger purposes (reserved here so no later lane invents a spelling)
# ---------------------------------------------------------------------------

#: The generator's provider calls (lane C1b) carry this ``ChatRequest.purpose``. Purposes
#: are open-valued; the ledger machinery already excludes a non-``turn`` purpose from strike
#: scoring, context tracking, effort classification and preflight (memo §2.12).
PURPOSE_RENDER: Final = "supplement_render"
#: RESERVED, NOT WRITTEN. The decision is a CLASSIFICATION HTTP call (``decide`` on the
#: shared ``ClassificationService``), not a provider stream request, so it never passes
#: through the request ledger -- its cost is the INFO line ``ClassificationService.decide``
#: already logs (memo §2.12: ``purpose="supplement_decision"`` "is *not* used"). The name
#: is held so a future ledger bridge for classification spend cannot spell it two ways.
PURPOSE_DECISION: Final = "supplement_decision"

# ---------------------------------------------------------------------------
# Defaults (memo §2.12 table). One constant per key, beside its consumer.
# ---------------------------------------------------------------------------

DEFAULT_ENABLED: Final = True
DEFAULT_FILES: Final = True
#: OFF by default in C1, against the memo table's ``true``, on the memo's own §6 ruling:
#: graphics spend money before any surface can render them, so the flip ships as a one-line
#: follow-up in the window that carries the first renderer. The C1a decision also has no
#: generator to hand a "yes" to.
DEFAULT_GRAPHICS: Final = False
DEFAULT_MODEL: Final = "auto"
DEFAULT_MAX_TURNS: Final = 2
MAX_TURNS_BOUNDS: Final = (1, 4)
DEFAULT_MAX_OUTPUT_TOKENS: Final = 6000
DEFAULT_TIMEOUT_S: Final = 90
#: Soft, BETWEEN-TURNS cap on one job's spend (memo §2.12): it stops a job that crosses it,
#: never a turn mid-flight. $1.00 covers roughly one Opus-class turn -- operator directive,
#: 2026-10-10 -- where the original $0.20 assumed a Sonnet-class generator envelope (2 x
#: the memo's measured $0.08 per-job high end plus $0.04 of headroom, §5.3). The generator
#: (C1b) is its reader; C1a writes no spend against it.
DEFAULT_MAX_COST_USD: Final = 1.00
#: The "N more" threshold: how many callouts are featured before the rest collapse.
DEFAULT_MAX_FEATURED: Final = 4
DEFAULT_DENY_PREFIXES: Final[tuple[str, ...]] = ()

#: ACCEPT RANGES, ``(low, high)``, for the keys below: ``from_values`` honours a stored value
#: inside as stored and falls back to the default outside. Each is MIRRORED by the registry
#: row's ``minimum``/``maximum`` in ``settings_io.py`` (carried as literals there, because
#: settings_io stays off this module's import path) and the pair is pinned equal by
#: ``tests/unit/test_settings_io.py::test_supplements_registry_bounds_are_honoured_by_the_reader``.
#: The invariant is one-directional: the page's range must sit INSIDE the reader's, because a
#: value the page accepts and the reader substitutes is a silent lie (agent review round 1,
#: R4 -- ``maxOutputTokens`` had no page maximum, so 2,000,000 came back as 6000).
MAX_OUTPUT_TOKENS_BOUNDS: Final = (1, 1_000_000)
TIMEOUT_S_BOUNDS: Final = (1, 3600)
MAX_FEATURED_BOUNDS: Final = (1, 12)


@dataclass(frozen=True)
class SupplementSettings:
    """A snapshot of ``values.supplements`` taken when the runner is built.

    Scope is ``NEW_SESSIONS`` (the runner is per runtime), so this is read once and never
    re-read: an edit lands on the next session, which is what the settings page promises.
    Every field falls back to its default on a missing, malformed or out-of-range value --
    a hand-edited ``config.yml`` must never turn the feature into an error.
    """

    enabled: bool = DEFAULT_ENABLED
    files: bool = DEFAULT_FILES
    graphics: bool = DEFAULT_GRAPHICS
    model: str = DEFAULT_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    timeout_s: int = DEFAULT_TIMEOUT_S
    max_cost_usd: float = DEFAULT_MAX_COST_USD
    max_featured: int = DEFAULT_MAX_FEATURED
    deny_prefixes: tuple[str, ...] = DEFAULT_DENY_PREFIXES

    @property
    def active(self) -> bool:
        """Whether any half of the feature is switched on (the master AND one sub-switch)."""
        return self.enabled and (self.files or self.graphics)

    @classmethod
    def from_values(cls, values: Mapping[str, Any] | None) -> "SupplementSettings":
        """Read the ``supplements`` section out of ``config.yml``'s ``values`` mapping."""
        section = values.get("supplements") if isinstance(values, Mapping) else None
        if not isinstance(section, Mapping):
            return cls()
        # Lazy: ``settings_io`` is the registry's home and this module is a leaf. The import
        # is paid when a runtime handle's runner is BUILT (``SupplementRunner.__init__``),
        # and only when the section exists at all -- an absent section returns ``cls()``
        # without touching the registry.
        from local_operator.settings_io import strict_bool

        model = section.get("model")
        prefixes = section.get("denyPrefixes")
        return cls(
            enabled=strict_bool(section.get("enabled"), DEFAULT_ENABLED),
            files=strict_bool(section.get("files"), DEFAULT_FILES),
            graphics=strict_bool(section.get("graphics"), DEFAULT_GRAPHICS),
            model=model.strip() if isinstance(model, str) and model.strip() else DEFAULT_MODEL,
            max_turns=_bounded_int(section.get("maxTurns"), DEFAULT_MAX_TURNS, *MAX_TURNS_BOUNDS),
            max_output_tokens=_bounded_int(
                section.get("maxOutputTokens"),
                DEFAULT_MAX_OUTPUT_TOKENS,
                *MAX_OUTPUT_TOKENS_BOUNDS,
            ),
            timeout_s=_bounded_int(section.get("timeoutS"), DEFAULT_TIMEOUT_S, *TIMEOUT_S_BOUNDS),
            max_cost_usd=_nonnegative_float(section.get("maxCostUsd"), DEFAULT_MAX_COST_USD),
            max_featured=_bounded_int(
                section.get("maxFeatured"), DEFAULT_MAX_FEATURED, *MAX_FEATURED_BOUNDS
            ),
            deny_prefixes=(
                tuple(item.strip() for item in prefixes if isinstance(item, str) and item.strip())
                if isinstance(prefixes, (list, tuple))
                else DEFAULT_DENY_PREFIXES
            ),
        )


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    # ``bool`` is an ``int`` in Python; ``maxTurns: true`` is a typo, not a request for 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if low <= value <= high else default


def _nonnegative_float(value: Any, default: float) -> float:
    # ``0`` is a VALID stored value here, not a malformed one (agent review round 1, R4):
    # the settings page accepts ``maxCostUsd: 0`` -- "never spend" -- and mapping it to the
    # default turned a block into a permit. Negative, bool and non-numeric stay
    # malformed -> default, and there is deliberately NO upper cutoff: the registry's 100.0
    # is the page's control ceiling, while a hand-edited higher cap is the user's own
    # written money guard -- substituting the default would silently change a limit they
    # stated. The invariant this keeps: every value the page accepts comes back as stored.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value) if value >= 0 else default
