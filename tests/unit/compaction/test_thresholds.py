"""Threshold math: reserve provenance, §C default, clamps, strategy, recovery band."""

from local_operator.compaction.thresholds import (
    RECOVERY_BAND,
    CompactionSettings,
    cleared_headroom,
    compaction_context_tokens,
    effective_reserve_tokens,
    resolve_budget_reserve_tokens,
    resolve_strategy,
    resolve_threshold_tokens,
    should_compact,
)


def test_defaults_match_contract():
    s = CompactionSettings()
    assert s.enabled is True
    assert s.strategy == "auto"
    assert s.reserve_tokens is None  # None = defaulted (provenance, not value)
    assert s.keep_recent_tokens == 20000
    assert s.threshold_percent == -1.0
    assert s.threshold_tokens == -1
    assert s.auto_continue is True
    assert s.mid_turn_enabled is True


def test_effective_reserve_floors_at_15_percent():
    s = CompactionSettings()
    # 15% of 200k = 30k > 16384 → floor applies.
    assert effective_reserve_tokens(200_000, s) == 30_000
    # 15% of 100k = 15k < 16384 → configured reserve wins.
    assert effective_reserve_tokens(100_000, s) == 16_384
    # Floor is int-floored, not rounded: 100001 * 0.15 = 15000.15 -> 15000.
    assert effective_reserve_tokens(100_001, s) == 16_384


def test_resolve_threshold_defaulted_reserve_section_c_default():
    """Defaulted reserve + feasible window: the docs/REWRITE.md §C default —
    the lesser of 80% of the window and 600k."""
    s = CompactionSettings()
    assert resolve_threshold_tokens(200_000, s) == 160_000
    assert resolve_threshold_tokens(1_000_000, s) == 600_000
    assert resolve_threshold_tokens(40_000, s) == 32_000


def test_resolve_threshold_defaulted_reserve_impossible_recovers_15_percent():
    """A defaulted reserve that is impossible
    for the window (>= window, or >= window - proportional) recovers to
    max(1, floor(w * 0.15))."""
    s = CompactionSettings()
    # w=2000: effective 16384 >= 2000 - 300 → recover: 2000 - max(1, 300) = 1700.
    assert resolve_threshold_tokens(2_000, s) == 1_700
    assert resolve_threshold_tokens(1, s) == 1
    assert resolve_budget_reserve_tokens(2_000, s) == 300
    assert resolve_budget_reserve_tokens(1, s) == 1


def test_resolve_threshold_explicit_reserve_always_honored():
    """An explicit reserve defines the usable budget and bypasses the §C
    default — including one that happens to equal the default value."""
    # Feasible window: w - effective (15% floor applies over 16384).
    s = CompactionSettings(reserve_tokens=16384)
    assert resolve_threshold_tokens(200_000, s) == 200_000 - 30_000
    # Reserve above the 15% floor wins outright.
    s_big = CompactionSettings(reserve_tokens=50_000)
    assert resolve_threshold_tokens(200_000, s_big) == 150_000
    # Explicit reserve >= window still falls back to proportional (the
    # threshold must stay strictly below the window).
    s_huge = CompactionSettings(reserve_tokens=5_000)
    assert resolve_threshold_tokens(2_000, s_huge) == 1_700


def test_resolve_budget_reserve_provenance_not_value():
    """Defaulted == None, never a comparison against 16384: an explicit
    reserve equal to the default stays explicit (feasible windows keep the
    §C default out of it)."""
    defaulted = CompactionSettings()
    explicit = CompactionSettings(reserve_tokens=16384)
    assert resolve_budget_reserve_tokens(100_000, defaulted) == 16_384
    assert resolve_budget_reserve_tokens(100_000, explicit) == 16_384
    # But the THRESHOLD diverges: defaulted takes the §C default.
    assert resolve_threshold_tokens(100_000, defaulted) == 80_000
    assert resolve_threshold_tokens(100_000, explicit) == 100_000 - 16_384


def test_resolve_threshold_explicit_tokens_wins_and_clamps():
    base = CompactionSettings(threshold_tokens=500, threshold_percent=50.0)
    assert resolve_threshold_tokens(10_000, base) == 500  # tokens > percent
    assert resolve_threshold_tokens(400, base) == 399  # clamped to window - 1
    assert resolve_threshold_tokens(10_000, CompactionSettings(threshold_tokens=10_000)) == 9_999


def test_resolve_threshold_percent_branch():
    s = CompactionSettings(threshold_percent=50.0)
    assert resolve_threshold_tokens(2_000, s) == 1_000
    # Percent clamps into [1, 99].
    assert resolve_threshold_tokens(2_000, CompactionSettings(threshold_percent=150.0)) == 1_980
    assert resolve_threshold_tokens(2_000, CompactionSettings(threshold_percent=0.1)) == 20
    # Floored at 1: tiny window + tiny percent never yields 0 (RC-23).
    assert resolve_threshold_tokens(1, CompactionSettings(threshold_percent=0.1)) == 1
    assert resolve_threshold_tokens(50, CompactionSettings(threshold_percent=1.0)) == 1


def test_should_compact_boundaries():
    s = CompactionSettings()
    window = 200_000
    threshold = resolve_threshold_tokens(window, s)
    assert should_compact(threshold, window, s) is False  # at threshold: stable
    assert should_compact(threshold + 1, window, s) is True
    assert should_compact(threshold - 1, window, s) is False
    assert should_compact(threshold + 1, window, CompactionSettings(enabled=False)) is False
    assert should_compact(threshold + 1, window, CompactionSettings(strategy="off")) is False
    assert should_compact(threshold + 1, 0, s) is False  # unknown window never fires


def test_compaction_context_tokens_provider_floored_by_local():
    assert compaction_context_tokens(None, 1000) == 1000
    assert compaction_context_tokens(500, 1000) == 1000  # local floor
    assert compaction_context_tokens(2000, 1000) == 2000  # provider ground truth
    assert compaction_context_tokens(0, 0) == 0


def test_recovery_band_and_headroom():
    assert RECOVERY_BAND == 0.8
    threshold = 100_000
    residual_cleared = int(threshold * RECOVERY_BAND)
    assert cleared_headroom(residual_cleared, threshold) == threshold - residual_cleared
    assert cleared_headroom(threshold, threshold) == 0
    assert cleared_headroom(threshold + 5, threshold) < 0


class _Model:
    """Duck-typed stand-in: resolve_strategy only reads ``supports_images``."""

    def __init__(self, supports_images: bool) -> None:
        self.supports_images = supports_images


def test_resolve_strategy_selection():
    auto = CompactionSettings()  # strategy='auto'
    assert resolve_strategy(auto, _Model(True)) == "snapcompact"
    assert resolve_strategy(auto, _Model(False)) == "context-full"
    assert (
        resolve_strategy(CompactionSettings(strategy="snapcompact"), _Model(False)) == "snapcompact"
    )
    context_full = CompactionSettings(strategy="context-full")
    assert resolve_strategy(context_full, _Model(True)) == "context-full"
    # 'off' is gated by should_compact; the resolver degrades to context-full.
    assert resolve_strategy(CompactionSettings(strategy="off"), _Model(True)) == "context-full"


def test_resolve_strategy_ignores_missing_attribute():
    """A model without supports_images is treated as non-vision."""
    assert resolve_strategy(CompactionSettings(), object()) == "context-full"
