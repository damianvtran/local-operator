"""Threshold math: the min() trigger rule, validation fallbacks, clamps,
strategy, recovery band."""

import logging

from local_operator.compaction import thresholds as thresholds_module
from local_operator.compaction.thresholds import (
    DEFAULT_THRESHOLD_PERCENT,
    DEFAULT_THRESHOLD_TOKENS,
    RECOVERY_BAND,
    CompactionSettings,
    cleared_headroom,
    compaction_context_tokens,
    effective_reserve_tokens,
    resolve_strategy,
    resolve_threshold_cap_tokens,
    resolve_threshold_percent,
    resolve_threshold_tokens,
    should_compact,
)


def test_defaults_match_contract():
    s = CompactionSettings()
    assert s.enabled is True
    assert s.strategy == "auto"
    assert s.reserve_tokens is None  # None = defaulted (provenance, not value)
    assert s.keep_recent_tokens == 20000
    assert s.threshold_percent == 0.80
    assert s.threshold_tokens == 600_000
    assert s.auto_continue is True
    assert s.mid_turn_enabled is True
    assert not hasattr(s, "max_threshold_tokens")  # superseded by threshold_tokens


def test_effective_reserve_floors_at_15_percent():
    s = CompactionSettings()
    # 15% of 200k = 30k > 16384 → floor applies.
    assert effective_reserve_tokens(200_000, s) == 30_000
    # 15% of 100k = 15k < 16384 → configured reserve wins.
    assert effective_reserve_tokens(100_000, s) == 16_384
    # Floor is int-floored, not rounded: 100001 * 0.15 = 15000.15 -> 15000.
    assert effective_reserve_tokens(100_001, s) == 16_384


def test_trigger_is_the_smaller_of_percent_and_absolute():
    """THE rule: ``min(percent x window, absolute)``.

    A 1M-context model resolves to 600k (the absolute ceiling: 80% would be
    800k, and re-sending 800k on every request is slow and expensive even
    though it fits). A 200k model resolves to 160k (the percentage: a 600k
    absolute trigger could never fire on a 200k window, and min() makes it
    inert rather than disabling compaction).
    """
    s = CompactionSettings()
    assert resolve_threshold_tokens(1_000_000, s) == 600_000
    assert resolve_threshold_tokens(200_000, s) == 160_000
    assert resolve_threshold_tokens(40_000, s) == 32_000
    assert resolve_threshold_tokens(1_050_000, s) == 600_000


def test_regression_1m_session_does_not_compact_at_235k():
    """A 1M-context session compacting at ~235k discarded three quarters of
    its usable context per pass.

    Observed on anthropic/claude-opus-5 (context_window 1_000_000): a
    ``compacting context… 234.8k -> 33.2k`` receipt, i.e. 23% of the window.
    The cause was a second absolute knob (a defensive ``max_threshold_tokens``
    ceiling of 250k) resolving the trigger independently of the percentage.
    With one resolver there is exactly one number, and it is 600k.
    """
    s = CompactionSettings()
    window = 1_000_000
    assert should_compact(234_800, window, s) is False
    assert should_compact(500_000, window, s) is False
    assert should_compact(600_000, window, s) is False  # at threshold: stable
    assert should_compact(600_001, window, s) is True


def test_small_window_absolute_knob_is_inert_not_disabling():
    """The 600k absolute default is larger than a 200k window entirely; min()
    must leave the percentage governing, never resolve to "never compact"."""
    s = CompactionSettings()
    for window in (200_000, 128_000, 32_000, 8_000):
        threshold = resolve_threshold_tokens(window, s)
        assert threshold == int(window * 0.80)
        assert threshold < window
        assert should_compact(threshold + 1, window, s) is True


def test_both_knobs_are_settable_and_either_can_win():
    # Percentage lowered: it now governs a 1M window.
    assert resolve_threshold_tokens(1_000_000, CompactionSettings(threshold_percent=0.5)) == 500_000
    # Percent spelling: 50 and 0.5 mean the same thing.
    assert resolve_threshold_tokens(1_000_000, CompactionSettings(threshold_percent=50)) == 500_000
    # Absolute lowered: it governs a small window too (forcing early passes).
    assert resolve_threshold_tokens(200_000, CompactionSettings(threshold_tokens=3_000)) == 3_000
    # Raising the absolute knob cannot push past the percentage.
    assert (
        resolve_threshold_tokens(200_000, CompactionSettings(threshold_tokens=10_000_000))
        == 160_000
    )


def test_legacy_max_threshold_tokens_is_read_as_threshold_tokens(caplog):
    """The superseded ceiling key keeps working, with a rename warning: the
    user config that caused the ~235k passes sets ``max_threshold_tokens:
    250000``, and silently dropping an explicit ceiling would let a session
    sail past a proxy's real serving limit."""
    with caplog.at_level(logging.WARNING, logger=thresholds_module.__name__):
        s = CompactionSettings.model_validate({"max_threshold_tokens": 250_000})
    assert s.threshold_tokens == 250_000
    assert resolve_threshold_tokens(1_000_000, s) == 250_000
    assert "superseded by" in caplog.text
    # An explicit threshold_tokens wins and the legacy key is dropped.
    both = CompactionSettings.model_validate(
        {"max_threshold_tokens": 250_000, "threshold_tokens": 600_000}
    )
    assert both.threshold_tokens == 600_000
    # The caller's config dict is never mutated.
    raw = {"max_threshold_tokens": 250_000}
    CompactionSettings.model_validate(raw)
    assert raw == {"max_threshold_tokens": 250_000}


def test_explicit_reserve_can_only_pull_the_trigger_earlier():
    """An explicit reserve caps the trigger at ``window - reserve``; it never
    pushes a pass later than the percentage."""
    # Reserve above the 20% the percentage already leaves → it governs.
    assert resolve_threshold_tokens(200_000, CompactionSettings(reserve_tokens=50_000)) == 150_000
    # Reserve inside that headroom → inert (the 15% floor applies first).
    assert resolve_threshold_tokens(200_000, CompactionSettings(reserve_tokens=16_384)) == 160_000
    # A reserve at or above the window is impossible to honour: ignored, never
    # "never compact".
    assert resolve_threshold_tokens(2_000, CompactionSettings(reserve_tokens=5_000)) == 1_600
    # Defaulted reserve (None) never constrains the trigger.
    assert resolve_threshold_tokens(40_000, CompactionSettings()) == 32_000


def test_resolve_threshold_clamps_to_window():
    assert resolve_threshold_tokens(0, CompactionSettings()) == 0
    assert resolve_threshold_tokens(-5, CompactionSettings()) == 0
    # Floored at 1: tiny window never yields 0 (RC-23).
    assert resolve_threshold_tokens(1, CompactionSettings()) == 1
    assert resolve_threshold_tokens(50, CompactionSettings(threshold_percent=0.01)) == 1
    # Never at or above the window itself: 100% of the window clamps to w-1 so
    # a trigger always has somewhere above it to fire from.
    assert resolve_threshold_tokens(10_000, CompactionSettings(threshold_percent=1.0)) == 9_999
    # An absolute knob at or above the percentage is simply dominated by it.
    assert resolve_threshold_tokens(10_000, CompactionSettings(threshold_tokens=10_000)) == 8_000
    assert resolve_threshold_tokens(400, CompactionSettings(threshold_tokens=500)) == 320


def test_invalid_percent_falls_back_to_default_with_warning(caplog):
    thresholds_module._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=thresholds_module.__name__):
        assert resolve_threshold_percent(CompactionSettings(threshold_percent=0)) == (
            DEFAULT_THRESHOLD_PERCENT
        )
        assert resolve_threshold_percent(CompactionSettings(threshold_percent=-0.5)) == (
            DEFAULT_THRESHOLD_PERCENT
        )
        assert resolve_threshold_percent(CompactionSettings(threshold_percent=250.0)) == (
            DEFAULT_THRESHOLD_PERCENT
        )
    assert caplog.text.count("threshold_percent") == 3  # one warning per bad value
    # The resolved trigger degrades to the documented default, not to 0 or to
    # "never compact".
    assert resolve_threshold_tokens(1_000_000, CompactionSettings(threshold_percent=0)) == 600_000
    assert resolve_threshold_tokens(200_000, CompactionSettings(threshold_percent=-1.0)) == 160_000
    # Valid range, both spellings, no warning.
    assert resolve_threshold_percent(CompactionSettings(threshold_percent=0.9)) == 0.9
    assert resolve_threshold_percent(CompactionSettings(threshold_percent=90)) == 0.9
    assert resolve_threshold_percent(CompactionSettings(threshold_percent=1.0)) == 1.0
    assert resolve_threshold_percent(CompactionSettings(threshold_percent=100)) == 1.0


def test_invalid_absolute_tokens_falls_back_to_default_with_warning(caplog):
    thresholds_module._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=thresholds_module.__name__):
        assert resolve_threshold_cap_tokens(CompactionSettings(threshold_tokens=0)) == (
            DEFAULT_THRESHOLD_TOKENS
        )
        assert resolve_threshold_cap_tokens(CompactionSettings(threshold_tokens=-1)) == (
            DEFAULT_THRESHOLD_TOKENS
        )
    assert "threshold_tokens" in caplog.text
    # -1 used to mean "unset"; it must not disable compaction now.
    assert resolve_threshold_tokens(1_000_000, CompactionSettings(threshold_tokens=-1)) == 600_000
    assert should_compact(600_001, 1_000_000, CompactionSettings(threshold_tokens=-1)) is True


def test_warnings_are_deduplicated_per_value(caplog):
    """The trigger is resolved at every tool-loop boundary — a misconfigured
    session must not log a warning per turn."""
    thresholds_module._WARNED.clear()
    bad = CompactionSettings(threshold_percent=0)
    with caplog.at_level(logging.WARNING, logger=thresholds_module.__name__):
        for _ in range(5):
            resolve_threshold_percent(bad)
    assert caplog.text.count("out of range") == 1


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
