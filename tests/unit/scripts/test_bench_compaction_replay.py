"""The replay benchmark must measure the incident, not merely print numbers."""

from local_operator.compaction.cutpoint import find_cut_point
from scripts.bench_compaction_replay import _legacy_find_cut_point, _run


def test_mid_turn_fix_reduces_cumulative_replay_without_starving():
    """Both arms share one workload; only cut selection differs."""
    common = dict(
        calls=30,
        output_chars=4_000,
        threshold_tokens=8_000,
        keep_recent_tokens=2_000,
        summary_repetitions=4,
    )

    before = _run(_legacy_find_cut_point, **common)
    after = _run(find_cut_point, **common)

    assert before.starved_boundaries > 0
    assert before.compactions == 0
    assert after.starved_boundaries == 0
    assert after.compactions > 0
    assert after.cumulative_replay_tokens < before.cumulative_replay_tokens
    assert after.peak_context_tokens < before.peak_context_tokens
