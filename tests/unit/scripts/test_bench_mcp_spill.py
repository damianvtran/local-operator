"""The MCP benchmark must prove prompt, transcript, and recovery savings."""

from scripts.bench_mcp_spill import (
    MIN_METADATA_BYTE_REDUCTION,
    MIN_MODEL_TOKEN_REDUCTION,
    run_benchmark,
)


def test_oversized_mcp_result_reduces_both_copies_and_remains_recoverable(tmp_path):
    result = run_benchmark(tmp_path / "cfg", rows=100, value_chars=64)

    assert result.model_token_reduction_percent >= MIN_MODEL_TOKEN_REDUCTION * 100
    assert result.metadata_reduction_percent >= MIN_METADATA_BYTE_REDUCTION * 100
    assert result.compact_model_tokens < result.baseline_model_tokens
    assert result.compact_metadata_bytes < result.baseline_metadata_bytes
    assert result.spill_complete is True
    assert result.recovery_matches is True
