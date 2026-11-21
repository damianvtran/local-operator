"""Other vendors must not inherit Claude's price or 1h write tariff."""

import pytest

from local_operator.harness.types import Usage
from local_operator.model import costs
from local_operator.model.configure import cost_for_usage, resolve_model_info
from local_operator.model.registry import ModelInfo, bedrock_models
from local_operator.providers.bedrock.converse import usage_from_converse

from .test_converse_body import NOVA


def test_unpriced_native_usage_stays_unknown_through_record_and_aggregation(monkeypatch):
    info = bedrock_models[NOVA]
    monkeypatch.setattr(costs, "_resolve_for_paint", lambda *args: info)
    usage = usage_from_converse(
        {
            "inputTokens": 100,
            "outputTokens": 20,
            "cacheReadInputTokens": 1000,
            "cacheWriteInputTokens": 200,
        }
    )
    assert costs.turn_cost("bedrock/" + NOVA, usage) is None
    stamped = costs.stamp_record_estimate(usage, "bedrock", NOVA)
    assert stamped.usd_cost is None and stamped.estimated_usd_cost is None
    aggregate = Usage(cost_components=[Usage(input_tokens=10, usd_cost=0.01), usage])
    assert costs.turn_cost("bedrock/" + NOVA, aggregate) is None


def test_disjoint_buckets_each_billed_once_without_claude_ttl_multiplier():
    info = ModelInfo(
        id=NOVA,
        name="synthetic known tariff",
        description="Test-only exact tariff, never published as a Bedrock price",
        input_price=2,
        output_price=8,
        cache_reads_price=0.5,
        cache_writes_price=3,
    )
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        cache_read_tokens=1000,
        cache_write_tokens=200,
        cache_write_5m_tokens=150,
        cache_write_1h_tokens=50,
    )
    expected = (100 * 2 + 20 * 8 + 1000 * 0.5 + 200 * 3) / 1_000_000
    assert cost_for_usage("bedrock", info, usage) == pytest.approx(expected)


def test_exact_global_claude_arn_matches_existing_claude_tariff():
    name = "global.anthropic.claude-sonnet-5-5"
    arn = "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/" + name
    row = resolve_model_info("bedrock", arn)
    direct = bedrock_models[name]
    assert row.input_price == direct.input_price
    assert row.output_price == direct.output_price
    assert row.cache_reads_price == direct.cache_reads_price
