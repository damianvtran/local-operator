"""Muse (Meta) pricing: every Muse model resolves Meta's published rates.

The defect: the shipped ``meta_muse_models`` registry rows carried no
prices at all, models.dev's ``meta`` key was not mapped for the
``meta-muse`` provider, and the OpenRouter secondary had no ``meta``
namespace — so every Muse call resolved 0/0, ``spend.price_call``
reported it unknown, and the status band rendered ``$—`` for a model the
session was actively using (observed: "Wonder 1.0  Medium  2.7%/1.0M  $—").

Prices pinned here are Meta's published pay-as-you-go Standard table
(https://dev.meta.ai/docs/pricing-rate-limits, read 2026-10-12): Wonder
$4.00/$20.00 with $0.20 cached input, Spark $1.25/$4.25 with $0.15 cached
input, Spark Contributor $0.10/$0.20 with $0.002 cached input. Muse Code
through this provider is a flat subscription, so the figures are the
API-equivalent value `session/spend.py` already labels for subscription
flavours — never a per-token debit claim.

Cache semantics (https://dev.meta.ai/docs/prompt-caching): caching is
automatic, there is NO cache-write rate (an uncached prefix is ordinary
input), and ``cached_tokens`` is a SUBSET of ``input_tokens``. So a
million input tokens with half cached must cost
``500k x input + 500k x cache_read`` — the cached half charged once, at
the read rate, never at both rates.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.model import prices
from local_operator.model.configure import cost_for_usage, resolve_model_info
from local_operator.model.discovery import DiscoveredModel
from local_operator.model.registry import (
    get_model_info,
    meta_muse_family_model_info,
    meta_muse_models,
)

# (model id, input, output, cache read) per Meta's published Standard table.
STANDARD = {
    "muse-wonder-1.0": (4.0, 20.0, 0.20),
    "muse-spark-1.3": (1.25, 4.25, 0.15),
    "muse-spark-1.2": (1.25, 4.25, 0.15),
    "muse-spark-1.1": (1.25, 4.25, 0.15),
}

# Contributor variants are NOT shipped registry rows: they reach a price only
# through the catalogue chain, so a namespace/catalogue-key regression would
# silently return them to 0/0 unless the chain itself is pinned here.
CONTRIBUTOR = {
    "muse-spark-1.3-contributor": (0.10, 0.20, 0.002),
    "muse-spark-1.2-contributor": (0.10, 0.20, 0.002),
}


def _usage(input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        usd_cost=None,
        at_ms=None,
    )


@pytest.mark.parametrize("model_id", sorted(STANDARD))
def test_registry_carries_published_standard_prices(model_id: str) -> None:
    info = get_model_info("meta-muse", model_id)
    input_price, output_price, cache_read = STANDARD[model_id]
    assert info.input_price == input_price
    assert info.output_price == output_price
    assert info.cache_reads_price == cache_read
    # No Meta cache-write premium exists: a cache-writing (uncached) token
    # bears the ordinary input rate.
    assert info.cache_writes_price == input_price
    assert info.supports_prompt_cache is True


@pytest.mark.parametrize("model_id", sorted(STANDARD))
def test_resolve_model_info_prices_every_shipped_muse(model_id: str) -> None:
    """The `$—` regression: resolution must not come back 0/0."""
    info = resolve_model_info("meta-muse", model_id)
    input_price, output_price, cache_read = STANDARD[model_id]
    assert (info.input_price, info.output_price, info.cache_reads_price) == (
        input_price,
        output_price,
        cache_read,
    )


@pytest.mark.parametrize(
    "model_id,expected",
    [
        # 1M input (500k cached) + 100k output, computed by hand from the
        # published table: uncached x input + cached x read + out x output.
        ("muse-wonder-1.0", 500_000 * 4.0 / 1e6 + 500_000 * 0.20 / 1e6 + 100_000 * 20.0 / 1e6),
        ("muse-spark-1.3", 500_000 * 1.25 / 1e6 + 500_000 * 0.15 / 1e6 + 100_000 * 4.25 / 1e6),
        ("muse-spark-1.1", 500_000 * 1.25 / 1e6 + 500_000 * 0.15 / 1e6 + 100_000 * 4.25 / 1e6),
    ],
)
def test_cached_tokens_priced_once_at_read_rate(model_id: str, expected: float) -> None:
    info = resolve_model_info("meta-muse", model_id)
    usage = _usage(input_tokens=1_000_000, output_tokens=100_000, cache_read=500_000)
    assert cost_for_usage("meta-muse", info, usage) == pytest.approx(expected)


def test_cache_extremes_for_wonder() -> None:
    info = resolve_model_info("meta-muse", "muse-wonder-1.0")
    # Nothing cached: the whole million at the input rate.
    assert cost_for_usage("meta-muse", info, _usage(input_tokens=1_000_000)) == pytest.approx(4.0)
    # Everything cached: the whole million at the read rate, not $4 + $0.20.
    assert cost_for_usage(
        "meta-muse", info, _usage(input_tokens=1_000_000, cache_read=1_000_000)
    ) == pytest.approx(0.20)


def test_spend_price_call_reports_muse_known() -> None:
    """The status band's source: priced, known, and split by cache."""
    from local_operator.session.spend import price_call

    micro, known = price_call(
        "meta-muse",
        "muse-wonder-1.0",
        _usage(input_tokens=1_000_000, output_tokens=100_000, cache_read=500_000),
    )
    assert known is True
    assert micro == 4_100_000  # $4.10, the hand-computed figure above


def test_wonder_full_gets_limits_but_no_invented_price() -> None:
    """Meta publishes no per-token price for subscription-only Wonder Full.

    The family resolver may inherit its limits (observed on the
    authenticated listing) but must not borrow Wonder 1.0's rate onto a
    different model — unpriced (`$—`) is the honest rendering there.
    """
    info = meta_muse_family_model_info("muse-wonder-full-1.0")
    assert info is not None
    assert info.context_window == 1_007_997
    assert info.max_tokens == 128_000
    assert info.input_price == 0.0 and info.output_price == 0.0
    assert info.cache_reads_price is None


def test_family_resolver_drops_prices_for_unshipped_ids() -> None:
    """A future Spark point release inherits limits, never today's price."""
    info = meta_muse_family_model_info("muse-spark-1.4")
    assert info is not None
    assert info.context_window == meta_muse_models["muse-spark-1.3"].context_window
    assert info.input_price == 0.0 and info.cache_reads_price is None
    # Non-text Muse lines and non-Muse ids are not this family's business.
    assert meta_muse_family_model_info("muse-image-1.0") is None
    assert meta_muse_family_model_info("llama-4") is None


def test_family_resolver_requires_a_separator_after_the_tier_name() -> None:
    """Ids merely starting with the tier letters are a different family.

    A bare-prefix match would hand Wonder/Spark limits and capabilities to
    ``muse-wonderful-…``/``muse-sparkle-…``; only the ``muse-wonder-`` /
    ``muse-spark-`` namespace (point releases, Wonder Full, Contributor)
    may inherit.
    """
    assert meta_muse_family_model_info("muse-wonderful-1.0") is None
    assert meta_muse_family_model_info("muse-sparkle-1.0") is None
    assert meta_muse_family_model_info("muse-wonder") is None
    assert meta_muse_family_model_info("muse-spark") is None
    assert meta_muse_family_model_info("muse-wonder-full-1.0") is not None
    assert meta_muse_family_model_info("muse-spark-1.3-contributor") is not None


def test_family_resolver_normalises_a_namespaced_id() -> None:
    """The returned row is keyed by the stripped spelling, not the raw ask."""
    info = meta_muse_family_model_info("meta/muse-spark-1.9")
    assert info is not None
    assert (info.id, info.name) == ("muse-spark-1.9", "muse-spark-1.9")


@pytest.mark.parametrize("model_id", sorted(CONTRIBUTOR))
def test_price_chain_reads_contributor_prices_from_models_dev(model_id: str) -> None:
    """Contributor ids price through the models.dev `meta` leg at 0.10/0.20/0.002."""
    body = {
        "meta": {
            "id": "meta",
            "name": "Meta",
            "models": {
                model_id: {
                    "id": model_id,
                    "name": model_id,
                    "limit": {"context": 1048576, "output": 131072},
                    "cost": {"input": 0.10, "output": 0.20, "cache_read": 0.002},
                }
            },
        }
    }
    projected = prices.project(body, None)["providers"]
    row = prices.price_row("meta-muse", model_id, models_dev=projected, openrouter=[])
    assert row is not None
    assert (row.input_price, row.output_price, row.cache_read_price) == CONTRIBUTOR[model_id]


def test_price_chain_reads_models_dev_meta_key() -> None:
    """models.dev is the primary leg for an id the registry has not taught."""
    body = {
        "meta": {
            "id": "meta",
            "name": "Meta",
            "models": {
                "muse-spark-1.4": {
                    "id": "muse-spark-1.4",
                    "name": "Muse Spark 1.4",
                    "limit": {"context": 1048576, "output": 131072},
                    "cost": {"input": 1.25, "output": 4.25, "cache_read": 0.15},
                }
            },
        }
    }
    projected = prices.project(body, None)["providers"]
    row = prices.price_row("meta-muse", "muse-spark-1.4", models_dev=projected, openrouter=[])
    assert row is not None
    assert (row.input_price, row.output_price, row.cache_read_price) == (1.25, 4.25, 0.15)


def test_price_chain_openrouter_secondary_for_meta_muse() -> None:
    """A gap in models.dev (Wonder has no `cost` there) reaches OpenRouter."""
    openrouter_row = DiscoveredModel(
        id="meta/muse-spark-1.3",
        name="Muse Spark 1.3",
        input_price=1.25,
        output_price=4.25,
        cache_read_price=0.15,
        cache_write_price=1.25,
    )
    row = prices.price_row(
        "meta-muse", "muse-spark-1.3", models_dev={}, openrouter=[openrouter_row]
    )
    assert row is not None
    assert (row.input_price, row.output_price, row.cache_read_price) == (1.25, 4.25, 0.15)
