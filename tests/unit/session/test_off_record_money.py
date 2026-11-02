"""Off-record publication must use the ledger's nullable money, like refresh.

These are store/real-accumulator regressions, not ordinary streaming or App
coverage. Runtime complete/asides feed this writer through serving.on_usage.
"""

from types import SimpleNamespace

import pytest

from local_operator.harness.types import Usage
from local_operator.model.registry import ModelInfo
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.spend import SessionSpend, has_reported_tokens

UNKNOWN_RECEIPTS = [
    {"input_tokens": 10},
    {"output_tokens": 10},
    {"cache_read_tokens": 10},
    {"cache_write_tokens": 10},
    {"reasoning_tokens": 10},
    {},  # A final unknown-tier stamp is receipt evidence even at all-zero counts.
]


def _host(monkeypatch, *, ledger):
    from local_operator.model import configure

    model = ModelInfo(
        id="gpt-6-astra", name="synthetic", description="synthetic", input_price=10, output_price=50
    )
    monkeypatch.setattr(configure, "resolve_model_info", lambda *_: model)
    monkeypatch.setattr(configure, "resolve_model_info_paint", lambda *_: (model, True))
    session = SimpleNamespace(effective_model=SimpleNamespace(provider="openai", model_id=model.id))
    if ledger:
        session.spend = SessionSpend()
        session.accrue_spend = session.spend.accrue
    store = FrontendStateStore(
        FrontendSessionState(session_id="off-record-test", epoch="synthetic")
    )
    return session, store


@pytest.mark.parametrize("counters", UNKNOWN_RECEIPTS)
@pytest.mark.parametrize("ledger", [False, True])
def test_off_record_unknown_publishes_none_before_and_after_refresh(monkeypatch, counters, ledger):
    session, store = _host(monkeypatch, ledger=ledger)
    receipt = Usage(openai_service_tier="unknown", **counters)
    assert has_reported_tokens(receipt)
    store.accrue_usage(session, receipt)
    assert store.state.cumulative_parent_cost is None
    assert store.state.cost_knowledge.value == "unknown"
    assert store._spend_of(session).calls == 1
    # Serialization cannot turn unavailable money back into numeric zero.
    restored = FrontendSessionState.model_validate_json(store.state.model_dump_json())
    assert restored.cumulative_parent_cost is None
    session.restored_usage = lambda: receipt
    store.refresh_restored_usage(session)
    assert store.state.cumulative_parent_cost is None
    assert store.state.cost_knowledge.value == "unknown"
    assert store._spend_of(session).calls == 1


@pytest.mark.parametrize("ledger", [False, True])
def test_off_record_mixed_unpriced_last_retains_known_floor(monkeypatch, ledger):
    session, store = _host(monkeypatch, ledger=ledger)
    for receipt in [
        Usage(input_tokens=10, openai_service_tier="unknown"),
        Usage(input_tokens=10, usd_cost=3.0),
        Usage(output_tokens=10, openai_service_tier="unknown"),
    ]:
        store.accrue_usage(session, receipt)
    assert store.state.cumulative_parent_cost == 3.0
    assert store.state.cost_knowledge.value == "partial"
    session.restored_usage = lambda: receipt
    store.refresh_restored_usage(session)
    assert store.state.cumulative_parent_cost == 3.0
    assert store.state.cost_knowledge.value == "partial"


def test_unused_known_zero_and_explicit_zero_cache_are_not_unknown(monkeypatch):
    session, store = _host(monkeypatch, ledger=True)
    assert store.state.last_usage is None
    assert store._spend_of(session).calls == 0
    assert not has_reported_tokens(Usage(cache_read_tokens=0))
    receipt = Usage(input_tokens=1, usd_cost=0.0)
    store.accrue_usage(session, receipt)
    assert store.state.cumulative_parent_cost == 0.0
    assert store.state.cost_knowledge.value == "exact"
    assert store._spend_of(session).calls == 1


@pytest.mark.parametrize(
    "micro,calls,priced,knowledge,expected",
    [
        (0, 1, 0, "unknown", "$—"),
        (0, 1, 1, "exact", None),
        (0, 0, 0, "unknown", None),
        (3_000_000, 2, 1, "partial", "≥$3.00"),
    ],
)
def test_record_only_root_controls_money_cell(micro, calls, priced, knowledge, expected):
    from types import MethodType

    from local_operator.session.channel_spend import InferenceSnapshot, combine
    from local_operator.session.frontend_state import FrontendSpendChannels
    from local_operator.tui.app import OperatorApp

    root = FrontendSpendChannels.model_validate(
        combine(
            [],
            inference=InferenceSnapshot(
                micro=micro, calls=calls, priced_calls=priced, knowledge=knowledge
            ),
        )
    )
    # No App construction/last-usage fiction: production fold and wire model
    # carry actual record-only activity, and the real consumer reads that root.
    host = SimpleNamespace(
        _spend_channels_state=root, _spend_total=lambda: root.total_micro / 1_000_000
    )
    host._spend_text = MethodType(OperatorApp._spend_text, host)
    assert OperatorApp._band_cost_cell(host, None, 0.0, False) == expected


@pytest.mark.parametrize("counters", UNKNOWN_RECEIPTS)
def test_legacy_unknown_receipt_activity_uses_shared_predicate(counters):
    from local_operator.tui.app import OperatorApp

    receipt = Usage(openai_service_tier="unknown", **counters)
    host = SimpleNamespace(_spend_channels_state=None)
    assert OperatorApp._band_cost_cell(host, None, 0.0, has_reported_tokens(receipt)) == "$—"
    assert OperatorApp._band_cost_cell(host, None, 0.0, has_reported_tokens(Usage())) is None
