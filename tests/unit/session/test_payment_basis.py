"""Per-call payment provenance survives dispatch, child folding and SQLite."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from local_operator.harness.jobs import _merge_accounting_component
from local_operator.harness.types import ModelSpec, StreamUsageEvent, Usage
from local_operator.model.costs import cost_summary_by_basis
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.failover import _stamp_serving_spec, stream_with_failover
from local_operator.session.channel_spend import ChildrenSnapshot, combine
from local_operator.session.frontend_state import _aggregate_usage, _usage_billing_basis
from local_operator.session.spend import inference_billing_labels


@pytest.mark.parametrize("provider", ["anthropic", "kimi", "openai"])
@pytest.mark.parametrize("kind", ["oauth", "api_key", None])
def test_credential_payment_relationship(provider, kind):
    basis = (
        "subscription_api_equivalent"
        if kind == "oauth" or (kind is None and provider == "openai")
        else "billed"
    )
    assert inference_billing_labels(provider, None, credential_kind=kind) == (basis, "catalogue")
    assert inference_billing_labels(provider, 0.79, credential_kind=kind) == (
        basis,
        "catalogue" if basis != "billed" else "provider_reported",
    )


def test_plan_key_is_not_a_cash_receipt():
    assert inference_billing_labels("zai", 1, credential_kind="api_key") == (
        "subscription_api_equivalent",
        "catalogue",
    )


@pytest.mark.asyncio
async def test_two_dispatches_same_provider_different_credentials():
    from tests.unit.providers.test_failover import FakeAuth, ScriptedClient, _request

    class Auth(FakeAuth):
        kind = "oauth"

        async def get_oauth_access(self, *args, **kwargs):
            return OAuthAccess("synthetic", 1, kind=self.kind)

    auth = Auth({"anthropic": ["synthetic"]})
    from local_operator.harness.types import StreamEndEvent

    client = ScriptedClient(
        [StreamUsageEvent(usage=Usage(usd_cost=0.79)), StreamEndEvent(stop_reason="stop")]
    )

    async def client_for(spec):
        return client

    usages = []
    for kind in ("oauth", "api_key"):
        auth.kind = kind
        events = [
            e
            async for e in stream_with_failover(
                _request("anthropic", "claude-test"),
                auth,
                {"retry": {"enabled": False}},
                client_for,
            )
        ]
        usages.extend(e.usage for e in events if isinstance(e, StreamUsageEvent))
    assert [u.billing_basis for u in usages] == ["subscription_api_equivalent", "billed"]
    aggregate = _aggregate_usage(usages)
    assert aggregate.billing_basis is None
    assert [c.billing_basis for c in aggregate.cost_components] == [
        "subscription_api_equivalent",
        "billed",
    ]


def test_stamp_serialization_and_validation():
    spec = ModelSpec(provider="anthropic", model_id="claude-test")
    event = StreamUsageEvent(usage=Usage(usd_cost=1))
    stamped = _stamp_serving_spec(event, spec, access=SimpleNamespace(kind="oauth"))
    assert isinstance(stamped, StreamUsageEvent)
    assert (
        Usage.model_validate(stamped.usage.model_dump()).billing_basis
        == "subscription_api_equivalent"
    )
    assert _usage_billing_basis(stamped.usage) == "subscription_api_equivalent"
    assert _usage_billing_basis(Usage(provider="anthropic", billing_basis="foreign")) == "billed"
    assert "billing_basis" not in Usage().model_dump()


def test_mixed_child_components_reconcile():
    grouped: dict[tuple[str | None, str | None, str, str | None], Usage] = {}
    for basis, cost in [("subscription_api_equivalent", 11.62), ("billed", 0.79)]:
        _merge_accounting_component(
            grouped,
            Usage(provider="anthropic", model_id="claude-test", billing_basis=basis, usd_cost=cost),
        )
    assert len(grouped) == 2
    total, unknown, bases = cost_summary_by_basis(grouped.values())
    assert total == pytest.approx(12.41) and not unknown
    assert bases["subscription_api_equivalent"] == 11620000
    snapshot = combine(
        [], children=ChildrenSnapshot(total_micro=12410000, by_basis=bases), tracked=True
    )
    assert snapshot["children"]["by_basis"]["subscription_api_equivalent"] == 11620000
    assert sum(snapshot["children"]["by_basis"].values()) == 12410000
    assert (
        sum(v for k, v in snapshot["by_basis"].items() if k != "not_tracked_calls")
        == snapshot["total_micro"]
    )
    assert snapshot["breakdown"]["by_agent"][0]["bases"] == [
        "billed",
        "subscription_api_equivalent",
    ]


def test_report_splits_one_model_by_payment_basis(tmp_path):
    from local_operator.analytics.store import AnalyticsStore
    from tests.unit.analytics.test_store import _snap

    store = AnalyticsStore(tmp_path / "ledger.db")
    store.record_batch(
        [
            replace(_snap(), billing_basis=basis, request_id=basis)
            for basis in ("billed", "subscription_api_equivalent")
        ]
    )
    report = store.session_report("s1")
    assert report.by_model["anthropic", "claude"].calls == 2
    assert {key[2] for key in report.by_model_basis} == {"billed", "subscription_api_equivalent"}
    assert all(value.calls == 1 for value in report.by_model_basis.values())
    child = replace(
        _snap(session_id="child"),
        billing_basis="subscription_api_equivalent",
        parent_session_id="s1",
        request_id="child-call",
        cost_micro=11620000,
    )
    store.record_batch([child])
    report = store.session_report("s1")
    assert report.descendants_aggregate is not None
    assert report.descendants_aggregate.cost_micro == 11620000
    assert report.descendants_by_basis["subscription_api_equivalent"].cost_micro == 11620000
    store.close()


def test_env_oauth_overrides_legacy_key_wrapper(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "synthetic-env-oauth")
    spec = ModelSpec(provider="anthropic", model_id="claude-test")
    stamped = _stamp_serving_spec(
        StreamUsageEvent(usage=Usage(usd_cost=1)),
        spec,
        access=OAuthAccess("synthetic-env-oauth", 0, kind="api_key"),
        key="synthetic-env-oauth",
    )
    assert isinstance(stamped, StreamUsageEvent)
    assert stamped.usage.billing_basis == "subscription_api_equivalent"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-env-oauth")
    ambiguous = _stamp_serving_spec(
        StreamUsageEvent(usage=Usage(usd_cost=1)),
        spec,
        access=OAuthAccess("synthetic-env-oauth", 0, kind="api_key"),
        key="synthetic-env-oauth",
    )
    assert isinstance(ambiguous, StreamUsageEvent)
    assert ambiguous.usage.billing_basis == "billed"


def test_child_rounding_and_unknown_basis_reconcile():
    total, unknown, bases = cost_summary_by_basis(
        [
            Usage(usd_cost=0.0000006, billing_basis="billed"),
            Usage(usd_cost=0.0000006, billing_basis="subscription_api_equivalent"),
            Usage(usd_cost=0.01, billing_basis="future-basis"),
        ],
        recorded_only=True,
    )
    assert not unknown
    assert all(micro >= 0 for micro in bases.values())
    assert total is not None
    assert sum(bases.values()) == round(total * 1_000_000)
    assert bases["not_tracked"] == 9999


def test_plan_key_provider_normalizes():
    assert inference_billing_labels(" ZAI ", 1, credential_kind="api_key") == (
        "subscription_api_equivalent",
        "catalogue",
    )


def test_turn_and_frozen_usage_keep_agreed_basis():
    from local_operator.session.frontend_state import _freeze_value

    usage = Usage(usd_cost=1, billing_basis="subscription_api_equivalent")
    aggregate = _aggregate_usage([usage, usage])
    assert aggregate.billing_basis == "subscription_api_equivalent"
    frozen = _freeze_value(aggregate)
    assert frozen.model_dump()["billing_basis"] == "subscription_api_equivalent"


def test_retained_job_receipt_fold_does_not_mix_payment_pools():
    from local_operator.session.frontend_state import _folded_components

    usages = [
        Usage(provider="anthropic", model_id="same", usd_cost=cost, billing_basis=basis)
        for cost, basis in [(11.62, "subscription_api_equivalent"), (0.79, "billed")]
    ]
    folded = _folded_components(usages)
    assert len(folded) == 2
    assert {u.billing_basis: u.usd_cost for u in folded} == {
        "subscription_api_equivalent": 11.62,
        "billed": 0.79,
    }
