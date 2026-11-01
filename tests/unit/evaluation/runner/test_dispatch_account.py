"""Per-episode dispatch authority must not confuse partial totals with free calls."""

import pytest


def test_unknown_paid_summary_blocks_only_finite_money_cap() -> None:
    from local_operator.evaluation.runner.dispatch_account import (
        DispatchAccount,
        DispatchBlocked,
    )

    capped = DispatchAccount(max_cost_micros=1_000_000, max_cycle_cost_micros=None)
    capped.start_cycle()
    capped.before_dispatch()
    capped.record("summary-1", cost_micros=0, cost_known=False)
    with pytest.raises(DispatchBlocked, match="cost-unreported"):
        capped.before_dispatch()

    uncapped = DispatchAccount(max_cost_micros=None, max_cycle_cost_micros=None)
    uncapped.start_cycle()
    uncapped.record("summary-1", cost_micros=0, cost_known=False)
    uncapped.before_dispatch()
    assert uncapped.cost_known is False


def test_final_snapshot_replaces_provisional_and_is_idempotent() -> None:
    from local_operator.evaluation.runner.dispatch_account import DispatchAccount

    account = DispatchAccount(max_cost_micros=100, max_cycle_cost_micros=None)
    account.start_cycle()
    account.record("call-1", cost_micros=7, cost_known=False)
    account.record("call-1", cost_micros=11, cost_known=True)
    account.record("call-1", cost_micros=11, cost_known=True)
    assert account.cost_micros == 11
    assert account.cost_known is True
    account.before_dispatch()


def test_summary_snapshots_replace_by_dispatch_id() -> None:
    from local_operator.evaluation.runner.dispatch_account import (
        DispatchAccount,
        SummaryCharge,
    )
    from local_operator.evaluation.runner.model import ModelUsage

    account = DispatchAccount(max_cost_micros=None, max_cycle_cost_micros=None)
    account.start_cycle()
    account.record_summary(SummaryCharge("same", ModelUsage(input_tokens=1), 7, False, "unknown"))
    account.record_summary(SummaryCharge("same", ModelUsage(input_tokens=4), 9, True, "provider"))
    assert len(account.completed_auxiliary) == 1
    assert account.completed_auxiliary[0].usage.input_tokens == 4
    assert account.completed_auxiliary[0].cost_micros == 9


def test_caps_apply_before_next_dispatch_and_cycle_reset() -> None:
    from local_operator.evaluation.runner.dispatch_account import (
        DispatchAccount,
        DispatchBlocked,
    )

    account = DispatchAccount(max_cost_micros=20, max_cycle_cost_micros=10)
    account.start_cycle()
    account.record("call-1", cost_micros=10, cost_known=True)
    with pytest.raises(DispatchBlocked, match="cost-cycle-cap"):
        account.before_dispatch()
    account.start_cycle()
    account.before_dispatch()
    account.record("call-2", cost_micros=10, cost_known=True)
    with pytest.raises(DispatchBlocked, match="budget-cap"):
        account.before_dispatch()
