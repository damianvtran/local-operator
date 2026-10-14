"""OperationBudget invariants: monotonic clock, reserve, stage caps, modes."""

import pytest

from local_operator.hub_sync.budget import (
    AUTOMATIC_ATTEMPT_SECONDS,
    MANUAL_ATTEMPT_SECONDS,
    OperationBudget,
    OperationDeadlineExceeded,
)


def test_start_sets_total_by_mode() -> None:
    clock = {"t": 1000.0}
    auto = OperationBudget.start("automatic", "op1", 1, clock=lambda: clock["t"])
    assert auto.deadline_monotonic - auto.started_monotonic == AUTOMATIC_ATTEMPT_SECONDS
    manual = OperationBudget.start("manual", "op2", 1, clock=lambda: clock["t"])
    assert manual.deadline_monotonic - manual.started_monotonic == MANUAL_ATTEMPT_SECONDS


def test_reserve_stops_a_final_stage_at_zero() -> None:
    clock = {"t": 1000.0}
    budget = OperationBudget.start("automatic", "op", 1, clock=lambda: clock["t"])
    clock["t"] = 1000.0 + AUTOMATIC_ATTEMPT_SECONDS - 1.0  # 1s left < 5s reserve
    with pytest.raises(OperationDeadlineExceeded):
        budget.require("commit")


def test_stage_caps_constrain_but_never_replenish() -> None:
    clock = {"t": 1000.0}
    budget = OperationBudget.start("manual", "op", 1, clock=lambda: clock["t"])
    # preview stage cap is 60s and cannot exceed the deadline minus reserve
    assert budget.request_deadline("preview") <= budget.deadline_monotonic - 5.0 + 1e-9
    assert budget.request_deadline("preview") <= clock["t"] + 60.0 + 1e-9


def test_invalid_budget_refused() -> None:
    with pytest.raises(ValueError):
        OperationBudget(
            deadline_monotonic=1.0,
            started_monotonic=2.0,
            mode="automatic",
            operation_id="o",
            attempt_generation=1,
        )
    with pytest.raises(ValueError):
        OperationBudget(
            deadline_monotonic=2.0,
            started_monotonic=1.0,
            mode="nope",
            operation_id="o",
            attempt_generation=1,
        )
