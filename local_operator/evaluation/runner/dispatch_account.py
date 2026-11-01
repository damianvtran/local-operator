"""Per-episode provider dispatch authority for evaluation budgets.

The runner owns this state; a client borrows it only for a decision. Provider
usage events are snapshots, so later updates for the same dispatch replace the
previous observation instead of billing it twice.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_operator.evaluation.runner.model import CompactionRecord, ModelUsage


@dataclass(frozen=True)
class SummaryCharge:
    dispatch_id: str
    usage: ModelUsage
    cost_micros: int
    cost_known: bool
    provider_request_id: str


class DispatchBlocked(Exception):
    """A further provider request would cross an enforced budget boundary."""


class DispatchAccount:
    def __init__(self, *, max_cost_micros: int | None, max_cycle_cost_micros: int | None) -> None:
        self.max_cost_micros = max_cost_micros
        self.max_cycle_cost_micros = max_cycle_cost_micros
        self._charges: dict[str, tuple[int, bool, int]] = {}
        self._cycle = 0
        # A blocked decision needs the already-paid summary, not a fabricated
        # response. This remains scoped to the active decision call.
        self.completed_auxiliary: list[SummaryCharge] = []
        self.compaction: CompactionRecord | None = None

    def start_cycle(self) -> None:
        self._cycle += 1
        self.completed_auxiliary = []
        self.compaction = None

    def record_summary(self, charge: SummaryCharge) -> None:
        # A terminal receipt replaces an earlier provisional snapshot of the
        # same call; a retry must never count as a second provider call.
        self.completed_auxiliary = [
            c for c in self.completed_auxiliary if c.dispatch_id != charge.dispatch_id
        ]
        self.completed_auxiliary.append(charge)

    def record(self, dispatch_id: str, *, cost_micros: int, cost_known: bool) -> None:
        previous = self._charges.get(dispatch_id)
        cycle = previous[2] if previous is not None else self._cycle
        self._charges[dispatch_id] = (cost_micros, cost_known, cycle)

    @property
    def cost_micros(self) -> int:
        return sum(cost for cost, _, _ in self._charges.values())

    @property
    def cost_known(self) -> bool:
        return all(known for _, known, _ in self._charges.values())

    def before_dispatch(self) -> None:
        cycle = [entry for entry in self._charges.values() if entry[2] == self._cycle]
        if self.max_cost_micros is not None:
            if not self.cost_known:
                raise DispatchBlocked("cost-unreported")
            if self.cost_micros >= self.max_cost_micros:
                raise DispatchBlocked("budget-cap")
        if self.max_cycle_cost_micros is not None:
            if any(not known for _, known, _ in cycle):
                raise DispatchBlocked("cost-unreported")
            if sum(cost for cost, _, _ in cycle) >= self.max_cycle_cost_micros:
                raise DispatchBlocked("cost-cycle-cap")
