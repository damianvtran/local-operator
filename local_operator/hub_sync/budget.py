"""One monotonic attempt deadline shared by the existing hub engine's stages.

Socket inactivity limits cannot bound a trickling response. This value is passed
through fetch/resolver/preview/commit/settlement rather than reconstructed by
callees; stage ceilings only constrain the remaining original attempt budget.
UI observation and awaiting human confirmation are not running attempts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Literal

AUTOMATIC_ATTEMPT_SECONDS = 120.0
MANUAL_ATTEMPT_SECONDS = 300.0
SETTLEMENT_RESERVE_SECONDS = 5.0
FETCH_STAGE_SECONDS = 30.0
PREVIEW_STAGE_SECONDS = 60.0
CONNECT_INACTIVITY_SECONDS = 5.0
READ_INACTIVITY_SECONDS = 15.0


class OperationDeadlineExceeded(TimeoutError):
    code = "operation-deadline-exceeded"


@dataclass(frozen=True)
class OperationBudget:
    deadline_monotonic: float
    started_monotonic: float
    mode: Literal["automatic", "manual"]
    operation_id: str
    attempt_generation: int
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if (
            self.mode not in ("automatic", "manual")
            or not math.isfinite(self.started_monotonic)
            or not math.isfinite(self.deadline_monotonic)
            or self.deadline_monotonic < self.started_monotonic
            or isinstance(self.attempt_generation, bool)
            or not isinstance(self.attempt_generation, int)
            or self.attempt_generation < 1
        ):
            raise ValueError("invalid hub operation budget")

    @classmethod
    def start(
        cls,
        mode: Literal["automatic", "manual"],
        operation_id: str,
        attempt_generation: int,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> OperationBudget:
        now = clock()
        seconds = AUTOMATIC_ATTEMPT_SECONDS if mode == "automatic" else MANUAL_ATTEMPT_SECONDS
        return cls(now + seconds, now, mode, operation_id, attempt_generation, clock)

    def remaining(self) -> float:
        return max(0.0, self.deadline_monotonic - self.clock())

    def require(self, stage: str, *, reserve_seconds: float = SETTLEMENT_RESERVE_SECONDS) -> float:
        remaining = self.remaining() - reserve_seconds
        if remaining <= 0:
            # Stage is an internal enum/name, never a payload/source/pin or secret.
            raise OperationDeadlineExceeded(f"no attempt budget remains for {stage}")
        return remaining

    def request_deadline(self, stage: Literal["fetch", "preview", "commit", "replay"]) -> float:
        usable = self.require(stage)
        cap = (
            FETCH_STAGE_SECONDS
            if stage == "fetch"
            else (PREVIEW_STAGE_SECONDS if stage == "preview" else usable)
        )
        return min(
            self.deadline_monotonic - SETTLEMENT_RESERVE_SECONDS, self.clock() + min(cap, usable)
        )
