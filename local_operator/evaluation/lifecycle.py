"""Pure cleanup contracts and the evaluation episode state machine.

The permit boundary constrains cooperative adapters: Python cannot stop hostile
code from bypassing a type contract, but normal allocation and execution APIs
can make possession of a factory-minted permit a mandatory argument.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import Field, ValidationInfo, field_validator, model_validator

from local_operator.evaluation.protocol import ArtifactRef, ProtocolModel
from local_operator.evaluation.receipts import (
    MAX_DECLARATIONS,
    ZERO_DIGEST,
    BudgetAuthorization,
    BudgetReconciliation,
    BudgetReservation,
    Digest,
    SafeCount,
    SealedPreflight,
    StrictIdentifier,
)

MAX_CLEANUP_ATTEMPTS = 32
MAX_CLEANUP_TIMEOUT_MS = 3_600_000
MAX_FAILURE_LENGTH = 2_000
_PERMIT_FACTORY = object()


def _identity(kind: str, payload: Any) -> str:
    encoded = json.dumps(
        {"identity_kind": kind, "payload": payload},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CleanupActionKind = Literal[
    "release_instance",
    "delete_volume",
    "delete_artifact",
    "revoke_lease",
    "close_session",
    "restore_snapshot",
]


class CleanupAction(ProtocolModel):
    """Bounded symbolic cleanup; commands and provider parameters stay in adapters."""

    action_id: StrictIdentifier
    kind: CleanupActionKind
    resource_ref: StrictIdentifier
    timeout_ms: SafeCount = Field(le=MAX_CLEANUP_TIMEOUT_MS)
    max_attempts: int = Field(ge=1, le=MAX_CLEANUP_ATTEMPTS)


class CleanupPlan(ProtocolModel):
    episode_id: StrictIdentifier
    actions: tuple[CleanupAction, ...] = Field(min_length=1, max_length=MAX_DECLARATIONS)
    cleanup_plan_id: Digest = ZERO_DIGEST

    @field_validator("actions", mode="before")
    @classmethod
    def _freeze_actions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _canonicalize_and_identify(self) -> Self:
        ordered = tuple(sorted(self.actions, key=lambda item: item.action_id))
        ids = [item.action_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("cleanup plan contains duplicate action IDs")
        targets = [(item.kind, item.resource_ref) for item in ordered]
        if len(targets) != len(set(targets)):
            raise ValueError("cleanup plan contains conflicting duplicate actions")
        object.__setattr__(self, "actions", ordered)
        payload = {
            "episode_id": self.episode_id,
            "actions": [item.model_dump(mode="json") for item in ordered],
        }
        expected = _identity("cleanup-plan-v1", payload)
        if self.cleanup_plan_id not in (ZERO_DIGEST, expected):
            raise ValueError("cleanup plan identity does not match its actions")
        object.__setattr__(self, "cleanup_plan_id", expected)
        return self


class CleanupReceipt(ProtocolModel):
    action_id: StrictIdentifier
    status: Literal["not_needed", "attempted", "succeeded", "failed"]
    evidence_code: StrictIdentifier
    duration_ms: SafeCount
    receipt_id: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _identify(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"receipt_id"})
        expected = _identity("cleanup-receipt-v1", payload)
        if self.receipt_id not in (ZERO_DIGEST, expected):
            raise ValueError("cleanup receipt identity does not match its result")
        object.__setattr__(self, "receipt_id", expected)
        return self


class CleanupResult(ProtocolModel):
    cleanup_plan_id: Digest
    succeeded_action_ids: tuple[StrictIdentifier, ...]
    not_needed_action_ids: tuple[StrictIdentifier, ...]
    incomplete_action_ids: tuple[StrictIdentifier, ...]
    receipt_digests: tuple[Digest, ...]
    rescue_required: bool
    cleanup_result_id: Digest

    @field_validator(
        "succeeded_action_ids",
        "not_needed_action_ids",
        "incomplete_action_ids",
        "receipt_digests",
        mode="before",
    )
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        groups = (
            self.succeeded_action_ids,
            self.not_needed_action_ids,
            self.incomplete_action_ids,
        )
        flattened = [item for group in groups for item in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("cleanup result action IDs must be unique")
        if any(tuple(sorted(group)) != group for group in groups):
            raise ValueError("cleanup result action IDs must be canonical")
        if tuple(sorted(self.receipt_digests)) != self.receipt_digests:
            raise ValueError("cleanup receipt digests must be canonical")
        if self.rescue_required != bool(self.incomplete_action_ids):
            raise ValueError("cleanup rescue flag disagrees with incomplete actions")
        expected = _identity(
            "cleanup-result-v1",
            self.model_dump(mode="json", exclude={"cleanup_result_id"}),
        )
        if self.cleanup_result_id != expected:
            raise ValueError("cleanup result identity does not match its receipts")
        return self


def aggregate_cleanup(
    plan: CleanupPlan,
    receipts: Sequence[CleanupReceipt],
) -> CleanupResult:
    """Require exact action coverage; attempted alone is never evidence of cleanup."""

    snapshot = tuple(receipts)
    by_id: dict[str, CleanupReceipt] = {}
    for receipt in snapshot:
        if receipt.action_id in by_id:
            raise ValueError("cleanup has duplicate receipts")
        by_id[receipt.action_id] = receipt
    expected_ids = {action.action_id for action in plan.actions}
    if set(by_id) != expected_ids:
        raise ValueError("cleanup requires exactly one receipt for every action")
    succeeded = tuple(sorted(key for key, item in by_id.items() if item.status == "succeeded"))
    not_needed = tuple(sorted(key for key, item in by_id.items() if item.status == "not_needed"))
    incomplete = tuple(
        sorted(key for key, item in by_id.items() if item.status in ("attempted", "failed"))
    )
    payload = {
        "cleanup_plan_id": plan.cleanup_plan_id,
        "succeeded_action_ids": succeeded,
        "not_needed_action_ids": not_needed,
        "incomplete_action_ids": incomplete,
        "receipt_digests": tuple(sorted(item.receipt_id for item in snapshot)),
        "rescue_required": bool(incomplete),
    }
    return CleanupResult(
        **payload,
        cleanup_result_id=_identity("cleanup-result-v1", payload),
    )


class SideEffectPermit(ProtocolModel):
    """Unforgeable-by-validation authority bound to one sealed episode budget."""

    episode_id: StrictIdentifier
    plan_id: Digest
    preflight_seal_id: Digest
    budget_id: Digest
    permit_id: Digest

    @model_validator(mode="after")
    def _factory_only(self, info: ValidationInfo) -> Self:
        context = info.context if isinstance(info.context, dict) else {}
        if context.get("permit_factory") is not _PERMIT_FACTORY:
            raise ValueError("side-effect permits can only be minted from validated authorities")
        expected = _identity(
            "side-effect-permit-v1",
            self.model_dump(mode="json", exclude={"permit_id"}),
        )
        if self.permit_id != expected:
            raise ValueError("side-effect permit identity does not match its authorities")
        return self


def mint_side_effect_permit(
    *,
    episode_id: StrictIdentifier,
    plan_id: Digest,
    preflight: SealedPreflight,
    budget: BudgetAuthorization,
) -> SideEffectPermit:
    """Mint authority only after successful preflight and explicit budget authorization."""

    if not preflight.successful:
        raise ValueError("failed preflight cannot authorize side effects")
    if preflight.plan_id != plan_id:
        raise ValueError("preflight seal belongs to another dependency plan")
    if budget.episode_id != episode_id:
        raise ValueError("budget authorization belongs to another episode")
    payload = {
        "episode_id": episode_id,
        "plan_id": plan_id,
        "preflight_seal_id": preflight.seal_id,
        "budget_id": budget.budget_id,
    }
    return SideEffectPermit.model_validate(
        {**payload, "permit_id": _identity("side-effect-permit-v1", payload)},
        context={"permit_factory": _PERMIT_FACTORY},
    )


class ScoreReceipt(ProtocolModel):
    """Content-addressed final score evidence bound to one episode and plan."""

    episode_id: StrictIdentifier
    plan_id: Digest
    score_artifact: ArtifactRef
    finalized_at_ms: SafeCount
    score_id: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _identify(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"score_id"})
        expected = _identity("score-receipt-v1", payload)
        if self.score_id not in (ZERO_DIGEST, expected):
            raise ValueError("score receipt identity does not match its artifact")
        object.__setattr__(self, "score_id", expected)
        return self


EpisodeState = Literal[
    "planned",
    "preflighted",
    "authorized",
    "running",
    "finalizing",
    "cleaning",
    "completed",
    "failed",
    "cancelled",
]
TerminalIntent = Literal["complete", "fail", "cancel"]


class EpisodeLifecycle(ProtocolModel):
    """Immutable episode state whose methods enforce the only legal transition graph."""

    episode_id: StrictIdentifier
    plan_id: Digest
    budget_id: Digest
    cleanup_plan_id: Digest
    state: EpisodeState = "planned"
    preflight_seal_id: Digest | None = None
    permit_id: Digest | None = None
    reservation_ids: tuple[Digest, ...] = ()
    reconciliation_id: Digest | None = None
    reconciliation_reportable: bool | None = None
    score_id: Digest | None = None
    cleanup_result_id: Digest | None = None
    rescue_required: bool | None = None
    terminal_intent: TerminalIntent | None = None
    failure_kind: (
        Literal[
            "preflight",
            "infrastructure",
            "crash",
            "ambiguous_finalization",
            "cleanup",
            "unreportable",
        ]
        | None
    ) = None
    failure_reason: str | None = Field(default=None, min_length=1, max_length=MAX_FAILURE_LENGTH)

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def _freeze_reservations(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_state_evidence(self) -> Self:
        if len(self.reservation_ids) != len(set(self.reservation_ids)):
            raise ValueError("episode contains duplicate reservations")
        if tuple(sorted(self.reservation_ids)) != self.reservation_ids:
            raise ValueError("episode reservations must be canonical")
        before_running = self.state in ("planned", "preflighted", "authorized")
        if before_running and any(
            value is not None
            for value in (
                self.reconciliation_id,
                self.score_id,
                self.cleanup_result_id,
                self.terminal_intent,
            )
        ):
            raise ValueError("pre-run state cannot carry finalization or cleanup evidence")
        if self.state == "planned" and any(
            value is not None for value in (self.preflight_seal_id, self.permit_id)
        ):
            raise ValueError("planned state cannot carry preflight authority")
        if self.state == "preflighted" and (
            self.preflight_seal_id is None or self.permit_id is not None
        ):
            raise ValueError("preflighted state requires only a preflight seal")
        if self.state in ("authorized", "running", "finalizing", "cleaning", "completed") and (
            self.preflight_seal_id is None or self.permit_id is None
        ):
            raise ValueError("post-authorization state requires preflight and permit identities")
        if self.state == "running" and not self.reservation_ids:
            raise ValueError("running state requires a prior budget reservation")
        if self.state == "finalizing" and self.terminal_intent is not None:
            raise ValueError("finalizing state cannot already carry a terminal intent")
        if self.state == "cleaning" and self.terminal_intent is None:
            raise ValueError("cleaning state requires a terminal intent")
        if self.state in ("completed", "cancelled") and self.cleanup_result_id is None:
            raise ValueError("post-run terminal state requires cleanup evidence")
        if self.state == "completed":
            if any(
                value is None
                for value in (
                    self.reconciliation_id,
                    self.score_id,
                    self.cleanup_result_id,
                )
            ):
                raise ValueError("completed state requires cost, score, and cleanup evidence")
            if self.reconciliation_reportable is not True or self.rescue_required is not False:
                raise ValueError("completed state must be reportable and fully cleaned")
            if self.failure_kind is not None or self.terminal_intent != "complete":
                raise ValueError("completed state cannot carry failure evidence")
        if self.state in ("failed", "cancelled") and self.failure_reason is None:
            raise ValueError("failed or cancelled state requires a reason")
        if self.state == "failed" and self.reservation_ids and self.cleanup_result_id is None:
            raise ValueError("post-start failure requires cleanup evidence")
        if self.failure_kind in ("preflight", "infrastructure") and self.score_id is not None:
            raise ValueError("preflight and infrastructure failures cannot carry a score")
        if self.state == "cancelled" and self.terminal_intent != "cancel":
            raise ValueError("cancelled state requires cancellation intent")
        return self

    @classmethod
    def planned(
        cls,
        *,
        episode_id: StrictIdentifier,
        plan_id: Digest,
        budget_id: Digest,
        cleanup_plan_id: Digest,
    ) -> Self:
        return cls(
            episode_id=episode_id,
            plan_id=plan_id,
            budget_id=budget_id,
            cleanup_plan_id=cleanup_plan_id,
        )

    def _transition(self, expected: EpisodeState, **updates: Any) -> Self:
        if self.state != expected:
            raise ValueError(f"illegal episode transition from {self.state}")
        payload = self.model_dump(mode="python")
        payload.update(updates)
        return type(self).model_validate(payload)

    def preflight(self, seal: SealedPreflight) -> Self:
        if not seal.successful:
            raise ValueError("failed preflight cannot enter preflighted state")
        if seal.plan_id != self.plan_id:
            raise ValueError("preflight seal belongs to another dependency plan")
        return self._transition("planned", state="preflighted", preflight_seal_id=seal.seal_id)

    def authorize(
        self, seal: SealedPreflight, budget: BudgetAuthorization
    ) -> tuple[Self, SideEffectPermit]:
        if self.state != "preflighted" or self.preflight_seal_id != seal.seal_id:
            raise ValueError("illegal or mismatched episode authorization")
        if budget.budget_id != self.budget_id:
            raise ValueError("budget authorization does not match the planned budget")
        permit = mint_side_effect_permit(
            episode_id=self.episode_id,
            plan_id=self.plan_id,
            preflight=seal,
            budget=budget,
        )
        return (
            self._transition("preflighted", state="authorized", permit_id=permit.permit_id),
            permit,
        )

    def start(
        self,
        permit: SideEffectPermit,
        reservations: Sequence[BudgetReservation],
    ) -> Self:
        if self.state != "authorized":
            raise ValueError(f"illegal episode transition from {self.state}")
        snapshot = tuple(reservations)
        if not snapshot:
            raise ValueError("episode start requires a prior budget reservation")
        if (
            permit.episode_id != self.episode_id
            or permit.plan_id != self.plan_id
            or permit.budget_id != self.budget_id
            or permit.permit_id != self.permit_id
        ):
            raise ValueError("side-effect permit does not match this episode")
        if any(
            item.episode_id != self.episode_id or item.budget_id != self.budget_id
            for item in snapshot
        ):
            raise ValueError("budget reservation does not match this episode")
        ids = tuple(sorted(item.reservation_id for item in snapshot))
        if len(ids) != len(set(ids)):
            raise ValueError("episode start contains duplicate reservations")
        return self._transition("authorized", state="running", reservation_ids=ids)

    def begin_finalization(self) -> Self:
        return self._transition("running", state="finalizing")

    def finish_finalization(
        self,
        reconciliation: BudgetReconciliation,
        score: ScoreReceipt,
    ) -> Self:
        self._validate_reconciliation(reconciliation)
        if score.episode_id != self.episode_id or score.plan_id != self.plan_id:
            raise ValueError("score receipt does not match this episode")
        return self._transition(
            "finalizing",
            state="cleaning",
            terminal_intent="complete",
            reconciliation_id=reconciliation.reconciliation_id,
            reconciliation_reportable=reconciliation.reportable,
            score_id=score.score_id,
        )

    def mark_ambiguous_finalization(
        self,
        reconciliation: BudgetReconciliation,
        reason: str,
    ) -> Self:
        self._validate_reconciliation(reconciliation)
        return self._transition(
            "finalizing",
            state="cleaning",
            terminal_intent="fail",
            reconciliation_id=reconciliation.reconciliation_id,
            reconciliation_reportable=reconciliation.reportable,
            failure_kind="ambiguous_finalization",
            failure_reason=reason,
        )

    def crash(self, reason: str) -> Self:
        return self._transition(
            "running",
            state="cleaning",
            terminal_intent="fail",
            failure_kind="crash",
            failure_reason=reason,
        )

    def cancel(self, reason: str) -> Self:
        return self._transition(
            "running",
            state="cleaning",
            terminal_intent="cancel",
            failure_reason=reason,
        )

    def fail_before_running(
        self,
        *,
        kind: Literal["preflight", "infrastructure"],
        reason: str,
    ) -> Self:
        if self.state not in ("planned", "preflighted", "authorized"):
            raise ValueError("post-start failure must enter cleaning first")
        payload = self.model_dump(mode="python")
        payload.update(state="failed", failure_kind=kind, failure_reason=reason)
        return type(self).model_validate(payload)

    def finish_cleanup(self, result: CleanupResult) -> Self:
        if result.cleanup_plan_id != self.cleanup_plan_id:
            raise ValueError("cleanup result belongs to another plan")
        updates: dict[str, Any] = {
            "cleanup_result_id": result.cleanup_result_id,
            "rescue_required": result.rescue_required,
        }
        if self.terminal_intent == "cancel":
            updates["state"] = "cancelled"
        elif (
            self.terminal_intent == "complete"
            and not result.rescue_required
            and self.reconciliation_reportable is True
            and self.score_id is not None
        ):
            updates["state"] = "completed"
        else:
            updates["state"] = "failed"
            if result.rescue_required:
                updates["failure_kind"] = "cleanup"
                updates["failure_reason"] = "cleanup requires operator rescue"
            elif self.reconciliation_reportable is False and self.failure_kind is None:
                updates["failure_kind"] = "unreportable"
                updates["failure_reason"] = "required usage was unavailable"
        return self._transition("cleaning", **updates)

    def _validate_reconciliation(self, reconciliation: BudgetReconciliation) -> None:
        if (
            reconciliation.episode_id != self.episode_id
            or reconciliation.budget_id != self.budget_id
            or reconciliation.reservation_ids != self.reservation_ids
        ):
            raise ValueError("cost reconciliation does not match this episode")
