"""Pure cleanup contracts and the evaluation episode state machine.

The permit boundary constrains cooperative adapters: Python cannot stop hostile
code from bypassing a type contract, but normal allocation and execution APIs
can make possession of a factory-minted permit a mandatory argument.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal, Self, cast

from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from local_operator.evaluation.protocol import ArtifactRef, ProtocolModel
from local_operator.evaluation.receipts import (
    MAX_DECLARATIONS,
    ZERO_DIGEST,
    BudgetAuthorization,
    BudgetCommitment,
    BudgetReconciliation,
    Digest,
    PositiveSafeCount,
    SafeCount,
    SealedPreflight,
    StrictIdentifier,
)

MAX_CLEANUP_ATTEMPTS = 32
MAX_CLEANUP_TIMEOUT_MS = 3_600_000
MAX_FAILURE_LENGTH = 2_000
_PERMIT_FACTORY = object()
_LIFECYCLE_FACTORY = object()


def _identity(kind: str, payload: Any) -> str:
    encoded = json.dumps(
        {"identity_kind": kind, "payload": payload},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_identity(payload: dict[str, Any]) -> str:
    complete = dict(payload)
    complete.setdefault("state", "planned")
    complete.setdefault("preflight_seal_id", None)
    complete.setdefault("permit_id", None)
    complete.setdefault("reservation_ids", ())
    complete.setdefault("reconciliation_id", None)
    complete.setdefault("reconciliation_reportable", None)
    complete.setdefault("score_id", None)
    complete.setdefault("cleanup_result_id", None)
    complete.setdefault("rescue_required", None)
    complete.setdefault("terminal_intent", None)
    complete.setdefault("failure_kind", None)
    complete.setdefault("failure_reason", None)
    complete.setdefault("previous_state_id", None)
    return _identity("episode-lifecycle-state-v1", complete)


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
    timeout_ms: PositiveSafeCount = Field(le=MAX_CLEANUP_TIMEOUT_MS)
    max_attempts: int = Field(ge=1, le=MAX_CLEANUP_ATTEMPTS)
    action_digest: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _identify(self) -> Self:
        expected = _identity(
            "cleanup-action-v1",
            self.model_dump(mode="json", exclude={"action_digest"}),
        )
        if self.action_digest not in (ZERO_DIGEST, expected):
            raise ValueError("cleanup action identity does not match its declaration")
        object.__setattr__(self, "action_digest", expected)
        return self


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
    cleanup_plan_id: Digest
    action_id: StrictIdentifier
    action_digest: Digest
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


def record_cleanup(
    plan: CleanupPlan,
    action_id: StrictIdentifier,
    *,
    status: Literal["not_needed", "attempted", "succeeded", "failed"],
    evidence_code: StrictIdentifier,
    duration_ms: int,
) -> CleanupReceipt:
    """Bind cleanup evidence to one exact symbolic action and plan."""

    action = next((item for item in plan.actions if item.action_id == action_id), None)
    if action is None:
        raise ValueError("cleanup action is not selected by the cleanup plan")
    return CleanupReceipt(
        cleanup_plan_id=plan.cleanup_plan_id,
        action_id=action_id,
        action_digest=action.action_digest,
        status=status,
        evidence_code=evidence_code,
        duration_ms=duration_ms,
    )


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
    actions = {action.action_id: action for action in plan.actions}
    for action_id, receipt in by_id.items():
        if receipt.cleanup_plan_id != plan.cleanup_plan_id:
            raise ValueError("cleanup receipt belongs to another cleanup plan")
        if receipt.action_digest != actions[action_id].action_digest:
            raise ValueError("cleanup receipt belongs to another action declaration")
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
    """Factory-only state authority with a content-addressed transition chain.

    Pydantic's ``model_construct`` can fabricate an object, but it lacks the
    private marker checked by every transition. This constrains cooperative
    adapters rather than hostile code which ignores the contract entirely.
    """

    _authority: object = PrivateAttr()

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
    previous_state_id: Digest | None = None
    operation: StrictIdentifier
    state_id: Digest

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def _freeze_reservations(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_state_evidence(self, info: ValidationInfo) -> Self:
        context = info.context if isinstance(info.context, dict) else {}
        if context.get("lifecycle_factory") is not _LIFECYCLE_FACTORY:
            raise ValueError("episode lifecycle authority can only be factory minted")
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
        if self.state == "planned":
            if self.previous_state_id is not None or self.operation != "plan":
                raise ValueError("planned lifecycle must be the transition-chain root")
        elif self.previous_state_id is None:
            raise ValueError("non-planned lifecycle requires a previous state identity")
        expected = _state_identity(self.model_dump(mode="json", exclude={"state_id"}))
        if self.state_id != expected:
            raise ValueError("episode lifecycle state identity does not match its evidence")
        object.__setattr__(self, "_authority", _LIFECYCLE_FACTORY)
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
        return cast(
            Self,
            plan_episode(
                episode_id=episode_id,
                plan_id=plan_id,
                budget_id=budget_id,
                cleanup_plan_id=cleanup_plan_id,
            ),
        )

    def __reduce__(self) -> Any:
        raise TypeError("episode lifecycle authority cannot be pickled")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        if memo is not None:
            memo[id(self)] = self
        return self

    def model_copy(self, *, update: Any = None, deep: bool = False) -> Self:
        if update:
            raise TypeError("episode lifecycle authority cannot be updated by copying")
        return self

    def _assert_authority(self) -> None:
        if getattr(self, "_authority", None) is not _LIFECYCLE_FACTORY:
            raise ValueError("episode lifecycle lacks transition authority")
        expected = _state_identity(self.model_dump(mode="json", exclude={"state_id"}))
        if self.state_id != expected:
            raise ValueError("episode lifecycle authority was mutated")

    def _transition(self, expected: EpisodeState, operation: str, **updates: Any) -> Self:
        self._assert_authority()
        if self.state != expected:
            raise ValueError(f"illegal episode transition from {self.state}")
        payload = self.model_dump(mode="python", exclude={"state_id"})
        payload.update(previous_state_id=self.state_id, operation=operation, **updates)
        payload["state_id"] = _state_identity(payload)
        return type(self).model_validate(
            payload,
            context={"lifecycle_factory": _LIFECYCLE_FACTORY},
        )

    def preflight(self, seal: SealedPreflight) -> Self:
        if not seal.successful:
            raise ValueError("failed preflight cannot enter preflighted state")
        if seal.plan_id != self.plan_id:
            raise ValueError("preflight seal belongs to another dependency plan")
        return self._transition(
            "planned", "seal-preflight", state="preflighted", preflight_seal_id=seal.seal_id
        )

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
            self._transition(
                "preflighted",
                "authorize-side-effects",
                state="authorized",
                permit_id=permit.permit_id,
            ),
            permit,
        )

    def start(
        self,
        permit: SideEffectPermit,
        authorization: BudgetAuthorization,
        commitment: BudgetCommitment,
    ) -> Self:
        self._assert_authority()
        if self.state != "authorized":
            raise ValueError(f"illegal episode transition from {self.state}")
        commitment.assert_authority(authorization)
        if commitment.budget_id != self.budget_id:
            raise ValueError("budget commitment does not match this episode")
        if (
            permit.episode_id != self.episode_id
            or permit.plan_id != self.plan_id
            or permit.budget_id != self.budget_id
            or permit.permit_id != self.permit_id
        ):
            raise ValueError("side-effect permit does not match this episode")
        return self._transition(
            "authorized",
            "start-episode",
            state="running",
            reservation_ids=commitment.reservation_ids,
        )

    def begin_finalization(self) -> Self:
        return self._transition("running", "begin-finalization", state="finalizing")

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
            "finish-finalization",
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
            "finish-finalization",
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
            "record-crash",
            state="cleaning",
            terminal_intent="fail",
            failure_kind="crash",
            failure_reason=reason,
        )

    def cancel(self, reason: str) -> Self:
        return self._transition(
            "running",
            "record-cancellation",
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
        return self._transition(
            self.state,
            "fail-before-running",
            state="failed",
            failure_kind=kind,
            failure_reason=reason,
        )

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
        return self._transition("cleaning", "finish-cleanup", **updates)

    def _validate_reconciliation(self, reconciliation: BudgetReconciliation) -> None:
        if (
            reconciliation.episode_id != self.episode_id
            or reconciliation.budget_id != self.budget_id
            or reconciliation.authorization_digest != self.budget_id
            or reconciliation.authorization.budget_id != self.budget_id
            or reconciliation.reservation_ids != self.reservation_ids
        ):
            raise ValueError("cost reconciliation does not match this episode")


def plan_episode(
    *,
    episode_id: StrictIdentifier,
    plan_id: Digest,
    budget_id: Digest,
    cleanup_plan_id: Digest,
) -> EpisodeLifecycle:
    """Mint the public plan as the root of a private transition authority chain."""

    payload = {
        "episode_id": episode_id,
        "plan_id": plan_id,
        "budget_id": budget_id,
        "cleanup_plan_id": cleanup_plan_id,
        "state": "planned",
        "operation": "plan",
        "state_id": ZERO_DIGEST,
    }
    payload["state_id"] = _state_identity(
        {key: value for key, value in payload.items() if key != "state_id"}
    )
    return EpisodeLifecycle.model_validate(
        payload,
        context={"lifecycle_factory": _LIFECYCLE_FACTORY},
    )
