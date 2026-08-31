"""Canonical evidence record contract tests."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from local_operator.evaluation.evidence.models import (
    AbandonmentRecord,
    ActionBatchPayload,
    BudgetCommitmentPayload,
    CancelPayload,
    CleanupPayload,
    EnvironmentStepPayload,
    ErrorPayload,
    EventRecord,
    EvidenceArtifactRef,
    EvidenceManifest,
    FinalizationStartPayload,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ObservationPayload,
    PreflightPayload,
    ReconciliationPayload,
    RouteIdentity,
    ScoreArtifact,
    ScoringResultPayload,
    ScoringStartPayload,
    UsageCostPayload,
    UserSimulatorExchangePayload,
)

DIGEST = "0123456789abcdef" * 4
OTHER_DIGEST = "abcdef0123456789" * 4
ROUTE = RouteIdentity(provider_id="provider", route_id="route", model_id="model")
ARTIFACT = EvidenceArtifactRef(sha256=DIGEST, media_type="text/plain", byte_count=4)


def manifest() -> EvidenceManifest:
    return EvidenceManifest(
        episode_id="episode-1",
        harness_version="0.44.10",
        harness_git_revision=DIGEST,
        adapter_id="adapter",
        adapter_version="1.0",
        benchmark_id="benchmark",
        benchmark_release="release-1",
        task_id="task-1",
        task_digest=DIGEST,
        input_digest=OTHER_DIGEST,
        requested_route=ROUTE,
        fallback_policy="forbid",
        environment_digest=DIGEST,
        environment_release="env-1",
        dependency_plan_id=DIGEST,
        budget_id=OTHER_DIGEST,
        cleanup_plan_id=DIGEST,
        config_digest=OTHER_DIGEST,
        created_wall_time_ms=1,
        metadata={"portable": [True, 7, "release"]},
    )


def test_manifest_identity_is_stable_and_input_isolated() -> None:
    metadata = {"portable": [True, 7, "release"]}
    first = manifest().model_copy(update={"metadata": metadata})
    metadata["portable"].append("mutated")
    second = manifest()
    assert first == second
    assert first.bundle_id == second.bundle_id
    assert first.manifest_digest == second.manifest_digest
    with pytest.raises(TypeError):
        first.metadata["new"] = True  # type: ignore[index]
    assert EvidenceManifest.from_canonical_json(first.to_canonical_json()) == first


def test_manifest_golden_bytes_are_deterministic() -> None:
    value = manifest()
    encoded = value.to_canonical_json()
    assert encoded.startswith(b'{"adapter_id":"adapter","adapter_version":"1.0"')
    assert b'"schema_version":"1.0"' in encoded
    assert encoded == EvidenceManifest.from_canonical_json(encoded).to_canonical_json()
    with pytest.raises(ValueError, match="canonical"):
        EvidenceManifest.from_canonical_json(b"{\n" + encoded[1:])


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (
            "preflight",
            PreflightPayload(
                sealed_preflight_id=DIGEST, plan_id=DIGEST, receipt_ids=(DIGEST,), passed=True
            ),
        ),
        (
            "lifecycle_transition",
            LifecycleTransitionPayload(previous_state_id=None, state_id=DIGEST, state="planned"),
        ),
        (
            "model_request",
            ModelRequestPayload(
                request_id="request",
                requested_route=ROUTE,
                tool_schema_digest=DIGEST,
                input_tokens=3,
                message_count=1,
                tool_count=1,
            ),
        ),
        (
            "model_response",
            ModelResponsePayload(
                request_id="request",
                provider_request_id="provider-request",
                requested_route=ROUTE,
                served_route=ROUTE,
                stop_reason="end",
                output_tokens=2,
                reasoning_tokens=0,
                tool_call_count=0,
            ),
        ),
        (
            "usage_cost",
            UsageCostPayload(
                request_id="request", input_tokens=3, output_tokens=2, cost_microusd=9
            ),
        ),
        (
            "budget_commitment",
            BudgetCommitmentPayload(
                commitment_id=OTHER_DIGEST,
                budget_id=OTHER_DIGEST,
                reservation_ids=(DIGEST,),
                reserved_summary_digest=DIGEST,
            ),
        ),
        (
            "reconciliation",
            ReconciliationPayload(
                reconciliation_id=DIGEST,
                budget_id=OTHER_DIGEST,
                commitment_id=OTHER_DIGEST,
                reportable=True,
                provider_cost_microusd=9,
                environment_cost_microusd=0,
                total_cost_microusd=9,
            ),
        ),
        (
            "observation",
            ObservationPayload(observation_id="observation", sequence=0, artifacts=(ARTIFACT,)),
        ),
        (
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch",
                observation_id="observation",
                action_count=1,
                action_artifact=ARTIFACT,
            ),
        ),
        (
            "environment_step",
            EnvironmentStepPayload(
                step_id="step",
                action_batch_id="batch",
                receipt_id=DIGEST,
                input_observation_id="observation",
                output_observation_id="observation-2",
                terminated=False,
                truncated=False,
            ),
        ),
        (
            "user_simulator_exchange",
            UserSimulatorExchangePayload(
                exchange_id="exchange",
                request_artifact=ARTIFACT,
                response_artifact=ARTIFACT,
                receipt_id=DIGEST,
            ),
        ),
        (
            "finalization_start",
            FinalizationStartPayload(
                finalization_id="final",
                intent="score",
                scoring_operation_id="score-op",
            ),
        ),
        (
            "scoring_start",
            ScoringStartPayload(
                finalization_id="final",
                scoring_operation_id="score-op",
                scorer_id="scorer",
                scorer_version="1",
                intent_digest=DIGEST,
            ),
        ),
        (
            "scoring_result",
            ScoringResultPayload(
                finalization_id="final",
                scoring_operation_id="score-op",
                score=ScoreArtifact(status="scored", binary=0),
            ),
        ),
        (
            "cleanup",
            CleanupPayload(
                cleanup_result_id=DIGEST,
                cleanup_plan_id=DIGEST,
                receipt_ids=(DIGEST,),
                rescue_required=False,
            ),
        ),
        (
            "error",
            ErrorPayload(
                error_id="error", category="provider", diagnostic_code="timeout", retryable=True
            ),
        ),
        (
            "cancel",
            CancelPayload(cancellation_id="cancel", source="operator", diagnostic_code="requested"),
        ),
    ],
)
def test_all_event_kinds_are_closed_canonical_and_round_trip(kind: str, payload: object) -> None:
    event = EventRecord(
        sequence=0,
        previous_event_sha256=DIGEST,
        monotonic_ns=1,
        wall_time_ms=2,
        kind=kind,  # type: ignore[arg-type]
        payload=payload,  # type: ignore[arg-type]
    )
    assert EventRecord.from_canonical_json(event.to_canonical_json()) == event
    assert event.event_id != DIGEST


def test_event_kind_payload_mismatch_and_extra_fields_fail() -> None:
    with pytest.raises(ValidationError):
        EventRecord(
            sequence=0,
            previous_event_sha256=DIGEST,
            monotonic_ns=1,
            wall_time_ms=2,
            kind="cancel",
            payload=ErrorPayload(
                error_id="error", category="internal", diagnostic_code="x", retryable=False
            ),
        )
    with pytest.raises(ValidationError):
        EventRecord.model_validate(
            {
                "sequence": 0,
                "previous_event_sha256": DIGEST,
                "monotonic_ns": 1,
                "wall_time_ms": 2,
                "kind": "cancel",
                "payload": {
                    "cancellation_id": "cancel",
                    "source": "operator",
                    "diagnostic_code": "x",
                    "raw_exception": "secret",
                },
            }
        )


def test_score_zero_is_distinct_from_unscored() -> None:
    zero = ScoreArtifact(status="scored", binary=0)
    unscored = ScoreArtifact(status="unscored", reason="infrastructure_failure")
    assert zero.binary == 0 and zero.reason is None
    assert unscored.binary is None and unscored.reason == "infrastructure_failure"
    assert zero.score_id != unscored.score_id
    for invalid in (
        {"status": "scored"},
        {"status": "scored", "binary": 0, "reason": "crash"},
        {"status": "unscored", "reason": "crash", "binary": 0},
        {"status": "unscored"},
    ):
        with pytest.raises(ValidationError):
            ScoreArtifact.model_validate(invalid)


def test_abandonment_recovery_authority_is_complete_and_canonical() -> None:
    start = FinalizationStartPayload(
        finalization_id="final", intent="score", scoring_operation_id="score-op"
    )
    record = AbandonmentRecord(
        bundle_id=DIGEST,
        manifest_digest=OTHER_DIGEST,
        reason="ambiguous_finalization",
        diagnostic_code="cutpoint",
        finalization_id=start.finalization_id,
        finalization_intent=start.intent,
        scoring_operation_id=start.scoring_operation_id,
        finalization_intent_digest=start.intent_digest,
        pre_finalization_event_count=2,
        pre_finalization_event_sha256=DIGEST,
        last_event_sequence=1,
        last_event_sha256=OTHER_DIGEST,
        event_count=2,
        abandoned_wall_time_ms=3,
    )
    assert AbandonmentRecord.from_canonical_json(record.to_canonical_json()) == record
    for field in (
        "finalization_id",
        "finalization_intent",
        "finalization_intent_digest",
        "pre_finalization_event_count",
        "pre_finalization_event_sha256",
    ):
        invalid = record.model_dump(mode="json")
        invalid[field] = None
        invalid["abandonment_id"] = "0" * 64
        with pytest.raises(ValidationError):
            AbandonmentRecord.model_validate(invalid, strict=True)
    with pytest.raises(ValidationError):
        AbandonmentRecord(
            bundle_id=DIGEST,
            manifest_digest=OTHER_DIGEST,
            reason="operator_abandoned",
            diagnostic_code="open",
            finalization_id="final",
            finalization_intent="unscored",
            finalization_intent_digest=FinalizationStartPayload(
                finalization_id="final", intent="unscored"
            ).intent_digest,
            pre_finalization_event_count=0,
            pre_finalization_event_sha256=DIGEST,
            last_event_sequence=None,
            last_event_sha256=DIGEST,
            event_count=0,
            abandoned_wall_time_ms=1,
        )


def test_models_are_deeply_immutable() -> None:
    event = EventRecord(
        sequence=0,
        previous_event_sha256=DIGEST,
        monotonic_ns=1,
        wall_time_ms=2,
        kind="observation",
        payload=ObservationPayload(observation_id="observation", sequence=0, artifacts=(ARTIFACT,)),
    )
    assert copy.deepcopy(event) == event
    with pytest.raises(ValidationError):
        event.sequence = 2  # type: ignore[misc]
