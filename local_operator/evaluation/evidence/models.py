"""Canonical records for benchmark-neutral evaluation evidence bundles.

The models in this module are values, not lifecycle authorities.  Process-local
permits still protect cooperative execution, while these records define the
portable, durable boundary that survives process death and can be verified by a
reader which trusts neither the writer nor ``state.json``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Self, TypeAlias

from pydantic import (
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from local_operator.evaluation.protocol import (
    MAX_SAFE_JSON_INTEGER,
    ArtifactRef,
    FrozenMapping,
    PortableMetadataObject,
    ProtocolModel,
)
from local_operator.evaluation.receipts import (
    ZERO_DIGEST,
    Digest,
    SafeCount,
    StrictIdentifier,
)

EVIDENCE_SCHEMA_VERSION = "1.0"
MAX_WALL_TIME_MS = MAX_SAFE_JSON_INTEGER
MAX_MONOTONIC_NS = MAX_SAFE_JSON_INTEGER
MAX_TEXT = 2_000

MediaType = Literal[
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/json",
    "text/plain",
    "application/octet-stream",
]


class EvidenceArtifactRef(ArtifactRef):
    """Artifact reference restricted to evidence media the verifier understands."""

    @field_validator("media_type")
    @classmethod
    def _closed_media(cls, value: str) -> str:
        if value not in {
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/webp",
            "application/json",
            "text/plain",
            "application/octet-stream",
        }:
            raise ValueError("unsupported evidence artifact media type")
        return value


EventKind = Literal[
    "preflight",
    "lifecycle_transition",
    "model_request",
    "model_response",
    "usage_cost",
    "budget_commitment",
    "reconciliation",
    "observation",
    "action_batch",
    "environment_step",
    "user_simulator_exchange",
    "finalization_start",
    "scoring_start",
    "scoring_result",
    "cleanup",
    "error",
    "cancel",
]
UnscoredReason = Literal[
    "preflight_failure",
    "infrastructure_failure",
    "cancelled",
    "crash",
    "ambiguous_finalization",
    "scorer_failure",
]
AbandonmentReason = Literal[
    "preflight_failure",
    "infrastructure_failure",
    "cancelled",
    "crash",
    "ambiguous_finalization",
    "operator_abandoned",
]
ReportabilityLabel = Literal[
    "reportable",
    "preflight_incomplete",
    "budget_unreconciled",
    "cleanup_incomplete",
    "unscored",
    "infrastructure_failure",
    "cancelled",
]
ComparabilityLabel = Literal[
    "comparable",
    "route_changed",
    "environment_unpinned",
    "input_mismatch",
    "adapter_mismatch",
    "benchmark_mismatch",
]


def canonical_bytes(value: ProtocolModel | dict[str, Any] | list[Any]) -> bytes:
    """Encode the evidence JSON subset exactly once for hashing and persistence."""

    payload = value.model_dump(mode="json") if isinstance(value, ProtocolModel) else value
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest(domain: str, value: ProtocolModel | dict[str, Any] | list[Any]) -> str:
    """Domain-separate identities so equal JSON cannot cross record roles."""

    return hashlib.sha256(domain.encode("ascii") + b"\0" + canonical_bytes(value)).hexdigest()


class _MetadataModel(ProtocolModel):
    @field_validator("metadata", mode="before", check_fields=False)
    @classmethod
    def _snapshot_metadata(cls, value: Any) -> Any:
        if isinstance(value, FrozenMapping):
            return dict(value.items())
        if not isinstance(value, dict):
            raise ValueError("portable metadata must be an object")
        return dict(value)

    @field_validator("metadata", check_fields=False)
    @classmethod
    def _freeze_metadata(cls, value: Any) -> FrozenMapping:
        return FrozenMapping(value)

    @field_serializer("metadata", check_fields=False)
    def _serialize_metadata(self, value: FrozenMapping) -> Any:
        def thaw(item: Any) -> Any:
            if isinstance(item, FrozenMapping):
                return {key: thaw(nested) for key, nested in item.items()}
            if isinstance(item, tuple):
                return [thaw(nested) for nested in item]
            return item

        return thaw(value)


class RouteIdentity(ProtocolModel):
    provider_id: StrictIdentifier
    route_id: StrictIdentifier
    model_id: StrictIdentifier


class EvidenceManifest(_MetadataModel):
    """Immutable provenance and declared authorities for one evidence bundle.

    Metadata is deliberately portable and may contain digests or public release
    pins, never resolved credentials, provider request bodies, or raw prompts.
    """

    schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    bundle_id: Digest = ZERO_DIGEST
    manifest_digest: Digest = ZERO_DIGEST
    episode_id: StrictIdentifier
    harness_version: StrictIdentifier
    harness_git_revision: Digest
    adapter_id: StrictIdentifier
    adapter_version: StrictIdentifier
    benchmark_id: StrictIdentifier
    benchmark_release: StrictIdentifier
    task_id: StrictIdentifier
    task_digest: Digest
    input_digest: Digest
    requested_route: RouteIdentity
    fallback_policy: Literal["forbid", "allow_compatible", "allow_any"]
    environment_digest: Digest
    environment_release: StrictIdentifier
    provider_image_digest: Digest | None = None
    provider_release: StrictIdentifier | None = None
    dependency_plan_id: Digest
    budget_id: Digest
    cleanup_plan_id: Digest
    config_digest: Digest
    created_wall_time_ms: SafeCount
    metadata: PortableMetadataObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _identify(self) -> Self:
        unsigned = self.model_dump(mode="json", exclude={"bundle_id", "manifest_digest"})
        digest = canonical_digest("evidence-manifest-v1", unsigned)
        bundle = canonical_digest(
            "evidence-bundle-v1",
            {"episode_id": self.episode_id, "manifest_digest": digest},
        )
        if self.manifest_digest not in (ZERO_DIGEST, digest):
            raise ValueError("manifest digest does not match canonical contents")
        if self.bundle_id not in (ZERO_DIGEST, bundle):
            raise ValueError("bundle identity does not match canonical manifest")
        object.__setattr__(self, "manifest_digest", digest)
        object.__setattr__(self, "bundle_id", bundle)
        return self


class PreflightPayload(ProtocolModel):
    sealed_preflight_id: Digest
    plan_id: Digest
    receipt_ids: tuple[Digest, ...]
    passed: bool

    @field_validator("receipt_ids", mode="before")
    @classmethod
    def _tuple_receipts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class LifecycleTransitionPayload(ProtocolModel):
    previous_state_id: Digest | None
    state_id: Digest
    state: Literal[
        "planned", "preflighted", "running", "finalizing", "completed", "failed", "cancelled"
    ]
    finalization_id: StrictIdentifier | None = None
    preflight_seal_id: Digest | None = None
    commitment_id: Digest | None = None
    reconciliation_id: Digest | None = None
    reconciliation_reportable: bool | None = None
    score_id: Digest | None = None
    cleanup_result_id: Digest | None = None
    rescue_required: bool | None = None
    failure_kind: (
        Literal[
            "preflight",
            "infrastructure",
            "crash",
            "ambiguous_finalization",
            "scorer",
            "cleanup",
            "cancelled",
        ]
        | None
    ) = None


class ModelRequestPayload(ProtocolModel):
    request_id: StrictIdentifier
    requested_route: RouteIdentity
    tool_schema_digest: Digest
    input_tokens: SafeCount
    message_count: SafeCount
    tool_count: SafeCount
    redacted_prompt: EvidenceArtifactRef | None = None


class ModelResponsePayload(ProtocolModel):
    request_id: StrictIdentifier
    provider_request_id: StrictIdentifier
    requested_route: RouteIdentity
    served_route: RouteIdentity
    stop_reason: StrictIdentifier
    output_tokens: SafeCount
    reasoning_tokens: SafeCount
    cache_read_tokens: SafeCount = 0
    cache_write_tokens: SafeCount = 0
    tool_call_count: SafeCount
    redacted_response: EvidenceArtifactRef | None = None


class UsageCostPayload(ProtocolModel):
    request_id: StrictIdentifier
    input_tokens: SafeCount
    output_tokens: SafeCount
    reasoning_tokens: SafeCount = 0
    cache_read_tokens: SafeCount = 0
    cache_write_tokens: SafeCount = 0
    cost_microusd: SafeCount


class BudgetCommitmentPayload(ProtocolModel):
    """Durable binding between the manifest budget and reserved execution authority."""

    commitment_id: Digest
    budget_id: Digest
    reservation_ids: tuple[Digest, ...]
    reserved_summary_digest: Digest

    @field_validator("reservation_ids", mode="before")
    @classmethod
    def _tuple_reservations(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _unique_reservations(self) -> Self:
        if len(set(self.reservation_ids)) != len(self.reservation_ids):
            raise ValueError("budget commitment contains duplicate reservations")
        return self


class ReconciliationPayload(ProtocolModel):
    """Durable usage receipt which closes one committed budget exactly once."""

    reconciliation_id: Digest
    budget_id: Digest
    commitment_id: Digest
    reportable: bool
    provider_cost_microusd: SafeCount
    environment_cost_microusd: SafeCount
    total_cost_microusd: SafeCount
    receipt_artifact: EvidenceArtifactRef | None = None

    @model_validator(mode="after")
    def _sum_costs(self) -> Self:
        if self.provider_cost_microusd + self.environment_cost_microusd != self.total_cost_microusd:
            raise ValueError("reconciliation total does not match component costs")
        return self


class ObservationPayload(ProtocolModel):
    observation_id: StrictIdentifier
    sequence: SafeCount
    artifacts: tuple[EvidenceArtifactRef, ...] = ()
    text_artifact: EvidenceArtifactRef | None = None

    @field_validator("artifacts", mode="before")
    @classmethod
    def _tuple_artifacts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ActionBatchPayload(ProtocolModel):
    action_batch_id: StrictIdentifier
    observation_id: StrictIdentifier
    action_count: SafeCount
    action_artifact: EvidenceArtifactRef
    terminal: Literal["finish", "ask_user"] | None = None


class EnvironmentStepPayload(ProtocolModel):
    step_id: StrictIdentifier
    action_batch_id: StrictIdentifier
    receipt_id: Digest
    input_observation_id: StrictIdentifier
    output_observation_id: StrictIdentifier
    terminated: bool
    truncated: bool


class UserSimulatorExchangePayload(ProtocolModel):
    exchange_id: StrictIdentifier
    previous_exchange_id: StrictIdentifier | None = None
    request_artifact: EvidenceArtifactRef
    response_artifact: EvidenceArtifactRef
    receipt_id: Digest


class FinalizationIntent(ProtocolModel):
    kind: Literal["score", "unscored"]
    scorer_id: StrictIdentifier | None = None
    scorer_version: StrictIdentifier | None = None
    intent_digest: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _validate_and_identify(self) -> Self:
        if self.kind == "score" and (self.scorer_id is None or self.scorer_version is None):
            raise ValueError("scoring intent requires a pinned scorer identity")
        if self.kind == "unscored" and (
            self.scorer_id is not None or self.scorer_version is not None
        ):
            raise ValueError("unscored intent cannot name a scorer")
        expected = canonical_digest(
            "finalization-intent-v1",
            self.model_dump(mode="json", exclude={"intent_digest"}),
        )
        if self.intent_digest not in (ZERO_DIGEST, expected):
            raise ValueError("finalization intent identity does not match declaration")
        object.__setattr__(self, "intent_digest", expected)
        return self


class FinalizationStartPayload(ProtocolModel):
    finalization_id: StrictIdentifier
    intent: Literal["score", "unscored"]
    scoring_operation_id: StrictIdentifier | None = None
    intent_digest: Digest

    @model_validator(mode="after")
    def _bind_operation(self) -> Self:
        if (self.intent == "score") != (self.scoring_operation_id is not None):
            raise ValueError("finalization intent and scoring operation disagree")
        return self


class ScoringStartPayload(ProtocolModel):
    finalization_id: StrictIdentifier
    scoring_operation_id: StrictIdentifier
    scorer_id: StrictIdentifier
    scorer_version: StrictIdentifier
    intent_digest: Digest


class ScoreArtifact(ProtocolModel):
    """Canonical score where zero and absence are different wire shapes."""

    status: Literal["scored", "unscored"]
    binary: Literal[0, 1] | None = None
    partial_ppm: int | None = Field(default=None, ge=0, le=1_000_000)
    reason: UnscoredReason | None = None
    details: EvidenceArtifactRef | None = None
    score_id: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _validate_and_identify(self) -> Self:
        if self.status == "scored":
            if self.reason is not None or (self.binary is None and self.partial_ppm is None):
                raise ValueError("scored result needs a score and cannot have an unscored reason")
        elif self.reason is None or self.binary is not None or self.partial_ppm is not None:
            raise ValueError("unscored result needs only a closed unscored reason")
        payload = self.model_dump(mode="json", exclude={"score_id"})
        expected = canonical_digest("score-artifact-v1", payload)
        if self.score_id not in (ZERO_DIGEST, expected):
            raise ValueError("score identity does not match canonical result")
        object.__setattr__(self, "score_id", expected)
        return self


class ScoringResultPayload(ProtocolModel):
    finalization_id: StrictIdentifier
    scoring_operation_id: StrictIdentifier
    score: ScoreArtifact


class CleanupPayload(ProtocolModel):
    cleanup_result_id: Digest
    cleanup_plan_id: Digest
    receipt_ids: tuple[Digest, ...]
    rescue_required: bool

    @field_validator("receipt_ids", mode="before")
    @classmethod
    def _tuple_receipts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ErrorPayload(ProtocolModel):
    error_id: StrictIdentifier
    category: Literal["adapter", "environment", "provider", "infrastructure", "scorer", "internal"]
    diagnostic_code: StrictIdentifier
    detail_artifact: EvidenceArtifactRef | None = None
    retryable: bool


class CancelPayload(ProtocolModel):
    cancellation_id: StrictIdentifier
    source: Literal["operator", "harness", "benchmark", "timeout"]
    diagnostic_code: StrictIdentifier


EventPayload: TypeAlias = (
    PreflightPayload
    | LifecycleTransitionPayload
    | ModelRequestPayload
    | ModelResponsePayload
    | UsageCostPayload
    | BudgetCommitmentPayload
    | ReconciliationPayload
    | ObservationPayload
    | ActionBatchPayload
    | EnvironmentStepPayload
    | UserSimulatorExchangePayload
    | FinalizationStartPayload
    | ScoringStartPayload
    | ScoringResultPayload
    | CleanupPayload
    | ErrorPayload
    | CancelPayload
)
_EVENT_PAYLOAD_TYPES: dict[str, type[ProtocolModel]] = {
    "preflight": PreflightPayload,
    "lifecycle_transition": LifecycleTransitionPayload,
    "model_request": ModelRequestPayload,
    "model_response": ModelResponsePayload,
    "usage_cost": UsageCostPayload,
    "budget_commitment": BudgetCommitmentPayload,
    "reconciliation": ReconciliationPayload,
    "observation": ObservationPayload,
    "action_batch": ActionBatchPayload,
    "environment_step": EnvironmentStepPayload,
    "user_simulator_exchange": UserSimulatorExchangePayload,
    "finalization_start": FinalizationStartPayload,
    "scoring_start": ScoringStartPayload,
    "scoring_result": ScoringResultPayload,
    "cleanup": CleanupPayload,
    "error": ErrorPayload,
    "cancel": CancelPayload,
}


class EventRecord(ProtocolModel):
    schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    sequence: SafeCount
    event_id: Digest = ZERO_DIGEST
    previous_event_sha256: Digest
    monotonic_ns: SafeCount
    wall_time_ms: SafeCount
    kind: EventKind
    payload: EventPayload

    @model_validator(mode="before")
    @classmethod
    def _parse_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        copied = dict(value)
        kind = copied.get("kind")
        payload = copied.get("payload")
        payload_type = _EVENT_PAYLOAD_TYPES.get(kind) if isinstance(kind, str) else None
        if payload_type is not None and not isinstance(payload, payload_type):
            copied["payload"] = payload_type.model_validate(payload, strict=True)
        return copied

    @model_validator(mode="after")
    def _validate_and_identify(self) -> Self:
        expected_type = _EVENT_PAYLOAD_TYPES[self.kind]
        if type(self.payload) is not expected_type:
            raise ValueError("event payload does not match its closed kind")
        unsigned = self.model_dump(mode="json", exclude={"event_id"})
        expected = canonical_digest("evidence-event-v1", unsigned)
        if self.event_id not in (ZERO_DIGEST, expected):
            raise ValueError("event identity does not match canonical record")
        object.__setattr__(self, "event_id", expected)
        return self


class ArtifactInventoryEntry(ProtocolModel):
    ref: EvidenceArtifactRef


class EvidenceCounters(ProtocolModel):
    event_count: SafeCount
    model_request_count: SafeCount
    model_response_count: SafeCount
    action_batch_count: SafeCount
    environment_step_count: SafeCount
    input_tokens: SafeCount
    output_tokens: SafeCount
    reasoning_tokens: SafeCount
    cache_read_tokens: SafeCount
    cache_write_tokens: SafeCount
    cost_microusd: SafeCount


class OutcomeSeal(ProtocolModel):
    schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    bundle_id: Digest
    manifest_digest: Digest
    event_count: SafeCount
    final_event_sha256: Digest
    artifacts: tuple[EvidenceArtifactRef, ...]
    finalization_id: StrictIdentifier
    preflight_seal_id: Digest
    commitment_id: Digest | None
    reconciliation_id: Digest | None
    score_id: Digest
    cleanup_result_id: Digest | None
    result: ScoreArtifact
    reportable: bool
    reportability_label: ReportabilityLabel
    comparable: bool
    comparability_label: ComparabilityLabel
    requested_route: RouteIdentity
    served_routes: tuple[RouteIdentity, ...]
    counters: EvidenceCounters
    started_wall_time_ms: SafeCount
    ended_wall_time_ms: SafeCount
    evidence_root: Digest = ZERO_DIGEST

    @field_validator("artifacts", "served_routes", mode="before")
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _canonical_and_identify(self) -> Self:
        artifacts = tuple(sorted(self.artifacts, key=lambda item: item.sha256))
        if len({item.sha256 for item in artifacts}) != len(artifacts):
            raise ValueError("outcome artifact inventory contains duplicates")
        routes = tuple(
            sorted(
                self.served_routes,
                key=lambda item: (item.provider_id, item.route_id, item.model_id),
            )
        )
        if len(set(routes)) != len(routes):
            raise ValueError("outcome served route inventory contains duplicates")
        if self.ended_wall_time_ms < self.started_wall_time_ms:
            raise ValueError("outcome wall times are reversed")
        if self.score_id != self.result.score_id:
            raise ValueError("outcome score identity disagrees with result")
        if self.reportable != (self.reportability_label == "reportable"):
            raise ValueError("outcome reportability label disagrees with flag")
        if self.comparable != (self.comparability_label == "comparable"):
            raise ValueError("outcome comparability label disagrees with flag")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "served_routes", routes)
        unsigned = self.model_dump(mode="json", exclude={"evidence_root"})
        expected = canonical_digest("evidence-root-v1", unsigned)
        if self.evidence_root not in (ZERO_DIGEST, expected):
            raise ValueError("outcome evidence root does not match canonical seal")
        object.__setattr__(self, "evidence_root", expected)
        return self


class OutcomeDraft(ProtocolModel):
    """Assertions supplied by a caller; sealing accepts only verifier agreement."""

    finalization_id: StrictIdentifier
    preflight_seal_id: Digest
    commitment_id: Digest | None
    reconciliation_id: Digest | None
    cleanup_result_id: Digest | None
    result: ScoreArtifact
    reportability_label: ReportabilityLabel
    comparability_label: ComparabilityLabel
    ended_wall_time_ms: SafeCount


class AbandonmentRecord(ProtocolModel):
    schema_version: Literal["1.0"] = EVIDENCE_SCHEMA_VERSION
    bundle_id: Digest
    manifest_digest: Digest
    reason: AbandonmentReason
    diagnostic_code: StrictIdentifier
    last_event_sequence: int | None = Field(default=None, ge=0, le=MAX_SAFE_JSON_INTEGER)
    last_event_sha256: Digest
    event_count: SafeCount
    abandoned_wall_time_ms: SafeCount
    abandonment_id: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _identify(self) -> Self:
        expected = canonical_digest(
            "evidence-abandonment-v1",
            self.model_dump(mode="json", exclude={"abandonment_id"}),
        )
        if self.abandonment_id not in (ZERO_DIGEST, expected):
            raise ValueError("abandonment identity does not match durable head")
        object.__setattr__(self, "abandonment_id", expected)
        return self


class StateMarker(ProtocolModel):
    state: Literal["open", "finalizing", "sealed", "abandoned"]
    bundle_id: Digest
    updated_wall_time_ms: SafeCount
    finalization_id: StrictIdentifier | None = None
    scoring_operation_id: StrictIdentifier | None = None
    intent: Literal["score", "unscored"] | None = None
    intent_digest: Digest | None = None
    terminal_id: Digest | None = None

    @model_validator(mode="after")
    def _state_fields(self) -> Self:
        if self.state == "finalizing" and self.finalization_id is None:
            raise ValueError("finalizing marker requires a finalization ID")
        if self.state in ("open", "sealed", "abandoned") and self.intent is not None:
            raise ValueError("only finalizing marker carries finalization intent")
        return self


VerificationSeverity = Literal["error", "warning"]
VerificationIssueCode = Literal[
    "unsupported_platform",
    "root_invalid",
    "unknown_root_entry",
    "unsafe_path",
    "unsafe_owner",
    "unsafe_permissions",
    "resource_limit_exceeded",
    "manifest_missing",
    "manifest_noncanonical",
    "manifest_invalid",
    "manifest_identity_mismatch",
    "journal_missing",
    "journal_truncated",
    "journal_noncanonical",
    "event_invalid",
    "event_sequence_mismatch",
    "event_chain_mismatch",
    "event_hash_mismatch",
    "event_time_reversed",
    "event_order_invalid",
    "lifecycle_invalid",
    "artifact_name_invalid",
    "artifact_unsafe",
    "artifact_hash_mismatch",
    "artifact_count_mismatch",
    "artifact_media_mismatch",
    "artifact_missing",
    "artifact_unreferenced",
    "finalization_invalid",
    "receipt_binding_invalid",
    "score_invalid",
    "counter_mismatch",
    "cost_mismatch",
    "route_mismatch",
    "terminal_conflict",
    "outcome_noncanonical",
    "outcome_invalid",
    "outcome_mismatch",
    "abandonment_noncanonical",
    "abandonment_invalid",
    "abandonment_mismatch",
    "state_missing",
    "state_invalid",
    "state_stale",
]


class VerificationIssue(ProtocolModel):
    code: VerificationIssueCode
    severity: VerificationSeverity
    location: str = Field(min_length=1, max_length=256)


class VerificationReport(ProtocolModel):
    valid: bool
    terminal_state: Literal["open", "finalizing", "sealed", "abandoned", "invalid"]
    issues: tuple[VerificationIssue, ...]
    manifest: EvidenceManifest | None = None
    events: tuple[EventRecord, ...] = ()
    artifacts: tuple[EvidenceArtifactRef, ...] = ()
    outcome: OutcomeSeal | None = None
    abandonment: AbandonmentRecord | None = None
    counters: EvidenceCounters | None = None

    @field_validator("issues", "events", "artifacts", mode="before")
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _valid_agrees(self) -> Self:
        expected = not any(issue.severity == "error" for issue in self.issues)
        if self.valid != expected:
            raise ValueError("verification validity disagrees with issue severities")
        return self


EVENT_PAYLOAD_ADAPTER = TypeAdapter(EventPayload)
