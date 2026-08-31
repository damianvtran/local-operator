"""Strict benchmark-neutral contracts shared by adapter hosts and workers.

Adapters are untrusted infrastructure plugins.  This module therefore contains
only immutable values, closed enums, and content-bound identities; importing it
must not discover or import an adapter distribution.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import Field, field_validator, model_validator

from local_operator.evaluation.evidence.models import ScoreArtifact, canonical_digest
from local_operator.evaluation.lifecycle import CleanupPlan, CleanupReceipt
from local_operator.evaluation.protocol import ActionBatch, Observation, ProtocolModel
from local_operator.evaluation.receipts import (
    ZERO_DIGEST,
    Digest,
    SafeCount,
    StrictIdentifier,
)

ADAPTER_SCHEMA_VERSION = "1.0"
ADAPTER_ENTRY_POINT_GROUP = "local_operator.evaluation_adapters.v1"
MAX_RESCUE_REFS = 256
MAX_REQUIREMENTS = 256
MAX_RECEIPTS = 256

AdapterMethod: TypeAlias = Literal[
    "hello",
    "inspect_requirements",
    "prepare",
    "reset_start",
    "observe",
    "execute",
    "ask_user_exchange",
    "score",
    "cleanup",
    "close",
]
AdapterState: TypeAlias = Literal[
    "NEW",
    "HANDSHAKEN",
    "INSPECTED",
    "PREPARED",
    "RUNNING",
    "FINALIZING",
    "CLEANING",
    "CLOSED",
    "POISONED",
]
RouteCapability: TypeAlias = Literal["computer", "browser", "terminal"]

# Both peers use this table.  Keeping it data, rather than distributed branches,
# makes an invalid transition fail before plugin code can acquire authority.
METHOD_STATES: Mapping[AdapterMethod, frozenset[AdapterState]] = {
    "hello": frozenset({"NEW"}),
    "inspect_requirements": frozenset({"HANDSHAKEN"}),
    "prepare": frozenset({"INSPECTED"}),
    "reset_start": frozenset({"PREPARED"}),
    "observe": frozenset({"RUNNING"}),
    "execute": frozenset({"RUNNING"}),
    "ask_user_exchange": frozenset({"RUNNING"}),
    "score": frozenset({"RUNNING", "FINALIZING"}),
    "cleanup": frozenset({"PREPARED", "RUNNING", "FINALIZING", "CLEANING"}),
    "close": frozenset(
        {"HANDSHAKEN", "INSPECTED", "PREPARED", "RUNNING", "FINALIZING", "CLEANING"}
    ),
}
METHOD_NEXT_STATE: Mapping[AdapterMethod, AdapterState] = {
    "hello": "HANDSHAKEN",
    "inspect_requirements": "INSPECTED",
    "prepare": "PREPARED",
    "reset_start": "RUNNING",
    "observe": "RUNNING",
    "execute": "RUNNING",
    "ask_user_exchange": "RUNNING",
    "score": "FINALIZING",
    "cleanup": "CLEANING",
    "close": "CLOSED",
}
KEYED_METHODS = frozenset(
    {"prepare", "reset_start", "execute", "ask_user_exchange", "score", "cleanup", "close"}
)
READ_ONLY_METHODS = frozenset({"hello", "inspect_requirements", "observe"})


def canonical_params_digest(method: AdapterMethod, params: ProtocolModel) -> Digest:
    return canonical_digest(f"adapter-rpc-params-{method}-v1", params)


class AdapterSelector(ProtocolModel):
    """Exact wheel and interpreter selection; ranges and fallback are absent."""

    schema_version: Literal["1.0"]
    adapter_id: StrictIdentifier
    distribution: StrictIdentifier
    version: StrictIdentifier
    entry_point: StrictIdentifier
    package_digest: Digest
    release_digest: Digest
    python_executable: str = Field(min_length=1, max_length=4096)
    workspace: str = Field(min_length=1, max_length=4096)
    route_capability: RouteCapability

    @field_validator("python_executable", "workspace")
    @classmethod
    def _absolute_path(cls, value: str) -> str:
        if not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("adapter paths must be normalized absolute paths")
        return value


class AdapterCapabilities(ProtocolModel):
    routes: tuple[RouteCapability, ...] = Field(min_length=1, max_length=3)
    ask_user: bool
    scoring: bool

    @field_validator("routes", mode="before")
    @classmethod
    def _freeze_routes(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _unique_routes(self) -> "AdapterCapabilities":
        if len(self.routes) != len(set(self.routes)):
            raise ValueError("adapter routes must be unique")
        return self


class AdapterMetadata(ProtocolModel):
    adapter_id: StrictIdentifier
    distribution: StrictIdentifier
    version: StrictIdentifier
    entry_point: StrictIdentifier
    package_digest: Digest
    release_digest: Digest
    schema_version: Literal["1.0"]
    capabilities: AdapterCapabilities


class PythonRuntime(ProtocolModel):
    executable: str = Field(min_length=1, max_length=4096)
    implementation: StrictIdentifier
    version: StrictIdentifier
    cache_tag: StrictIdentifier

    @field_validator("executable")
    @classmethod
    def _absolute_executable(cls, value: str) -> str:
        if not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("resolved Python executable must be normalized and absolute")
        return value

    @classmethod
    def current(cls) -> "PythonRuntime":
        cache_tag = sys.implementation.cache_tag
        if cache_tag is None:
            raise RuntimeError("selected Python does not expose a cache tag")
        return cls(
            executable=str(Path(sys.executable).resolve()),
            implementation=platform.python_implementation(),
            version=platform.python_version(),
            cache_tag=cache_tag,
        )


class Handshake(ProtocolModel):
    selector: AdapterSelector
    metadata: AdapterMetadata
    python: PythonRuntime
    workspace_digest: Digest
    selected_route: RouteCapability

    @model_validator(mode="after")
    def _repeat_exact_pins(self) -> "Handshake":
        selector = self.selector
        metadata = self.metadata
        repeated = (
            metadata.adapter_id,
            metadata.distribution,
            metadata.version,
            metadata.entry_point,
            metadata.package_digest,
            metadata.release_digest,
            metadata.schema_version,
        )
        selected = (
            selector.adapter_id,
            selector.distribution,
            selector.version,
            selector.entry_point,
            selector.package_digest,
            selector.release_digest,
            selector.schema_version,
        )
        if repeated != selected:
            raise ValueError("handshake does not repeat the exact adapter selection")
        if self.python.executable != selector.python_executable:
            raise ValueError("handshake resolved a different Python executable")
        if self.selected_route != selector.route_capability:
            raise ValueError("handshake selected a different route")
        if self.selected_route not in metadata.capabilities.routes:
            raise ValueError("selected route is not an adapter capability")
        return self


class SecretRef(ProtocolModel):
    """An opaque name resolved by infrastructure, never secret bytes."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Z][A-Z0-9_]*$")


InfraPurpose: TypeAlias = Literal[
    "benchmark_auth",
    "benchmark_compute",
    "benchmark_storage",
    "artifact_storage",
]


class ScopedInfraValue(ProtocolModel):
    """Non-model infrastructure input with a closed, purpose-limited vocabulary."""

    name: StrictIdentifier
    purpose: InfraPurpose
    value: str = Field(min_length=1, max_length=4096)


class RescueDescriptor(ProtocolModel):
    schema_version: Literal["1.0"]
    selector: AdapterSelector
    handshake: Handshake
    episode_id: StrictIdentifier
    cleanup_plan: CleanupPlan
    secret_refs: tuple[SecretRef, ...] = Field(max_length=MAX_RESCUE_REFS)
    infra_values: tuple[ScopedInfraValue, ...] = Field(max_length=MAX_RESCUE_REFS)
    artifact_root: str = Field(min_length=1, max_length=4096)
    descriptor_id: Digest = ZERO_DIGEST

    @field_validator("secret_refs", "infra_values", mode="before")
    @classmethod
    def _freeze_refs(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("artifact_root")
    @classmethod
    def _absolute_artifact_root(cls, value: str) -> str:
        if not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("artifact root must be normalized and absolute")
        return value

    @model_validator(mode="after")
    def _content_bind(self) -> "RescueDescriptor":
        if self.handshake.selector != self.selector:
            raise ValueError("rescue handshake and selector differ")
        if self.cleanup_plan.episode_id != self.episode_id:
            raise ValueError("rescue cleanup plan belongs to another episode")
        refs = [item.name for item in self.secret_refs]
        values = [(item.purpose, item.name) for item in self.infra_values]
        if len(refs) != len(set(refs)) or len(values) != len(set(values)):
            raise ValueError("rescue references must be unique")
        expected = canonical_digest(
            "adapter-rescue-descriptor-v1",
            self.model_dump(mode="json", exclude={"descriptor_id"}),
        )
        if self.descriptor_id not in (ZERO_DIGEST, expected):
            raise ValueError("rescue descriptor identity does not match its content")
        object.__setattr__(self, "descriptor_id", expected)
        return self


class EmptyParams(ProtocolModel):
    pass


class HelloParams(ProtocolModel):
    selector: AdapterSelector


class InspectRequirementsParams(ProtocolModel):
    pass


class Requirement(ProtocolModel):
    requirement_id: StrictIdentifier
    kind: Literal["secret", "infra"]
    name: StrictIdentifier
    required: bool


class RequirementsResult(ProtocolModel):
    requirements: tuple[Requirement, ...] = Field(max_length=MAX_REQUIREMENTS)

    @field_validator("requirements", mode="before")
    @classmethod
    def _freeze_requirements(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class OperationParams(ProtocolModel):
    operation_id: StrictIdentifier


class PrepareParams(OperationParams):
    episode_id: StrictIdentifier
    secret_refs: tuple[SecretRef, ...] = Field(max_length=MAX_RESCUE_REFS)
    infra_values: tuple[ScopedInfraValue, ...] = Field(max_length=MAX_RESCUE_REFS)

    @field_validator("secret_refs", "infra_values", mode="before")
    @classmethod
    def _freeze_values(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class PrepareResult(ProtocolModel):
    cleanup_plan: CleanupPlan


class ResetStartParams(OperationParams):
    episode_id: StrictIdentifier


class ObserveParams(ProtocolModel):
    episode_id: StrictIdentifier


class ObservationResult(ProtocolModel):
    observation: Observation


class ExecuteParams(OperationParams):
    action_batch: ActionBatch
    action_batch_id: Digest

    @model_validator(mode="after")
    def _bind_batch(self) -> "ExecuteParams":
        expected = canonical_digest("adapter-action-batch-v1", self.action_batch)
        if self.action_batch_id != expected:
            raise ValueError("action batch identity does not match canonical batch")
        return self


class ExecutionReceipt(ProtocolModel):
    operation_id: StrictIdentifier
    action_batch_id: Digest
    input_observation_id: StrictIdentifier
    output_observation_id: StrictIdentifier
    sequence: SafeCount
    receipt_id: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _identify(self) -> "ExecutionReceipt":
        if self.input_observation_id == self.output_observation_id:
            raise ValueError("execution must advance to a distinct observation")
        expected = canonical_digest(
            "adapter-execution-receipt-v1",
            self.model_dump(mode="json", exclude={"receipt_id"}),
        )
        if self.receipt_id not in (ZERO_DIGEST, expected):
            raise ValueError("execution receipt identity does not match content")
        object.__setattr__(self, "receipt_id", expected)
        return self


class ExecuteResult(ProtocolModel):
    observation: Observation
    receipt: ExecutionReceipt


class AskUserExchangeParams(OperationParams):
    episode_id: StrictIdentifier
    ask_id: StrictIdentifier
    prompt: str = Field(min_length=1, max_length=100_000)
    answer: str | None = Field(default=None, min_length=1, max_length=100_000)


class AskUserExchangeResult(ProtocolModel):
    ask_id: StrictIdentifier
    accepted: bool


class ScoreParams(OperationParams):
    episode_id: StrictIdentifier


class ScoreResult(ProtocolModel):
    score: ScoreArtifact


class CleanupParams(OperationParams):
    cleanup_plan: CleanupPlan
    action_ids: tuple[StrictIdentifier, ...] = Field(min_length=1, max_length=MAX_RECEIPTS)

    @field_validator("action_ids", mode="before")
    @classmethod
    def _freeze_actions(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _selected_actions_exist(self) -> "CleanupParams":
        if len(self.action_ids) != len(set(self.action_ids)):
            raise ValueError("cleanup selection contains duplicate action IDs")
        known = {item.action_id for item in self.cleanup_plan.actions}
        if not set(self.action_ids) <= known:
            raise ValueError("cleanup selection contains unknown action IDs")
        return self


class CleanupResult(ProtocolModel):
    receipts: tuple[CleanupReceipt, ...] = Field(max_length=MAX_RECEIPTS)

    @field_validator("receipts", mode="before")
    @classmethod
    def _freeze_receipts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class CloseParams(OperationParams):
    episode_id: StrictIdentifier | None = None


class AckResult(ProtocolModel):
    accepted: Literal[True] = True


AdapterParams: TypeAlias = (
    HelloParams
    | InspectRequirementsParams
    | PrepareParams
    | ResetStartParams
    | ObserveParams
    | ExecuteParams
    | AskUserExchangeParams
    | ScoreParams
    | CleanupParams
    | CloseParams
)
AdapterResult: TypeAlias = (
    Handshake
    | RequirementsResult
    | PrepareResult
    | ObservationResult
    | ExecuteResult
    | AskUserExchangeResult
    | ScoreResult
    | CleanupResult
    | AckResult
)
PARAM_MODELS: Mapping[AdapterMethod, type[ProtocolModel]] = {
    "hello": HelloParams,
    "inspect_requirements": InspectRequirementsParams,
    "prepare": PrepareParams,
    "reset_start": ResetStartParams,
    "observe": ObserveParams,
    "execute": ExecuteParams,
    "ask_user_exchange": AskUserExchangeParams,
    "score": ScoreParams,
    "cleanup": CleanupParams,
    "close": CloseParams,
}
RESULT_MODELS: Mapping[AdapterMethod, type[ProtocolModel]] = {
    "hello": Handshake,
    "inspect_requirements": RequirementsResult,
    "prepare": PrepareResult,
    "reset_start": AckResult,
    "observe": ObservationResult,
    "execute": ExecuteResult,
    "ask_user_exchange": AskUserExchangeResult,
    "score": ScoreResult,
    "cleanup": CleanupResult,
    "close": AckResult,
}


class ReplayEntry(ProtocolModel):
    request_id: int = Field(gt=0, le=2**53 - 1)
    method: AdapterMethod
    params_digest: Digest
    operation_id: StrictIdentifier | None = None

    @model_validator(mode="after")
    def _keyed_operation(self) -> "ReplayEntry":
        if (self.method in KEYED_METHODS) != (self.operation_id is not None):
            raise ValueError("only keyed methods carry operation IDs")
        return self


@runtime_checkable
class EvaluationAdapter(Protocol):
    """Async benchmark implementation loaded only inside the worker."""

    metadata: AdapterMetadata

    async def inspect_requirements(
        self, params: InspectRequirementsParams
    ) -> RequirementsResult: ...

    async def prepare(self, params: PrepareParams) -> PrepareResult: ...

    async def reset_start(self, params: ResetStartParams) -> AckResult: ...

    async def observe(self, params: ObserveParams) -> ObservationResult: ...

    async def execute(self, params: ExecuteParams) -> ExecuteResult: ...

    async def ask_user_exchange(self, params: AskUserExchangeParams) -> AskUserExchangeResult: ...

    async def score(self, params: ScoreParams) -> ScoreResult: ...

    async def cleanup(self, params: CleanupParams) -> CleanupResult: ...

    async def close(self, params: CloseParams) -> AckResult: ...


def observation_content_id(observation: Observation) -> Digest:
    """Identify adapter content while excluding its self-asserted identifier."""

    return canonical_digest(
        "adapter-observation-v1",
        observation.model_dump(mode="json", exclude={"observation_id"}),
    )


def validate_observation(observation: Observation) -> None:
    if observation.observation_id != observation_content_id(observation):
        raise ValueError("observation identity does not match canonical content")


def validate_execution(
    params: ExecuteParams,
    result: ExecuteResult,
    current: Observation,
    *,
    seen_sequences: set[int],
) -> None:
    params.action_batch.validate_for(current)
    validate_observation(result.observation)
    receipt = result.receipt
    if receipt.operation_id != params.operation_id:
        raise ValueError("execution receipt operation ID differs")
    if receipt.action_batch_id != params.action_batch_id:
        raise ValueError("execution receipt action batch ID differs")
    if receipt.input_observation_id != current.observation_id:
        raise ValueError("execution receipt input observation differs")
    if receipt.output_observation_id != result.observation.observation_id:
        raise ValueError("execution receipt output observation differs")
    if receipt.sequence != result.observation.sequence or receipt.sequence in seen_sequences:
        raise ValueError("execution receipt sequence is stale or duplicated")
