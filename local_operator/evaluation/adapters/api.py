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
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import AfterValidator, Field, field_validator, model_validator

from local_operator.evaluation.evidence.models import ScoreArtifact, canonical_digest
from local_operator.evaluation.lifecycle import CleanupPlan
from local_operator.evaluation.protocol import ActionBatch, Observation, ProtocolModel
from local_operator.evaluation.receipts import (
    ZERO_DIGEST,
    Digest,
    SafeCount,
    StrictIdentifier,
)

# 1.1 added the parent-chosen ``artifact_root`` to ``ResetStartParams``.
#
# 1.2 adds ``secrets`` -- resolved secret material -- to ``ResetStartParams`` and
# ``BeginRescueParams``. Before 1.2 the boundary had NO secret delivery path at
# all: ``SecretRef`` is a name only, the worker environment is built from a
# closed allowlist (``supervisor._ENV_ALLOW``) that strips every credential-like
# name, and nothing resolved a ref into bytes the worker could use. An adapter
# that must call a cloud API (the OSWorld AWS provider) therefore could not run
# out-of-process. The field is defaulted to ``()`` so an adapter that needs no
# secrets is unaffected on the wire, but the bump is still required: a 1.1
# worker's ``extra="forbid"`` model REJECTS the new key the moment a parent
# sends a non-empty tuple, and it would do so as an opaque ``invalid_request``
# mid-episode rather than at the handshake. This boundary pins exact versions
# (see ``AdapterSelector``), so the bump is what locks a 1.1 adapter out up
# front instead of letting it fail after resources may exist.
#
# The field is deliberately NOT on ``PrepareParams`` (allocation-free, so it
# has no use for credentials) and NOT on ``RescueDescriptor`` (persisted to
# disk; a rescue worker receives secrets fresh from the parent's resolver via
# ``BeginRescueParams`` instead).
#
# 1.3 adds ``detail`` -- a bounded, worker-redacted structured cause -- to
# ``RpcError`` (rpc.py). A fatal ``adapter_error`` previously recorded nothing
# but the fixed sentence "adapter operation failed": two consecutive paid
# episodes (ep-e46c789ca818 at 19 billed steps, ep-ffda3fc88f81 at 16) died
# there and diagnosing either would have cost another paid run.
#
# The bump is REQUIRED even though the field is optional and defaults to None,
# and the reason is specific to this transport rather than to pydantic's
# leniency. Every RPC line must round-trip byte-identically through
# ``parse_canonical_line``, and a model always serialises its full field set --
# so a 1.2 worker's error line (no ``detail`` key) fails a 1.3 parent's
# canonicality check, and a 1.3 worker's line (``"detail":null``) fails a 1.2
# parent's ``extra="forbid"``. Mixed versions break in BOTH directions, on the
# error path, which is the worst possible place to discover a mismatch. The
# exact-version pin in ``AdapterSelector`` turns that into a refusal at
# selection time, before a worker is spawned or a resource allocated.
#
# 1.4 adds the OBSERVATION-PHASE RECOVERY contract: ``phase`` on
# ``RpcErrorDetail`` (rpc.py) and ``resume_observation`` on ``ExecuteParams``.
# Together they let an adapter say "my mutation committed; only the read-back
# failed", which is the one case where a repeat call is safe -- see
# ``ObservationPhaseError`` below for why the harness cannot infer this on its
# own. Five paid episodes died at steps 9-32 because a burstable benchmark VM
# starved its screenshot server for ~25s and the adapter, correctly, refused to
# build a frameless observation.
#
# The bump is required for exactly the reason 1.3's was, in both directions: a
# 1.3 worker's error line carries no ``phase`` key and fails a 1.4 parent's
# canonicality check, and a 1.4 worker's ``"phase":"unknown"`` fails a 1.3
# parent's ``extra="forbid"``. ``resume_observation`` is the same story on the
# request side.
ADAPTER_SCHEMA_VERSION = "1.4"
# One alias for the three models that pin the version, so a future bump cannot
# move the constant while leaving a model silently accepting the older literal.
SchemaVersion: TypeAlias = Literal["1.4"]
ADAPTER_ENTRY_POINT_GROUP = "local_operator.evaluation_adapters.v1"
MAX_RESCUE_REFS = 256
MAX_REQUIREMENTS = 256
MAX_RECEIPTS = 256

AdapterMethod: TypeAlias = Literal[
    "hello",
    "inspect_requirements",
    "begin_rescue",
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
    "begin_rescue": frozenset({"HANDSHAKEN"}),
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
    "begin_rescue": "CLEANING",
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
    {
        "begin_rescue",
        "prepare",
        "reset_start",
        "execute",
        "ask_user_exchange",
        "score",
        "cleanup",
        "close",
    }
)
READ_ONLY_METHODS = frozenset({"hello", "inspect_requirements", "observe"})


def canonical_params_digest(method: AdapterMethod, params: ProtocolModel) -> Digest:
    return canonical_digest(f"adapter-rpc-params-{method}-v1", params)


class AdapterSelector(ProtocolModel):
    """Exact wheel and interpreter selection; ranges and fallback are absent."""

    schema_version: SchemaVersion
    adapter_id: StrictIdentifier
    distribution: StrictIdentifier
    version: StrictIdentifier
    entry_point: StrictIdentifier
    package_digest: Digest
    release_digest: Digest
    python_executable: str = Field(min_length=1, max_length=4096)
    workspace: str = Field(min_length=1, max_length=4096)
    workspace_digest: Digest
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
    schema_version: SchemaVersion
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
        if self.workspace_digest != selector.workspace_digest:
            raise ValueError("handshake workspace digest differs from selection")
        if self.selected_route != selector.route_capability:
            raise ValueError("handshake selected a different route")
        if self.selected_route not in metadata.capabilities.routes:
            raise ValueError("selected route is not an adapter capability")
        return self


class SecretRef(ProtocolModel):
    """An opaque name resolved by infrastructure, never secret bytes."""

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Z][A-Z0-9_]*$")


class ResolvedSecret(ProtocolModel):
    """A ``SecretRef`` resolved to its bytes, for the private RPC pipe ONLY.

    This is the one model on the boundary that carries secret material, and it
    is admitted on exactly two calls: ``reset_start`` (the side-effect boundary,
    where a provider first needs a credential) and ``begin_rescue`` (a fresh
    worker that must tear down with nothing but the descriptor). It never
    appears in a persisted structure -- ``RescueDescriptor`` carries refs -- and
    the worker must never write a value into ``os.environ`` or a log; the
    parent's ``RedactionSet`` canaries the evidence bundle against every value.

    ``name`` reuses the ``SecretRef`` pattern so a resolved secret is always
    addressable by the ref it satisfied. The value bound is generous enough for
    a PEM-shaped credential while still refusing an arbitrary blob.
    """

    name: str = Field(min_length=1, max_length=256, pattern=r"^[A-Z][A-Z0-9_]*$")
    value: str = Field(min_length=1, max_length=8192)


# ``benchmark_judge`` and ``benchmark_user_simulator`` name the NON-SECRET
# settings (provider name, model name, base URL) of a benchmark's own scoring
# model and simulated user. They are model settings, but for a role the agent
# under test never plays -- ``ModelCapabilityRequirement.role`` already
# distinguishes ``agent | judge | user_simulator`` -- and the key itself still
# travels as a ``SecretRef``/``ResolvedSecret``, never as infra. The agent's own
# provider/model purposes remain structurally absent from this vocabulary.
InfraPurpose: TypeAlias = Literal[
    "benchmark_auth",
    "benchmark_compute",
    "benchmark_storage",
    "benchmark_judge",
    "benchmark_user_simulator",
    "artifact_storage",
]


class ScopedInfraValue(ProtocolModel):
    """Non-model infrastructure input with a closed, purpose-limited vocabulary."""

    name: StrictIdentifier
    purpose: InfraPurpose
    value: str = Field(min_length=1, max_length=4096)


def _normalized_absolute_root(value: str) -> str:
    # Normalized AND absolute, not merely absolute: the parent opens this
    # directory with O_DIRECTORY and then resolves artifact names against that
    # descriptor, so a path containing ".." would let the root itself denote a
    # different directory than the one the parent believes it authorized.
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        raise ValueError("artifact root must be normalized and absolute")
    return value


# The single definition of a parent-authorized artifact directory. Both the
# rescue descriptor and the live episode's reset carry one, and they must agree
# on what makes a root acceptable -- a second, slightly different validator is
# exactly how a confinement boundary develops a gap.
ArtifactRoot: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=4096),
    AfterValidator(_normalized_absolute_root),
]


class RescueDescriptor(ProtocolModel):
    schema_version: SchemaVersion
    selector: AdapterSelector
    handshake: Handshake
    episode_id: StrictIdentifier
    cleanup_plan: CleanupPlan
    secret_refs: tuple[SecretRef, ...] = Field(max_length=MAX_RESCUE_REFS)
    infra_values: tuple[ScopedInfraValue, ...] = Field(max_length=MAX_RESCUE_REFS)
    artifact_root: ArtifactRoot
    descriptor_id: Digest = ZERO_DIGEST

    @field_validator("secret_refs", "infra_values", mode="before")
    @classmethod
    def _freeze_refs(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

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


class BeginRescueParams(OperationParams):
    """Content pins plus the secrets a rescue worker needs to tear down.

    The descriptor itself stays secret-free (it is persisted to disk). A rescue
    worker enters at HANDSHAKEN having never run ``prepare``, so the parent's
    resolver re-resolves ``descriptor.secret_refs`` at rescue time and hands the
    values over here, on the private pipe, for the cleanup calls that follow.
    """

    descriptor: RescueDescriptor
    descriptor_id: Digest
    episode_id: StrictIdentifier
    cleanup_plan_id: Digest
    selector_digest: Digest
    handshake_digest: Digest
    secrets: tuple[ResolvedSecret, ...] = Field(default=(), max_length=MAX_RESCUE_REFS)

    @field_validator("secrets", mode="before")
    @classmethod
    def _freeze_secrets(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _bind_descriptor(self) -> "BeginRescueParams":
        if (
            self.descriptor_id != self.descriptor.descriptor_id
            or self.episode_id != self.descriptor.episode_id
            or self.cleanup_plan_id != self.descriptor.cleanup_plan.cleanup_plan_id
            or self.selector_digest
            != canonical_digest("adapter-rescue-selector-v1", self.descriptor.selector)
            or self.handshake_digest
            != canonical_digest("adapter-rescue-handshake-v1", self.descriptor.handshake)
        ):
            raise ValueError("rescue initialization pins differ from descriptor")
        return self


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
    """Begin the episode, and tell the worker where its frame bytes may go.

    WHY ``reset_start`` AND NOT ``prepare``. The worker's environment is built
    from a closed allowlist (``supervisor._ENV_ALLOW`` is locale and temp only),
    so a genuinely out-of-process adapter has no ambient way to learn the
    directory the parent will read frames from. The path therefore has to arrive
    as validated RPC input, and this is the call it belongs on:

    * ``prepare`` is contractually ALLOCATION-FREE -- it creates nothing, so it
      produces no observation and has no bytes to publish. Handing it a writable
      root would invite exactly the allocation the two-stage rescue persistence
      depends on it not doing, and would widen the window in which a resource
      exists that no persisted descriptor names.
    * ``reset_start`` is the side-effect boundary AND the first call that yields
      an observation: the parent calls ``observe`` immediately after it, and
      every later ``observe``/``execute`` runs in ``RUNNING``, which only
      ``reset_start`` can enter. So it is the single point that precedes every
      frame-producing call while still being the first one that needs the root.
    * ``RescueDescriptor`` keeps its own copy because rescue runs in a NEW
      process against a persisted file, with no live session to have been told.

    Confinement is not established by this field. The parent treats the root as
    the only directory it will ever open, and resolves each artifact by its
    content digest relative to that directory descriptor with ``O_NOFOLLOW``
    (``supervisor.verify_artifact``); the worker names bytes by digest and never
    supplies a path. Sending the root is what makes the worker able to write
    where the parent already looks -- it grants no new read authority.

    ``secrets`` (schema 1.2) rides this call for the same reason the root does:
    it is the side-effect boundary, the first call at which a provider needs a
    credential, and the one call that precedes every allocation. ``prepare``
    stays secret-free so a persisted plan can never have been produced with
    credentials in hand.
    """

    task_id: StrictIdentifier
    episode_id: StrictIdentifier
    artifact_root: ArtifactRoot
    secrets: tuple[ResolvedSecret, ...] = Field(default=(), max_length=MAX_RESCUE_REFS)

    @field_validator("secrets", mode="before")
    @classmethod
    def _freeze_secrets(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ObserveParams(ProtocolModel):
    episode_id: StrictIdentifier


class ObservationResult(ProtocolModel):
    observation: Observation


class ObservationPhaseError(Exception):
    """The mutation COMMITTED; only reading the resulting state failed.

    An adapter raises this from ``execute`` to state a fact only it can know:
    the guest actions were applied and the environment moved, and the failure
    happened afterwards, while reading the new state back. That distinction is
    the whole safety argument for recovery -- ``execute`` is otherwise treated
    as an AMBIGUOUS mutation, where a repeat call could apply the actions a
    second time, and that treatment must not be weakened.

    WHY A DECLARED CONTRACT AND NOT INFERENCE. The parent's only other evidence
    is ``RpcErrorDetail``: an exception type name, a message, and file/line
    frames. Keying recovery off any of those would hard-code one benchmark's
    private internals into a benchmark-neutral harness -- adapter exception
    names are adapter-owned, unversioned, and free to change in a patch
    release. The adapter is the code that ordered the two phases, so it is the
    only honest source of the fact, and it says so in a type the boundary
    defines. ``AdapterRescueUnsupported`` (worker.py) is the same pattern.

    Raising this WITHOUT having committed the mutation is a contract violation
    that can double-apply an action. It is safe to raise only from after the
    point of no return -- typically ``raise ObservationPhaseError(...) from
    error`` around the read-back that follows the guest call.

    An adapter that never raises it is completely unaffected: the phase
    defaults to ``unknown`` and every failure poisons exactly as before.
    """


class ExecuteParams(OperationParams):
    """One action batch to apply, or one committed batch's read-back to resume.

    ``resume_observation`` is set ONLY by the parent, and only after this exact
    batch already failed with ``phase == "observation"`` -- that is, after the
    adapter declared the mutation committed. It carries one obligation, which is
    the contract an adapter accepts by ever raising ``ObservationPhaseError``:

        The adapter MUST NOT apply ``action_batch`` again. It must only re-read
        the environment and return the observation the failed call could not
        build, as the exact next sequence after the parent's current one.

    The batch still rides along because the receipt binds to it
    (``validate_execution`` checks ``action_batch_id``, the input observation,
    and that the sequence is fresh), so the parent's verification of a resumed
    call is identical to that of a normal one -- nothing is relaxed to let
    recovery through.

    Each attempt carries a FRESH ``operation_id`` derived from the original.
    Reusing the original key would replay the worker's cached error verbatim,
    which is precisely what the operation cache is for and precisely wrong
    here: a resumed read is a new call whose outcome is not yet decided.
    """

    action_batch: ActionBatch
    action_batch_id: Digest
    resume_observation: bool = False

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


class CleanupOutcome(ProtocolModel):
    action_id: StrictIdentifier
    status: Literal["not_needed", "attempted", "succeeded", "failed"]
    evidence_code: StrictIdentifier
    duration_ms: SafeCount


class CleanupResult(ProtocolModel):
    """Adapter evidence primitives; only the parent may mint receipts."""

    outcomes: tuple[CleanupOutcome, ...] = Field(max_length=MAX_RECEIPTS)

    @field_validator("outcomes", mode="before")
    @classmethod
    def _freeze_outcomes(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class CloseParams(OperationParams):
    episode_id: StrictIdentifier | None = None


class AckResult(ProtocolModel):
    accepted: Literal[True] = True


AdapterParams: TypeAlias = (
    HelloParams
    | InspectRequirementsParams
    | BeginRescueParams
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
    "begin_rescue": BeginRescueParams,
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
    "begin_rescue": AckResult,
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


@runtime_checkable
class RescuableAdapter(Protocol):
    """An adapter that can accept the rescue handoff.

    Deliberately SEPARATE from ``EvaluationAdapter`` rather than a tenth method
    on it. ``EvaluationAdapter`` is ``runtime_checkable`` and gates every
    handshake in ``load_selected_adapter``, so adding ``begin_rescue`` there
    would refuse an otherwise valid adapter at hello -- breaking ordinary
    episodes for a capability only ever used during teardown, and doing so on
    an existing installed wheel. Rescue is genuinely optional at load time and
    mandatory only once a rescue is actually requested, which is exactly the
    shape of a second protocol the worker checks at that moment.

    ``begin_rescue`` is the only call carrying the descriptor's infra values
    together with freshly resolved secrets, so it is the sole opportunity a
    rescue worker (which enters at HANDSHAKEN and never runs prepare or
    reset_start) has to build a teardown provider.
    """

    async def begin_rescue(self, params: BeginRescueParams) -> AckResult: ...


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
