"""Pure dependency, preflight, and budget contracts for evaluation episodes.

These models deliberately describe facts and authorities without discovering or
executing anything.  Adapters remain responsible for performing work, but a
cooperative adapter can require the receipts and permits defined here at every
side-effect boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    Field,
    JsonValue,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from local_operator.evaluation.protocol import (
    MAX_METADATA_BYTES,
    MAX_SAFE_JSON_INTEGER,
    ArtifactRef,
    FrozenMapping,
    PortableMetadataObject,
    ProtocolModel,
)

MAX_DECLARATIONS = 256
MAX_PORTS = 32
MAX_MODALITIES = 16
MAX_TOOLS = 128
MAX_REASON_LENGTH = 2_000
MAX_CANARIES = 256
ZERO_DIGEST = "0" * 64

StrictIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SafeCount = Annotated[int, Field(ge=0, le=MAX_SAFE_JSON_INTEGER)]
PositiveSafeCount = Annotated[int, Field(ge=1, le=MAX_SAFE_JSON_INTEGER)]
RequirementNecessity = Literal["required", "optional"]
ReportabilityClass = Literal["required", "optional"]


def _thaw(value: Any) -> JsonValue:
    if isinstance(value, FrozenMapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


class _MetadataModel(ProtocolModel):
    """Apply protocol metadata bounds to every declarative metadata field."""

    @field_validator("metadata", "evidence", mode="before", check_fields=False)
    @classmethod
    def _snapshot_metadata(cls, value: Any) -> dict[str, JsonValue]:
        if not isinstance(value, (FrozenMapping, Mapping)):
            raise ValueError("portable metadata must be a mapping")
        thawed = _thaw(value if isinstance(value, FrozenMapping) else FrozenMapping(value))
        assert isinstance(thawed, dict)
        return thawed

    @field_validator("metadata", "evidence", check_fields=False)
    @classmethod
    def _freeze_metadata(cls, value: Mapping[str, JsonValue]) -> FrozenMapping:
        frozen = FrozenMapping(value)
        encoded = json.dumps(
            _thaw(frozen),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > MAX_METADATA_BYTES:
            raise ValueError(f"portable metadata exceeds {MAX_METADATA_BYTES} canonical bytes")
        return frozen

    @field_serializer("metadata", "evidence", check_fields=False)
    def _serialize_metadata(self, value: FrozenMapping) -> dict[str, JsonValue]:
        thawed = _thaw(value)
        assert isinstance(thawed, dict)
        return thawed


class DisplayRequirement(ProtocolModel):
    """Display geometry and symbolic platform capability, never an instance SKU."""

    native_width: PositiveSafeCount
    native_height: PositiveSafeCount
    model_width: PositiveSafeCount
    model_height: PositiveSafeCount
    platform_capability: StrictIdentifier


class _RequirementBase(_MetadataModel):
    requirement_id: StrictIdentifier
    necessity: RequirementNecessity
    reportability: ReportabilityClass
    metadata: PortableMetadataObject = Field(default_factory=dict, validate_default=True)

    def conflict_key(self) -> tuple[str, ...]:
        raise NotImplementedError


class ComputeRequirement(_RequirementBase):
    kind: Literal["compute"] = "compute"
    cpu_class: StrictIdentifier
    memory_class: StrictIdentifier
    disk_bytes: SafeCount
    display: DisplayRequirement | None = None

    def conflict_key(self) -> tuple[str, ...]:
        return (self.kind, "episode-compute")


class NetworkRequirement(_RequirementBase):
    kind: Literal["network"] = "network"
    endpoint_id: StrictIdentifier
    service_id: StrictIdentifier
    protocol: Literal["http", "https", "tcp", "udp", "websocket"]
    ports: tuple[Annotated[int, Field(ge=1, le=65535)], ...] = Field(
        min_length=1, max_length=MAX_PORTS
    )
    proxy_capability: Literal["forbidden", "allowed", "required"]
    geography: StrictIdentifier | None = None

    @field_validator("ports", mode="before")
    @classmethod
    def _freeze_ports(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _unique_ports(self) -> "NetworkRequirement":
        if len(self.ports) != len(set(self.ports)):
            raise ValueError("network ports must be unique")
        return self

    def conflict_key(self) -> tuple[str, ...]:
        return (self.kind, self.service_id, self.endpoint_id)


class ExternalServiceRequirement(_RequirementBase):
    kind: Literal["external_service"] = "external_service"
    service_id: StrictIdentifier
    capability: StrictIdentifier
    # This is a lookup key only. Resolved credentials must never enter a model.
    account_ref: StrictIdentifier

    def conflict_key(self) -> tuple[str, ...]:
        return (self.kind, self.service_id, self.capability)


class ModelCapabilityRequirement(_RequirementBase):
    kind: Literal["model_capability"] = "model_capability"
    role: Literal["agent", "judge", "user_simulator"]
    modalities: tuple[StrictIdentifier, ...] = Field(min_length=1, max_length=MAX_MODALITIES)
    tools: tuple[StrictIdentifier, ...] = Field(default=(), max_length=MAX_TOOLS)
    min_context_tokens: SafeCount
    min_output_tokens: SafeCount
    route_pin: StrictIdentifier | None = None
    fallback_policy: Literal["forbid", "same_capability", "explicit_routes"]
    fallback_routes: tuple[StrictIdentifier, ...] = Field(default=(), max_length=16)

    @field_validator("modalities", "tools", "fallback_routes", mode="before")
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_routes(self) -> "ModelCapabilityRequirement":
        for name, values in (
            ("modalities", self.modalities),
            ("tools", self.tools),
            ("fallback routes", self.fallback_routes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if self.fallback_policy == "explicit_routes" and not self.fallback_routes:
            raise ValueError("explicit fallback policy requires fallback routes")
        if self.fallback_policy != "explicit_routes" and self.fallback_routes:
            raise ValueError("fallback routes require the explicit_routes policy")
        return self

    def conflict_key(self) -> tuple[str, ...]:
        return (self.kind, self.role)


class ClockRequirement(_RequirementBase):
    kind: Literal["clock"] = "clock"
    timezone: StrictIdentifier
    date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    fixed_clock: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
    )

    @model_validator(mode="after")
    def _require_clock_constraint(self) -> "ClockRequirement":
        if self.date is None and self.fixed_clock is None:
            raise ValueError("clock requirement needs a date or fixed clock")
        if self.date is not None and self.fixed_clock is not None:
            raise ValueError("clock requirement chooses either date or fixed clock")
        return self

    def conflict_key(self) -> tuple[str, ...]:
        return (self.kind, "episode-clock")


class PinnedInputRequirement(_RequirementBase):
    kind: Literal["pinned_input"] = "pinned_input"
    release_id: StrictIdentifier
    artifact: ArtifactRef
    content_sha256: Digest

    def conflict_key(self) -> tuple[str, ...]:
        return (self.kind, self.release_id)


Requirement: TypeAlias = Annotated[
    ComputeRequirement
    | NetworkRequirement
    | ExternalServiceRequirement
    | ModelCapabilityRequirement
    | ClockRequirement
    | PinnedInputRequirement,
    Field(discriminator="kind"),
]
_REQUIREMENT_ADAPTER = TypeAdapter(Requirement)


def _identity(kind: str, payload: Any) -> str:
    encoded = json.dumps(
        {"identity_kind": kind, "payload": payload},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DependencyPlan(ProtocolModel):
    """Canonical selected episode dependencies with an order-independent identity."""

    release_id: StrictIdentifier
    task_id: StrictIdentifier
    attempt_id: StrictIdentifier
    requirements: tuple[Requirement, ...] = Field(min_length=1, max_length=MAX_DECLARATIONS)
    plan_id: Digest = ZERO_DIGEST

    @field_validator("requirements", mode="before")
    @classmethod
    def _parse_requirements(cls, value: Any) -> Any:
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple):
            return value
        return tuple(_REQUIREMENT_ADAPTER.validate_python(item, strict=True) for item in value)

    @model_validator(mode="after")
    def _canonicalize_and_identify(self) -> "DependencyPlan":
        ordered = tuple(sorted(self.requirements, key=lambda item: item.requirement_id))
        ids = [item.requirement_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("dependency plan contains duplicate requirement IDs")
        seen_targets: dict[tuple[str, ...], Requirement] = {}
        for item in ordered:
            target = item.conflict_key()
            if target in seen_targets:
                raise ValueError("dependency plan contains conflicting requirements")
            seen_targets[target] = item
        object.__setattr__(self, "requirements", ordered)
        payload = {
            "release_id": self.release_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "requirements": [item.model_dump(mode="json") for item in ordered],
        }
        expected = _identity("dependency-plan-v1", payload)
        if self.plan_id not in (ZERO_DIGEST, expected):
            raise ValueError("dependency plan identity does not match its declarations")
        object.__setattr__(self, "plan_id", expected)
        return self


class PreflightReceipt(_MetadataModel):
    requirement_id: StrictIdentifier
    necessity: RequirementNecessity
    status: Literal["pass", "fail", "skip"]
    evidence: PortableMetadataObject = Field(default_factory=dict, validate_default=True)
    started_at_ms: SafeCount | None = None
    ended_at_ms: SafeCount | None = None
    duration_ms: SafeCount | None = None
    receipt_id: Digest = ZERO_DIGEST

    @model_validator(mode="after")
    def _validate_timing_and_identity(self) -> "PreflightReceipt":
        if self.status == "skip" and self.necessity != "optional":
            raise ValueError("only optional requirements may be skipped")
        has_range = self.started_at_ms is not None or self.ended_at_ms is not None
        if has_range:
            if self.started_at_ms is None or self.ended_at_ms is None:
                raise ValueError("preflight timing range requires both start and end")
            if self.ended_at_ms < self.started_at_ms:
                raise ValueError("preflight end must not precede start")
            elapsed = self.ended_at_ms - self.started_at_ms
            if self.duration_ms is not None and self.duration_ms != elapsed:
                raise ValueError("preflight duration disagrees with start/end")
        elif self.duration_ms is None:
            raise ValueError("preflight receipt requires a duration or start/end range")
        payload = self.model_dump(mode="json", exclude={"receipt_id"})
        expected = _identity("preflight-receipt-v1", payload)
        if self.receipt_id not in (ZERO_DIGEST, expected):
            raise ValueError("preflight receipt identity does not match its evidence")
        object.__setattr__(self, "receipt_id", expected)
        return self


class RedactionSet:
    """Ephemeral resolved canaries used only while sealing portable evidence."""

    __slots__ = ("_canaries",)

    def __init__(self, canaries: tuple[str, ...]) -> None:
        self._canaries = canaries

    @classmethod
    def from_resolved_values(cls, values: Iterable[str]) -> "RedactionSet":
        snapshot = tuple(values)
        if len(snapshot) > MAX_CANARIES:
            raise ValueError("too many redaction canaries")
        if any(not isinstance(value, str) or not value for value in snapshot):
            raise ValueError("redaction canaries must be non-empty strings")
        return cls(tuple(sorted(set(snapshot), key=lambda value: (-len(value), value))))

    def __repr__(self) -> str:
        return f"RedactionSet(count={len(self._canaries)})"

    def assert_clear(self, value: Any) -> None:
        def strings(item: Any) -> Iterable[str]:
            if isinstance(item, str):
                yield item
            elif isinstance(item, Mapping):
                for key, nested in item.items():
                    yield str(key)
                    yield from strings(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    yield from strings(nested)

        for candidate in strings(value):
            if any(canary in candidate for canary in self._canaries):
                # Never include the candidate or canary: validation errors often
                # cross process boundaries and become durable logs.
                raise ValueError("secret canary survived evidence redaction")


class SealedPreflight(ProtocolModel):
    plan_id: Digest
    passed_requirement_ids: tuple[StrictIdentifier, ...]
    failed_requirement_ids: tuple[StrictIdentifier, ...]
    skipped_requirement_ids: tuple[StrictIdentifier, ...]
    receipt_digests: tuple[Digest, ...]
    redaction_attested: Literal[True]
    seal_id: Digest

    @field_validator(
        "passed_requirement_ids",
        "failed_requirement_ids",
        "skipped_requirement_ids",
        "receipt_digests",
        mode="before",
    )
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_seal(self) -> "SealedPreflight":
        groups = (
            self.passed_requirement_ids,
            self.failed_requirement_ids,
            self.skipped_requirement_ids,
        )
        flattened = [item for group in groups for item in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("sealed requirement outcome IDs must be unique")
        if any(tuple(sorted(group)) != group for group in groups):
            raise ValueError("sealed requirement outcome IDs must be canonical")
        if tuple(sorted(self.receipt_digests)) != self.receipt_digests:
            raise ValueError("sealed receipt digests must be canonical")
        expected = _identity(
            "sealed-preflight-v1",
            self.model_dump(mode="json", exclude={"seal_id"}),
        )
        if self.seal_id != expected:
            raise ValueError("preflight seal identity does not match its receipts")
        return self

    @property
    def successful(self) -> bool:
        return not self.failed_requirement_ids


def seal_preflight(
    plan: DependencyPlan,
    receipts: Sequence[PreflightReceipt],
    redactions: RedactionSet,
) -> SealedPreflight:
    """Validate complete exact evidence and attest that resolved canaries are absent."""

    snapshot = tuple(receipts)
    by_id: dict[str, PreflightReceipt] = {}
    for receipt in snapshot:
        if receipt.requirement_id in by_id:
            raise ValueError("preflight has duplicate receipts")
        by_id[receipt.requirement_id] = receipt
    expected_ids = {item.requirement_id for item in plan.requirements}
    if set(by_id) != expected_ids:
        raise ValueError("preflight requires exactly one receipt for every selected requirement")
    requirements = {item.requirement_id: item for item in plan.requirements}
    for requirement_id, receipt in by_id.items():
        requirement = requirements[requirement_id]
        if receipt.necessity != requirement.necessity:
            raise ValueError("preflight receipt necessity disagrees with its requirement")
        if receipt.status == "skip" and requirement.necessity != "optional":
            raise ValueError("required dependencies cannot be skipped")
    redactions.assert_clear(plan.model_dump(mode="json"))
    redactions.assert_clear([receipt.model_dump(mode="json") for receipt in snapshot])
    passed = tuple(sorted(key for key, value in by_id.items() if value.status == "pass"))
    failed = tuple(sorted(key for key, value in by_id.items() if value.status == "fail"))
    skipped = tuple(sorted(key for key, value in by_id.items() if value.status == "skip"))
    required_failures = [key for key in failed if requirements[key].necessity == "required"]
    if required_failures:
        raise ValueError("required dependency failure prevents preflight sealing")
    payload = {
        "plan_id": plan.plan_id,
        "passed_requirement_ids": passed,
        "failed_requirement_ids": failed,
        "skipped_requirement_ids": skipped,
        "receipt_digests": tuple(sorted(receipt.receipt_id for receipt in snapshot)),
        "redaction_attested": True,
    }
    redactions.assert_clear(payload)
    return SealedPreflight(
        **payload,
        seal_id=_identity("sealed-preflight-v1", payload),
    )


BudgetResource = Literal[
    "provider_input_tokens",
    "provider_output_tokens",
    "provider_cache_tokens",
    "provider_usd_micros",
    "cloud_usd_micros",
    "instance_milliseconds",
    "wall_milliseconds",
    "model_cycles",
    "guest_actions",
    "user_simulator_turns",
]
BUDGET_RESOURCES: tuple[BudgetResource, ...] = (
    "provider_input_tokens",
    "provider_output_tokens",
    "provider_cache_tokens",
    "provider_usd_micros",
    "cloud_usd_micros",
    "instance_milliseconds",
    "wall_milliseconds",
    "model_cycles",
    "guest_actions",
    "user_simulator_turns",
)


class CappedAllowance(ProtocolModel):
    kind: Literal["capped"] = "capped"
    resource: BudgetResource
    value: SafeCount
    reporting: ReportabilityClass


class UncappedAllowance(ProtocolModel):
    kind: Literal["uncapped"] = "uncapped"
    resource: BudgetResource
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH, pattern=r"\S")
    authorized_by: StrictIdentifier
    authorized_at_ms: SafeCount
    reporting: ReportabilityClass


Allowance: TypeAlias = Annotated[CappedAllowance | UncappedAllowance, Field(discriminator="kind")]
_ALLOWANCE_ADAPTER = TypeAdapter(Allowance)


class BudgetAuthorization(ProtocolModel):
    episode_id: StrictIdentifier
    allowances: tuple[Allowance, ...] = Field(
        min_length=len(BUDGET_RESOURCES), max_length=len(BUDGET_RESOURCES)
    )
    budget_id: Digest = ZERO_DIGEST

    @field_validator("allowances", mode="before")
    @classmethod
    def _parse_allowances(cls, value: Any) -> Any:
        if isinstance(value, list):
            value = tuple(value)
        if not isinstance(value, tuple):
            return value
        return tuple(_ALLOWANCE_ADAPTER.validate_python(item, strict=True) for item in value)

    @model_validator(mode="after")
    def _canonicalize_and_identify(self) -> "BudgetAuthorization":
        ordered = tuple(sorted(self.allowances, key=lambda item: item.resource))
        resources = [item.resource for item in ordered]
        if set(resources) != set(BUDGET_RESOURCES) or len(resources) != len(set(resources)):
            raise ValueError("budget authorization requires each resource exactly once")
        object.__setattr__(self, "allowances", ordered)
        payload = {
            "episode_id": self.episode_id,
            "allowances": [item.model_dump(mode="json") for item in ordered],
        }
        expected = _identity("budget-authorization-v1", payload)
        if self.budget_id not in (ZERO_DIGEST, expected):
            raise ValueError("budget identity does not match its allowances")
        object.__setattr__(self, "budget_id", expected)
        return self

    def allowance_for(self, resource: BudgetResource) -> Allowance:
        return next(item for item in self.allowances if item.resource == resource)


class ResourceAmount(ProtocolModel):
    resource: BudgetResource
    value: SafeCount


class BudgetReservation(ProtocolModel):
    episode_id: StrictIdentifier
    budget_id: Digest
    amounts: tuple[ResourceAmount, ...] = Field(min_length=1, max_length=len(BUDGET_RESOURCES))
    reservation_id: Digest = ZERO_DIGEST

    @field_validator("amounts", mode="before")
    @classmethod
    def _freeze_amounts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _canonicalize_and_identify(self) -> "BudgetReservation":
        ordered = tuple(sorted(self.amounts, key=lambda item: item.resource))
        resources = [item.resource for item in ordered]
        if len(resources) != len(set(resources)):
            raise ValueError("reservation contains duplicate resources")
        object.__setattr__(self, "amounts", ordered)
        payload = self.model_dump(mode="json", exclude={"reservation_id"})
        expected = _identity("budget-reservation-v1", payload)
        if self.reservation_id not in (ZERO_DIGEST, expected):
            raise ValueError("reservation identity does not match its amounts")
        object.__setattr__(self, "reservation_id", expected)
        return self


def reserve_budget(
    authorization: BudgetAuthorization,
    request: Sequence[ResourceAmount],
    existing: Sequence[BudgetReservation] = (),
) -> BudgetReservation:
    """Reserve capacity before work; a zero cap remains distinct from uncapped."""

    requested = tuple(request)
    if not requested:
        raise ValueError("reservation request must not be empty")
    totals: dict[BudgetResource, int] = {resource: 0 for resource in BUDGET_RESOURCES}
    for reservation in existing:
        if reservation.budget_id != authorization.budget_id:
            raise ValueError("existing reservation belongs to another budget")
        for amount in reservation.amounts:
            totals[amount.resource] += amount.value
    seen: set[BudgetResource] = set()
    for amount in requested:
        if amount.resource in seen:
            raise ValueError("reservation request contains duplicate resources")
        seen.add(amount.resource)
        totals[amount.resource] += amount.value
    for resource, total in totals.items():
        allowance = authorization.allowance_for(resource)
        if isinstance(allowance, CappedAllowance) and total > allowance.value:
            raise ValueError("reservation exceeds an authorized cap")
    return BudgetReservation(
        episode_id=authorization.episode_id,
        budget_id=authorization.budget_id,
        amounts=requested,
    )


class AvailableUsage(ProtocolModel):
    kind: Literal["available"] = "available"
    resource: BudgetResource
    value: SafeCount


class UnavailableUsage(ProtocolModel):
    kind: Literal["unavailable"] = "unavailable"
    resource: BudgetResource
    reason: str = Field(min_length=1, max_length=MAX_REASON_LENGTH, pattern=r"\S")


Usage: TypeAlias = Annotated[AvailableUsage | UnavailableUsage, Field(discriminator="kind")]
_USAGE_ADAPTER = TypeAdapter(Usage)


class ReconciliationEntry(ProtocolModel):
    resource: BudgetResource
    allowance: Allowance
    reserved: SafeCount
    usage: Usage
    overrun: SafeCount

    @model_validator(mode="after")
    def _same_resource(self) -> "ReconciliationEntry":
        if self.allowance.resource != self.resource or self.usage.resource != self.resource:
            raise ValueError("reconciliation entry resources disagree")
        return self


class BudgetReconciliation(ProtocolModel):
    episode_id: StrictIdentifier
    budget_id: Digest
    reservation_ids: tuple[Digest, ...]
    entries: tuple[ReconciliationEntry, ...]
    reportable: bool
    reconciliation_id: Digest = ZERO_DIGEST

    @field_validator("reservation_ids", "entries", mode="before")
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _canonicalize_and_identify(self) -> "BudgetReconciliation":
        entries = tuple(sorted(self.entries, key=lambda item: item.resource))
        if tuple(item.resource for item in entries) != tuple(sorted(BUDGET_RESOURCES)):
            raise ValueError("reconciliation requires every resource exactly once")
        reservations = tuple(sorted(self.reservation_ids))
        if len(reservations) != len(set(reservations)):
            raise ValueError("reconciliation contains duplicate reservations")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "reservation_ids", reservations)
        expected_reportable = all(
            not (
                entry.allowance.reporting == "required"
                and isinstance(entry.usage, UnavailableUsage)
            )
            for entry in entries
        )
        if self.reportable != expected_reportable:
            raise ValueError("reconciliation reportability disagrees with required usage")
        payload = self.model_dump(mode="json", exclude={"reconciliation_id"})
        expected = _identity("budget-reconciliation-v1", payload)
        if self.reconciliation_id not in (ZERO_DIGEST, expected):
            raise ValueError("reconciliation identity does not match usage")
        object.__setattr__(self, "reconciliation_id", expected)
        return self


def reconcile_budget(
    authorization: BudgetAuthorization,
    reservations: Sequence[BudgetReservation],
    usage: Sequence[Usage],
) -> BudgetReconciliation:
    """Record exact/under/over usage; integer USD micros avoid floating ambiguity."""

    reservation_snapshot = tuple(reservations)
    reserved: dict[BudgetResource, int] = {resource: 0 for resource in BUDGET_RESOURCES}
    for item in reservation_snapshot:
        if item.budget_id != authorization.budget_id:
            raise ValueError("reservation belongs to another budget")
        for amount in item.amounts:
            reserved[amount.resource] += amount.value
    parsed_usage = tuple(_USAGE_ADAPTER.validate_python(item, strict=True) for item in usage)
    by_resource = {item.resource: item for item in parsed_usage}
    if len(by_resource) != len(parsed_usage) or set(by_resource) != set(BUDGET_RESOURCES):
        raise ValueError("reconciliation requires one usage result per resource")
    entries: list[ReconciliationEntry] = []
    for resource in BUDGET_RESOURCES:
        allowance = authorization.allowance_for(resource)
        actual = by_resource[resource]
        overrun = 0
        if isinstance(actual, AvailableUsage) and isinstance(allowance, CappedAllowance):
            overrun = max(0, actual.value - allowance.value)
        entries.append(
            ReconciliationEntry(
                resource=resource,
                allowance=allowance,
                reserved=reserved[resource],
                usage=actual,
                overrun=overrun,
            )
        )
    reportable = all(
        not (entry.allowance.reporting == "required" and isinstance(entry.usage, UnavailableUsage))
        for entry in entries
    )
    return BudgetReconciliation(
        episode_id=authorization.episode_id,
        budget_id=authorization.budget_id,
        reservation_ids=tuple(item.reservation_id for item in reservation_snapshot),
        entries=tuple(entries),
        reportable=reportable,
    )
