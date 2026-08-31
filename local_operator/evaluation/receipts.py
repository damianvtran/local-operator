"""Pure dependency, preflight, and budget contracts for evaluation episodes.

These models deliberately describe facts and authorities without discovering or
executing anything.  Adapters remain responsible for performing work, but a
cooperative adapter can require the receipts and permits defined here at every
side-effect boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timezone
from threading import RLock
from typing import Annotated, Any, Literal, Self, TypeAlias
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


@dataclass(slots=True)
class _AuthorityRecord:
    """Non-serializable process-local capability stored outside model instances."""

    kind: str
    lock: RLock = field(default_factory=RLock)
    consumed: bool = False
    lineage: Any = None
    receipts: tuple[Any, ...] = ()


_AUTHORITY_REGISTRY_LOCK = RLock()
_AUTHORITY_REGISTRY: dict[int, tuple[weakref.ReferenceType[AuthorityModel], _AuthorityRecord]] = {}


def _remove_authority(model_id: int, reference: weakref.ReferenceType[AuthorityModel]) -> None:
    """Drop only the exact dead reference, protecting a subsequently reused id."""

    with _AUTHORITY_REGISTRY_LOCK:
        current = _AUTHORITY_REGISTRY.get(model_id)
        if current is not None and current[0] is reference:
            del _AUTHORITY_REGISTRY[model_id]


def _register_authority(
    model: AuthorityModel,
    kind: str,
    *,
    lineage: Any = None,
    receipts: tuple[Any, ...] = (),
) -> _AuthorityRecord:
    model_id = id(model)

    def remove(reference: weakref.ReferenceType[AuthorityModel]) -> None:
        _remove_authority(model_id, reference)

    reference = weakref.ref(model, remove)
    record = _AuthorityRecord(kind=kind, lineage=lineage, receipts=receipts)
    with _AUTHORITY_REGISTRY_LOCK:
        _AUTHORITY_REGISTRY[model_id] = (reference, record)
    return record


def _lookup_authority(
    model: AuthorityModel,
    kind: str,
    *,
    allow_consumed: bool = False,
) -> _AuthorityRecord:
    with _AUTHORITY_REGISTRY_LOCK:
        current = _AUTHORITY_REGISTRY.get(id(model))
        if current is None or current[0]() is not model or current[1].kind != kind:
            raise ValueError("model lacks process-local authority")
        record = current[1]
    if record.consumed and not allow_consumed:
        raise ValueError("model lacks process-local authority")
    return record


def _authority_registry_size() -> int:
    with _AUTHORITY_REGISTRY_LOCK:
        return len(_AUTHORITY_REGISTRY)


class AuthorityModel(ProtocolModel):
    """Guard normal APIs while authority lives only in an identity registry.

    Cooperative callers cannot mint or duplicate capabilities through model
    APIs. Hostile Python mutating this private module registry is out of scope;
    durable cross-process authority belongs to a future evidence store.
    """

    def copy(
        self,
        *,
        include: Any = None,
        exclude: Any = None,
        update: dict[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        raise ValueError("authority models cannot be copied")

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        raise ValueError("authority models cannot be copied")

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        # model_construct intentionally skips validation. Permit pure evidence
        # construction, but discard every attempted private marker so the result
        # can never pass assert_authority or transition methods.
        public_values = {key: value for key, value in values.items() if not key.startswith("_")}
        return super().model_construct(_fields_set=_fields_set, **public_values)

    def __copy__(self) -> Self:
        raise TypeError("authority models cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        raise TypeError("authority models cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("authority models cannot be pickled")


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


TimezoneName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9_+-]+(?:/[A-Za-z0-9_+-]+)*$",
    ),
]


class ClockRequirement(_RequirementBase):
    kind: Literal["clock"] = "clock"
    timezone: TimezoneName
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
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("clock timezone must be a canonical IANA zone") from error
        if self.date is not None:
            try:
                parsed_date = Date.fromisoformat(self.date)
            except ValueError as error:
                raise ValueError("clock date must be a valid ISO calendar date") from error
            if parsed_date.isoformat() != self.date:
                raise ValueError("clock date must use canonical ISO syntax")
        if self.fixed_clock is not None:
            try:
                parsed_clock = datetime.fromisoformat(self.fixed_clock.replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError("fixed clock must be a valid UTC instant") from error
            canonical = parsed_clock.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if parsed_clock.microsecond or canonical != self.fixed_clock:
                raise ValueError("fixed clock must use canonical UTC second syntax")
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


def requirement_digest(requirement: Requirement) -> str:
    """Bind observations to the entire selected declaration, not only its label."""

    return _identity("requirement-v1", requirement.model_dump(mode="json"))


class PreflightReceipt(_MetadataModel):
    plan_id: Digest
    requirement_id: StrictIdentifier
    requirement_digest: Digest
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


def record_preflight(
    plan: DependencyPlan,
    requirement_id: StrictIdentifier,
    *,
    status: Literal["pass", "fail", "skip"],
    evidence: Mapping[str, JsonValue] | None = None,
    started_at_ms: int | None = None,
    ended_at_ms: int | None = None,
    duration_ms: int | None = None,
) -> PreflightReceipt:
    """Record evidence against the exact declaration selected by this plan."""

    requirement = next(
        (item for item in plan.requirements if item.requirement_id == requirement_id), None
    )
    if requirement is None:
        raise ValueError("preflight requirement is not selected by the dependency plan")
    return PreflightReceipt.model_validate(
        {
            "plan_id": plan.plan_id,
            "requirement_id": requirement_id,
            "requirement_digest": requirement_digest(requirement),
            "necessity": requirement.necessity,
            "status": status,
            "evidence": {} if evidence is None else dict(evidence),
            "started_at_ms": started_at_ms,
            "ended_at_ms": ended_at_ms,
            "duration_ms": duration_ms,
        }
    )


_PERCENT_ESCAPE_RE = re.compile(r"%([0-9A-Fa-f]{2})")


def _canonical_percent_escapes(value: str) -> str:
    """Normalize escape digits without changing literal plaintext case."""

    return _PERCENT_ESCAPE_RE.sub(lambda match: "%" + match.group(1).upper(), value)


class RedactionSet:
    """Ephemeral plaintext and common deterministic encodings checked at sealing.

    This bounded substring check is not a general secret scanner. It covers the
    resolved plaintext plus UTF-8 base64, URL-safe base64, percent-encoding, and
    hexadecimal variants generated here; transformations outside that set remain
    an adapter responsibility.
    """

    __slots__ = (
        "_exact_encoded_canaries",
        "_hex_canaries",
        "_percent_canaries",
        "_plaintext_canaries",
    )

    def __init__(
        self,
        plaintext_canaries: tuple[str, ...],
        exact_encoded_canaries: tuple[str, ...],
        percent_canaries: tuple[str, ...],
        hex_canaries: tuple[str, ...],
    ) -> None:
        self._plaintext_canaries = plaintext_canaries
        self._exact_encoded_canaries = exact_encoded_canaries
        self._percent_canaries = percent_canaries
        self._hex_canaries = hex_canaries

    @classmethod
    def from_resolved_values(cls, values: Iterable[str]) -> "RedactionSet":
        snapshot = tuple(values)
        if len(snapshot) > MAX_CANARIES:
            raise ValueError("too many redaction canaries")
        if any(not isinstance(value, str) or not value for value in snapshot):
            raise ValueError("redaction canaries must be non-empty strings")
        exact_encoded_variants: set[str] = set()
        percent_variants: set[str] = set()
        hex_variants: set[str] = set()
        for value in snapshot:
            raw = value.encode("utf-8")
            # Short encoded fragments collide too readily with ordinary evidence;
            # plaintext is still checked regardless of this conservative floor.
            if len(raw) < 8:
                continue
            standard = base64.b64encode(raw).decode("ascii")
            urlsafe = base64.urlsafe_b64encode(raw).decode("ascii")
            exact_encoded_variants.update(
                {standard, standard.rstrip("="), urlsafe, urlsafe.rstrip("=")}
            )
            # Escape DIGITS are case-insensitive, while unescaped literals are
            # not. Raw hexadecimal has no literal/plaintext distinction.
            percent_variants.add(_canonical_percent_escapes(quote(value, safe="")))
            hex_variants.add(raw.hex().casefold())
        return cls(
            tuple(sorted(set(snapshot), key=lambda value: (-len(value), value))),
            tuple(sorted(exact_encoded_variants, key=lambda value: (-len(value), value))),
            tuple(sorted(percent_variants, key=lambda value: (-len(value), value))),
            tuple(sorted(hex_variants, key=lambda value: (-len(value), value))),
        )

    def __repr__(self) -> str:
        return f"RedactionSet(count={len(self._plaintext_canaries)})"

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
            if (
                any(canary in candidate for canary in self._plaintext_canaries)
                or any(canary in candidate for canary in self._exact_encoded_canaries)
                or any(
                    canary in _canonical_percent_escapes(candidate)
                    for canary in self._percent_canaries
                )
                or any(canary in candidate.casefold() for canary in self._hex_canaries)
            ):
                # Never include the candidate or canary: validation errors often
                # cross process boundaries and become durable logs.
                raise ValueError("secret canary survived evidence redaction")


class SealedPreflight(AuthorityModel):
    """Factory-only attestation over the exact validated preflight receipts."""

    plan_id: Digest
    required_requirement_ids: tuple[StrictIdentifier, ...]
    passed_requirement_ids: tuple[StrictIdentifier, ...]
    failed_requirement_ids: tuple[StrictIdentifier, ...]
    skipped_requirement_ids: tuple[StrictIdentifier, ...]
    receipt_digests: tuple[Digest, ...]
    redaction_attested: Literal[True]
    seal_id: Digest

    @field_validator(
        "required_requirement_ids",
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
        if tuple(sorted(self.required_requirement_ids)) != self.required_requirement_ids:
            raise ValueError("sealed required requirement IDs must be canonical")
        if len(self.required_requirement_ids) != len(set(self.required_requirement_ids)):
            raise ValueError("sealed required requirement IDs must be unique")
        if not set(self.required_requirement_ids).issubset(flattened):
            raise ValueError("sealed required requirement IDs must have outcomes")
        if set(self.required_requirement_ids).intersection(self.failed_requirement_ids):
            raise ValueError("sealed preflight cannot contain required failures")
        if set(self.required_requirement_ids).intersection(self.skipped_requirement_ids):
            raise ValueError("sealed preflight cannot contain required skips")
        if tuple(sorted(self.receipt_digests)) != self.receipt_digests:
            raise ValueError("sealed receipt digests must be canonical")
        expected = _identity(
            "sealed-preflight-v1",
            self.model_dump(mode="json", exclude={"seal_id"}),
        )
        if self.seal_id != expected:
            raise ValueError("preflight seal identity does not match its receipts")
        return self

    def __copy__(self) -> "SealedPreflight":
        raise TypeError("preflight seal authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "SealedPreflight":
        raise TypeError("preflight seal authority cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("preflight seal authority cannot be pickled")

    def assert_authority(self) -> None:
        try:
            record = _lookup_authority(self, "preflight-seal")
        except ValueError as error:
            raise ValueError("preflight seal lacks factory authority") from error
        receipts = record.receipts
        actual_digests: list[str] = []
        for receipt in receipts:
            expected = _identity(
                "preflight-receipt-v1",
                receipt.model_dump(mode="json", exclude={"receipt_id"}),
            )
            if receipt.receipt_id != expected:
                raise ValueError("preflight receipt authority was mutated")
            actual_digests.append(receipt.receipt_id)
        if tuple(sorted(actual_digests)) != self.receipt_digests:
            raise ValueError("preflight seal receipt evidence does not match")
        expected_seal = _identity(
            "sealed-preflight-v1",
            self.model_dump(mode="json", exclude={"seal_id"}),
        )
        if self.seal_id != expected_seal:
            raise ValueError("preflight seal authority was mutated")

    @property
    def successful(self) -> bool:
        # Required failures and skips are structurally forbidden. Optional
        # failures remain visible without blocking cooperative adapters.
        return True


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
        if receipt.plan_id != plan.plan_id:
            raise ValueError("preflight receipt belongs to another dependency plan")
        if receipt.requirement_digest != requirement_digest(requirement):
            raise ValueError("preflight receipt belongs to another requirement declaration")
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
        "required_requirement_ids": tuple(
            sorted(
                item.requirement_id for item in plan.requirements if item.necessity == "required"
            )
        ),
        "passed_requirement_ids": passed,
        "failed_requirement_ids": failed,
        "skipped_requirement_ids": skipped,
        "receipt_digests": tuple(sorted(receipt.receipt_id for receipt in snapshot)),
        "redaction_attested": True,
    }
    redactions.assert_clear(payload)
    sealed = SealedPreflight.model_validate(
        {**payload, "seal_id": _identity("sealed-preflight-v1", payload)}
    )
    _register_authority(sealed, "preflight-seal", receipts=snapshot)
    return sealed


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
    reservation_key: StrictIdentifier
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


def _validate_reservations(
    authorization: BudgetAuthorization,
    reservations: Sequence[BudgetReservation],
) -> dict[BudgetResource, int]:
    totals: dict[BudgetResource, int] = {resource: 0 for resource in BUDGET_RESOURCES}
    keys: set[str] = set()
    for reservation in reservations:
        if (
            reservation.budget_id != authorization.budget_id
            or reservation.episode_id != authorization.episode_id
        ):
            raise ValueError("reservation belongs to another budget authorization")
        if reservation.reservation_key in keys:
            # Keys are operation-level idempotency identities. Reusing one is an
            # ambiguous retry, so callers must retrieve the first reservation.
            raise ValueError("duplicate reservation key")
        keys.add(reservation.reservation_key)
        for amount in reservation.amounts:
            totals[amount.resource] += amount.value
    for resource, total in totals.items():
        allowance = authorization.allowance_for(resource)
        if isinstance(allowance, CappedAllowance) and total > allowance.value:
            raise ValueError("reservation exceeds an authorized cap")
    return totals


def reserve_budget(
    authorization: BudgetAuthorization,
    reservation_key: StrictIdentifier,
    request: Sequence[ResourceAmount],
    existing: Sequence[BudgetReservation] = (),
) -> BudgetReservation:
    """Reserve capacity before work; unique keys distinguish equal operations."""

    requested = tuple(request)
    if not requested:
        raise ValueError("reservation request must not be empty")
    prior = tuple(existing)
    _validate_reservations(authorization, prior)
    if any(item.reservation_key == reservation_key for item in prior):
        raise ValueError("duplicate reservation key")
    seen: set[BudgetResource] = set()
    for amount in requested:
        if amount.resource in seen:
            raise ValueError("reservation request contains duplicate resources")
        seen.add(amount.resource)
    reservation = BudgetReservation(
        episode_id=authorization.episode_id,
        budget_id=authorization.budget_id,
        reservation_key=reservation_key,
        amounts=requested,
    )
    _validate_reservations(authorization, (*prior, reservation))
    return reservation


class BudgetCommitment(AuthorityModel):
    """Factory-only authority over the complete validated reservation set."""

    episode_id: StrictIdentifier
    budget_id: Digest
    authorization_digest: Digest
    reservation_ids: tuple[Digest, ...]
    reserved: tuple[ResourceAmount, ...]
    commitment_id: Digest

    @field_validator("reservation_ids", "reserved", mode="before")
    @classmethod
    def _freeze_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_commitment(self) -> "BudgetCommitment":
        resources = tuple(item.resource for item in self.reserved)
        if resources != tuple(sorted(BUDGET_RESOURCES)):
            raise ValueError("budget commitment requires every resource exactly once")
        if self.reservation_ids != tuple(sorted(self.reservation_ids)):
            raise ValueError("budget commitment reservation IDs must be canonical")
        expected = _identity(
            "budget-commitment-v1",
            self.model_dump(mode="json", exclude={"commitment_id"}),
        )
        if self.commitment_id != expected:
            raise ValueError("budget commitment identity does not match reservations")
        return self

    def __copy__(self) -> "BudgetCommitment":
        raise TypeError("budget commitment authority cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "BudgetCommitment":
        raise TypeError("budget commitment authority cannot be copied")

    def __reduce__(self) -> Any:
        raise TypeError("budget commitment authority cannot be pickled")

    def assert_authority(self, authorization: BudgetAuthorization) -> None:
        try:
            _lookup_authority(self, "budget-commitment")
        except ValueError as error:
            raise ValueError("budget commitment lacks factory authority") from error
        if (
            self.episode_id != authorization.episode_id
            or self.budget_id != authorization.budget_id
            or self.authorization_digest != authorization.budget_id
        ):
            raise ValueError("budget commitment belongs to another authorization")
        expected = _identity(
            "budget-commitment-v1",
            self.model_dump(mode="json", exclude={"commitment_id"}),
        )
        if self.commitment_id != expected:
            raise ValueError("budget commitment authority was mutated")


def commit_budget(
    authorization: BudgetAuthorization,
    reservations: Sequence[BudgetReservation],
) -> BudgetCommitment:
    """Seal the exact reservation set cooperative adapters may allocate against."""

    snapshot = tuple(reservations)
    totals = _validate_reservations(authorization, snapshot)
    payload = {
        "episode_id": authorization.episode_id,
        "budget_id": authorization.budget_id,
        "authorization_digest": authorization.budget_id,
        "reservation_ids": tuple(sorted(item.reservation_id for item in snapshot)),
        "reserved": tuple(
            ResourceAmount(resource=resource, value=totals[resource]).model_dump(mode="json")
            for resource in sorted(BUDGET_RESOURCES)
        ),
    }
    commitment = BudgetCommitment.model_validate(
        {**payload, "commitment_id": _identity("budget-commitment-v1", payload)}
    )
    _register_authority(commitment, "budget-commitment")
    return commitment


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
    authorization_digest: Digest
    authorization: BudgetAuthorization
    commitment_id: Digest
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
        if self.authorization.episode_id != self.episode_id:
            raise ValueError("reconciliation authorization belongs to another episode")
        if self.authorization.budget_id != self.budget_id:
            raise ValueError("reconciliation budget identity disagrees with authorization")
        if self.authorization_digest != self.authorization.budget_id:
            raise ValueError("reconciliation authorization digest is invalid")
        commitment_payload = {
            "episode_id": self.episode_id,
            "budget_id": self.budget_id,
            "authorization_digest": self.authorization_digest,
            "reservation_ids": reservations,
            "reserved": tuple(
                ResourceAmount(resource=entry.resource, value=entry.reserved).model_dump(
                    mode="json"
                )
                for entry in entries
            ),
        }
        if self.commitment_id != _identity("budget-commitment-v1", commitment_payload):
            raise ValueError("reconciliation commitment does not match reservations")
        for entry in entries:
            if entry.allowance != self.authorization.allowance_for(entry.resource):
                raise ValueError("reconciliation allowance disagrees with authorization")
            expected_overrun = 0
            if isinstance(entry.usage, AvailableUsage) and isinstance(
                entry.allowance, CappedAllowance
            ):
                expected_overrun = max(0, entry.usage.value - entry.allowance.value)
            if entry.overrun != expected_overrun:
                raise ValueError("reconciliation overrun disagrees with actual usage")
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
    totals = _validate_reservations(authorization, reservation_snapshot)
    commitment = commit_budget(authorization, reservation_snapshot)
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
                reserved=totals[resource],
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
        authorization_digest=authorization.budget_id,
        authorization=authorization,
        commitment_id=commitment.commitment_id,
        reservation_ids=tuple(item.reservation_id for item in reservation_snapshot),
        entries=tuple(entries),
        reportable=reportable,
    )
