"""Strict, benchmark-neutral observations and structured computer actions.

The protocol carries references to frame bytes rather than embedding them. This
keeps a future adapter RPC small and prevents transport details (filesystem
paths, object-store URLs, or inline base64) from becoming model-controlled
capabilities. Every coordinate is interpreted against geometry captured in the
observation; callers must never substitute the dimensions of a current display.
Metadata intentionally uses a portable JSON subset: strings, booleans, null,
and signed integers exactly representable by IEEE-754 (up to 2^53 - 1). Floats
are excluded so Python and future non-Python adapters produce identical bytes
without depending on an RFC 8785 implementation. Canonical JSON is UTF-8 with
literal Unicode, sorted ASCII object keys, compact ``,:`` separators, standard
JSON control/quote/backslash escaping, and unescaped slashes.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType

PROTOCOL_VERSION = "1.0"
MAX_IDENTIFIER_LENGTH = 256
MAX_TEXT_LENGTH = 100_000
MAX_METADATA_BYTES = 64_000
MAX_METADATA_DEPTH = 16
MAX_SAFE_JSON_INTEGER = 2**53 - 1
MAX_ENVELOPE_BYTES = 16 * 1024 * 1024
MAX_BATCH_SIZE = 64
MAX_DIMENSION = 1_000_000
MAX_COORDINATE = 1_000_000
MAX_SCROLL_DELTA = 100_000
MAX_WAIT_MS = 60_000

Identifier = Annotated[
    str,
    Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, pattern=r"\S"),
]
Coordinate = Annotated[int, Field(ge=0, le=MAX_COORDINATE)]
MouseButton = Literal["left", "middle", "right"]
_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_METADATA_DESCRIPTION = (
    f"Portable metadata; runtime maximum depth {MAX_METADATA_DEPTH} and canonical size "
    f"{MAX_METADATA_BYTES} bytes."
)
_SafeMetadataInteger = Annotated[
    int,
    Field(ge=-MAX_SAFE_JSON_INTEGER, le=MAX_SAFE_JSON_INTEGER),
]
PortableMetadataObject = TypeAliasType(
    "PortableMetadataObject",
    Annotated[
        dict[str, "PortableMetadataValue"],
        Field(
            description=_METADATA_DESCRIPTION,
            json_schema_extra={"propertyNames": {"pattern": r"^[A-Za-z0-9_.:-]{1,256}$"}},
        ),
    ],
)
PortableMetadataValue = TypeAliasType(
    "PortableMetadataValue",
    str
    | bool
    | None
    | _SafeMetadataInteger
    | list["PortableMetadataValue"]  # pyright: ignore[reportInvalidTypeForm]
    | PortableMetadataObject,
)


class FrozenMapping(Mapping[str, JsonValue]):
    """A recursively immutable mapping that remains copy and pickle friendly.

    ``MappingProxyType`` protects writes but cannot be deep-copied or pickled,
    both of which Pydantic callers reasonably use. This value object owns its
    canonical tuple of key/value pairs, exposes no mutable backing container,
    and reconstructs itself through the same recursive freezer when unpickled.
    """

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        _validate_metadata(values)
        object.__setattr__(
            self,
            "_items",
            tuple((key, _freeze_json(value)) for key, value in sorted(values.items())),
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __getitem__(self, key: str) -> JsonValue:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenMapping({_thaw_json(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        try:
            return _typed_json_key(self) == _typed_json_key(other)
        except (TypeError, ValueError):
            return False

    def __hash__(self) -> int:
        return hash(_typed_json_key(self))

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenMapping":
        # Every reachable value is immutable, so identity is a valid deep copy.
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[type["FrozenMapping"], tuple[dict[str, Any]]]:
        return type(self), (dict(self._items),)


class ProtocolModel(BaseModel):
    """Immutable, coercion-free wire model shared by every protocol shape."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)

    def to_canonical_json(self) -> bytes:
        """Return compact sorted UTF-8 JSON with literal Unicode and slashes."""
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_canonical_json(cls, payload: bytes | str) -> Self:
        """Parse only canonical bytes so signatures and digests cannot disagree."""
        raw, decoded = _decode_canonical_json(payload)
        parsed = cls.model_validate(decoded, strict=True)
        if parsed.to_canonical_json() != raw:
            raise ValueError("payload is valid JSON but is not canonical protocol JSON")
        return parsed

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Copy through validation so updates cannot bypass wire invariants.

        Pydantic's default ``model_copy(update=...)`` deliberately skips
        validation. That is unsafe for frozen protocol values because it can
        inject mutable metadata or invalid bounds. A serialized reconstruction
        also gives nested protocol models the same validation as RPC input;
        ``deep`` is accepted for API compatibility but immutability makes the
        reconstructed result safe either way.
        """
        fields_set = self.model_fields_set | (set(update) if update else set())
        values = self.model_dump(mode="python", round_trip=True)
        if update:
            values.update(deepcopy(dict(update)) if deep else update)
        copied = type(self).model_validate(values, strict=True)
        # Validation applies defaults, so restore which fields the caller
        # supplied to preserve exclude_unset and Pydantic's copy contract.
        object.__setattr__(copied, "__pydantic_fields_set__", fields_set)
        return copied


class ArtifactRef(ProtocolModel):
    """Content-addressed bytes whose retrieval remains an adapter concern."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(
        min_length=3,
        max_length=127,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$",
    )
    byte_count: int = Field(ge=1, le=10_000_000_000)


class FrameSize(ProtocolModel):
    """A top-left-origin pixel space captured with a specific frame."""

    width: int = Field(ge=1, le=MAX_DIMENSION)
    height: int = Field(ge=1, le=MAX_DIMENSION)
    origin: Literal["top_left"] = "top_left"
    unit: Literal["pixel"] = "pixel"


class PixelPoint(ProtocolModel):
    """An integer pixel selected under the protocol's floor-and-clamp policy."""

    x: Coordinate
    y: Coordinate


class FrameGeometry(ProtocolModel):
    """Native capture geometry and the exact resized frame shown to the model.

    Each directed conversion scales an input pixel coordinate by
    ``destination_size / source_size``, rounds toward negative infinity, then
    clamps to the destination's inclusive pixel bounds. The methods are not
    mathematical inverses when dimensions differ: downscaling merges pixels and
    floor rounding can move a round trip toward the top-left. Clamping makes a
    conversion boundary safe, but action validation rejects out-of-frame model
    coordinates rather than silently repairing model output.
    """

    native: FrameSize
    model_visible: FrameSize

    def model_to_native(self, x: int | float, y: int | float) -> PixelPoint:
        """Map from this frame's model-visible pixels into its native pixels."""
        return self._convert(x, y, source=self.model_visible, destination=self.native)

    def native_to_model(self, x: int | float, y: int | float) -> PixelPoint:
        """Map from this frame's native pixels into its model-visible pixels."""
        return self._convert(x, y, source=self.native, destination=self.model_visible)

    @staticmethod
    def _convert(
        x: int | float,
        y: int | float,
        *,
        source: FrameSize,
        destination: FrameSize,
    ) -> PixelPoint:
        return PixelPoint(
            x=_scale_clamped_axis(x, source.width, destination.width, name="x"),
            y=_scale_clamped_axis(y, source.height, destination.height, name="y"),
        )


class FrameRef(ProtocolModel):
    """An ordered observation frame and the geometry used to interpret it."""

    frame_id: Identifier
    artifact: ArtifactRef
    geometry: FrameGeometry


class Observation(ProtocolModel):
    """One immutable environment state presented during a stable episode."""

    task_id: Identifier
    episode_id: Identifier
    sequence: int = Field(ge=0, le=MAX_SAFE_JSON_INTEGER)
    observation_id: Identifier
    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_LENGTH, pattern=r"\S")
    frames: tuple[FrameRef, ...] = Field(default=(), max_length=32)
    metadata: PortableMetadataObject = Field(default_factory=dict, validate_default=True)

    @field_validator("frames", mode="before")
    @classmethod
    def _freeze_json_frames(cls, frames: Any) -> Any:
        # JSON has arrays but no tuples. Freeze exactly that wire shape while
        # strict mode continues to reject strings, iterators, and coercions.
        return tuple(frames) if isinstance(frames, list) else frames

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_wire_metadata(cls, metadata: Any) -> Any:
        # IEEE-754 implementations disagree with Python about formatting and
        # exactness outside this subset. Restricting metadata to strings,
        # booleans, null, and signed safe integers makes the existing sorted,
        # compact UTF-8 encoding portable without an RFC 8785 dependency.
        _validate_metadata(metadata)
        # JsonValue accepts concrete dict/list values only. Normalize safe
        # Mapping/tuple inputs for Pydantic, then freeze the validated result.
        return _thaw_json(metadata)

    @field_validator("metadata")
    @classmethod
    def _freeze_metadata(cls, metadata: Mapping[str, JsonValue]) -> FrozenMapping:
        frozen = FrozenMapping(metadata)
        canonical = _canonical_json_value(_thaw_json(frozen))
        if len(canonical) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata exceeds {MAX_METADATA_BYTES} canonical bytes")
        return frozen

    @field_serializer("metadata")
    def _serialize_metadata(self, metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        # Pydantic knows JsonValue but not MappingProxyType. Thaw only for wire
        # output; the model continues to expose recursively immutable values.
        thawed = _thaw_json(metadata)
        assert isinstance(thawed, dict)
        return thawed

    @model_validator(mode="after")
    def _unique_frames(self) -> Self:
        ids = [frame.frame_id for frame in self.frames]
        if len(ids) != len(set(ids)):
            raise ValueError("frame_id values must be unique within an observation")
        return self


class _Action(ProtocolModel):
    """Every action names the state it was selected from to expose staleness."""

    observation_id: Identifier


class _FrameAction(_Action):
    frame_id: Identifier


class _PointerAction(_FrameAction):
    x: Coordinate
    y: Coordinate
    button: MouseButton = "left"


class ClickAction(_PointerAction):
    kind: Literal["click"] = "click"


class DoubleClickAction(_FrameAction):
    """A distinct action retained because existing compilers emit double-click."""

    kind: Literal["double_click"] = "double_click"
    x: Coordinate
    y: Coordinate
    # Historical compilers only emit left-button double clicks; closing this
    # now avoids inventing adapter semantics that no implementation has proved.
    button: Literal["left"] = "left"


class TypeAction(_Action):
    kind: Literal["type"] = "type"
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH, pattern=r"\S")


_NAMED_KEYS = frozenset(
    {
        "ALT",
        "BACKSPACE",
        "CAPSLOCK",
        "CTRL",
        "DELETE",
        "DOWN",
        "END",
        "ENTER",
        "ESC",
        "HOME",
        "INSERT",
        "LEFT",
        "META",
        "PAGEDOWN",
        "PAGEUP",
        "RIGHT",
        "SHIFT",
        "SPACE",
        "TAB",
        "UP",
        *(f"F{index}" for index in range(1, 25)),
    }
)
_PRINTABLE_KEYS = frozenset(string.ascii_letters + string.digits + string.punctuation)


class KeyAction(_Action):
    kind: Literal["key"] = "key"
    keys: tuple[str, ...] = Field(min_length=1, max_length=8)

    @field_validator("keys", mode="before")
    @classmethod
    def _freeze_json_keys(cls, keys: Any) -> Any:
        return tuple(keys) if isinstance(keys, list) else keys

    @field_validator("keys")
    @classmethod
    def _known_unique_keys(cls, keys: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for key in keys:
            candidate = key.upper() if len(key) > 1 else key
            if candidate not in _NAMED_KEYS and candidate not in _PRINTABLE_KEYS:
                raise ValueError(f"unknown key: {key!r}")
            normalized.append(candidate)
        if len(normalized) != len(set(normalized)):
            raise ValueError("a key chord cannot contain duplicate keys")
        return tuple(normalized)


class ScrollAction(_FrameAction):
    kind: Literal["scroll"] = "scroll"
    x: Coordinate
    y: Coordinate
    delta_x: int = Field(default=0, ge=-MAX_SCROLL_DELTA, le=MAX_SCROLL_DELTA)
    delta_y: int = Field(default=0, ge=-MAX_SCROLL_DELTA, le=MAX_SCROLL_DELTA)

    @model_validator(mode="after")
    def _requires_motion(self) -> Self:
        if self.delta_x == 0 and self.delta_y == 0:
            raise ValueError("scroll requires a non-zero delta")
        return self


class WaitAction(_Action):
    kind: Literal["wait"] = "wait"
    duration_ms: int = Field(ge=1, le=MAX_WAIT_MS)


class FinishAction(_Action):
    """A terminal model claim; grading remains the benchmark adapter's job."""

    kind: Literal["finish"] = "finish"
    status: Literal["done", "failed", "infeasible"]
    reason: str = Field(min_length=1, max_length=10_000, pattern=r"\S")


class AskUserAction(_Action):
    """A terminal pause identified independently from retries or re-delivery."""

    kind: Literal["ask_user"] = "ask_user"
    request_id: Identifier
    question: str = Field(min_length=1, max_length=10_000, pattern=r"\S")


ComputerAction: TypeAlias = Annotated[
    ClickAction
    | DoubleClickAction
    | TypeAction
    | KeyAction
    | ScrollAction
    | WaitAction
    | FinishAction
    | AskUserAction,
    Field(discriminator="kind"),
]


class ObservationEnvelope(ProtocolModel):
    """Versioned adapter message carrying one environment observation."""

    # RPC senders must declare their version; accepting an omitted default
    # would make version negotiation ambiguous during rolling upgrades.
    protocol_version: Literal["1.0"]
    kind: Literal["observation"] = "observation"
    observation: Observation


class ActionBatch(ProtocolModel):
    """Versioned ordered actions selected from exactly one observation."""

    protocol_version: Literal["1.0"]
    kind: Literal["action_batch"] = "action_batch"
    task_id: Identifier
    episode_id: Identifier
    observation_id: Identifier
    actions: tuple[ComputerAction, ...] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    _TERMINAL_TYPES: ClassVar[tuple[type[ProtocolModel], ...]] = (FinishAction, AskUserAction)

    @field_validator("actions", mode="before")
    @classmethod
    def _freeze_json_actions(cls, actions: Any) -> Any:
        return tuple(actions) if isinstance(actions, list) else actions

    @model_validator(mode="after")
    def _bind_and_isolate_actions(self) -> Self:
        mismatches = [
            index
            for index, action in enumerate(self.actions)
            if action.observation_id != self.observation_id
        ]
        if mismatches:
            raise ValueError(f"actions at indexes {mismatches} bind to a different observation_id")
        if (
            any(isinstance(action, self._TERMINAL_TYPES) for action in self.actions)
            and len(self.actions) != 1
        ):
            raise ValueError("finish and ask_user must be the only action in their batch")
        return self

    def validate_for(self, observation: Observation) -> None:
        """Reject stale episodes and coordinates before an adapter executes."""
        expected = (observation.task_id, observation.episode_id, observation.observation_id)
        actual = (self.task_id, self.episode_id, self.observation_id)
        if actual != expected:
            raise ValueError(
                "action batch does not bind to the current task, episode, and observation"
            )
        frames = {frame.frame_id: frame for frame in observation.frames}
        for action in self.actions:
            if isinstance(action, (ClickAction, DoubleClickAction, ScrollAction)):
                frame = frames.get(action.frame_id)
                if frame is None:
                    raise ValueError(f"action references unknown frame_id {action.frame_id!r}")
                visible = frame.geometry.model_visible
                if action.x >= visible.width or action.y >= visible.height:
                    raise ValueError(
                        f"action coordinate {action.x},{action.y} is outside model-visible frame "
                        f"{visible.width}x{visible.height}"
                    )


ProtocolEnvelope: TypeAlias = Annotated[
    ObservationEnvelope | ActionBatch,
    Field(discriminator="kind"),
]
_ENVELOPE_ADAPTER = TypeAdapter(ProtocolEnvelope, config=ConfigDict(strict=True))


def parse_envelope(payload: bytes | str) -> ProtocolEnvelope:
    """Parse one exact, explicitly versioned canonical RPC envelope.

    Decoding with an object-pairs hook rejects duplicate names at every nesting
    level before model validation can normalize them away. Re-encoding catches
    other noncanonical representations as well as action validators that
    normalize a value (for example, a lowercase named key).
    """
    raw, decoded = _decode_canonical_json(payload)
    parsed = _ENVELOPE_ADAPTER.validate_python(decoded, strict=True)
    if parsed.to_canonical_json() != raw:
        raise ValueError("payload is valid JSON but is not canonical protocol JSON")
    return parsed


def _decode_canonical_json(payload: bytes | str) -> tuple[bytes, Any]:
    # The cap applies to transport bytes before any decoder allocation or
    # recursive parsing. Strings are measured in their protocol UTF-8 encoding.
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = payload
    if len(raw) > MAX_ENVELOPE_BYTES:
        raise ValueError(f"protocol envelope exceeds {MAX_ENVELOPE_BYTES} bytes")
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("protocol envelope is not valid UTF-8") from exc

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    decoded = json.loads(
        raw,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonstandard JSON constant: {value}")
        ),
    )
    return raw, decoded


def _scale_clamped_axis(
    value: int | float,
    source_size: int,
    destination_size: int,
    *,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    # Compare before multiplying. This prevents enormous ints and finite floats
    # near 1e308 from overflowing while preserving exact integer comparisons.
    if value <= 0:
        return 0
    if value >= source_size - 1:
        return destination_size - 1
    return math.floor(value * destination_size / source_size)


def _validate_metadata(value: Any, *, path: str = "metadata", depth: int = 0) -> None:
    """Enforce the portable, resource-bounded JSON subset at the RPC boundary."""
    if depth > MAX_METADATA_DEPTH:
        raise ValueError(f"{path} exceeds maximum nesting depth {MAX_METADATA_DEPTH}")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        if not -MAX_SAFE_JSON_INTEGER <= value <= MAX_SAFE_JSON_INTEGER:
            raise ValueError(f"{path} integer exceeds the portable JSON-safe range")
        return
    if isinstance(value, float):
        raise ValueError(f"{path} floats are not supported; use a string or integer")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_metadata(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            # Dynamic keys use an ASCII subset whose code-point and UTF-16 sort
            # orders are identical across Python and JavaScript adapters.
            if _METADATA_KEY_RE.fullmatch(key) is None:
                raise ValueError(
                    f"{path} keys must match [A-Za-z0-9_.:-]{{1,{MAX_IDENTIFIER_LENGTH}}}"
                )
            _validate_metadata(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _typed_json_key(value: Any) -> tuple[Any, ...]:
    """Return a hash/equality key that preserves JSON scalar type identity."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("integer", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, Mapping):
        return (
            "object",
            tuple((key, _typed_json_key(item)) for key, item in sorted(value.items())),
        )
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_typed_json_key(item) for item in value))
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _canonical_json_value(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
