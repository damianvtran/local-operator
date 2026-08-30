"""Strict, benchmark-neutral observations and structured computer actions.

The protocol carries references to frame bytes rather than embedding them. This
keeps a future adapter RPC small and prevents transport details (filesystem
paths, object-store URLs, or inline base64) from becoming model-controlled
capabilities. Every coordinate is interpreted against geometry captured in the
observation; callers must never substitute the dimensions of a current display.
"""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping
from types import MappingProxyType
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

PROTOCOL_VERSION = "1.0"
MAX_IDENTIFIER_LENGTH = 256
MAX_TEXT_LENGTH = 100_000
MAX_METADATA_BYTES = 64_000
MAX_METADATA_DEPTH = 16
MAX_SAFE_JSON_INTEGER = 2**53 - 1
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


class ProtocolModel(BaseModel):
    """Immutable, coercion-free wire model shared by every protocol shape."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def to_canonical_json(self) -> bytes:
        """Return the one stable UTF-8 representation used for adapter RPCs."""
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
        for name, value in (("x", x), ("y", y)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be an integer or float")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        converted_x = math.floor(x * destination.width / source.width)
        converted_y = math.floor(y * destination.height / source.height)
        return PixelPoint(
            x=min(destination.width - 1, max(0, converted_x)),
            y=min(destination.height - 1, max(0, converted_y)),
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
    sequence: int = Field(ge=0, le=2**63 - 1)
    observation_id: Identifier
    text: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_LENGTH, pattern=r"\S")
    frames: tuple[FrameRef, ...] = Field(default=(), max_length=32)
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)

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
        return metadata

    @field_validator("metadata")
    @classmethod
    def _freeze_metadata(cls, metadata: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        frozen = _freeze_json(metadata)
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
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload

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
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_metadata(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            if not key or len(key) > MAX_IDENTIFIER_LENGTH:
                raise ValueError(f"{path} contains an empty or overlong key")
            _validate_metadata(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_value(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
