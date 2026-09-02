"""Canonical, transport-neutral state consumed by every full terminal UI.

Raw agent events remain the animation stream. This module owns everything a
newly attached ``OperatorApp`` needs before the next event arrives, so local and
remote terminals hydrate from the same typed facts instead of reconstructing
session semantics from the phone's deliberately capped projection.
"""

from __future__ import annotations

import copy
import os
import time
import uuid
from collections import deque
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
)
from pydantic_core import PydanticSerializationError, to_jsonable_python

from local_operator.harness.subagent import TRAJECTORY_CAP as _TRAJECTORY_CAP
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    CompactionEndEvent,
    MessageEndEvent,
    ModelSpec,
    Usage,
)
from local_operator.tui.costs import job_cost, turn_cost

FRONTEND_STATE_VERSION = 1
FRONTEND_CAPABILITY = "tui_state_v1"
FRONTEND_CHECKPOINT_CUSTOM_TYPE = "frontend_state_checkpoint_v1"

# Commands whose effect belongs to the process drawing the widgets. Every other
# advertised slash is routed to the authoritative session owner; keeping this a
# complement means adding a command without classification fails the test rather
# than silently acquiring follower-local behavior.
_FRONTEND_LOCAL_SLASHES = {
    "help",
    "exit",
    "clear",
    # The clipboard is the machine the USER IS SITTING AT, and every part of
    # this command is already here: the transcript it reads is painted by this
    # frontend, the picker it opens is painted by this frontend, and the OSC 52
    # write goes out this terminal. Routed to the owner it would draw a chooser
    # on a screen nobody is looking at and copy onto a host nobody is at, which
    # is the same argument `/theme` and `/settings` make about config.yml.
    "copy",
    "new",
    "reload",
    "update",
    "resume",
    # A fork opens a window on THIS machine and reads THIS machine's config.yml
    # for where to put it — the same argument `/settings` and `/theme` make. On a
    # follower attached to a remote owner, forking must open a window here, not
    # on the owner's host where nobody is sitting.
    "fork",
    "theme",
    # The settings page reads and writes THIS machine's config.yml, exactly
    # like `/theme` and `/search` above it. Routed to the owner it would open
    # against the owner's config and persist a default governing a machine the
    # user is not sitting at — the same rule `/model default` states explicitly
    # when it refuses to run on a follower.
    "settings",
    "provider",
    "search",
    "accounts",
    # Reads config.yml and the local credential COUNTS, and compares the
    # frontend's own effective model — all of it available on a follower, so
    # routing it to the owner would only add a hop.
    "failovers",
    "usage",
    "analytics",
    "skills",
    "login",
    "logout",
    "credential",
    # The overlay is local UI; its provider request crosses the authoritative
    # complete_aside operation on RemoteSession.
    "btw",
}
# Bare ``/mcp`` renders the canonical server list locally, but its grant
# subcommands mutate OAuth state that lives on the authoritative owner — the
# follower's MCP facade is a read-only snapshot with no config accessor, so
# routing the mutation (not faking it locally) is the only non-crashing,
# non-divergent answer. The dispatch splits the two shapes by argument.
_MCP_GRANT_SUBCOMMANDS = {"login", "logout", "reauth"}
_IMAGE_SLASHES = {"agent", "team"}


class CommandScope(StrEnum):
    """Where one advertised slash command executes."""

    FRONTEND_LOCAL = "frontend_local"
    AUTHORITATIVE_SESSION = "authoritative_session"
    UNAVAILABLE = "unavailable"


class CostKnowledge(StrEnum):
    """How confidently the cumulative dollar amount is known."""

    UNKNOWN = "unknown"
    EXACT = "exact"
    PARTIAL = "partial"
    FLOOR = "floor"


class SlashCapability(BaseModel):
    model_config = ConfigDict(extra="allow")

    command: str
    scope: CommandScope
    operation: str | None = None
    reason: str | None = None
    supports_images: bool = False


class SlashResult(BaseModel):
    """The typed outcome of one slash command run on the authoritative owner.

    The v5 replacement for the synthetic ``ran /…`` receipt: the owner runs a
    shared slash command and returns WHAT happened as data, so the terminal
    that asked renders it locally instead of the answer painting in another
    process's transcript. Every product-facing string is produced by the
    standard handlers, so a follower's ``/goal``, ``/rename``, ``/mcp login``
    or ``/context`` says exactly what a local session would — there is no
    attach-specific vocabulary anywhere in the fields.

    ``kind`` is one of ``notice`` (the invoker prints ``text`` through the
    normal notice path), ``block`` (``data`` is a renderable payload the
    follower builds its standard block from), or ``noop`` (nothing to print —
    e.g. a picker the invoker opens itself).
    """

    model_config = ConfigDict(extra="allow")

    kind: str = "notice"
    text: str = ""
    style: str = "info"
    data: dict[str, Any] = Field(default_factory=dict)


class TodoItemState(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    status: str = "pending"
    reason: str | None = None


class TodoPhaseState(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "Todos"
    items: list[TodoItemState] = Field(default_factory=list)


class WakeState(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    message: str
    next_due_at: int
    created_at: int = 0
    every_ms: int | None = None
    remaining: int | None = None


class McpServerState(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    status: str
    error: str | None = None
    tool_count: int | None = None


class PendingGateState(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    kind: str
    title: str
    detail: str = ""
    options: list[dict[str, Any]] = Field(default_factory=list)
    secret: bool = False
    question_index: int = 0
    question_total: int = 1


class _FrozenSequence(tuple[Any, ...]):
    """Tuple-backed sequence with list-compatible equality and wire shape."""

    __slots__ = ()

    def __new__(cls, values: Iterable[Any]) -> "_FrozenSequence":
        return tuple.__new__(cls, tuple(values))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return False

    def __hash__(self) -> int:
        return tuple.__hash__(self)

    def __copy__(self) -> "_FrozenSequence":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenSequence":
        return self

    def __reduce__(self) -> tuple[type["_FrozenSequence"], tuple[tuple[Any, ...]]]:
        return type(self), (tuple(self),)


class _FrozenMapping(tuple[tuple[str, Any], ...]):
    """Tuple-of-pairs mapping with no writable or rebindable backing storage."""

    __slots__ = ()

    def __new__(cls, values: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> "_FrozenMapping":
        items = values.items() if isinstance(values, Mapping) else values
        return tuple.__new__(cls, tuple(items))

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        for candidate, value in tuple.__iter__(self):
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return (key for key, _value in tuple.__iter__(self))

    def __len__(self) -> int:
        return tuple.__len__(self)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> Iterator[str]:
        return self.__iter__()

    def values(self) -> Iterator[Any]:
        return (value for _key, value in tuple.__iter__(self))

    def items(self) -> Iterator[tuple[str, Any]]:
        return tuple.__iter__(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenMapping):
            return tuple(tuple.__iter__(self)) == tuple(tuple.__iter__(other))
        if isinstance(other, Mapping):
            return dict(self.items()) == {key: other[key] for key in other}
        return False

    def __hash__(self) -> int:
        return tuple.__hash__(self)

    def __copy__(self) -> "_FrozenMapping":
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenMapping":
        return self

    def __reduce__(self) -> tuple[type["_FrozenMapping"], tuple[tuple[tuple[str, Any], ...]]]:
        return type(self), (tuple(tuple.__iter__(self)),)


# Virtual registration keeps the runtime mapping contract without inheriting a
# second, incompatible Collection generic beside tuple's pair iteration.
Mapping.register(_FrozenMapping)  # type: ignore[attr-defined]


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, (_FrozenSequence, _FrozenMapping)):
        return value
    if isinstance(value, FrontendUsage):
        return _freeze_frontend_usage(value)
    if isinstance(value, Usage):
        return _freeze_usage(value)
    if isinstance(value, BaseModel):
        # Unknown future job extras have no stable attribute contract. Preserve
        # all their data as a recursively immutable mapping rather than sharing
        # a mutable model the older follower cannot know how to freeze safely.
        return _freeze_value(value.model_dump(mode="python"))
    if isinstance(value, MutableMapping):
        return _FrozenMapping({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, bytearray):
        # bytearray is the remaining built-in mutable container accepted by an
        # unconstrained extra; bytes preserves its Pydantic JSON string shape.
        return bytes(value)
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, (MutableSequence, deque)):
        return _FrozenSequence(_freeze_value(item) for item in value)
    if isinstance(value, MutableSet):
        # ``frozenset`` preserves set equality/membership and Pydantic's JSON
        # serializer emits the same array shape as a mutable set. Nested values
        # are frozen first so a hashable mutable wrapper cannot retain an alias.
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(item) for item in value)
    # Closed boundary: Pydantic is the wire serializer, so anything beyond the
    # known fast paths is admitted only if that serializer can reduce it to JSON
    # values. Dataclasses/namedtuples become mappings/lists; datetime/path/enums
    # become scalars. Arbitrary __dict__/slots objects are rejected rather than
    # shared into canonical state with an unknowable mutation surface.
    try:
        serialized = to_jsonable_python(value)
    except PydanticSerializationError as exc:
        raise TypeError(
            f"unsupported canonical frontend value: {type(value).__qualname__}"
        ) from exc
    if serialized is value:
        raise TypeError(f"unsupported canonical frontend value: {type(value).__qualname__}")
    return _freeze_value(serialized)


def _is_frozen_value(value: Any) -> bool:
    if isinstance(value, (_FrozenSequence, _FrozenMapping, _FrozenUsage, _FrozenFrontendUsage)):
        return True
    if isinstance(value, (str, int, float, bool, bytes, type(None), frozenset)):
        return True
    if isinstance(value, tuple):
        return all(_is_frozen_value(item) for item in value)
    return False


def _wire_value(value: Any) -> Any:
    """Return ordinary JSON containers at the serialization boundary only."""
    if isinstance(value, _FrozenSequence):
        return [_wire_value(item) for item in value]
    if isinstance(value, _FrozenMapping):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    if isinstance(value, frozenset):
        return [_wire_value(item) for item in value]
    if isinstance(value, BaseModel):
        return _thaw_model(value)
    return value


def _job_summary(job: "JobState") -> dict[str, Any]:
    """Serialize one roster row without visiting its retained trajectory."""
    values = {
        name: _wire_value(value) for name, value in job.__dict__.items() if name != "trajectory"
    }
    values.update({name: _wire_value(value) for name, value in (job.model_extra or {}).items()})
    return to_jsonable_python(values)


def _jobs_equal(current: Sequence["JobState"], candidate: Sequence["JobState"]) -> bool:
    """Compare changed rows only; stable refreshes reuse frozen job identities."""
    return len(current) == len(candidate) and all(
        old is new or old == new for old, new in zip(current, candidate)
    )


def _freeze_job(job: "JobState") -> "JobState":
    """Detach the owning model and freeze every nested canonical value."""
    values = {
        name: _freeze_value(value)
        for name, value in job.__dict__.items()
        if name not in {"usage", "descendant_usage"}
    }
    values["usage"] = _freeze_usage(job.usage) if job.usage is not None else None
    values["descendant_usage"] = _FrozenSequence(
        _freeze_frontend_usage(component) for component in job.descendant_usage
    )
    for name, value in (job.model_extra or {}).items():
        values[name] = _freeze_value(value)
    return job.model_copy(update=values)


class JobState(BaseModel):
    """Read-only job shape used by the existing job widgets.

    ``extra='allow'`` is intentional: retained trajectory fields can grow without
    forcing an older follower to reject a newer owner. Unknowns stay attached to
    the DTO and survive a round trip rather than being discarded.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    type: str
    status: str = "running"
    queued: bool = False
    label: str = ""
    agent: str = ""
    intent: str = ""
    latest_details: dict[str, Any] | str | None = None
    error_text: str = ""
    result_text: str = ""
    model_label: str | None = None
    context_window: int | None = None
    usage: Usage | None = None
    start_time: float = 0.0
    started_at: float | None = None
    settled_at: float | None = None
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    # Nested spend (#297): a finished grandchild's usage folds into its root's
    # row here. Without carrying it, follower-side child-cost pricing counted
    # only the direct child while the owner priced the whole subtree.
    descendant_usage: list[FrontendUsage] = Field(default_factory=list)
    prompt: str | None = None
    agent_role: str | None = None
    effort: str | None = None
    output_tail: str = ""
    output_seq: int = 0
    restored: bool = False
    # Canonical lineage (U5): the owner's subagent-comms tree is not itself
    # serializable, but its one fact — who launched whom — is. Stamping the
    # parent's job id (and the child's session/role for the page header) lets
    # a follower rebuild the full parent/peer/child graph from ``state.jobs``
    # alone, so the hierarchy keys navigate the authoritative structure rather
    # than silently doing nothing.
    parent_job_id: str | None = None
    session_id: str | None = None

    @model_serializer(mode="wrap")
    def _serialize_frozen_values(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Pydantic's field serializers expect concrete lists/dicts. Thaw only
        # the ephemeral wire copy; canonical state keeps the immutable wrappers.
        mutable = self.model_copy(
            update={name: _wire_value(value) for name, value in self.__dict__.items()}
        )
        if self.model_extra:
            mutable.__pydantic_extra__ = {
                name: _wire_value(value) for name, value in self.model_extra.items()
            }
        return handler(mutable)

    @classmethod
    def from_job(cls, job: Any) -> "JobState":
        trajectory = []
        for event in list(getattr(job, "trajectory", None) or []):
            if hasattr(event, "model_dump"):
                trajectory.append(event.model_dump(mode="json"))
            elif isinstance(event, dict):
                trajectory.append(copy.deepcopy(event))
        details = getattr(job, "latest_details", None)
        if isinstance(details, dict):
            details = copy.deepcopy(details)
        elif details is not None and not isinstance(details, str):
            details = {"progress": str(details)}
        usage = getattr(job, "usage", None)
        if isinstance(usage, dict):
            usage = Usage.model_validate(usage)
        descendants = []
        for component in list(getattr(job, "descendant_usage", None) or []):
            if isinstance(component, dict):
                descendants.append(FrontendUsage.model_validate(component))
            elif isinstance(component, Usage):
                descendants.append(FrontendUsage.model_validate(component.model_dump(mode="json")))
        return cls(
            id=str(getattr(job, "id", "") or ""),
            type=str(getattr(job, "type", "") or ""),
            status=str(getattr(job, "status", "running") or "running"),
            queued=bool(getattr(job, "queued", False)),
            label=str(getattr(job, "label", "") or getattr(job, "agent", "") or ""),
            agent=str(getattr(job, "agent", "") or ""),
            intent=str(getattr(job, "intent", "") or ""),
            latest_details=details,
            error_text=str(getattr(job, "error_text", "") or getattr(job, "error", "") or ""),
            result_text=str(getattr(job, "result_text", "") or getattr(job, "result", "") or ""),
            model_label=getattr(job, "model_label", None),
            context_window=getattr(job, "context_window", None),
            usage=usage,
            start_time=float(
                getattr(job, "start_time", 0.0)
                or getattr(job, "started_at", 0.0)
                or getattr(job, "created_at", 0.0)
                or 0.0
            ),
            started_at=getattr(job, "started_at", None),
            settled_at=getattr(job, "settled_at", None) or getattr(job, "finished_at", None),
            trajectory=trajectory,
            descendant_usage=descendants,
            prompt=getattr(job, "prompt", None),
            agent_role=getattr(job, "agent_role", None),
            effort=getattr(job, "effort", None),
            output_tail=str(getattr(job, "output_tail", "") or ""),
            output_seq=int(getattr(job, "output_seq", 0) or 0),
            restored=bool(getattr(job, "restored", False)),
        )


class FrontendModelSpec(ModelSpec):
    """Wire model spec that preserves fields introduced by newer owners."""

    model_config = ConfigDict(extra="allow")


class FrontendUsage(Usage):
    """Lossless wire usage, including future cost component metadata."""

    model_config = ConfigDict(extra="allow")


class _FrozenUsage(Usage):
    """Usage-compatible immutable value retained inside a shared job snapshot."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_serializer(mode="wrap")
    def _serialize_frozen_values(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return handler(_thaw_model(self))


class _FrozenFrontendUsage(FrontendUsage):
    """FrontendUsage-compatible immutable descendant accounting value."""

    model_config = ConfigDict(extra="allow", frozen=True)

    @model_serializer(mode="wrap")
    def _serialize_frozen_values(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return handler(_thaw_model(self))


def _thaw_model(value: BaseModel) -> BaseModel:
    mutable = value.model_copy(
        update={name: _wire_value(item) for name, item in value.__dict__.items()}
    )
    if value.model_extra:
        mutable.__pydantic_extra__ = {
            name: _wire_value(item) for name, item in value.model_extra.items()
        }
    return mutable


def _freeze_usage(value: Usage) -> _FrozenUsage:
    frozen = _FrozenUsage.model_validate(value.model_dump(mode="python"))
    updates = {name: _freeze_value(item) for name, item in frozen.__dict__.items()}
    for name, item in (frozen.model_extra or {}).items():
        updates[name] = _freeze_value(item)
    return frozen.model_copy(update=updates)


def _freeze_frontend_usage(value: FrontendUsage) -> _FrozenFrontendUsage:
    frozen = _FrozenFrontendUsage.model_validate(value.model_dump(mode="python"))
    updates = {name: _freeze_value(item) for name, item in frozen.__dict__.items()}
    for name, item in (frozen.model_extra or {}).items():
        updates[name] = _freeze_value(item)
    return frozen.model_copy(update=updates)


class FrontendSessionState(BaseModel):
    """Versioned JSON-safe source of truth for one standard terminal UI."""

    model_config = ConfigDict(extra="allow")

    state_version: int = FRONTEND_STATE_VERSION
    session_id: str
    epoch: str
    sequence: int = 0
    checkpoint_id: str | None = None
    cwd: str = ""
    conversation_title: str = ""
    conversation_title_user_set: bool = False
    #: True while the session is a FORK still wearing its parent's title. The
    #: band/tab tag rides on the SAME refresh as the name so a rename that
    #: clears the session flag reaches every frontend in the same frame —
    #: without it a follower's snapshot could not heal a stale `[fork]` tab
    #: (review round 1, R1). `extra="allow"` keeps older readers tolerant of
    #: the new field, and `getattr` at the source keeps a reduced facade
    #: (embedded SDK, test double) from failing the whole refresh.
    conversation_title_forked: bool = False
    goal: str = ""
    active_agent: str = ""
    active_team: str = ""
    selected_model: FrontendModelSpec | None = None
    effective_model: FrontendModelSpec | None = None
    last_usage: FrontendUsage | None = None
    usage_components: list[FrontendUsage] = Field(default_factory=list)
    context_tokens: int | None = None
    context_is_estimate: bool | None = None
    context_window: int | None = None
    context_breakdown: dict[str, int] | None = None
    cumulative_parent_cost: float | None = None
    child_costs: dict[str, float] = Field(default_factory=dict)
    cost_knowledge: CostKnowledge = CostKnowledge.UNKNOWN
    streaming: bool = False
    generation: int = 0
    activity_started_at: float | None = None
    active_duration_s: float = 0.0
    current_turn_accrued_cost: float = 0.0
    queued_steering: list[dict[str, Any]] = Field(default_factory=list)
    # Bounded transient seed for a frontend that joins mid-turn. Existing
    # frontends consume raw events; only the atomic snapshot needs this fold.
    live_events: list[dict[str, Any]] = Field(default_factory=list)
    jobs: list[JobState] = Field(default_factory=list)
    todos: list[TodoPhaseState] = Field(default_factory=list)
    wakes: list[WakeState] = Field(default_factory=list)
    mcp_servers: list[McpServerState] = Field(default_factory=list)
    mcp_startup: dict[str, Any] | None = None
    pending_gate: PendingGateState | None = None
    slash_capabilities: list[SlashCapability] = Field(default_factory=list)
    # The owner's provider-catalogue rows, so an attached terminal's bare
    # ``/model`` picker lists the models the SESSION can actually switch to
    # (owner credentials/aggregators), never the follower's own possibly-
    # credential-less registry (D3, review round 2). Bounded to the direct
    # (non-aggregator) rows; a follower's current model and its own live
    # refresh stay authoritative for their own rows.
    model_catalogue: list[dict[str, Any]] = Field(default_factory=list)
    history_cursor: str | None = None
    attachment_root: str | None = None

    @model_serializer(mode="wrap")
    def _serialize_frozen_jobs(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        mutable = self.model_copy(update={"jobs": list(self.jobs)})
        return handler(mutable)

    @field_validator("selected_model", "effective_model", mode="before")
    @classmethod
    def _model_wire(cls, value: Any) -> Any:
        return value.model_dump(mode="json") if isinstance(value, ModelSpec) else value

    @field_validator("last_usage", mode="before")
    @classmethod
    def _usage_wire(cls, value: Any) -> Any:
        return value.model_dump(mode="json") if isinstance(value, Usage) else value

    @field_validator("usage_components", mode="before")
    @classmethod
    def _usage_components_wire(cls, value: Any) -> Any:
        return [
            item.model_dump(mode="json") if isinstance(item, Usage) else item
            for item in value or []
        ]

    @property
    def cumulative_cost(self) -> float | None:
        if self.cumulative_parent_cost is None and not self.child_costs:
            return None
        return float(self.cumulative_parent_cost or 0.0) + sum(self.child_costs.values())

    @property
    def model_label(self) -> str:
        spec = self.selected_model
        return f"{spec.provider}/{spec.model_id}" if spec is not None else ""

    @property
    def effective_model_label(self) -> str:
        spec = self.effective_model or self.selected_model
        return f"{spec.provider}/{spec.model_id}" if spec is not None else ""


def _freeze_state_jobs(
    state: FrontendSessionState, *, jobs_are_canonical: bool = False
) -> FrontendSessionState:
    """Detach incoming owners, or preserve already-owned jobs on scalar updates."""
    jobs = state.jobs if jobs_are_canonical else (_freeze_job(job) for job in state.jobs)
    return state.model_copy(update={"jobs": _FrozenSequence(jobs)})


def _public_job(job: JobState) -> JobState:
    """Detach every Pydantic owner while sharing immutable retained payloads."""
    values = {
        name: copy.deepcopy(value) if isinstance(value, BaseModel) else value
        for name, value in job.__dict__.items()
        if name != "descendant_usage"
    }
    values["descendant_usage"] = _FrozenSequence(
        copy.deepcopy(component) for component in job.descendant_usage
    )
    detached = job.model_copy(update=values)
    if job.model_extra:
        detached.__pydantic_extra__ = {
            name: copy.deepcopy(value) if isinstance(value, BaseModel) else value
            for name, value in job.model_extra.items()
        }
    return detached


class FrontendSync(BaseModel):
    model_config = ConfigDict(extra="allow")

    state_version: int = FRONTEND_STATE_VERSION
    epoch: str
    sequence: int
    snapshot: FrontendSessionState
    live_cursor: str | None = None


class FrontendUpdate(BaseModel):
    """One typed field delta in the canonical stream.

    Deltas and raw events share one ordered transport queue. A sequence is
    consumed only when canonical fields actually change, so any missing number
    is a real transport gap and forces a fresh snapshot rather than being
    mistaken for intentional coalescing.
    """

    model_config = ConfigDict(extra="allow")

    epoch: str
    sequence: int
    changes: dict[str, Any]
    job_trajectory_appends: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # Jobs whose appended events are a REPLACEMENT, not a suffix. The owner's
    # ``AsyncJob.trajectory`` evicts oldest past ``subagent.TRAJECTORY_CAP``, so
    # once a child crosses the cap the prefix check can never hold again;
    # without this marker a follower would extend forever (500 → 1000 → 1500…)
    # while duplicating rows in its click-through view.
    job_trajectory_replacements: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class FrontendSubscription:
    sync: FrontendSync
    unsubscribe: Callable[[], None]


class SnapshotJobs:
    """Small manager facade preserving the existing widgets' one renderer."""

    def __init__(self, values: Iterable[JobState] = ()) -> None:
        self.replace(values)

    def replace(self, values: Iterable[JobState]) -> None:
        # Canonical jobs already own recursively immutable retained payloads.
        # Route copies through the public detacher instead of asking Pydantic to
        # deep-copy tuple-backed Mapping/Sequence wrappers: the wrappers must
        # stay immutable while consumers retain their abstract container API.
        self._values = [_public_job(value) for value in values]

    def list(self) -> list[JobState]:
        return [_public_job(value) for value in self._values]

    def get(self, job_id: str) -> JobState | None:
        return next((_public_job(value) for value in self._values if value.id == job_id), None)


class SnapshotWakeScheduler:
    def __init__(self, values: Iterable[WakeState] = ()) -> None:
        self.replace(values)

    def replace(self, values: Iterable[WakeState]) -> None:
        self.schedules = [SimpleNamespace(**value.model_dump()) for value in values]


class SnapshotSubagentComms:
    """A follower's read-only job-graph facade, rebuilt from canonical jobs.

    The owner's ``SubagentComms`` is a live registry of running children and
    cannot cross the socket, but every navigation the full-page view needs —
    parent/peer/child, ancestors, the node's session/role — is pure graph over
    ``(job_id, parent_job_id, label, session_id, prompt, agent_role, effort)``,
    all of which ``JobState`` now carries. This facade answers the SAME methods
    the app calls on ``_subagent_comms`` from ``state.jobs``, so the hierarchy
    keys work identically on a follower (U5) with no attach-specific code path.
    """

    def __init__(self, jobs: Iterable[JobState] = ()) -> None:
        self.replace(jobs)

    def replace(self, jobs: Iterable[JobState]) -> None:
        self._nodes = {job.id: self._node_for(job) for job in jobs}

    @staticmethod
    def _node_for(job: JobState) -> Any:
        return SimpleNamespace(
            job_id=job.id,
            label=job.label or job.agent or job.id,
            parent_job_id=job.parent_job_id,
            session_id=job.session_id,
            session_dir=None,
            prompt=job.prompt or "",
            agent_role=job.agent_role or "",
            effort=job.effort or "",
        )

    def node(self, job_id: str) -> Any | None:
        return self._nodes.get(job_id)

    def job(self, job_id: str) -> Any | None:
        # The page reads live job fields from the jobs facade, not here; the
        # comms lookup exists on the owner for a manager cross-reference the
        # follower resolves through its own SnapshotJobs instead.
        return None

    def parent(self, job_id: str) -> Any | None:
        node = self._nodes.get(job_id)
        if node is None or not node.parent_job_id:
            return None
        return self._nodes.get(node.parent_job_id)

    def children(self, job_id: str | None) -> list[Any]:
        return [node for node in self._nodes.values() if node.parent_job_id == job_id]

    def peers(self, job_id: str) -> list[Any]:
        node = self._nodes.get(job_id)
        if node is None:
            return []
        return [peer for peer in self.children(node.parent_job_id) if peer.job_id != job_id]

    def ancestors(self, job_id: str) -> list[Any]:
        rows: list[Any] = []
        seen = {job_id}
        current = self.parent(job_id)
        while current is not None and current.job_id not in seen:
            seen.add(current.job_id)
            rows.append(current)
            current = self.parent(current.job_id)
        rows.reverse()
        return rows

    def session_dir_of(self, job_id: str) -> Any | None:
        return None


class SnapshotMcpManager:
    """Read-only manager API used by status and ``/mcp`` reporting."""

    def __init__(self, values: Iterable[McpServerState] = ()) -> None:
        self.replace(values)
        self._callback: Callable[..., Any] | None = None

    def replace(self, values: Iterable[McpServerState]) -> None:
        self._values = [value.model_copy(deep=True) for value in values]

    def get_all_server_names(self) -> list[str]:
        return sorted(value.name for value in self._values)

    def get_connected_servers(self) -> list[str]:
        return sorted(value.name for value in self._values if value.status == "connected")

    def get_connection_status(self, name: str) -> str:
        match = next((value for value in self._values if value.name == name), None)
        return match.status if match is not None else "disconnected"

    def set_on_tools_changed(self, callback: Callable[..., Any]) -> None:
        self._callback = callback

    @property
    def on_tools_changed(self) -> Callable[..., Any] | None:
        return self._callback


class FrontendStateStore:
    """Atomic snapshot/update store shared by local and remote sessions.

    Initial joins receive one immutable snapshot. Later mutations publish only
    typed field deltas, keeping high-frequency transport bounded while preserving
    one reducer and a strict sequence suitable for gap detection.
    """

    def __init__(self, state: FrontendSessionState) -> None:
        self._state = _freeze_state_jobs(state.model_copy(deep=True))
        self._subscribers: list[Callable[[FrontendUpdate], None]] = []

    @property
    def state(self) -> FrontendSessionState:
        # Never share an owning Pydantic instance. Clone the small state/job/usage
        # shells while immutable retained payload wrappers remain shared.
        snapshot = self._state.model_copy(deep=True)
        return snapshot.model_copy(
            update={"jobs": _FrozenSequence(_public_job(job) for job in self._state.jobs)}
        )

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subscribers)

    def replace(self, state: FrontendSessionState) -> None:
        self._state = _freeze_state_jobs(state.model_copy(deep=True))

    def replace_and_notify(self, state: FrontendSessionState) -> None:
        """Install a proven wire snapshot without reaching into subscribers."""
        self._state = _freeze_state_jobs(state.model_copy(deep=True))
        update = FrontendUpdate(
            epoch=state.epoch,
            sequence=state.sequence,
            changes=state.model_dump(mode="json"),
        )
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))

    def apply_update(self, update: FrontendUpdate) -> FrontendSessionState:
        """Apply one already-validated ordered delta from an owner."""
        if update.epoch != self._state.epoch or update.sequence != self._state.sequence + 1:
            raise ValueError("frontend update is not the next state sequence")
        changes = copy.deepcopy(update.changes)
        if "jobs" in changes:
            previous = {job.id: job for job in self._state.jobs}
            replacements = set(update.job_trajectory_replacements)
            rebuilt = []
            for raw in changes["jobs"]:
                job_id = str(raw.get("id", ""))
                prior = previous.get(job_id)
                if job_id in replacements:
                    trajectory = []
                else:
                    trajectory = list(prior.trajectory if prior is not None else [])
                trajectory.extend(update.job_trajectory_appends.get(job_id, []))
                # Defensive mirror of the owner-side eviction: even a
                # misbehaving owner cannot grow a follower without bound.
                if len(trajectory) > _TRAJECTORY_CAP:
                    del trajectory[: len(trajectory) - _TRAJECTORY_CAP]
                raw["trajectory"] = trajectory
                rebuilt.append(raw)
            changes["jobs"] = rebuilt
        payload = self._state.model_dump()
        payload.update(changes)
        payload["epoch"] = update.epoch
        payload["sequence"] = update.sequence
        self._state = _freeze_state_jobs(FrontendSessionState.model_validate(payload))
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))
        return self.state

    def mutate(self, **changes: Any) -> FrontendUpdate | None:
        normalized: dict[str, Any] = {}
        wire_changes: dict[str, Any] = {}
        trajectory_appends: dict[str, list[dict[str, Any]]] = {}
        trajectory_replacements: list[str] = []
        for key, value in changes.items():
            if key == "jobs":
                # JobState equality walks the bounded trajectories without first
                # cloning them into JSON. On a 100-child roster at the 500-event
                # cap this is ~20x cheaper for the common unchanged refresh.
                candidate_jobs = [
                    _freeze_job(
                        item if isinstance(item, JobState) else JobState.model_validate(item)
                    )
                    for item in value
                ]
                if not _jobs_equal(self._state.jobs, candidate_jobs):
                    normalized[key] = candidate_jobs
                    wire_changes[key] = candidate_jobs
                continue
            candidate = _json_value(value)
            # Serialize only fields the caller proposes changing. Dumping the
            # complete state here cloned every retained trajectory even for a
            # one-bit streaming update, turning unrelated UI reads into stalls.
            current_value = _json_value(getattr(self._state, key))
            if current_value != candidate:
                normalized[key] = _validate_state_field(key, candidate)
                wire_changes[key] = candidate
        if "jobs" in wire_changes:
            previous = {job.id: job for job in self._state.jobs}
            summaries = []
            for job in wire_changes["jobs"]:
                job_id = job.id
                trajectory = job.trajectory
                prior = previous.get(job_id)
                old = prior.trajectory if prior is not None else []
                if trajectory[: len(old)] == old:
                    appended = trajectory[len(old) :]
                else:
                    # The owner list rotated past its cap (or was rebuilt):
                    # a suffix no longer exists, so ship a replacement once
                    # rather than the whole list disguised as appends forever.
                    appended = trajectory
                    trajectory_replacements.append(job_id)
                if appended:
                    trajectory_appends[job_id] = appended
                summaries.append(_job_summary(job))
            wire_changes["jobs"] = summaries
        if not normalized:
            return None
        # Unchanged fields are immutable snapshot components and can be shared.
        # Re-validating a full model here deep-copied all job trajectories for
        # every small delta; each changed field was validated above instead.
        jobs_changed = "jobs" in normalized
        self._state = _freeze_state_jobs(
            self._state.model_copy(update={**normalized, "sequence": self._state.sequence + 1}),
            jobs_are_canonical=not jobs_changed,
        )
        update = FrontendUpdate(
            epoch=self._state.epoch,
            sequence=self._state.sequence,
            changes=wire_changes,
            job_trajectory_appends=trajectory_appends,
            job_trajectory_replacements=trajectory_replacements,
        )
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))
        return update

    def subscribe(self, callback: Callable[[FrontendUpdate], None]) -> FrontendSubscription:
        # Capture and register without yielding. Session mutations happen on one
        # event loop, so this is the atomic boundary: no update can fit between
        # the snapshot's sequence and the subscriber becoming visible.
        self._subscribers.append(callback)
        state = self.state
        sync = FrontendSync(
            epoch=state.epoch,
            sequence=state.sequence,
            snapshot=state,
            live_cursor=state.history_cursor,
        )

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return FrontendSubscription(sync=sync, unsubscribe=unsubscribe)

    @classmethod
    def from_session(cls, session: Any) -> "FrontendStateStore":
        store = cls(cls._restored_state(session))
        store.refresh_from_session(session, initial=True)
        return store

    @classmethod
    def from_checkpoint(cls, session: Any) -> "FrontendStateStore":
        """Headless construction: durable restore only, no live source scan.

        A headless host (scheduler, owned session, exec CLI) must stay cheap —
        ``refresh_from_session`` walks jobs/todos/MCP and imports the TUI
        registry — but its turn-end checkpoint is unconditional, so the store
        MUST begin from the richest durable state or a single headless turn
        would persist a bare checkpoint over the TUI's spend/duration/title.
        """
        return cls(cls._restored_state(session))

    @staticmethod
    def _restored_state(session: Any) -> FrontendSessionState:
        transcript = getattr(session, "_transcript", None)
        checkpoint = (
            transcript.latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE) if transcript else None
        )
        restored = None
        if isinstance(checkpoint, dict):
            raw = checkpoint.get("state")
            try:
                restored = (
                    FrontendSessionState.model_validate(raw) if isinstance(raw, dict) else None
                )
            except Exception:
                restored = None
        epoch = uuid.uuid4().hex
        state = restored or FrontendSessionState(session_id=str(session.session_id), epoch=epoch)
        # A new owner epoch invalidates stale wire updates while preserving the
        # durable checkpoint identity used to reconcile takeover without addition.
        return state.model_copy(update={"epoch": epoch, "sequence": 0})

    def refresh_from_session(self, session: Any, *, initial: bool = False) -> FrontendSessionState:
        current = self._state
        selected = getattr(session, "model", None)
        effective = getattr(session, "effective_model", None) or selected
        last_usage = None
        restore = getattr(session, "restored_usage", None)
        if callable(restore):
            try:
                last_usage = restore()
            except Exception:
                last_usage = None
        jobs = self._jobs(session)
        if initial:
            # The atomic join snapshot folds canonical lineage in once so a
            # follower's job graph is complete from its first frame; steady-
            # state updates re-fold on the 50 ms jobs coalesce (see
            # ``refresh_jobs``), never on the per-event session-loop refresh.
            comms = getattr(session, "_subagent_comms", None)
            if comms is not None:
                jobs = [_with_lineage(job, comms) for job in jobs]
        child_costs: dict[str, float] = dict(current.child_costs)
        for job in jobs:
            cost = _job_subtree_cost(job, default_model_label=_label(selected))
            if cost is not None:
                child_costs[job.id] = cost
        parent_cost = current.cumulative_parent_cost
        knowledge = current.cost_knowledge
        if parent_cost is None and last_usage is not None:
            cost = turn_cost(_label(effective), last_usage)
            if cost is not None:
                parent_cost = cost
                knowledge = CostKnowledge.FLOOR
        title_state = getattr(session, "conversation_name_state", None)
        title = str(getattr(session, "conversation_name", "") or "")
        todos = _todo_state(str(getattr(session, "session_id", current.session_id)))
        wakes = _wake_state(getattr(session, "wake_scheduler", None))
        mcp_servers = _mcp_state(
            getattr(session, "mcp_manager", None), getattr(session, "mcp_startup", None)
        )
        mcp_startup = _json_value(getattr(session, "mcp_startup", None))
        queued = []
        try:
            for message in session.queued_steering():
                content = list(getattr(message, "content", ()) or ())
                text = str(getattr(message, "text", "") or "")
                if not text:
                    # Not everything on the steering queue is a plain user
                    # Message: a busy-path wake and a busy-path peer steer
                    # queue their CustomMessage, which carries no ``text``
                    # property or ``content`` blocks — only ``details``.
                    # Without this a follower saw those entries as
                    # ``{"text": ""}`` and a renderer would paint a blank
                    # queued row. ``body`` is the raw human-facing text a
                    # peer row keeps for the UIs, preferred over ``text``
                    # (the model-facing provenance envelope); a wake only has
                    # ``text``, which IS its human-facing message.
                    details = getattr(message, "details", None)
                    if isinstance(details, dict):
                        text = str(details.get("body") or details.get("text") or "")
                queued.append(
                    {
                        "id": str(getattr(message, "id", "") or ""),
                        "text": text,
                        "image_count": sum(
                            1 for block in content if block.__class__.__name__ == "ImageContent"
                        ),
                        "status": "queued",
                    }
                )
        except Exception:
            pass
        history_cursor = None
        transcript = getattr(session, "_transcript", None)
        if transcript is not None:
            try:
                entries = transcript.entries()
                history_cursor = entries[-1].id if entries else None
            except Exception:
                pass
        # `/context` remains an on-demand operation. Computing its schema
        # breakdown on every unrelated mutation would serialize the unbounded
        # tool inventory on the session loop.
        context_breakdown = current.context_breakdown
        # A compaction creates a newer occupancy estimate than the last provider
        # receipt while that receipt remains authoritative for billing. Generic
        # source refreshes run after `agent_end`; replaying the unchanged receipt
        # there caused every frontend to paint `tokens_after` and then rebound to
        # the pre-pass level. Only a genuinely newer receipt may supersede it.
        receipt_context = getattr(last_usage, "context_tokens", None) if last_usage else None
        current_receipt_context = (
            current.last_usage.context_tokens if current.last_usage is not None else None
        )
        preserve_settled_context = bool(
            current.context_is_estimate
            and current.context_tokens
            and receipt_context
            and receipt_context == current_receipt_context
        )
        changes = dict(
            cwd=str(getattr(session, "cwd", "") or getattr(session, "_cwd", "") or os.getcwd()),
            conversation_title=title,
            conversation_title_user_set=bool(getattr(title_state, "user_set", False)),
            conversation_title_forked=bool(getattr(session, "wears_inherited_title", False)),
            goal=str(getattr(session, "goal", "") or ""),
            active_agent=str(getattr(session, "active_agent", "") or ""),
            active_team=str(getattr(session, "active_team_name", "") or ""),
            selected_model=(
                selected.model_dump(mode="json") if isinstance(selected, ModelSpec) else selected
            ),
            effective_model=(
                effective.model_dump(mode="json") if isinstance(effective, ModelSpec) else effective
            ),
            last_usage=(
                last_usage.model_dump(mode="json") if isinstance(last_usage, Usage) else last_usage
            ),
            context_tokens=(current.context_tokens if preserve_settled_context else receipt_context)
            or current.context_tokens,
            context_is_estimate=(
                current.context_is_estimate
                if preserve_settled_context
                else False if receipt_context else current.context_is_estimate
            ),
            context_window=(
                getattr(effective, "context_window", None) if effective is not None else None
            ),
            context_breakdown=context_breakdown,
            cumulative_parent_cost=parent_cost,
            child_costs=child_costs,
            cost_knowledge=knowledge,
            streaming=bool(getattr(session, "is_streaming", False)),
            generation=int(getattr(session, "_generation", current.generation) or 0),
            activity_started_at=(
                current.activity_started_at
                if bool(getattr(session, "is_streaming", False))
                else None
            ),
            queued_steering=queued,
            jobs=jobs,
            todos=todos,
            wakes=wakes,
            mcp_servers=mcp_servers,
            mcp_startup=mcp_startup,
            history_cursor=history_cursor,
            attachment_root=str(getattr(transcript, "directory", "") or "") or None,
            slash_capabilities=_slash_capabilities(),
        )
        if initial:
            payload = current.model_dump()
            payload.update(changes)
            self._state = _freeze_state_jobs(FrontendSessionState.model_validate(payload))
        else:
            self.mutate(**changes)
        return self.state

    def refresh_restored_usage(self, session: Any) -> FrontendUpdate | None:
        """Price the restored point-in-time reading without rescanning state."""
        restore = getattr(session, "restored_usage", None)
        usage = restore() if callable(restore) else None
        if not isinstance(usage, Usage):
            return None
        state = self._state
        changes: dict[str, Any] = {
            "last_usage": usage.model_dump(mode="json"),
            "context_tokens": usage.context_tokens,
            "context_is_estimate": False if usage.context_tokens else state.context_is_estimate,
        }
        if state.cumulative_parent_cost is None:
            cost = turn_cost(_label(getattr(session, "effective_model", None)), usage)
            if cost is not None:
                changes.update(
                    cumulative_parent_cost=cost,
                    cost_knowledge=CostKnowledge.FLOOR,
                )
        return self.mutate(**changes)

    def accrue_usage(self, session: Any, usage: Usage) -> FrontendUpdate | None:
        """Accrue a provider call outside the ordinary agent event stream."""
        state = self._state
        cost = turn_cost(_label(getattr(session, "effective_model", None)), usage)
        changes: dict[str, Any] = {
            "last_usage": usage.model_dump(mode="json"),
            "context_tokens": usage.context_tokens or usage.input_tokens or state.context_tokens,
            "context_is_estimate": False,
        }
        if cost is not None:
            changes.update(
                cumulative_parent_cost=(state.cumulative_parent_cost or 0.0) + cost,
                cost_knowledge=(
                    CostKnowledge.EXACT
                    if state.cost_knowledge in {CostKnowledge.UNKNOWN, CostKnowledge.EXACT}
                    else state.cost_knowledge
                ),
                usage_components=list(state.usage_components)
                + list(usage.cost_components or [usage]),
            )
        elif usage.input_tokens or usage.output_tokens:
            changes["cost_knowledge"] = CostKnowledge.PARTIAL
        return self.mutate(**changes)

    def refresh_jobs(self, session: Any) -> FrontendUpdate | None:
        """Publish the job roster without rescanning unrelated session state.

        Canonical lineage (parent/child identity) is folded in HERE rather than
        in ``refresh_from_session``: the comms lookup is per-job, and the
        per-event refresh runs on the session loop for every streaming edge —
        folding there stalled concurrent children by over a second. The jobs
        roster is republished on a 50 ms coalesce, so lineage lands on the same
        cadence the page that needs it already refreshes on, at a fraction of
        the cost.
        """
        jobs = self._jobs(session)
        comms = getattr(session, "_subagent_comms", None)
        if comms is not None:
            jobs = [_with_lineage(job, comms) for job in jobs]
        child_costs = dict(self._state.child_costs)
        selected = getattr(session, "model", None)
        for job in jobs:
            cost = _job_subtree_cost(job, default_model_label=_label(selected))
            if cost is not None:
                child_costs[job.id] = cost
        return self.mutate(jobs=jobs, child_costs=child_costs)

    def refresh_model_catalogue(self, entries: Iterable[Any]) -> FrontendUpdate | None:
        """Publish the owner's offerable model rows as canonical state.

        The catalogue answers one question — which models may this SESSION
        switch to — and only the owner's provider controller knows it (the
        owner's credentials, its aggregators, its registry). Kept out of
        ``refresh_from_session`` because that runs on the session loop for
        every streaming edge; the catalogue changes on credential/registry
        timescales, so the TUI publishes it on adoption and after login-style
        mutations instead.
        """
        rows: list[dict[str, Any]] = []
        for entry in entries:
            rows.append(
                {
                    "provider": str(getattr(entry, "provider", "") or ""),
                    "model_id": str(getattr(entry, "model_id", "") or ""),
                    "label": str(getattr(entry, "label", "") or ""),
                    "context_window": int(getattr(entry, "context_window", 0) or 0),
                    "input_price": float(getattr(entry, "input_price", 0.0) or 0.0),
                    "output_price": float(getattr(entry, "output_price", 0.0) or 0.0),
                    "connected": bool(getattr(entry, "connected", False)),
                    "aggregated": bool(getattr(entry, "aggregated", False)),
                }
            )
        return self.mutate(model_catalogue=rows)

    def observe_event(self, session: Any, event: AgentEvent[Any]) -> FrontendUpdate | None:
        now = time.time()
        state = self._state
        changes: dict[str, Any] = {}
        self._fold_live_event(event)
        if isinstance(event, AgentStartEvent):
            changes.update(
                streaming=True,
                generation=int(event.generation or state.generation + 1),
                activity_started_at=now,
            )
        elif isinstance(event, AgentEndEvent):
            duration = state.active_duration_s
            if state.activity_started_at is not None:
                duration += max(0.0, now - state.activity_started_at)
            changes.update(streaming=False, activity_started_at=None, active_duration_s=duration)
            # Reconcile the whole turn once. Per-call receipts are retained so a
            # mixed-provider aggregate never loses which call owned which price.
            usages = [
                usage
                for message in (event.messages or [])
                if isinstance((usage := getattr(message, "usage", None)), Usage)
            ]
            if usages:
                aggregate = _aggregate_usage(usages)
                total = turn_cost(_label(getattr(session, "effective_model", None)), aggregate)
                if total is not None:
                    remainder = max(0.0, total - state.current_turn_accrued_cost)
                    previous = state.cumulative_parent_cost or 0.0
                    changes.update(
                        cumulative_parent_cost=previous + remainder,
                        current_turn_accrued_cost=0.0,
                        usage_components=(
                            list(state.usage_components)
                            if state.current_turn_accrued_cost > 0
                            else list(state.usage_components) + list(aggregate.cost_components)
                        ),
                        cost_knowledge=(
                            CostKnowledge.EXACT
                            if state.cost_knowledge in {CostKnowledge.UNKNOWN, CostKnowledge.EXACT}
                            else state.cost_knowledge
                        ),
                    )
                elif any(u.input_tokens or u.output_tokens for u in usages):
                    changes["cost_knowledge"] = CostKnowledge.PARTIAL
                    changes["current_turn_accrued_cost"] = 0.0
                # `messages` remain the billing authority even when a post-turn
                # compaction invalidates their occupancy. The session stamps the
                # settled level separately so this late boundary cannot rebound a
                # frontend to the provider reading from before the rewrite.
                settled_context = event.context_tokens or aggregate.context_tokens
                changes.update(
                    last_usage=aggregate.model_dump(mode="json"),
                    context_tokens=settled_context,
                    context_is_estimate=(
                        True
                        if event.context_tokens is not None
                        else False if aggregate.context_tokens else state.context_is_estimate
                    ),
                )
        elif (
            isinstance(event, MessageEndEvent) and getattr(event.message, "usage", None) is not None
        ):
            usage = getattr(event.message, "usage", None)
            if not isinstance(usage, Usage):
                return
            # Occupancy is a level. Cost accrues per call so arbitrary joins see
            # the same lifetime figure; AgentEnd adds only the final remainder.
            call_cost = turn_cost(_label(getattr(session, "effective_model", None)), usage)
            changes.update(
                last_usage=usage.model_dump(mode="json"),
                context_tokens=usage.context_tokens or usage.input_tokens or state.context_tokens,
                context_is_estimate=False,
            )
            if call_cost is not None:
                changes.update(
                    cumulative_parent_cost=(state.cumulative_parent_cost or 0.0) + call_cost,
                    current_turn_accrued_cost=state.current_turn_accrued_cost + call_cost,
                    cost_knowledge=(
                        CostKnowledge.EXACT
                        if state.cost_knowledge in {CostKnowledge.UNKNOWN, CostKnowledge.EXACT}
                        else state.cost_knowledge
                    ),
                    usage_components=list(state.usage_components)
                    + list(usage.cost_components or [usage]),
                )
            elif usage.input_tokens or usage.output_tokens:
                changes["cost_knowledge"] = CostKnowledge.PARTIAL
        elif isinstance(event, CompactionEndEvent) and event.success:
            changes.update(
                context_tokens=event.tokens_after or None,
                context_is_estimate=True,
            )
        update = self.mutate(**changes) if changes else None
        # Expensive source snapshots are explicit mutation hooks. Turn edges and
        # tool/result boundaries are the defensive fallback; token deltas never
        # rescan transcript/jobs or publish replacement state. Subagent progress
        # is excluded too: the job manager already coalesces that live row onto
        # ``refresh_jobs`` within 50 ms. Re-scanning the entire session here
        # published a duplicate frame for every child boundary and made remote
        # followers pay a second wire update for the same activity string.
        kind = str(getattr(event, "type", ""))
        if isinstance(event, (AgentEndEvent, MessageEndEvent)) or kind in {
            "tool_execution_end",
            "subagent_end",
            "model_change",
        }:
            before = self._state.sequence
            self.refresh_from_session(session)
            if self._state.sequence != before:
                update = None
        return update

    def _fold_live_event(self, event: AgentEvent[Any]) -> None:
        """Maintain only the bounded in-flight seed, without publishing deltas.

        Connected frontends already receive the raw event. Updating this local
        snapshot before event fan-out makes a join at that exact boundary see
        the accumulated assistant/tool state without flooding every peer with a
        second frame for every token.
        """
        data = event.model_dump(mode="json")
        kind = str(data.get("type", ""))
        live = list(self._state.live_events)
        if kind in {"agent_start", "agent_end"}:
            live = []
        elif kind == "message_start":
            message = data.get("message") or {}
            if message.get("role") != "user":
                live = [
                    item
                    for item in live
                    if item.get("type") not in {"message_start", "message_update"}
                ]
                live.append(data)
        elif kind == "message_update":
            live = [item for item in live if item.get("type") != "message_update"]
            live.append(data)
        elif kind == "message_end":
            live = [
                item for item in live if item.get("type") not in {"message_start", "message_update"}
            ]
        elif kind in {"tool_call_compose", "tool_execution_start"}:
            call_id = str(data.get("tool_call_id") or "")
            live = [item for item in live if str(item.get("tool_call_id") or "") != call_id]
            live.append(data)
        elif kind == "tool_execution_end":
            call_id = str(data.get("tool_call_id") or "")
            live = [item for item in live if str(item.get("tool_call_id") or "") != call_id]
        # Shallow copy on purpose: this runs per streaming delta on the session
        # loop, and a deep copy re-clones a 500-event trajectory each token.
        # ``live`` is freshly built above, and every other field is replaced
        # (never mutated in place) by ``mutate``/``apply_update``.
        self._state = self._state.model_copy(update={"live_events": live})

    async def checkpoint(self, transcript: Any) -> None:
        state = self.state
        checkpoint_id = uuid.uuid4().hex
        state.checkpoint_id = checkpoint_id
        self.replace(state)
        # Trajectories are reconstructable from durable child transcripts and
        # live_events are transient by definition; persisting them appended
        # ~71 KiB per busy child to the transcript at EVERY turn end.
        durable = state.model_copy(
            update={
                "live_events": [],
                "jobs": [job.model_copy(update={"trajectory": []}) for job in state.jobs],
            }
        )
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {"checkpoint_id": checkpoint_id, "state": durable.model_dump(mode="json")},
        )

    @staticmethod
    def _jobs(session: Any) -> list[JobState]:
        manager = getattr(session, "jobs", None)
        try:
            rows = manager.list() if manager else []
        except Exception:
            return []
        values: list[JobState] = []
        for job in rows:
            try:
                values.append(JobState.from_job(job))
            except Exception:
                # One malformed extension row cannot erase unrelated jobs.
                continue
        return values


def _slash_capabilities() -> list[SlashCapability]:
    # Imported lazily so module import remains headless-safe; a full frontend
    # store needs the authoritative registry rather than a duplicated name list.
    from local_operator.tui.app import SLASH_COMMANDS

    values = []
    for command in SLASH_COMMANDS:
        scope = (
            CommandScope.FRONTEND_LOCAL
            if command.name in _FRONTEND_LOCAL_SLASHES
            else CommandScope.AUTHORITATIVE_SESSION
        )
        values.append(
            SlashCapability(
                command=command.name,
                scope=scope,
                operation=(None if scope is CommandScope.FRONTEND_LOCAL else "slash"),
                supports_images=command.name in _IMAGE_SLASHES,
            )
        )
    # ``/mcp`` is advertised once but its scope is argument-dependent: bare it
    # is a local listing, with a grant subcommand it is authoritative. The
    # capability records the authoritative shape (the one that needs routing);
    # the follower's dispatch keeps the bare listing local by inspecting args.
    for capability in values:
        if capability.command == "mcp":
            capability.scope = CommandScope.AUTHORITATIVE_SESSION
            capability.operation = "slash"
    return values


def _with_lineage(job: JobState, comms: Any) -> JobState:
    """Stamp one job's canonical parent/child identity from the comms tree.

    The comms registry — not the job manager — knows who launched whom, so the
    lineage is merged here at snapshot time rather than carried on the job
    itself. Read defensively: a job with no comms record (a bash job, a swept
    child) keeps ``parent_job_id=None`` and simply has no navigation targets.
    """
    try:
        node = comms.node(job.id)
    except Exception:
        node = None
    if node is None:
        return job
    return job.model_copy(
        update={
            "parent_job_id": getattr(node, "parent_job_id", None),
            "session_id": getattr(node, "session_id", None),
        }
    )


def _job_subtree_cost(job: Any, *, default_model_label: str) -> float | None:
    """Direct plus nested descendant spend for one root job row.

    Mirrors the harness accounting (`jobs.py`): each descendant component is
    priced at ITS OWN serving identity, never the parent's rate. Any
    unpriceable component returns ``None`` so the prior figure is retained
    rather than silently undercounted — the same honesty rule the legacy
    harvest applied.
    """
    direct = job_cost(job, default_model_label=default_model_label)
    components = list(getattr(job, "descendant_usage", None) or [])
    if direct is None and not components:
        return None
    descendant = 0.0
    for component in components:
        provider = getattr(component, "provider", None) or ""
        model_id = getattr(component, "model_id", None) or ""
        cost = turn_cost(f"{provider}/{model_id}" if provider else model_id, component)
        if cost is None:
            return None
        descendant += cost
    return (direct or 0.0) + descendant


def _label(spec: Any) -> str:
    if spec is None:
        return ""
    return f"{getattr(spec, 'provider', '')}/{getattr(spec, 'model_id', '')}".strip("/")


_STATE_FIELD_ADAPTERS: dict[str, TypeAdapter[Any]] = {}


def _validate_state_field(key: str, value: Any) -> Any:
    """Validate one changed field without rebuilding the unrelated state."""
    field = FrontendSessionState.model_fields.get(key)
    if field is None:
        raise ValueError(f"unknown frontend state field: {key}")
    adapter = _STATE_FIELD_ADAPTERS.get(key)
    if adapter is None:
        # The key space is the model's finite field set, so this cache is
        # intrinsically bounded and avoids rebuilding pydantic schemas per delta.
        adapter = _STATE_FIELD_ADAPTERS[key] = TypeAdapter(field.annotation)
    return adapter.validate_python(value)


def _json_value(value: Any) -> Any:
    """Fully JSON-shape a candidate so equality against dumped state is real.

    ``mutate`` decides "changed" by comparing this against the state's
    ``model_dump(mode="json")``. A list of pydantic models (jobs, capabilities)
    left as model instances can NEVER equal its dict form, which made change
    detection constant-true and published a sequence-consuming frame on every
    no-op refresh — so recurse into containers, not just the top level.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return copy.deepcopy(value.__dict__)
    return copy.deepcopy(value)


def _todo_state(session_id: str) -> list[TodoPhaseState]:
    try:
        from local_operator.tools.builtin import todo_snapshot

        raw = todo_snapshot(session_id)
    except Exception:
        return []
    phases: list[TodoPhaseState] = []
    for phase in raw or []:
        if "items" in phase:
            phases.append(TodoPhaseState.model_validate(phase))
        else:
            phases.append(TodoPhaseState(name="Todos", items=[TodoItemState.model_validate(phase)]))
    return phases


def _wake_state(scheduler: Any) -> list[WakeState]:
    try:
        return [WakeState.model_validate(schedule.model_dump()) for schedule in scheduler.schedules]
    except Exception:
        return []


def _mcp_state(manager: Any, startup: Any) -> list[McpServerState]:
    names: set[str] = set()
    failures = dict(getattr(startup, "failures", {}) or {})
    if manager is not None:
        try:
            names.update(manager.get_all_server_names())
        except Exception:
            pass
    names.update(getattr(startup, "configured", ()) or ())
    values = []
    for name in sorted(names):
        status = "failed" if name in failures else "disconnected"
        if manager is not None:
            try:
                status = str(manager.get_connection_status(name) or status)
            except Exception:
                pass
        values.append(McpServerState(name=name, status=status, error=failures.get(name)))
    return values


def _aggregate_usage(usages: list[Usage]) -> Usage:
    last_context = next(
        (u.context_tokens for u in reversed(usages) if u.context_tokens is not None), None
    )
    components: list[Usage] = []
    for usage in usages:
        components.extend(
            component.model_copy(deep=True) for component in (usage.cost_components or [usage])
        )
    return Usage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cache_read_tokens=sum(u.cache_read_tokens for u in usages),
        cache_write_tokens=sum(u.cache_write_tokens for u in usages),
        # Same contract as ``cache_write_tokens`` above: the TTL split folds
        # wherever the write count does, or the frontend's split would read
        # as the first call's value only (see ``Usage.cache_write_1h_tokens``).
        cache_write_5m_tokens=sum(u.cache_write_5m_tokens for u in usages),
        cache_write_1h_tokens=sum(u.cache_write_1h_tokens for u in usages),
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
        context_tokens=last_context,
        cost_components=components,
    )
