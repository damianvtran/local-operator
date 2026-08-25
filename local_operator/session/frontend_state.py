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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    "new",
    "reload",
    "update",
    "resume",
    "theme",
    "provider",
    "search",
    "accounts",
    "usage",
    "analytics",
    "skills",
    "login",
    "logout",
    "credential",
    # These render canonical data or a local widget. Their owner-dependent work
    # is exposed through SessionProtocol operations rather than opening UI in a
    # different process.
    "mcp",
    # The overlay is local UI; its provider request crosses the authoritative
    # complete_aside operation on RemoteSession.
    "btw",
}
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


class JobState(BaseModel):
    """Read-only job shape used by the existing job widgets.

    ``extra='allow'`` is intentional: retained trajectory fields can grow without
    forcing an older follower to reject a newer owner. Unknowns stay attached to
    the DTO and survive a round trip rather than being discarded.
    """

    model_config = ConfigDict(extra="allow")

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
    prompt: str | None = None
    agent_role: str | None = None
    effort: str | None = None
    output_tail: str = ""
    output_seq: int = 0
    restored: bool = False

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
    history_cursor: str | None = None
    attachment_root: str | None = None

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


@dataclass(frozen=True, slots=True)
class FrontendSubscription:
    sync: FrontendSync
    unsubscribe: Callable[[], None]


class SnapshotJobs:
    """Small manager facade preserving the existing widgets' one renderer."""

    def __init__(self, values: Iterable[JobState] = ()) -> None:
        self.replace(values)

    def replace(self, values: Iterable[JobState]) -> None:
        self._values = [value.model_copy(deep=True) for value in values]

    def list(self) -> list[JobState]:
        return [value.model_copy(deep=True) for value in self._values]

    def get(self, job_id: str) -> JobState | None:
        return next(
            (value.model_copy(deep=True) for value in self._values if value.id == job_id), None
        )


class SnapshotWakeScheduler:
    def __init__(self, values: Iterable[WakeState] = ()) -> None:
        self.replace(values)

    def replace(self, values: Iterable[WakeState]) -> None:
        self.schedules = [SimpleNamespace(**value.model_dump()) for value in values]


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
        self._state = state.model_copy(deep=True)
        self._subscribers: list[Callable[[FrontendUpdate], None]] = []

    @property
    def state(self) -> FrontendSessionState:
        return self._state.model_copy(deep=True)

    def replace(self, state: FrontendSessionState) -> None:
        self._state = state.model_copy(deep=True)

    def replace_and_notify(self, state: FrontendSessionState) -> None:
        """Install a proven wire snapshot without reaching into subscribers."""
        self._state = state.model_copy(deep=True)
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
            rebuilt = []
            for raw in changes["jobs"]:
                job_id = str(raw.get("id", ""))
                prior = previous.get(job_id)
                trajectory = list(prior.trajectory if prior is not None else [])
                trajectory.extend(update.job_trajectory_appends.get(job_id, []))
                raw["trajectory"] = trajectory
                rebuilt.append(raw)
            changes["jobs"] = rebuilt
        payload = self._state.model_dump()
        payload.update(changes)
        payload["epoch"] = update.epoch
        payload["sequence"] = update.sequence
        self._state = FrontendSessionState.model_validate(payload)
        for subscriber in list(self._subscribers):
            subscriber(update.model_copy(deep=True))
        return self.state

    def mutate(self, **changes: Any) -> FrontendUpdate | None:
        normalized: dict[str, Any] = {}
        wire_changes: dict[str, Any] = {}
        trajectory_appends: dict[str, list[dict[str, Any]]] = {}
        current = self._state.model_dump(mode="json")
        for key, value in changes.items():
            candidate = _json_value(value)
            if current.get(key) != candidate:
                normalized[key] = candidate
                wire_changes[key] = candidate
        if "jobs" in wire_changes:
            previous = {str(job.get("id", "")): job for job in current.get("jobs", [])}
            summaries = []
            for job in wire_changes["jobs"]:
                if hasattr(job, "model_dump"):
                    job = job.model_dump(mode="json")
                else:
                    job = copy.deepcopy(job)
                job_id = str(job.get("id", ""))
                trajectory = list(job.pop("trajectory", []) or [])
                old = list(previous.get(job_id, {}).get("trajectory", []) or [])
                if trajectory[: len(old)] == old:
                    appended = trajectory[len(old) :]
                else:
                    appended = trajectory
                if appended:
                    trajectory_appends[job_id] = appended
                summaries.append(job)
            wire_changes["jobs"] = summaries
        if not normalized:
            return None
        payload = self._state.model_dump()
        payload.update(normalized)
        payload["sequence"] = self._state.sequence + 1
        self._state = FrontendSessionState.model_validate(payload)
        update = FrontendUpdate(
            epoch=self._state.epoch,
            sequence=self._state.sequence,
            changes=wire_changes,
            job_trajectory_appends=trajectory_appends,
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
        state = state.model_copy(update={"epoch": epoch, "sequence": 0})
        store = cls(state)
        store.refresh_from_session(session, initial=True)
        return store

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
        child_costs: dict[str, float] = dict(current.child_costs)
        for job in jobs:
            cost = job_cost(job, default_model_label=_label(selected))
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
                queued.append(
                    {
                        "id": str(getattr(message, "id", "") or ""),
                        "text": str(getattr(message, "text", "") or ""),
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
        changes = dict(
            cwd=str(getattr(session, "cwd", "") or getattr(session, "_cwd", "") or os.getcwd()),
            conversation_title=title,
            conversation_title_user_set=bool(getattr(title_state, "user_set", False)),
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
            context_tokens=(
                getattr(last_usage, "context_tokens", None)
                if last_usage
                else current.context_tokens
            ),
            context_is_estimate=(
                False
                if isinstance(last_usage, Usage) and last_usage.context_tokens
                else current.context_is_estimate
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
            self._state = FrontendSessionState.model_validate(payload)
        else:
            self.mutate(**changes)
        return self.state

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
        """Publish the job roster without rescanning unrelated session state."""
        jobs = self._jobs(session)
        child_costs = dict(self._state.child_costs)
        selected = getattr(session, "model", None)
        for job in jobs:
            cost = job_cost(job, default_model_label=_label(selected))
            if cost is not None:
                child_costs[job.id] = cost
        return self.mutate(jobs=jobs, child_costs=child_costs)

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
                changes.update(
                    last_usage=aggregate.model_dump(mode="json"),
                    context_tokens=aggregate.context_tokens,
                    context_is_estimate=(
                        False if aggregate.context_tokens else state.context_is_estimate
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
        # rescan transcript/jobs or publish replacement state.
        kind = str(getattr(event, "type", ""))
        if isinstance(event, (AgentEndEvent, MessageEndEvent)) or kind in {
            "tool_execution_end",
            "subagent_progress",
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
        self._state = self._state.model_copy(update={"live_events": live}, deep=True)

    async def checkpoint(self, transcript: Any) -> None:
        state = self.state
        checkpoint_id = uuid.uuid4().hex
        state.checkpoint_id = checkpoint_id
        self.replace(state)
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE,
            {"checkpoint_id": checkpoint_id, "state": state.model_dump(mode="json")},
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
    return values


def _label(spec: Any) -> str:
    if spec is None:
        return ""
    return f"{getattr(spec, 'provider', '')}/{getattr(spec, 'model_id', '')}".strip("/")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
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
        reasoning_tokens=sum(u.reasoning_tokens for u in usages),
        context_tokens=last_context,
        cost_components=components,
    )
