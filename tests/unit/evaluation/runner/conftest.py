"""Shared fixtures for episode-runner tests.

The in-process fake supervisor follows ``test_supervisor.py``'s ``RawSupervisor``
pattern: it speaks the adapter protocol at the ``_call_raw`` seam, so every
test drives the REAL ``VerifiedAdapterSession``, the REAL lifecycle
authorities, and a REAL ``EvidenceWriter`` on ``tmp_path``. Only the subprocess
boundary is faked, because that is the one thing a unit test cannot afford per
case; everything the runner is responsible for stays under test.
"""

from __future__ import annotations

import hashlib
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import pytest

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    AdapterSelector,
    AskUserExchangeResult,
    CleanupOutcome,
    CleanupResult,
    ExecuteResult,
    ExecutionReceipt,
    Handshake,
    ObservationResult,
    PrepareResult,
    PythonRuntime,
    RequirementsResult,
    ScoreResult,
    observation_content_id,
)
from local_operator.evaluation.evidence.models import RouteIdentity, ScoreArtifact
from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan
from local_operator.evaluation.protocol import ActionBatch, Observation
from local_operator.evaluation.receipts import (
    BUDGET_RESOURCES,
    BudgetAuthorization,
    CappedAllowance,
    ComputeRequirement,
    DependencyPlan,
    RedactionSet,
    ResourceAmount,
    record_preflight,
    reserve_budget,
    seal_preflight,
)
from local_operator.evaluation.runner.episode import EpisodeConfig, EpisodeSpec
from local_operator.evaluation.runner.model import ModelDecision, ModelUsage

_P = TypeVar("_P")

DIGEST = "0123456789abcdef" * 4
OTHER_DIGEST = "abcdef0123456789" * 4
ROUTE = RouteIdentity(provider_id="provider", route_id="route", model_id="model")
TASK_ID = "task-1"


def selector(tmp_path: Path) -> AdapterSelector:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AdapterSelector(
        schema_version="1.1",
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.0",
        entry_point="tiny_adapter:create",
        package_digest="a" * 64,
        release_digest="b" * 64,
        python_executable=str(Path(sys.executable).resolve()),
        workspace=str(workspace),
        workspace_digest="c" * 64,
        route_capability="computer",
    )


def handshake(tmp_path: Path) -> Handshake:
    return Handshake(
        selector=selector(tmp_path),
        metadata=AdapterMetadata(
            adapter_id="tiny",
            distribution="tiny-adapter",
            version="1.0",
            entry_point="tiny_adapter:create",
            package_digest="a" * 64,
            release_digest="b" * 64,
            schema_version="1.1",
            capabilities=AdapterCapabilities(routes=("computer",), ask_user=True, scoring=True),
        ),
        python=PythonRuntime.current(),
        workspace_digest="c" * 64,
        selected_route="computer",
    )


def observation(episode_id: str, sequence: int, *, text: str = "state") -> Observation:
    provisional = Observation(
        task_id=TASK_ID,
        episode_id=episode_id,
        sequence=sequence,
        observation_id="provisional",
        text=f"{text}-{sequence}",
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def cleanup_plan(episode_id: str) -> CleanupPlan:
    return CleanupPlan(
        episode_id=episode_id,
        actions=(
            CleanupAction(
                action_id="release",
                kind="release_instance",
                resource_ref="resource",
                timeout_ms=1000,
                max_attempts=2,
            ),
        ),
    )


def build_spec(episode_id: str) -> EpisodeSpec:
    plan = DependencyPlan(
        release_id="release-1",
        task_id=TASK_ID,
        attempt_id=episode_id,
        requirements=(
            ComputeRequirement(
                requirement_id="compute",
                necessity="required",
                reportability="optional",
                cpu_class="standard",
                memory_class="small",
                disk_bytes=10_000,
            ),
        ),
    )
    receipts = (record_preflight(plan, "compute", status="pass", duration_ms=1),)
    redactions = RedactionSet.from_resolved_values(())
    preflight = seal_preflight(plan, receipts, redactions)
    budget = BudgetAuthorization(
        episode_id=episode_id,
        allowances=tuple(
            CappedAllowance(resource=resource, value=1_000_000, reporting="optional")
            for resource in BUDGET_RESOURCES
        ),
    )
    reservation = reserve_budget(
        budget,
        "episode",
        [ResourceAmount(resource=resource, value=1) for resource in BUDGET_RESOURCES],
    )
    return EpisodeSpec(
        episode_id=episode_id,
        task_id=TASK_ID,
        task_digest=DIGEST,
        input_digest=OTHER_DIGEST,
        benchmark_id="benchmark",
        benchmark_release="release-1",
        environment_digest=DIGEST,
        environment_release="env-1",
        config_digest=OTHER_DIGEST,
        harness_version="0.0.0",
        harness_git_revision=DIGEST,
        requested_route=ROUTE,
        dependency_plan=plan,
        budget=budget,
        preflight=preflight,
        reservations=(reservation,),
    )


def build_config(tmp_path: Path, **overrides: Any) -> EpisodeConfig:
    evidence = tmp_path / "evidence"
    artifacts = tmp_path / "artifacts"
    rescue = tmp_path / "rescue"
    for path in (evidence, artifacts, rescue):
        path.mkdir(parents=True, exist_ok=True)
    defaults: dict[str, Any] = {
        "evidence_root": evidence,
        "artifact_root": artifacts,
        "rescue_root": rescue,
        "max_steps": 4,
        "prepare_timeout": 5.0,
        "reset_timeout": 5.0,
        "step_timeout": 5.0,
        "score_timeout": 5.0,
        "cleanup_timeout": 5.0,
        "ask_deadline_ms": 1000,
        "handshake_timeout": 5.0,
    }
    defaults.update(overrides)
    return EpisodeConfig(**defaults)


class FakeAdapter:
    """An in-process adapter that answers the protocol exactly as a worker does.

    Failure injection is per method and per call index so a test can name the
    precise cutpoint it cares about ("die on the second execute") without
    reimplementing the protocol for each scenario.
    """

    def __init__(
        self,
        tmp_path: Path,
        episode_id: str,
        *,
        score: ScoreArtifact | None = None,
        cleanup_status: str = "succeeded",
        failures: dict[str, BaseException] | None = None,
        fail_after: dict[str, int] | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.episode_id = episode_id
        self.score = score or ScoreArtifact(status="scored", binary=1)
        self.cleanup_status = cleanup_status
        self.failures = failures or {}
        self.fail_after = fail_after or {}
        self.calls: list[str] = []
        self.terminated = False
        self.sequence = 0
        self.current = observation(episode_id, 0)
        self._counts: dict[str, int] = {}

    async def handshake(self, *, timeout: float = 10.0) -> Handshake:
        del timeout
        self.calls.append("handshake")
        self._maybe_fail("handshake")
        return handshake(self.tmp_path)

    async def terminate(self) -> None:
        self.terminated = True

    def _maybe_fail(self, method: str) -> None:
        count = self._counts.get(method, 0)
        self._counts[method] = count + 1
        error = self.failures.get(method)
        if error is None:
            return
        if count >= self.fail_after.get(method, 0):
            raise error

    async def _call_raw(self, method: Any, params: Any, result_type: Any, *, timeout: float) -> Any:
        del result_type, timeout
        self.calls.append(method)
        self._maybe_fail(method)
        if method == "inspect_requirements":
            return RequirementsResult(requirements=())
        if method == "prepare":
            return PrepareResult(cleanup_plan=cleanup_plan(self.episode_id))
        if method == "reset_start":
            return AckResult()
        if method == "observe":
            return ObservationResult(observation=self.current)
        if method == "execute":
            self.sequence += 1
            output = observation(self.episode_id, self.sequence)
            receipt = ExecutionReceipt(
                operation_id=params.operation_id,
                action_batch_id=params.action_batch_id,
                input_observation_id=self.current.observation_id,
                output_observation_id=output.observation_id,
                sequence=output.sequence,
            )
            self.current = output
            return ExecuteResult(observation=output, receipt=receipt)
        if method == "ask_user_exchange":
            return AskUserExchangeResult(ask_id=params.ask_id, accepted=True)
        if method == "score":
            return ScoreResult(score=self.score)
        if method == "cleanup":
            return CleanupResult(
                outcomes=tuple(
                    CleanupOutcome(
                        action_id=action_id,
                        status=self.cleanup_status,  # pyright: ignore[reportArgumentType]
                        evidence_code="released",
                        duration_ms=1,
                    )
                    for action_id in params.action_ids
                )
            )
        if method in ("close", "begin_rescue"):
            return AckResult()
        raise AssertionError(f"unexpected adapter method {method}")


class ScriptedModel:
    """Emits a fixed sequence of decisions, then finishes."""

    def __init__(self, script: Sequence[str] | None = None, *, error: BaseException | None = None):
        self.script = list(script or ["finish"])
        self.error = error
        self.calls = 0
        # Overridable so a test can simulate a provider serving a route other
        # than the pinned one.
        self.route = ROUTE

    async def decide(
        self, observation: Observation, transcript: Sequence[Observation]
    ) -> ModelDecision:
        del transcript
        if self.error is not None:
            raise self.error
        kind = self.script[self.calls] if self.calls < len(self.script) else "finish"
        self.calls += 1
        return ModelDecision(
            action_batch=_batch(observation, kind),
            route=self.route,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
            cost_micros=7,
        )


def _batch(current: Observation, kind: str) -> ActionBatch:
    if kind == "finish":
        actions: list[dict[str, Any]] = [
            {
                "kind": "finish",
                "observation_id": current.observation_id,
                "status": "done",
                "reason": "task complete",
            }
        ]
    elif kind == "ask":
        actions = [
            {
                "kind": "ask_user",
                "observation_id": current.observation_id,
                "request_id": f"ask-{uuid.uuid4().hex[:8]}",
                "question": "What next?",
            }
        ]
    else:
        actions = [
            {
                "kind": "wait",
                "observation_id": current.observation_id,
                "duration_ms": 1,
            }
        ]
    return ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": current.task_id,
            "episode_id": current.episode_id,
            "observation_id": current.observation_id,
            "actions": actions,
        },
        strict=True,
    )


class RecordingResponder:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def ask(self, prompt: str, deadline_ms: int) -> str | None:
        del deadline_ms
        self.prompts.append(prompt)
        return self.answer


@pytest.fixture
def episode_id() -> str:
    # Lineage is process-global per episode ID, so every test needs its own.
    return f"ep-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def fake_rescue() -> Callable[..., Any]:
    async def _rescue(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs

        class _Aggregate:
            complete = True
            descriptor_id = descriptor.descriptor_id

        return _Aggregate()

    return _rescue


def artifact_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payloads(root: Path, payload_type: type[_P]) -> list[_P]:
    """Return every event payload of one concrete type from a bundle.

    ``EventRecord.payload`` is a wide union, so reading a field off it directly
    neither type-checks nor asserts what the test means. Selecting by type does
    both, and keeps a renamed payload field a compile-time failure rather than
    an ``AttributeError`` at run time.
    """

    from local_operator.evaluation.evidence.verify import verify_bundle

    return [
        event.payload
        for event in verify_bundle(root).events
        if isinstance(event.payload, payload_type)
    ]
