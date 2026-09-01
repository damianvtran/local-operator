"""In-process episodes: the REAL adapter over FakeProvider through a runner.

The headline evidence for this adapter is ``test_spawn.py``, which runs the
same chain in a genuinely spawned worker. These tests keep their place for a
different reason: they are FAST and they can inject failures the spawned path
cannot reach cheaply.

The adapter is called through a thin ``_call_raw`` shim, so everything the
PARENT is responsible for is still real — ``VerifiedAdapterSession``, the
``HostVerifier``, the lifecycle authorities, ``verify_artifact``, the evidence
writer, and the independent ``verify_bundle``. What the shim removes is the
process boundary and the RPC serialization, which is exactly what the spawned
tests exist to cover.

That division is deliberate. An episode per failure mode (evaluator raising,
an unanswered ask, a partial score) costs a venv build and two interpreter
spawns out-of-process; in-process it costs milliseconds. The failure modes
here are adapter-internal and provider-driven, so nothing about them depends
on the boundary — they would prove the same thing more slowly. The claims that
DO depend on the boundary (frames reaching the parent's artifact root, the
schema-1.1 field arriving over RPC, the shipped wheel loading at all) are
asserted only in ``test_spawn.py``, and none of them are asserted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterSelector,
    Handshake,
    PythonRuntime,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.adapters.osworld import fixtures
from tests.unit.evaluation.runner.conftest import (
    RecordingResponder,
    ScriptedModel,
    build_config,
    build_spec,
)

RELEASE_DIGEST = "d" * 64


class _AdapterSupervisorShim:
    """A ``_call_raw`` seam that drives the real in-process adapter.

    The real ``AdapterSupervisor`` spawns a worker and does RPC; this shim
    answers the same method/params/result_type signature by calling the real
    ``OSWorldV2Adapter`` directly, so every parent-side check (the verifier,
    the state machine, the receipts) runs against the adapter's real returns.
    """

    def __init__(self, adapter: OSWorldV2Adapter, selector: AdapterSelector) -> None:
        self._adapter = adapter
        self.selector = selector
        self.terminated = False

    async def handshake(self, *, timeout: float = 10.0) -> Handshake:
        del timeout
        return Handshake(
            selector=self.selector,
            metadata=self._adapter.metadata,
            python=PythonRuntime.current(),
            workspace_digest=self.selector.workspace_digest,
            selected_route="computer",
        )

    async def terminate(self) -> None:
        self.terminated = True

    async def _call_raw(self, method: str, params: Any, result_type: Any, *, timeout: float) -> Any:
        del timeout
        handler = getattr(self._adapter, method)
        return await handler(params)


def _selector(tmp_path: Path, workspace: Path, adapter: OSWorldV2Adapter) -> AdapterSelector:
    # The handshake re-validates that the adapter's metadata repeats the
    # selector's exact pins, so the selector must carry the digests the real
    # adapter computed for itself (the installed wheel's package digest and
    # the workspace's release digest), not fixture placeholders.
    metadata = adapter.metadata
    return AdapterSelector(
        schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.0",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=metadata.package_digest,
        release_digest=metadata.release_digest,
        python_executable=str(Path(__import__("sys").executable).resolve()),
        workspace=str(workspace),
        workspace_digest="c" * 64,
        route_capability="computer",
    )


def _adapter(tmp_path: Path, provider: FakeProvider) -> OSWorldV2Adapter:
    workspace = tmp_path / "workspace"
    tasks = workspace / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    (tasks / "task_plain.py").write_text(fixtures.PLAIN)
    # No artifact root is handed to the constructor: the adapter learns it from
    # ``ResetStartParams.artifact_root`` like any worker, so even in-process the
    # root travels the same path it does out-of-process.
    return OSWorldV2Adapter(
        provider_factory=lambda: provider,
        workspace_root=workspace,
    )


def _spec_with_task(episode_id: str) -> Any:
    from local_operator.evaluation.adapters.api import ScopedInfraValue

    spec = build_spec(episode_id)
    # The runner passes task_id to reset_start; the adapter loads
    # tasks/<task_id>.py from the workspace. Point the spec at the fixture.
    object.__setattr__(spec, "task_id", "task_plain")
    # The adapter's provisioning is derived from the task plus declared infra
    # values; the spec carries the non-secret account facts (subnet, SG, role,
    # client password, file base URL) exactly as a real host would resolve
    # them from the adapter's inspect_requirements names.
    infra = tuple(
        ScopedInfraValue(name=name, purpose="benchmark_compute", value=f"test-{name}")
        for name in (
            "AWS_REGION",
            "AWS_SUBNET_ID",
            "AWS_SECURITY_GROUP_ID",
            "AWS_SCHEDULER_ROLE_ARN",
            "OSWORLD_CLIENT_PASSWORD",
            "OSWORLD_FILE_BASE_URL",
        )
    )
    object.__setattr__(spec, "infra_values", infra)
    return spec


@pytest.mark.asyncio
async def test_fake_provider_episode_seals_a_valid_bundle(tmp_path: Path, episode_id: str) -> None:
    provider = FakeProvider(scripted_score=1.0)
    adapter = _adapter(tmp_path, provider)
    selector = _selector(tmp_path, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)

    runner = EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["step", "step", "finish"]),
        launch=lambda _selector: shim,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    assert outcome.score.binary == 1
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None
    assert report.counters.environment_step_count == 2
    # The fake exercised the real allocate/execute/evaluate/terminate path.
    assert provider.allocated is True
    assert provider.evaluate_calls == 1
    assert provider.terminated_refs == [f"lop-ep-{episode_id}"]


@pytest.mark.asyncio
async def test_partial_score_maps_to_ppm_in_a_real_bundle(tmp_path: Path, episode_id: str) -> None:
    provider = FakeProvider(scripted_score=0.5)
    adapter = _adapter(tmp_path, provider)
    selector = _selector(tmp_path, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)
    runner = EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["finish"]),
        launch=lambda _selector: shim,
    )
    outcome = await runner.run()
    assert outcome.score is not None and outcome.score.partial_ppm == 500_000


@pytest.mark.asyncio
async def test_an_evaluator_failure_abandons_rather_than_scores_zero(
    tmp_path: Path, episode_id: str
) -> None:
    # The evaluator raising must abandon the episode (rescue + teardown), NOT
    # seal a 0.0 the agent did not commit.
    provider = FakeProvider(fail_evaluate=True)
    adapter = _adapter(tmp_path, provider)
    selector = _selector(tmp_path, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)

    rescued: list[Any] = []

    async def recording_rescue(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        rescued.append(descriptor)

        class _Aggregate:
            complete = True
            descriptor_id = descriptor.descriptor_id

        return _Aggregate()

    runner = EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["finish"]),
        launch=lambda _selector: shim,
        rescue=recording_rescue,
    )
    outcome = await runner.run()
    assert outcome.status == "abandoned", outcome.diagnostic
    assert outcome.rescue_required is True
    # The abandoned bundle is still a coherent terminal the verifier reads.
    assert outcome.bundle_root is not None
    report = verify_bundle(outcome.bundle_root)
    assert report.abandonment is not None


@pytest.mark.asyncio
async def test_an_unanswered_ask_cancels_the_episode(tmp_path: Path, episode_id: str) -> None:
    provider = FakeProvider()
    adapter = _adapter(tmp_path, provider)
    selector = _selector(tmp_path, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)
    runner = EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["ask"]),
        responder=RecordingResponder(None),  # the user never answers
        launch=lambda _selector: shim,
    )
    outcome = await runner.run()
    assert outcome.status == "cancelled", outcome.diagnostic
    assert outcome.bundle_root is not None
    report = verify_bundle(outcome.bundle_root)
    assert report.valid, [issue.code for issue in report.issues]


@pytest.mark.asyncio
async def test_an_answered_ask_runs_the_exchange(tmp_path: Path, episode_id: str) -> None:
    # The task declares no user_simulator, so the adapter refuses the exchange
    # honestly (accepted=False) while the harness responder still supplies the
    # answer the model sees — the documented one-sided PR 1 wiring.
    provider = FakeProvider(has_user_simulator=False)
    adapter = _adapter(tmp_path, provider)
    selector = _selector(tmp_path, adapter._workspace_root, adapter)
    shim = _AdapterSupervisorShim(adapter, selector)
    runner = EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["ask", "finish"]),
        responder=RecordingResponder("the answer"),
        launch=lambda _selector: shim,
    )
    outcome = await runner.run()
    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.bundle_root is not None
    assert verify_bundle(outcome.bundle_root).valid
