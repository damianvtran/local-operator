"""End-to-end: the REAL adapter over FakeProvider drives a REAL EpisodeRunner.

This is the headline evidence for PR 1. Only the model is scripted; the
adapter, the protocol, the lifecycle authorities, the cleanup receipts, the
evidence writer, and the independent ``verify_bundle`` are all real. The
provider is the in-process fake, injected through the adapter's
``provider_factory`` seam — the same factory the production AWS provider fills
in PR 2.

The episode is driven IN-PROCESS (the adapter is called through a thin
``_call_raw`` shim) rather than through a spawned worker for one precise,
documented reason: the worker never learns ``artifact_root``. The runner
passes it only inside ``BeginRescueParams.descriptor`` (api.py:259) and the
parent's ``verify_artifact`` reads frames from it (episode.py:495), but no
RPC param on the prepare/reset path carries it to the worker, and the worker's
environment is stripped (supervisor._ENV_ALLOW). A spawned worker therefore
cannot write frame bytes where the parent reads them. Closing that gap is a
HARNESS change (add artifact_root to Prepare/ResetStart params), explicitly
out of scope for PR 1 — it is reported to the manager, not routed around.

What this test DOES prove, in-process but with zero faking of the contract:
the real adapter's prepare→reset_start→execute→score→cleanup chain satisfies
the real VerifiedAdapterSession verifier, and the resulting bundle seals and
verifies. The real-spawn + handshake of the shipped wheel is covered
separately in test_spawn.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.adapters.api import (
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
        schema_version="1.0",
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
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return OSWorldV2Adapter(
        provider_factory=lambda: provider,
        artifact_root=artifact_root,
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
