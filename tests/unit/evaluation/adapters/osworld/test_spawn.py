"""Out-of-process coverage: the REAL adapter wheel in a REAL spawned worker.

Every other adapter test drives the protocol in this process, where the adapter
shares the parent's memory. These spawn a genuine interpreter running the
genuine installed wheel, exactly as production does, because that is the only
configuration in which the boundary is actually load-bearing: the supervisor
builds the child's environment from a closed allowlist (locale and temp), so a
spawned worker learns nothing ambiently — not the artifact root, not which
provider to run, not where the task corpus is.

That makes the sealed-bundle test below the real evidence for this adapter.
The worker must:

* be told the artifact root over RPC (``ResetStartParams.artifact_root``,
  schema 1.1) and write real PNG frames into the parent's directory;
* select its backend from the digest-pinned workspace, since an env var cannot
  survive the allowlist;
* produce observations the parent independently verifies (``verify_artifact``
  re-opens each frame under ``O_NOFOLLOW``/``O_NONBLOCK`` and re-hashes it);
* reach a sealed bundle that the independent ``verify_bundle`` accepts.

Marked ``slow`` like the existing subprocess suite: each spawn starts a real
CPython and imports pydantic and local_operator before answering.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import Handshake, ScopedInfraValue
from local_operator.evaluation.adapters.supervisor import AdapterSupervisor
from local_operator.evaluation.evidence.models import ObservationPayload
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers
from tests.unit.evaluation.runner.conftest import ScriptedModel, build_spec, payloads

pytestmark = pytest.mark.slow

_TASKS = {"task_plain": fixtures.PLAIN}


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the shipped wheel once; each test installs it into its own venv."""

    return spawn_helpers.build_adapter_wheel(tmp_path_factory.mktemp("wheel"))


def _spec_for_spawn(episode_id: str) -> Any:
    """An episode spec pointing at the fixture task, with the infra values the
    adapter's own ``inspect_requirements`` names."""

    spec = build_spec(episode_id)
    object.__setattr__(spec, "task_id", "task_plain")
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
async def test_real_wheel_handshakes_and_reaps(tmp_path: Path, adapter_wheel: Path) -> None:
    selector = spawn_helpers.build_spawnable_adapter(tmp_path, adapter_wheel, _TASKS)
    supervisor = AdapterSupervisor.launch(selector)
    try:
        handshake = await supervisor.handshake(timeout=60)
        assert isinstance(handshake, Handshake)
        assert handshake.selector == selector
        assert handshake.metadata.adapter_id == "osworld-v2"
        assert handshake.metadata.distribution == "lop-osworld-v2-adapter"
        assert handshake.workspace_digest == selector.workspace_digest
        assert handshake.metadata.capabilities.routes == ("computer",)
        assert handshake.metadata.capabilities.ask_user is True
        assert handshake.metadata.capabilities.scoring is True
    finally:
        await supervisor.terminate()
    assert supervisor.process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(supervisor.pgid, 0)


@pytest.mark.asyncio
async def test_spawned_worker_seals_a_verified_bundle(
    tmp_path: Path, adapter_wheel: Path, episode_id: str
) -> None:
    """THE headline: a real out-of-process episode ends in a verifiable bundle.

    This asserts the whole chain end to end with nothing shared in memory: the
    spawned worker resolved its provider from the pinned workspace, wrote 1920x1080
    PNG frames into the root the PARENT chose and named over RPC, the parent
    re-opened and re-hashed every one of those frames, and the sealed bundle
    verifies independently.
    """

    selector = spawn_helpers.build_spawnable_adapter(
        tmp_path,
        adapter_wheel,
        _TASKS,
        # The worker cannot be told this any other way: the environment is
        # stripped, so the backend selection has to live in the workspace whose
        # digest the handshake pins.
        provider={"provider": "fake", "scripted_score": 1.0},
    )
    config = spawn_helpers.spawn_config(tmp_path)

    runner = EpisodeRunner(
        _spec_for_spawn(episode_id),
        config,
        selector=selector,
        model=ScriptedModel(["step", "step", "finish"]),
        launch=AdapterSupervisor.launch,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    assert outcome.score.binary == 1
    assert outcome.reportability_label == "reportable"

    # The frames exist as real files in the PARENT's directory, named by their
    # own digest. A worker that never learned the root would have published
    # nothing here, and the episode would have failed at the first observation.
    published = sorted(item for item in config.artifact_root.iterdir() if item.is_file())
    assert published, "the spawned worker published no frames into the parent's root"

    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None
    assert report.counters.environment_step_count == 2

    # Every observation the bundle recorded carries the frame the worker wrote,
    # which is what proves the bytes survived the boundary rather than the
    # episode merely completing with empty observations.
    observations = payloads(root, ObservationPayload)
    assert len(observations) == 3  # reset + two steps
    assert all(payload.artifacts for payload in observations)

    # And those artifacts really are the PNG frames, not text or JSON.
    bundled = {artifact.sha256: artifact for report_ in (report,) for artifact in report_.artifacts}
    frame_refs = [
        bundled[artifact.sha256]
        for payload in observations
        for artifact in payload.artifacts
        if artifact.sha256 in bundled
    ]
    assert frame_refs, "no observation artifact resolved to a bundled artifact"
    assert any(ref.media_type == "image/png" for ref in frame_refs)


@pytest.mark.asyncio
async def test_spawned_worker_reports_a_partial_score(
    tmp_path: Path, adapter_wheel: Path, episode_id: str
) -> None:
    """A fractional evaluator score survives the boundary as exact ppm.

    Worth spawning separately because the protocol's metadata subset excludes
    floats: a fraction can only cross as ``partial_ppm``, and this is the only
    test that proves it does so through a real serialized RPC rather than a
    shared-memory object.
    """

    selector = spawn_helpers.build_spawnable_adapter(
        tmp_path,
        adapter_wheel,
        _TASKS,
        provider={"provider": "fake", "scripted_score": 0.5},
    )
    runner = EpisodeRunner(
        _spec_for_spawn(episode_id),
        spawn_helpers.spawn_config(tmp_path),
        selector=selector,
        model=ScriptedModel(["finish"]),
        launch=AdapterSupervisor.launch,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None
    assert outcome.score.partial_ppm == 500_000
    assert outcome.bundle_root is not None
    assert verify_bundle(outcome.bundle_root).valid
