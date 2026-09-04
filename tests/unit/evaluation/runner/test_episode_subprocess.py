"""Real-worker coverage for the episode runner.

Every other runner test drives the protocol in-process. These spawn a genuine
interpreter running a genuine installed adapter, exactly as production does,
because the in-process fake cannot exercise process death: the poison, rescue,
and abandonment paths only mean something if a worker can actually stop
existing mid-operation.

The fixture construction follows ``adapters/test_launch.py``, which is the
established way to build a real interpreter and a real distribution here.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector
from local_operator.evaluation.adapters.discovery import workspace_digest
from local_operator.evaluation.adapters.supervisor import AdapterSupervisor
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.runner.episode import EpisodeRunner
from tests.unit.evaluation.copied_interpreter import (
    copied_interpreter,
    site_packages_of,
)
from tests.unit.evaluation.runner.conftest import (
    ScriptedModel,
    build_config,
    build_spec,
)

pytestmark = pytest.mark.slow

RELEASE_DIGEST = "b" * 64

# A functioning adapter: it holds one observation, advances it on execute, and
# grades trivially. Everything it returns has to satisfy the parent's verifier,
# so the observation identity is recomputed the same way the protocol does.
_ADAPTER_SOURCE = """
from importlib.metadata import distribution

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    CleanupOutcome,
    CleanupResult,
    ExecuteResult,
    ExecutionReceipt,
    ObservationPhaseError,
    ObservationResult,
    PrepareResult,
    RequirementsResult,
    ScoreResult,
    observation_content_id,
)
from local_operator.evaluation.adapters.discovery import distribution_digest
from local_operator.evaluation.evidence.models import ScoreArtifact
from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan
from local_operator.evaluation.protocol import (
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)


# A genuinely valid 1x1 PNG, constructed rather than pasted so its CRCs are
# right: the parent runs the real media validator over these bytes, and a
# hand-mangled blob would be refused for the wrong reason.
def _png_bytes():
    import struct
    import zlib

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\\x89PNG\\r\\n\\x1a\\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\\x00\\x00\\x00\\x00"))
        + chunk(b"IEND", b"")
    )


# Write frame bytes into the PARENT-SUPPLIED root, named by digest.
#
# This is the point of the artifact_root field: this process's environment was
# stripped by the supervisor, so `root` arrived on ResetStartParams and is the
# only way the worker knows where the parent will look. Content addressing is
# the adapter's half of confinement -- it publishes under a name it cannot
# choose, so it cannot aim the parent at anything but these bytes. `name` exists
# only so the escape tests can publish under a WRONG name.
def _publish(root, payload, name=None):
    import hashlib
    import os

    digest = hashlib.sha256(payload).hexdigest()
    path = os.path.join(root, name or digest)
    with open(path, "wb") as handle:
        handle.write(payload)
    return digest

# The supervisor builds the worker environment from a closed allowlist, so a
# cutpoint cannot ride in as an env var. The adapter reads it from a file beside
# its own module instead, which is ordinary adapter-owned input and leaves the
# production environment policy untouched.
def _cutpoint():
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_cutpoint")
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return ""


# Which frame behaviour this run exercises, read the same adapter-owned way as
# the cutpoint. "" means the original no-frame adapter, so every pre-existing
# test in this module keeps its exact previous behaviour.
def _frame_mode():
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_frame_mode")
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return ""


# How many times the OBSERVATION phase of execute should fail before it starts
# succeeding, armed the same adapter-owned way as the cutpoint. This reproduces
# the paid-episode failure exactly: the guest actions land, and the read-back
# that follows raises. The countdown lives in a FILE because it must survive
# across separate execute calls and be observable from the parent.
def _observation_failures():
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tiny_observation_failures"
    )
    try:
        with open(path) as handle:
            return int(handle.read().strip() or "0")
    except (OSError, ValueError):
        return 0


# Burn one armed failure and report whether this call should fail. Decrementing
# on disk is what makes the failure TRANSIENT rather than permanent: the
# exhausted-retries test arms more failures than the runner has attempts, and
# the recovery test arms fewer.
def _consume_observation_failure():
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tiny_observation_failures"
    )
    remaining = _observation_failures()
    if remaining <= 0:
        return False
    with open(path, "w") as handle:
        handle.write(str(remaining - 1))
    return True


# Whether reset_start should import a task module from the WORKSPACE (the
# worker's cwd), the way upstream OSWorld's ``instantiate_task`` does. Armed
# through a file beside the module like the cutpoint. This is the import that
# left ``tasks/__pycache__/task_001.cpython-312.pyc`` in the first paid
# episode's pinned workspace and broke its rescue.
def _maybe_import_workspace_task():
    import importlib.util
    import os

    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_workspace_import")
    if not os.path.exists(marker):
        return
    for relative, expected in (
        (("tasks", "task_001.py"), "source"),
        (("tasks", "pkg", "mod.py"), "nested"),
    ):
        path = os.path.join(os.getcwd(), *relative)
        spec = importlib.util.spec_from_file_location(
            "osworld_task_" + relative[-1].replace(".py", ""), path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.MARK == expected


# Build the frame list for one observation, publishing real bytes into the root
# the PARENT supplied on reset_start. The dishonest modes each break exactly one
# clause of the parent's verification so the refusals cannot pass for each other.
def _frames(root, sequence):
    import hashlib
    import os

    mode = _frame_mode()
    if not mode or root is None:
        return ()
    payload = _png_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    byte_count = len(payload)
    if mode == "honest":
        _publish(root, payload)
    elif mode == "outside_root":
        # Publish into a sibling directory and name it by its true digest. The
        # parent only ever opens `root`, so the bytes are simply not there.
        outside = os.path.join(os.path.dirname(root), "outside")
        os.makedirs(outside, exist_ok=True)
        _publish(outside, payload)
    elif mode == "symlink_escape":
        # Real bytes outside the root, with an in-root symlink pointing at them.
        # Only O_NOFOLLOW distinguishes this from the honest case.
        outside = os.path.join(os.path.dirname(root), "outside")
        os.makedirs(outside, exist_ok=True)
        target = os.path.join(outside, digest)
        with open(target, "wb") as handle:
            handle.write(payload)
        link = os.path.join(root, digest)
        if not os.path.lexists(link):
            os.symlink(target, link)
    elif mode == "digest_mismatch":
        # Different bytes of the SAME length under the honest digest's name.
        # Length-changing tampering would be caught by the size check first,
        # leaving the digest comparison itself unexercised.
        tampered = payload[:-1] + bytes([payload[-1] ^ 0xFF])
        _publish(root, tampered, name=digest)
    elif mode == "byte_count_mismatch":
        _publish(root, payload)
        byte_count = byte_count + 1
    elif mode == "fifo":
        # A FIFO under an honest-looking digest name. Nothing ever writes to it,
        # so a parent that opens it without O_NONBLOCK blocks in the kernel
        # forever -- the liveness attack, not a content one.
        path = os.path.join(root, digest)
        if not os.path.lexists(path):
            os.mkfifo(path)
    return (
        FrameRef(
            frame_id="frame-%%d" %% sequence,
            artifact=ArtifactRef(
                sha256=digest, media_type="image/png", byte_count=byte_count
            ),
            geometry=FrameGeometry(
                native=FrameSize(width=1, height=1),
                model_visible=FrameSize(width=1, height=1),
            ),
        ),
    )


def _observation(task_id, episode_id, sequence, root=None):
    provisional = Observation(
        task_id=task_id,
        episode_id=episode_id,
        sequence=sequence,
        observation_id="provisional",
        text="state-%%d" %% sequence,
        frames=_frames(root, sequence),
    )
    return provisional.model_copy(
        update={"observation_id": observation_content_id(provisional)}
    )


class TinyAdapter:
    def __init__(self, metadata):
        self.metadata = metadata
        self.task_id = "task-1"
        self.episode_id = None
        self.sequence = 0
        self.current = None
        # Learned ONLY from reset_start; the stripped environment offers no
        # other route to it, which is exactly what this fixture proves.
        self.artifact_root = None
        # How many times a batch was actually APPLIED, published beside the
        # module so the parent can assert a resumed call did not re-apply one.
        self.applied = 0
        self.applied_path = __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
            "tiny_applied_count",
        )

    def _maybe_die(self, name):
        if _cutpoint() == name:
            __import__("os").kill(__import__("os").getpid(), 9)

    async def inspect_requirements(self, params):
        self._maybe_die("inspect_requirements")
        return RequirementsResult(requirements=())

    async def prepare(self, params):
        self._maybe_die("prepare")
        self.episode_id = params.episode_id
        return PrepareResult(
            cleanup_plan=CleanupPlan(
                episode_id=params.episode_id,
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
        )

    async def reset_start(self, params):
        self._maybe_die("reset_start")
        _maybe_import_workspace_task()
        self.task_id = params.task_id
        self.episode_id = params.episode_id
        self.artifact_root = params.artifact_root
        self.sequence = 0
        self.current = _observation(
            self.task_id, self.episode_id, 0, self.artifact_root
        )
        return AckResult()

    async def observe(self, params):
        return ObservationResult(observation=self.current)

    async def execute(self, params):
        self._maybe_die("execute")
        # The MUTATION. Recorded so a test can prove a resumed call did not
        # apply the batch a second time -- the safety property the whole
        # observation-phase contract rests on.
        if not params.resume_observation:
            self.applied += 1
            with open(self.applied_path, "w") as handle:
                handle.write(str(self.applied))

        # PAST THE POINT OF NO RETURN: from here a failure is a failure to
        # READ, which is what ObservationPhaseError declares. The sequence is
        # advanced only after the observation is built, so a failed attempt
        # leaves the adapter where it was and the resumed attempt yields the
        # sequence the parent expects.
        if _consume_observation_failure():
            raise ObservationPhaseError("environment returned no screenshot frame")
        output = _observation(
            self.task_id, self.episode_id, self.sequence + 1, self.artifact_root
        )
        self.sequence += 1
        receipt = ExecutionReceipt(
            operation_id=params.operation_id,
            action_batch_id=params.action_batch_id,
            input_observation_id=self.current.observation_id,
            output_observation_id=output.observation_id,
            sequence=output.sequence,
        )
        self.current = output
        return ExecuteResult(observation=output, receipt=receipt)

    async def ask_user_exchange(self, params):
        raise NotImplementedError

    async def score(self, params):
        self._maybe_die("score")
        return ScoreResult(score=ScoreArtifact(status="scored", binary=1))

    async def cleanup(self, params):
        self._maybe_die("cleanup")
        return CleanupResult(
            outcomes=tuple(
                CleanupOutcome(
                    action_id=action_id,
                    status="succeeded",
                    evidence_code="released",
                    duration_ms=1,
                )
                for action_id in params.action_ids
            )
        )

    async def close(self, params):
        return AckResult()


def create():
    installed = distribution("tiny-runner-adapter")
    return TinyAdapter(
        AdapterMetadata(
            adapter_id="tiny-runner",
            distribution="tiny-runner-adapter",
            version="1.0",
            entry_point="tiny_runner_adapter:create",
            package_digest=distribution_digest(installed),
            release_digest="%s",
            schema_version="1.4",
            capabilities=AdapterCapabilities(
                routes=("computer",), ask_user=False, scoring=True
            ),
        )
    )
""" % RELEASE_DIGEST


def _install_adapter(site: Path) -> str:
    # The copied venv already carries the repo .pth (``copied_interpreter``);
    # only the adapter distribution is written here.
    module = site / "tiny_runner_adapter.py"
    module.write_text(_ADAPTER_SOURCE)
    info = site / "tiny_runner_adapter-1.0.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: tiny-runner-adapter\nVersion: 1.0\n"
    )
    (info / "entry_points.txt").write_text(
        "[local_operator.evaluation_adapters.v1]\n" "tiny-runner = tiny_runner_adapter:create\n"
    )
    rows: list[list[str]] = []
    for path in sorted([module, info / "METADATA", info / "entry_points.txt"]):
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append([str(path.relative_to(site)), f"sha256={digest}", str(len(data))])
    rows.append([str((info / "RECORD").relative_to(site)), "", ""])
    target = io.StringIO()
    csv.writer(target, lineterminator="\n").writerows(rows)
    (info / "RECORD").write_text(target.getvalue())
    from importlib.metadata import PathDistribution

    return distribution_digest_of(PathDistribution(info))


def distribution_digest_of(distribution: Any) -> str:
    from local_operator.evaluation.adapters.discovery import distribution_digest

    return distribution_digest(distribution)


@pytest.fixture
def adapter_site(tmp_path: Path) -> Path:
    """The installed adapter's site-packages, where the cutpoint file lives."""

    return site_packages_of(copied_interpreter(tmp_path / "venv"))


@pytest.fixture
def real_selector(tmp_path: Path, adapter_site: Path) -> AdapterSelector:
    executable = next(
        item
        for item in sorted((tmp_path / "venv" / "bin").glob("python3.*"))
        if item.is_file() and not item.is_symlink()
    )
    site = adapter_site
    package_digest = _install_adapter(site)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    # Task modules in the pinned workspace, as the OSWorld layout has, so a
    # test can make the worker import from the verified tree. The nested
    # module proves ``-B`` covers subdirectory caches, not only the top level.
    (workspace / "tasks").mkdir(exist_ok=True)
    (workspace / "tasks" / "task_001.py").write_text("MARK = 'source'\n")
    (workspace / "tasks" / "pkg").mkdir(exist_ok=True)
    (workspace / "tasks" / "pkg" / "mod.py").write_text("MARK = 'nested'\n")
    return AdapterSelector(
        schema_version="1.4",
        adapter_id="tiny-runner",
        distribution="tiny-runner-adapter",
        version="1.0",
        entry_point="tiny_runner_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


def _subprocess_config(tmp_path: Path, **overrides: Any) -> Any:
    """Config with timeouts sized for a REAL interpreter spawn.

    The in-process default of 5s is ample for a fake adapter but marginal here:
    a real worker must start CPython and import pydantic and ``local_operator``
    before it can answer the handshake. Under parallel load that can exceed 5s,
    and the episode then fails at ``prepare`` -- reporting ``failed_pre_bundle``
    instead of reaching the cutpoint under test, which turns a genuine
    assertion into a flake about machine speed rather than about the runner.
    """

    return build_config(
        tmp_path,
        handshake_timeout=60.0,
        prepare_timeout=60.0,
        reset_timeout=60.0,
        step_timeout=60.0,
        score_timeout=60.0,
        cleanup_timeout=60.0,
        **overrides,
    )


def _arm_cutpoint(site: Path, cutpoint: str | None) -> None:
    """Tell the installed adapter which operation should kill its process."""

    marker = site / "tiny_cutpoint"
    if cutpoint is None:
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(cutpoint)


async def _accepting_rescue(descriptor: Any, **kwargs: Any) -> Any:
    """Stand in for a real rescue so a refusal is not misread as a leak.

    A refused frame poisons the session, which legitimately demands rescue. The
    tests using this are about confinement, not about reclamation, so rescue is
    reported complete and the assertions stay on the refusal itself.
    """

    del kwargs

    class _Aggregate:
        complete = True
        descriptor_id = descriptor.descriptor_id

    return _Aggregate()


def _arm_workspace_import(site: Path, enabled: bool) -> None:
    marker = site / "tiny_workspace_import"
    if enabled:
        marker.write_text("1")
    else:
        marker.unlink(missing_ok=True)


def _arm_frames(site: Path, mode: str | None) -> None:
    """Tell the installed adapter how to publish observation frames.

    Armed through a file beside the adapter module for the same reason the
    cutpoint is: the worker's environment is built from a closed allowlist, so a
    test cannot hand it a mode any other way without weakening the policy under
    test. ``None`` restores the frameless adapter the other tests here expect.
    """

    marker = site / "tiny_frame_mode"
    if mode is None:
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(mode)


def _arm_observation_failures(site: Path, count: int) -> None:
    """Arm N consecutive observation-phase failures in the installed adapter.

    Armed through a file for the same reason the cutpoint is: the worker's
    environment is built from a closed allowlist, so a test cannot hand it a
    mode any other way without weakening the policy under test.
    """

    marker = site / "tiny_observation_failures"
    if count <= 0:
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(str(count))


def _applied_count(site: Path) -> int:
    """How many times the adapter actually APPLIED a batch."""

    marker = site / "tiny_applied_count"
    try:
        return int(marker.read_text().strip())
    except (OSError, ValueError):
        return 0


def _retry_events(root: Path) -> list[dict[str, Any]]:
    """Every journalled observation-phase retry in a sealed bundle."""

    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines() if line]
    return [
        event
        for event in events
        if event["kind"] == "error"
        and event["payload"]["diagnostic_code"] == "observation-phase-retry"
    ]


@pytest.mark.asyncio
async def test_transient_observation_failure_does_not_destroy_the_episode(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
) -> None:
    """The paid-episode failure, reproduced end to end against a real worker.

    Five real episodes died between steps 9 and 32 like this: a burstable VM
    starved its guest screenshot server, the read-back after a committed step
    failed, and the whole episode was destroyed along with every valid step
    before it. Here the first step's observation phase fails twice and then
    recovers; the episode must reach a normal scored ending.
    """

    _arm_cutpoint(adapter_site, None)
    _arm_frames(adapter_site, "honest")
    _arm_observation_failures(adapter_site, 2)
    # No sleeping in a unit test; the bound under test is the ATTEMPT count.
    config = _subprocess_config(tmp_path, observation_retry_delay=0.0)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=real_selector,
        model=ScriptedModel(["step", "step", "finish"]),
        launch=AdapterSupervisor.launch,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    # Never poisoned, so no rescue was demanded and no VM was abandoned.
    assert outcome.rescue_required is False
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None
    assert report.counters.environment_step_count == 2

    # THE SAFETY PROPERTY: two steps ran, so the batch was applied exactly
    # twice. A resumed call that re-applied its batch would show more.
    assert _applied_count(adapter_site) == 2

    # THE HONESTY PROPERTY: both degraded reads are in the evidence, so a
    # reader sees the environment faltered rather than a suspiciously clean run.
    retries = _retry_events(root)
    assert len(retries) == 2
    assert all(event["payload"]["category"] == "environment" for event in retries)
    assert all(event["payload"]["retryable"] is True for event in retries)


@pytest.mark.asyncio
async def test_exhausted_observation_retries_still_fail_and_seal_the_bundle(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
) -> None:
    """Recovery is BOUNDED: past the bound the episode still dies cleanly.

    A harness that retried forever would hold a paid VM open against a dead
    environment. The bundle must still seal and verify, so an unrecoverable
    outage stays auditable rather than merely survived.
    """

    _arm_cutpoint(adapter_site, None)
    _arm_frames(adapter_site, "honest")
    # One more failure than the runner has attempts.
    _arm_observation_failures(adapter_site, 5)
    config = _subprocess_config(tmp_path, observation_retry_delay=0.0)
    rescued: list[Any] = []

    async def recording_rescue(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        rescued.append(descriptor)

        class _Aggregate:
            complete = True
            descriptor_id = descriptor.descriptor_id

        return _Aggregate()

    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=real_selector,
        model=ScriptedModel(["step", "step", "finish"]),
        launch=AdapterSupervisor.launch,
        rescue=recording_rescue,
    )

    outcome = await runner.run()

    assert outcome.status == "failed", outcome.diagnostic
    # The ORIGINAL cause is recorded, not the recovery that could not fix it.
    assert outcome.diagnostic is not None
    assert "no screenshot frame" in outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]

    # Exactly the configured bound: 3 retries, then the failure propagates.
    assert len(_retry_events(root)) == 3
    # The step never completed, so no step event was written for it.
    assert report.counters is not None
    assert report.counters.environment_step_count == 0


@pytest.mark.asyncio
async def test_real_worker_completes_a_scored_episode(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
) -> None:
    _arm_cutpoint(adapter_site, None)
    runner = EpisodeRunner(
        build_spec(episode_id),
        _subprocess_config(tmp_path),
        selector=real_selector,
        model=ScriptedModel(["step", "step", "finish"]),
        launch=AdapterSupervisor.launch,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None
    assert report.counters.environment_step_count == 2


@pytest.mark.parametrize(
    "cutpoint",
    ["prepare", "reset_start", "execute", "score", "cleanup"],
)
@pytest.mark.asyncio
async def test_worker_killed_at_each_cutpoint_stays_coherent(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
    cutpoint: str,
) -> None:
    """SIGKILL mid-operation must still yield a coherent bundle or abandonment."""

    _arm_cutpoint(adapter_site, cutpoint)
    rescued: list[Any] = []

    async def recording_rescue(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        rescued.append(descriptor)

        class _Aggregate:
            complete = True
            descriptor_id = descriptor.descriptor_id

        return _Aggregate()

    runner = EpisodeRunner(
        build_spec(episode_id),
        _subprocess_config(tmp_path),
        selector=real_selector,
        model=ScriptedModel(["step", "finish"]),
        launch=AdapterSupervisor.launch,
        rescue=recording_rescue,
    )

    outcome = await runner.run()

    # The episode must never claim success when its worker was killed.
    assert outcome.status in ("failed", "abandoned", "failed_pre_bundle")
    if cutpoint == "cleanup":
        # Scoring already happened, so the grade stands; what the kill costs is
        # the cleanup guarantee, and the run is labelled for exactly that.
        assert outcome.reportability_label == "cleanup_incomplete"
        assert outcome.rescue_required is True
    else:
        assert outcome.score is None or outcome.score.status == "unscored"
    if cutpoint == "prepare":
        # No manifest is possible before prepare returns a cleanup plan.
        assert outcome.status == "failed_pre_bundle"
        assert outcome.bundle_root is None
        return
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    if report.abandonment is not None:
        # An abandoned bundle is a coherent terminal in its own right.
        assert report.abandonment.reason in (
            "ambiguous_finalization",
            "infrastructure_failure",
        )
    else:
        assert report.valid, [issue.code for issue in report.issues]
        assert report.outcome is not None
        # A kill during cleanup happens after grading, so the sealed score is
        # real; every earlier cutpoint dies before a score can exist.
        expected = "scored" if cutpoint == "cleanup" else "unscored"
        assert report.outcome.result.status == expected
        assert report.outcome.reportable is False
    # A dead worker leaves resources only the persisted descriptor can reclaim.
    assert rescued, "a killed worker must trigger rescue"


@pytest.mark.asyncio
async def test_killed_worker_process_group_is_reaped(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
) -> None:
    _arm_cutpoint(adapter_site, "execute")
    supervisors: list[AdapterSupervisor] = []

    def recording_launch(selector: AdapterSelector) -> AdapterSupervisor:
        supervisor = AdapterSupervisor.launch(selector)
        supervisors.append(supervisor)
        return supervisor

    async def rescue(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs

        class _Aggregate:
            complete = True
            descriptor_id = descriptor.descriptor_id

        return _Aggregate()

    runner = EpisodeRunner(
        build_spec(episode_id),
        _subprocess_config(tmp_path),
        selector=real_selector,
        model=ScriptedModel(["step", "finish"]),
        launch=recording_launch,
        rescue=rescue,
    )

    await runner.run()

    assert supervisors
    supervisor = supervisors[0]
    assert await asyncio.to_thread(supervisor.process.poll) is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(supervisor.pgid, 0)


@pytest.mark.asyncio
async def test_real_worker_publishes_a_frame_the_parent_verifies_and_bundles(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
) -> None:
    """A genuinely spawned adapter delivers observation frames end to end.

    This is the regression guard for the gap that ``artifact_root`` on
    ``ResetStartParams`` closes. The worker is a real process whose environment
    was built from the supervisor's closed allowlist, so the ONLY way it can
    know where to write frame bytes is the field the parent sent it. Before that
    field existed an out-of-process adapter could not publish a frame at all,
    and the in-process fakes hid it because they shared the parent's memory.

    The assertion chain is deliberately the full one: the parent read the bytes
    (``verify_artifact``), accepted them (digest, size, media), copied them into
    the bundle, and the sealed bundle still verifies. A test that stopped at
    "the episode completed" would pass against an adapter that published no
    frames whatsoever, which is precisely the broken state.
    """

    _arm_cutpoint(adapter_site, None)
    _arm_frames(adapter_site, "honest")
    config = _subprocess_config(tmp_path)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=real_selector,
        model=ScriptedModel(["step", "finish"]),
        launch=AdapterSupervisor.launch,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]

    # The worker really wrote into the parent-chosen directory, and the bytes
    # there are the exact PNG the parent then admitted into the bundle.
    published = [item for item in config.artifact_root.iterdir() if item.is_file()]
    assert published, "the spawned worker published nothing into the parent's root"
    payload = published[0].read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert published[0].name == hashlib.sha256(payload).hexdigest()

    # Every observation carried a frame, and the frame's bytes are in the
    # bundle: an episode that silently dropped frames would fail here.
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    observations = [item for item in events if item.get("kind") == "observation"]
    assert observations, "no observation events were recorded"
    assert all(event["payload"]["artifacts"] for event in observations)
    assert {
        artifact["sha256"] for event in observations for artifact in event["payload"]["artifacts"]
    } == {hashlib.sha256(payload).hexdigest()}
    assert all(
        artifact["media_type"] == "image/png" and artifact["byte_count"] == len(payload)
        for event in observations
        for artifact in event["payload"]["artifacts"]
    )
    # The bundle stores artifacts content-addressed, so the frame's digest being
    # present is proof the parent ingested these exact bytes rather than
    # recording a reference to something it never read.
    assert (root / "artifacts" / hashlib.sha256(payload).hexdigest()).read_bytes() == payload


@pytest.mark.parametrize(
    ("mode", "refusal"),
    [
        # Bytes exist, but only outside the one directory the parent opens.
        ("outside_root", "artifact path is unsafe or unavailable"),
        # An in-root name that resolves outside it; O_NOFOLLOW is the guard.
        ("symlink_escape", "artifact path is unsafe or unavailable"),
        # In-root bytes of the declared length whose content is not the digest.
        ("digest_mismatch", "artifact digest differs"),
        # In-root bytes of the right content but a lied-about length.
        ("byte_count_mismatch", "artifact is not a matching regular file"),
        # A FIFO nobody writes to: the LIVENESS case. The S_ISREG check sits
        # behind the open, so only O_NONBLOCK stops this wedging the parent
        # forever -- and it is refused by that same existing clause.
        ("fifo", "artifact is not a matching regular file"),
    ],
)
@pytest.mark.asyncio
async def test_worker_cannot_deliver_frames_from_outside_the_root(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
    mode: str,
    refusal: str,
) -> None:
    """Being told the root grants no authority to read outside it, or to lie.

    Sending the worker a writable path is only safe because the parent keeps
    resolving artifacts by content, inside that one directory descriptor, with
    ``O_NOFOLLOW``. Each mode here breaks a different clause of that check, so a
    weakened ``verify_artifact`` cannot be masked by another clause still
    holding -- verified by reintroducing the weakness: dropping ``O_NOFOLLOW``
    let ``symlink_escape`` reach ``completed``, and removing the digest
    comparison turned ``digest_mismatch`` red.

    ``fifo`` is the odd one out and belongs here anyway: it attacks LIVENESS
    rather than content. The refusal it lands on is shared with
    ``byte_count_mismatch``, so the assertion that matters is that the episode
    TERMINATES at all -- without ``O_NONBLOCK`` the open never returns, and
    because ``verify_artifact`` runs on the event-loop thread after the mutating
    call's ``wait_for`` has closed, nothing upstream can time it out.

    The episode must not report success, and it must never seal a bundle that
    claims frames it could not verify.
    """

    _arm_cutpoint(adapter_site, None)
    _arm_frames(adapter_site, mode)
    runner = EpisodeRunner(
        build_spec(episode_id),
        _subprocess_config(tmp_path),
        selector=real_selector,
        model=ScriptedModel(["step", "finish"]),
        launch=AdapterSupervisor.launch,
        rescue=_accepting_rescue,
    )

    outcome = await runner.run()

    assert outcome.status != "completed"
    # Pinning the exact refusal keeps each mode honest about WHICH clause caught
    # it. Without this, length-changing tampering is rejected by the size check
    # and the digest comparison goes untested while the suite stays green.
    assert outcome.diagnostic is not None and refusal in outcome.diagnostic
    assert outcome.score is None or outcome.score.status == "unscored"
    root = outcome.bundle_root
    if root is None:
        return
    report = verify_bundle(root)
    # Whatever terminal it reached must be coherent: a refused frame may not
    # leave a bundle that both verifies and reports a reportable success.
    if report.abandonment is None:
        assert report.valid, [issue.code for issue in report.issues]
        assert report.outcome is not None
        assert report.outcome.reportable is False


@pytest.mark.asyncio
async def test_worker_importing_from_the_workspace_leaves_no_bytecode_behind(
    tmp_path: Path,
    episode_id: str,
    real_selector: AdapterSelector,
    adapter_site: Path,
) -> None:
    """The workspace-digest drift from bundle ep-6ea01a117eee, reproduced with
    a real worker: the adapter imports ``tasks/task_001.py`` from the pinned
    workspace during ``reset_start``. The worker runs with ``-B`` (and sets
    ``sys.dont_write_bytecode``), so no ``__pycache__`` may appear under the
    workspace, and the workspace still hashes to the selector's pin afterwards
    -- which is what a later rescue re-checks."""

    _arm_cutpoint(adapter_site, None)
    _arm_workspace_import(adapter_site, True)
    try:
        runner = EpisodeRunner(
            build_spec(episode_id),
            _subprocess_config(tmp_path),
            selector=real_selector,
            model=ScriptedModel(["step", "finish"]),
            launch=AdapterSupervisor.launch,
        )

        outcome = await runner.run()
    finally:
        _arm_workspace_import(adapter_site, False)

    assert outcome.status == "completed", outcome.diagnostic
    workspace = Path(real_selector.workspace)
    caches = [path for path in workspace.rglob("*") if path.name == "__pycache__"]
    assert caches == [], caches
    assert not list(workspace.rglob("*.pyc"))
    assert workspace_digest(str(workspace)) == real_selector.workspace_digest
