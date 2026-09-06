"""Real spawn coverage for the only path that can start an adapter worker.

Every other adapter test drives the protocol in-process. This module builds a
genuine interpreter and a genuine installed distribution, then launches through
``AdapterSupervisor`` exactly as production does, because a manufactured
single-link interpreter fixture previously hid a policy that rejected every real
CPython install.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector, Handshake
from local_operator.evaluation.adapters.discovery import (
    distribution_digest,
    resolve_launch,
    validate_resolved_launch,
    workspace_digest,
)
from local_operator.evaluation.adapters.supervisor import AdapterSupervisor
from tests.unit.evaluation.copied_interpreter import (
    copied_interpreter,
    site_packages_of,
)

RELEASE_DIGEST = "b" * 64
_ADAPTER_SOURCE = """from importlib.metadata import distribution

from local_operator.evaluation.adapters.api import AdapterCapabilities, AdapterMetadata
from local_operator.evaluation.adapters.discovery import distribution_digest


class TinyAdapter:
    def __init__(self, metadata):
        self.metadata = metadata

    async def inspect_requirements(self, params): raise NotImplementedError

    async def prepare(self, params): raise NotImplementedError

    async def reset_start(self, params): raise NotImplementedError

    async def observe(self, params): raise NotImplementedError

    async def execute(self, params): raise NotImplementedError

    async def ask_user_exchange(self, params): raise NotImplementedError

    async def score(self, params): raise NotImplementedError

    async def cleanup(self, params): raise NotImplementedError

    async def close(self, params): raise NotImplementedError


def create():
    installed = distribution("tiny-e2e-adapter")
    return TinyAdapter(
        AdapterMetadata(
            adapter_id="tiny-e2e",
            distribution="tiny-e2e-adapter",
            version="1.0",
            entry_point="tiny_e2e_adapter:create",
            package_digest=distribution_digest(installed),
            release_digest="%s",
            schema_version="1.6",
            capabilities=AdapterCapabilities(
                routes=("computer",), ask_user=False, scoring=False
            ),
        )
    )
""" % RELEASE_DIGEST


# A second spawnable distribution whose teardown is only possible with what
# ``begin_rescue`` delivers. It stands in for the AWS provider without a cloud:
# ``cleanup`` can confirm a release ONLY when begin_rescue handed it the
# descriptor's refs AND the freshly resolved secret, mirroring a rescue worker
# that enters at HANDSHAKEN holding no credential from any other source. If the
# worker stores the descriptor without forwarding it, ``_provider`` stays None
# and cleanup reports the honest "could not look" code -- which is exactly the
# production defect this module now pins.
_RESCUE_ADAPTER_SOURCE = '''from importlib.metadata import distribution

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    CleanupOutcome,
    CleanupResult,
)
from local_operator.evaluation.adapters.discovery import distribution_digest


class _Provider:
    """Stands in for a teardown-only cloud client built from delivered secrets."""

    def __init__(self, token):
        self._token = token
        self.released = []

    def release(self, ref):
        self.released.append(ref)
        return "released-with-" + self._token


class RescueSpawnAdapter:
    def __init__(self, metadata):
        self.metadata = metadata
        self._provider = None

    async def begin_rescue(self, params):
        # The ONLY place a rescue worker can obtain a credential. Build the
        # teardown client here or teardown is impossible.
        secrets = {s.name: s.value for s in params.secrets}
        token = secrets.get("TEARDOWN_TOKEN")
        if token is not None:
            self._provider = _Provider(token)
        return AckResult()

    async def cleanup(self, params):
        action_id = params.action_ids[0]
        if self._provider is None:
            # The honest fallback: cleanup was reached but nothing could look.
            return CleanupResult(outcomes=(CleanupOutcome(
                action_id=action_id, status="attempted",
                evidence_code="terminate-unconfirmed", duration_ms=1),))
        action = next(a for a in params.cleanup_plan.actions if a.action_id == action_id)
        code = self._provider.release(action.resource_ref)
        return CleanupResult(outcomes=(CleanupOutcome(
            action_id=action_id, status="succeeded",
            evidence_code=code, duration_ms=1),))

    async def close(self, params): return AckResult()

    async def inspect_requirements(self, params): raise NotImplementedError

    async def prepare(self, params): raise NotImplementedError

    async def reset_start(self, params): raise NotImplementedError

    async def observe(self, params): raise NotImplementedError

    async def execute(self, params): raise NotImplementedError

    async def ask_user_exchange(self, params): raise NotImplementedError

    async def score(self, params): raise NotImplementedError


def create():
    installed = distribution("rescue-e2e-adapter")
    return RescueSpawnAdapter(
        AdapterMetadata(
            adapter_id="rescue-e2e",
            distribution="rescue-e2e-adapter",
            version="1.0",
            entry_point="rescue_e2e_adapter:create",
            package_digest=distribution_digest(installed),
            release_digest="{release}",
            schema_version="1.6",
            capabilities=AdapterCapabilities(
                routes=("computer",), ask_user=False, scoring=False
            ),
        )
    )
'''.replace("{release}", RELEASE_DIGEST)


def _install_named_adapter(
    site: Path, module_name: str, dist_name: str, source: str, adapter_id: str
) -> str:
    """Install a single-module distribution with a real hashed RECORD.

    Same shape as ``_install_adapter`` (which predates it and stays as the
    canonical tiny case); parameterised so a second spawnable adapter does not
    duplicate the RECORD-writing logic. ``distribution_digest`` verifies these
    rows, so a hand-written unhashed RECORD would be rejected as an editable
    install rather than testing the shipped path.
    """

    module = site / f"{module_name}.py"
    module.write_text(source)
    # The dist-info directory is keyed by the DISTRIBUTION name (normalised
    # with underscores), not the module name: importlib.metadata resolves
    # ``verify_distribution``'s lookup through it, so naming it after the
    # module makes the distribution invisible and the handshake fails with
    # "selected adapter distribution is not installed".
    info = site / f"{dist_name.replace('-', '_')}-1.0.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: 1.0\n")
    # The entry point NAME must equal the selector's adapter_id (not the
    # distribution name): load_selected_adapter matches on name AND value, and
    # a mismatch fails as "must expose exactly one exact entry point".
    (info / "entry_points.txt").write_text(
        f"[local_operator.evaluation_adapters.v1]\n{adapter_id} = {module_name}:create\n"
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

    return distribution_digest(PathDistribution(info))


def _install_adapter(site: Path) -> str:
    # The copied venv already carries the repo .pth (``copied_interpreter``);
    # only the adapter distribution is written here.
    module = site / "tiny_e2e_adapter.py"
    module.write_text(_ADAPTER_SOURCE)
    info = site / "tiny_e2e_adapter-1.0.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: tiny-e2e-adapter\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(
        "[local_operator.evaluation_adapters.v1]\ntiny-e2e = tiny_e2e_adapter:create\n"
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

    return distribution_digest(PathDistribution(info))


def test_real_running_interpreter_resolves_and_revalidates() -> None:
    """A stock CPython install ships hardlinked names; it must still launch."""

    executable = Path(sys.executable).resolve()
    assert os.lstat(executable).st_nlink >= 1
    workspace = Path(__file__).resolve().parent
    selector = AdapterSelector.model_construct(
        python_executable=str(executable), workspace=str(workspace)
    )
    resolved = resolve_launch(selector)
    assert resolved.executable == str(executable)
    assert resolved.executable_sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    validate_resolved_launch(resolved)


@pytest.mark.asyncio
async def test_supervisor_launch_completes_real_handshake_and_reaps(tmp_path: Path) -> None:
    executable = copied_interpreter(tmp_path / "venv")
    site = site_packages_of(executable)
    package_digest = _install_adapter(site)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    selector = AdapterSelector(
        schema_version="1.6",
        adapter_id="tiny-e2e",
        distribution="tiny-e2e-adapter",
        version="1.0",
        entry_point="tiny_e2e_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )
    supervisor = AdapterSupervisor.launch(selector)
    try:
        handshake = await supervisor.handshake(timeout=60)
        assert isinstance(handshake, Handshake)
        assert handshake.selector == selector
        assert handshake.metadata.adapter_id == "tiny-e2e"
        assert handshake.workspace_digest == selector.workspace_digest
    finally:
        await supervisor.terminate()
    assert supervisor.process.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(supervisor.pgid, 0)
    assert await asyncio.to_thread(supervisor.process.poll) is not None


def _rescue_selector(tmp_path: Path) -> AdapterSelector:
    """A real interpreter with the rescue adapter installed and a pinned workspace."""

    executable = copied_interpreter(tmp_path / "venv")
    site = site_packages_of(executable)
    package_digest = _install_named_adapter(
        site,
        "rescue_e2e_adapter",
        "rescue-e2e-adapter",
        _RESCUE_ADAPTER_SOURCE,
        "rescue-e2e",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    return AdapterSelector(
        schema_version="1.6",
        adapter_id="rescue-e2e",
        distribution="rescue-e2e-adapter",
        version="1.0",
        entry_point="rescue_e2e_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


def _rescue_descriptor(selector: AdapterSelector, handshake: Handshake, root: Path) -> Any:
    from local_operator.evaluation.adapters.api import RescueDescriptor, SecretRef
    from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan

    plan = CleanupPlan(
        episode_id="episode",
        actions=(
            CleanupAction(
                action_id="release-instance",
                kind="release_instance",
                resource_ref="lop-ep-episode",
                timeout_ms=30_000,
                max_attempts=1,
            ),
        ),
    )
    return RescueDescriptor(
        schema_version="1.6",
        selector=selector,
        handshake=handshake,
        episode_id="episode",
        cleanup_plan=plan,
        secret_refs=(SecretRef(name="TEARDOWN_TOKEN"),),
        infra_values=(),
        artifact_root=str(root),
    )


@pytest.mark.asyncio
async def test_spawned_rescue_worker_builds_a_provider_and_releases(tmp_path: Path) -> None:
    """A REAL spawned rescue worker tears down using begin_rescue's secrets.

    This is the regression that a green suite previously missed entirely: every
    other rescue test called ``adapter.begin_rescue`` in-process, so nothing
    exercised the worker's dispatch, and the worker answered begin_rescue
    itself without ever handing it to adapter code. The adapter's provider
    therefore stayed None in production and every cleanup action reported
    ``attempted``/``terminate-unconfirmed`` -- a sweep never confirmed teardown,
    and a genuinely leaked instance would never have been terminated.

    Asserting the evidence code (not merely ``complete``) is what makes this
    proof: the code can only be produced by a provider that begin_rescue built
    from the delivered secret, so it cannot pass via the None fallback.
    """

    from local_operator.evaluation.adapters.api import ResolvedSecret
    from local_operator.evaluation.adapters.supervisor import run_rescue

    selector = _rescue_selector(tmp_path)
    supervisor = AdapterSupervisor.launch(selector)
    try:
        handshake = await supervisor.handshake(timeout=60)
    finally:
        await supervisor.terminate()

    descriptor = _rescue_descriptor(selector, handshake, tmp_path)
    aggregate = await run_rescue(
        descriptor,
        secrets=(ResolvedSecret(name="TEARDOWN_TOKEN", value="tok-4b1e"),),
    )

    assert aggregate.complete
    assert [receipt.evidence_code for receipt in aggregate.receipts] == ["released-with-tok-4b1e"]
    assert [receipt.status for receipt in aggregate.receipts] == ["succeeded"]


@pytest.mark.asyncio
async def test_spawned_rescue_worker_refuses_an_adapter_without_begin_rescue(
    tmp_path: Path,
) -> None:
    """An adapter that cannot accept the handoff fails LOUDLY, never cleanly.

    ``TinyAdapter`` implements the episode methods but not ``begin_rescue``. It
    can store a descriptor and can never build a provider, so letting the
    rescue proceed would report an orderly ``attempted`` for a resource nothing
    is able to release. The worker refuses instead, and the failure surfaces to
    the sweep as an error rather than as reassuring evidence.
    """

    from local_operator.evaluation.adapters.rpc import RpcRemoteError
    from local_operator.evaluation.adapters.supervisor import run_rescue

    executable = copied_interpreter(tmp_path / "venv")
    site = site_packages_of(executable)
    package_digest = _install_adapter(site)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    selector = AdapterSelector(
        schema_version="1.6",
        adapter_id="tiny-e2e",
        distribution="tiny-e2e-adapter",
        version="1.0",
        entry_point="tiny_e2e_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )
    supervisor = AdapterSupervisor.launch(selector)
    try:
        handshake = await supervisor.handshake(timeout=60)
    finally:
        await supervisor.terminate()

    descriptor = _rescue_descriptor(selector, handshake, tmp_path)
    # The reason must survive to the caller. An earlier draft raised
    # RpcProtocolError here, which killed the channel and surfaced as a bare
    # TimeoutError -- indistinguishable from a slow worker, and useless to an
    # operator holding a descriptor nothing can tear down.
    with pytest.raises(RpcRemoteError) as excinfo:
        await run_rescue(descriptor)
    assert "begin_rescue" in str(excinfo.value)


# A third spawnable distribution whose ``prepare`` FAILS, with a realistic
# wrapped cause and a secret in scope. It exists because the defect it pins is
# invisible in-process: the information loss happens in the worker's own
# ``except`` clause and is only observable after the failure has crossed a real
# pipe into a real parent. A mock adapter raising in the parent's own process
# never exercises the encode/serialise/parse round trip that discards it.
_FAILING_ADAPTER_SOURCE = '''from importlib.metadata import distribution

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    RequirementsResult,
)
from local_operator.evaluation.adapters.discovery import distribution_digest


class _CloudTimeout(Exception):
    """Stands in for a provider SDK's own error type."""


class FailingAdapter:
    def __init__(self, metadata):
        self.metadata = metadata
        self.token = None

    async def inspect_requirements(self, params):
        return RequirementsResult(requirements=())

    async def prepare(self, params):
        # The realistic shape: an SDK error the adapter wraps in its own type.
        # The diagnosable text is one link DOWN the chain, which is precisely
        # what reporting only the outermost type would throw away.
        try:
            raise _CloudTimeout("connect timeout to ec2.us-east-1: i-0abc123")
        except _CloudTimeout as error:
            raise RuntimeError("could not allocate the guest instance") from error

    async def reset_start(self, params): raise NotImplementedError

    async def observe(self, params): raise NotImplementedError

    async def execute(self, params): raise NotImplementedError

    async def ask_user_exchange(self, params): raise NotImplementedError

    async def score(self, params): raise NotImplementedError

    async def cleanup(self, params): raise NotImplementedError

    async def close(self, params): return AckResult()


def create():
    installed = distribution("failing-e2e-adapter")
    return FailingAdapter(
        AdapterMetadata(
            adapter_id="failing-e2e",
            distribution="failing-e2e-adapter",
            version="1.0",
            entry_point="failing_e2e_adapter:create",
            package_digest=distribution_digest(installed),
            release_digest="{release}",
            schema_version="1.6",
            capabilities=AdapterCapabilities(
                routes=("computer",), ask_user=False, scoring=False
            ),
        )
    )
'''.replace("{release}", RELEASE_DIGEST)


def _failing_selector(tmp_path: Path) -> AdapterSelector:
    """A real interpreter with the failing adapter installed and a pinned workspace."""

    executable = copied_interpreter(tmp_path / "venv")
    site = site_packages_of(executable)
    package_digest = _install_named_adapter(
        site,
        "failing_e2e_adapter",
        "failing-e2e-adapter",
        _FAILING_ADAPTER_SOURCE,
        "failing-e2e",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    return AdapterSelector(
        schema_version="1.6",
        adapter_id="failing-e2e",
        distribution="failing-e2e-adapter",
        version="1.0",
        entry_point="failing_e2e_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


@pytest.mark.asyncio
async def test_spawned_worker_reports_the_real_adapter_cause_across_the_boundary(
    tmp_path: Path,
) -> None:
    """A REAL spawned worker's adapter failure arrives NAMED, not generic.

    This is the regression for the largest cause of lost paid episodes. Two
    consecutive real runs (ep-e46c789ca818, ep-ffda3fc88f81) died on a fatal
    ``adapter_error`` whose entire recorded diagnostic was::

        RpcRemoteError: adapter_error: adapter operation failed

    The worker knew the exception type, the message, the failing method and the
    operation ID, and the generic ``except`` clause discarded all four. It has
    to be proven through a real subprocess: the loss happened inside the
    worker's own handler, so an in-process mock adapter never crosses the
    encode/serialise/parse round trip where the information actually vanished.

    Asserting the CAUSE chain (not merely the outer type) is what makes this
    evidence: ``_CloudTimeout`` is raised inside the spawned adapter, wrapped in
    a ``RuntimeError``, and can only appear here if the chain survived the wire.
    """

    from local_operator.evaluation.adapters.api import (
        InspectRequirementsParams,
        PrepareParams,
        PrepareResult,
        RequirementsResult,
    )
    from local_operator.evaluation.adapters.rpc import RpcRemoteError

    selector = _failing_selector(tmp_path)
    supervisor = AdapterSupervisor.launch(selector)
    try:
        await supervisor.handshake(timeout=60)
        await supervisor._call_raw(
            "inspect_requirements",
            InspectRequirementsParams(),
            RequirementsResult,
            timeout=60,
        )
        with pytest.raises(RpcRemoteError) as excinfo:
            await supervisor._call_raw(
                "prepare",
                PrepareParams(
                    operation_id="prepare-op-7",
                    episode_id="episode",
                    secret_refs=(),
                    infra_values=(),
                ),
                PrepareResult,
                timeout=60,
            )
    finally:
        await supervisor.terminate()

    error = excinfo.value
    # The closed envelope is unchanged: same code, same fixed message.
    assert error.code == "adapter_error"
    detail = error.detail
    assert detail is not None
    # The four facts the episodes needed and did not get.
    assert detail.exception_type == "RuntimeError"
    assert "could not allocate the guest instance" in detail.message
    assert detail.method == "prepare"
    assert detail.operation_id == "prepare-op-7"
    # The wrapped SDK error, one link down, is what actually names the fault.
    assert detail.causes[0].exception_type == "_CloudTimeout"
    assert "connect timeout to ec2.us-east-1" in detail.causes[0].message
    # Worker-side frames are basename/line/function only -- never a path.
    assert detail.frames, "frames carry the raising call site"
    assert any(frame.function == "prepare" for frame in detail.frames)
    assert all("/" not in frame.file for frame in detail.frames)
    # And the rendered form -- what reaches the evidence artifact -- names it.
    rendered = str(error)
    assert "RuntimeError" in rendered and "_CloudTimeout" in rendered
    assert "prepare-op-7" in rendered


# A fourth spawnable distribution: it is HANDED a secret on ``reset_start`` and
# then raises an exception whose message embeds that exact value. It is the
# adversarial case for the new error detail -- a well-meaning adapter that
# interpolates a credential into its own error text -- and the only way to prove
# the worker-side canary check runs before anything crosses the pipe.
_LEAKY_ADAPTER_SOURCE = """from importlib.metadata import distribution

from local_operator.evaluation.adapters.api import (
    AckResult,
    AdapterCapabilities,
    AdapterMetadata,
    RequirementsResult,
)
from local_operator.evaluation.adapters.discovery import distribution_digest
from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan
from local_operator.evaluation.adapters.api import PrepareResult


class LeakyAdapter:
    def __init__(self, metadata):
        self.metadata = metadata

    async def inspect_requirements(self, params):
        return RequirementsResult(requirements=())

    async def prepare(self, params):
        return PrepareResult(cleanup_plan=CleanupPlan(
            episode_id=params.episode_id,
            actions=(CleanupAction(
                action_id="release", kind="release_instance",
                resource_ref="resource", timeout_ms=100, max_attempts=1),)))

    async def reset_start(self, params):
        # The adapter does exactly what a careless one does: puts the
        # credential it was just handed into its own error message. The
        # padding places the secret so the message field's 512-character bound
        # cuts THROUGH it: truncating before the canary scan would sever the
        # match and emit the surviving prefix verbatim (round 1, F1).
        token = {s.name: s.value for s in params.secrets}["AWS_SECRET_ACCESS_KEY"]
        raise RuntimeError("auth rejected: " + ("x" * 500) + token + " (retry)")

    async def observe(self, params): raise NotImplementedError

    async def execute(self, params): raise NotImplementedError

    async def ask_user_exchange(self, params): raise NotImplementedError

    async def score(self, params): raise NotImplementedError

    async def cleanup(self, params): raise NotImplementedError

    async def close(self, params): return AckResult()


def create():
    installed = distribution("leaky-e2e-adapter")
    return LeakyAdapter(
        AdapterMetadata(
            adapter_id="leaky-e2e",
            distribution="leaky-e2e-adapter",
            version="1.0",
            entry_point="leaky_e2e_adapter:create",
            package_digest=distribution_digest(installed),
            release_digest="{release}",
            schema_version="1.6",
            capabilities=AdapterCapabilities(
                routes=("computer",), ask_user=False, scoring=False
            ),
        )
    )
""".replace("{release}", RELEASE_DIGEST)


def _leaky_selector(tmp_path: Path) -> AdapterSelector:
    """A real interpreter with the leaky adapter installed and a pinned workspace."""

    executable = copied_interpreter(tmp_path / "venv")
    site = site_packages_of(executable)
    package_digest = _install_named_adapter(
        site,
        "leaky_e2e_adapter",
        "leaky-e2e-adapter",
        _LEAKY_ADAPTER_SOURCE,
        "leaky-e2e",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "adapter-release.json").write_text(
        json.dumps({"release_digest": RELEASE_DIGEST}, separators=(",", ":"), sort_keys=True)
    )
    return AdapterSelector(
        schema_version="1.6",
        adapter_id="leaky-e2e",
        distribution="leaky-e2e-adapter",
        version="1.0",
        entry_point="leaky_e2e_adapter:create",
        package_digest=package_digest,
        release_digest=RELEASE_DIGEST,
        python_executable=str(executable.resolve()),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


@pytest.mark.asyncio
async def test_error_detail_withholds_a_secret_the_adapter_put_in_its_message(
    tmp_path: Path,
) -> None:
    """The canary check runs on the WORKER side, before anything crosses.

    Carrying a real cause across the boundary is only safe if the value a
    delivered secret has cannot ride out with it. This adapter interpolates the
    credential it was handed into its own exception message -- the realistic
    careless case, not a contrived one -- and the assertion is that the parent
    never receives those bytes at all: not in the message, not in the cause
    chain, not through ``str()``, and not in the rendered evidence artifact.

    It fails CLOSED (the whole string is replaced) rather than masking, because
    a partial mask still narrows the secret for whoever holds the bundle.
    """

    from local_operator.evaluation.adapters.api import (
        AckResult,
        InspectRequirementsParams,
        PrepareParams,
        PrepareResult,
        RequirementsResult,
        ResetStartParams,
        ResolvedSecret,
    )
    from local_operator.evaluation.adapters.rpc import WITHHELD, RpcRemoteError

    marker = "AKIA-canary-secret-value-8823xyz"
    selector = _leaky_selector(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    supervisor = AdapterSupervisor.launch(selector)
    try:
        await supervisor.handshake(timeout=60)
        await supervisor._call_raw(
            "inspect_requirements", InspectRequirementsParams(), RequirementsResult, timeout=60
        )
        await supervisor._call_raw(
            "prepare",
            PrepareParams(
                operation_id="prepare-op",
                episode_id="episode",
                secret_refs=(),
                infra_values=(),
            ),
            PrepareResult,
            timeout=60,
        )
        with pytest.raises(RpcRemoteError) as excinfo:
            await supervisor._call_raw(
                "reset_start",
                ResetStartParams(
                    operation_id="reset-op",
                    task_id="task",
                    episode_id="episode",
                    artifact_root=str(artifact_root),
                    secrets=(ResolvedSecret(name="AWS_SECRET_ACCESS_KEY", value=marker),),
                ),
                AckResult,
                timeout=60,
            )
    finally:
        await supervisor.terminate()

    error = excinfo.value
    detail = error.detail
    assert detail is not None
    # The structure survives -- the reader still learns type, method and key.
    assert detail.exception_type == "RuntimeError"
    assert detail.method == "reset_start"
    assert detail.operation_id == "reset-op"
    # ...but the message that embedded the credential was withheld WHOLE.
    assert detail.message == WITHHELD
    # No surface the parent can see carries the value.
    for surface in (
        str(error),
        json.dumps(detail.model_dump(mode="json")),
        supervisor.stdout_tail.bytes().decode(errors="replace"),
        supervisor.stderr_tail.bytes().decode(errors="replace"),
    ):
        assert marker not in surface
        assert "AKIA-canary" not in surface
        # No PREFIX of the credential survives either. The adapter pads its
        # message so the field bound cuts through the secret, which is the
        # shape a truncate-then-scan ordering leaks (round 1, F1): the canary
        # stops matching once severed, and the surviving prefix crosses the
        # pipe verbatim.
        assert not any(
            marker[:length] in surface for length in range(8, len(marker) + 1)
        ), "a prefix of the credential crossed the boundary"
