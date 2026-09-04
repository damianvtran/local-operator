"""Secret delivery, judge preflight, inputs re-verification, and rescue-by-descriptor.

The secret path is new on the boundary (schema 1.2), so its non-leak
properties are asserted here at every layer the adapter touches: the values
reach the provider constructor and NOTHING else -- not ``os.environ`` (bar the
documented judge-key exception, which is scrubbed on close), not the rescue
descriptor, not any exception message. ``test_spawn.py`` extends the same
assertions across a real process boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter import cleanup as cleanup_mod
from lop_osworld_v2_adapter.adapter import (
    AdapterStateError,
    InputsMismatch,
    JudgeUnavailable,
    OSWorldV2Adapter,
)
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.fake import FakeProvider

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    AdapterCapabilities,
    AdapterMetadata,
    AdapterSelector,
    BeginRescueParams,
    CleanupParams,
    CloseParams,
    Handshake,
    PrepareParams,
    PythonRuntime,
    RescueDescriptor,
    ResetStartParams,
    ResolvedSecret,
    ScopedInfraValue,
)
from local_operator.evaluation.evidence.models import canonical_digest
from tests.unit.evaluation.adapters.osworld import fixtures

_INFRA = tuple(
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
_JUDGE_INFRA = _INFRA + (
    ScopedInfraValue(name="OSWORLD_EVAL_MODEL_PROVIDER", purpose="benchmark_judge", value="openai"),
    ScopedInfraValue(name="OSWORLD_EVAL_MODEL_NAME", purpose="benchmark_judge", value="gpt-x"),
)
_AWS_SECRETS = (
    ResolvedSecret(name="AWS_ACCESS_KEY_ID", value="AKIAMARKER"),
    ResolvedSecret(name="AWS_SECRET_ACCESS_KEY", value="marker-aws-secret"),
)
_MARKERS = ("AKIAMARKER", "marker-aws-secret", "marker-judge-key")


def _workspace(tmp_path: Path, tasks: dict[str, str], *, provider: dict[str, Any] | None) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "tasks").mkdir(parents=True, exist_ok=True)
    for task_id, source in tasks.items():
        (workspace / "tasks" / f"{task_id}.py").write_text(source)
    if provider is not None:
        (workspace / "adapter-provider.json").write_text(json.dumps(provider))
    return workspace


async def _prepared(adapter: OSWorldV2Adapter, episode_id: str, infra: Any = _INFRA) -> None:
    await adapter.prepare(
        PrepareParams(
            operation_id=f"prepare-{episode_id}",
            episode_id=episode_id,
            secret_refs=(),
            infra_values=infra,
        )
    )


def _reset(episode_id: str, task_id: str, tmp_path: Path, secrets: Any = ()) -> ResetStartParams:
    # The parent owns and creates the artifact root; these tests stand in for it.
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    return ResetStartParams(
        operation_id=f"reset-{episode_id}",
        task_id=task_id,
        episode_id=episode_id,
        artifact_root=str(tmp_path / "artifacts"),
        secrets=secrets,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OSWORLD_EVAL_MODEL_API_KEY", "OSWORLD_USER_SIM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# judge preflight refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_judged_task_without_the_key_is_refused_before_allocation(
    tmp_path: Path, episode_id: str
) -> None:
    workspace = _workspace(tmp_path, {"task_judged": fixtures.JUDGED}, provider=None)
    provider = FakeProvider()
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    await _prepared(adapter, episode_id, _JUDGE_INFRA)
    with pytest.raises(JudgeUnavailable, match="OSWORLD_EVAL_MODEL_API_KEY"):
        await adapter.reset_start(_reset(episode_id, "task_judged", tmp_path, _AWS_SECRETS))
    assert provider.allocated is False


@pytest.mark.asyncio
async def test_a_judged_task_without_provider_settings_is_refused(
    tmp_path: Path, episode_id: str
) -> None:
    workspace = _workspace(tmp_path, {"task_judged": fixtures.JUDGED}, provider=None)
    provider = FakeProvider()
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    await _prepared(adapter, episode_id, _INFRA)  # no judge infra
    secrets = _AWS_SECRETS + (ResolvedSecret(name="OSWORLD_EVAL_MODEL_API_KEY", value="k"),)
    with pytest.raises(JudgeUnavailable, match="OSWORLD_EVAL_MODEL_PROVIDER"):
        await adapter.reset_start(_reset(episode_id, "task_judged", tmp_path, secrets))
    assert provider.allocated is False


@pytest.mark.asyncio
async def test_the_judge_key_takes_the_documented_env_path_and_is_scrubbed_on_close(
    tmp_path: Path, episode_id: str
) -> None:
    workspace = _workspace(tmp_path, {"task_judged": fixtures.JUDGED}, provider=None)
    provider = FakeProvider()
    adapter = OSWorldV2Adapter(provider_factory=lambda: provider, workspace_root=workspace)
    await _prepared(adapter, episode_id, _JUDGE_INFRA)
    secrets = _AWS_SECRETS + (
        ResolvedSecret(name="OSWORLD_EVAL_MODEL_API_KEY", value="marker-judge-key"),
    )
    await adapter.reset_start(_reset(episode_id, "task_judged", tmp_path, secrets))
    assert provider.allocated is True
    # The ONE documented exception: OSWorld reads this from the env only.
    assert os.environ["OSWORLD_EVAL_MODEL_API_KEY"] == "marker-judge-key"
    # The AWS secrets never take that path.
    assert not any(v in _MARKERS[:2] for v in os.environ.values())
    await adapter.close(CloseParams(operation_id=f"close-{episode_id}", episode_id=episode_id))
    assert "OSWORLD_EVAL_MODEL_API_KEY" not in os.environ


# ---------------------------------------------------------------------------
# secrets reach the AWS provider and nothing else
# ---------------------------------------------------------------------------


class _CapturingProvider(FakeProvider):
    """A fake that records the credentials the adapter handed it."""

    captured: list[Any] = []


@pytest.mark.asyncio
async def test_aws_secrets_build_the_provider_and_never_touch_the_environment(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, {"task_plain": fixtures.PLAIN}, provider=None)
    captured: list[dict[str, Any]] = []

    def fake_aws_provider(credentials: Any, **kwargs: Any) -> FakeProvider:
        captured.append({"credentials": credentials, **kwargs})
        return FakeProvider()

    monkeypatch.setattr(aws_mod, "AwsProvider", fake_aws_provider)
    adapter = OSWorldV2Adapter(workspace_root=workspace)  # no factory: production path
    await _prepared(adapter, episode_id)
    before = dict(os.environ)
    await adapter.reset_start(_reset(episode_id, "task_plain", tmp_path, _AWS_SECRETS))
    assert len(captured) == 1
    creds = captured[0]["credentials"]
    assert creds.access_key_id == "AKIAMARKER"
    assert creds.secret_access_key == "marker-aws-secret"
    assert creds.session_token is None
    assert captured[0]["region"] == "test-AWS_REGION"
    assert captured[0]["lease_ref"] == f"lop-ttl-{episode_id}"
    assert captured[0]["ttl_seconds"] == aws_mod.DEFAULT_TTL_SECONDS
    # Nothing new in the environment carries a secret value.
    added = {k: v for k, v in os.environ.items() if before.get(k) != v}
    assert not any(marker in v for v in added.values() for marker in _MARKERS)


@pytest.mark.asyncio
async def test_a_missing_aws_secret_names_the_ref_not_a_value(
    tmp_path: Path, episode_id: str
) -> None:
    workspace = _workspace(tmp_path, {"task_plain": fixtures.PLAIN}, provider=None)
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    await _prepared(adapter, episode_id)
    only_key = (ResolvedSecret(name="AWS_ACCESS_KEY_ID", value="AKIAMARKER"),)
    with pytest.raises(AdapterStateError) as info:
        await adapter.reset_start(_reset(episode_id, "task_plain", tmp_path, only_key))
    assert "AWS_SECRET_ACCESS_KEY" in str(info.value)
    assert "AKIAMARKER" not in str(info.value)


@pytest.mark.asyncio
async def test_ttl_override_reaches_the_provider(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, {"task_plain": fixtures.PLAIN}, provider=None)
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        aws_mod,
        "AwsProvider",
        lambda credentials, **kw: (captured.append(kw), FakeProvider())[1],
    )
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    infra = _INFRA + (
        ScopedInfraValue(name="OSWORLD_TTL_SECONDS", purpose="benchmark_compute", value="5400"),
    )
    await _prepared(adapter, episode_id, infra)
    await adapter.reset_start(_reset(episode_id, "task_plain", tmp_path, _AWS_SECRETS))
    assert captured[0]["ttl_seconds"] == 5400


# ---------------------------------------------------------------------------
# inputs.json re-verification
# ---------------------------------------------------------------------------


def _inputs_root(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "inputs"
    (root / "gated" / "manifests").mkdir(parents=True)
    (root / "gated" / "tasks" / "manifests").mkdir(parents=True)
    assets = b'{"kind": "assets"}'
    tasks = b'{"files": {}}'
    (root / "gated" / "manifests" / "assets.json").write_bytes(assets)
    (root / "gated" / "tasks" / "manifests" / "task_hashes.json").write_bytes(tasks)
    return root, hashlib.sha256(assets).hexdigest(), hashlib.sha256(tasks).hexdigest()


@pytest.mark.asyncio
async def test_reset_reverifies_the_inputs_root_against_the_workspace_pin(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, {"task_plain": fixtures.PLAIN}, provider=None)
    root, assets_sha, tasks_sha = _inputs_root(tmp_path)
    (workspace / "inputs.json").write_text(
        json.dumps({"assets_manifest_sha256": assets_sha, "tasks_manifest_sha256": tasks_sha})
    )
    monkeypatch.setattr(aws_mod, "AwsProvider", lambda credentials, **kw: FakeProvider())
    infra = _INFRA + (
        ScopedInfraValue(name="OSWORLD_INPUTS_ROOT", purpose="benchmark_storage", value=str(root)),
    )
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    await _prepared(adapter, episode_id, infra)
    await adapter.reset_start(_reset(episode_id, "task_plain", tmp_path, _AWS_SECRETS))

    # Tamper with the live root: one byte in the assets manifest.
    (root / "gated" / "manifests" / "assets.json").write_bytes(b'{"kind": "assetz"}')
    adapter2 = OSWorldV2Adapter(workspace_root=workspace)
    await _prepared(adapter2, f"{episode_id}-2", infra)
    with pytest.raises(InputsMismatch, match="assets.json"):
        await adapter2.reset_start(_reset(f"{episode_id}-2", "task_plain", tmp_path, _AWS_SECRETS))


# ---------------------------------------------------------------------------
# rescue from descriptor + secrets
# ---------------------------------------------------------------------------


def _descriptor(tmp_path: Path, workspace: Path, adapter: OSWorldV2Adapter, episode_id: str) -> Any:
    import sys

    refs = cleanup_mod.CleanupRefs.mint(episode_id)
    plan = cleanup_mod.build_cleanup_plan(episode_id, refs)
    selector = AdapterSelector(
        schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_id="osworld-v2",
        distribution="lop-osworld-v2-adapter",
        version="0.1.1",
        entry_point="lop_osworld_v2_adapter:create",
        package_digest=adapter.metadata.package_digest,
        release_digest=adapter.metadata.release_digest,
        python_executable=str(Path(sys.executable).resolve()),
        workspace=str(workspace),
        workspace_digest="c" * 64,
        route_capability="computer",
    )
    handshake = Handshake(
        selector=selector,
        metadata=AdapterMetadata(
            adapter_id="osworld-v2",
            distribution="lop-osworld-v2-adapter",
            version="0.1.1",
            entry_point="lop_osworld_v2_adapter:create",
            package_digest=adapter.metadata.package_digest,
            release_digest=adapter.metadata.release_digest,
            schema_version=ADAPTER_SCHEMA_VERSION,
            capabilities=AdapterCapabilities(routes=("computer",), ask_user=True, scoring=True),
        ),
        python=PythonRuntime.current(),
        workspace_digest="c" * 64,
        selected_route="computer",
    )
    return RescueDescriptor(
        schema_version=ADAPTER_SCHEMA_VERSION,
        selector=selector,
        handshake=handshake,
        episode_id=episode_id,
        cleanup_plan=plan,
        secret_refs=(),
        infra_values=_INFRA,
        artifact_root=str(tmp_path / "artifacts"),
    )


@pytest.mark.asyncio
async def test_rescue_builds_the_teardown_provider_from_descriptor_and_secrets(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rescue worker has ONLY the descriptor + freshly delivered secrets.

    The adapter must build a teardown-only AWS provider from the descriptor's
    infra values (region) and refs (lease), with credentials from the RPC
    field, and terminate by tag -- never needing prepare, a plan, or a task.
    """

    workspace = _workspace(tmp_path, {"task_plain": fixtures.PLAIN}, provider=None)
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    descriptor = _descriptor(tmp_path, workspace, adapter, episode_id)
    assert "marker" not in descriptor.to_canonical_json().decode()

    captured: list[dict[str, Any]] = []
    teardown = FakeProvider()
    # Pre-register the instance so terminate finds it "by tag".
    teardown._instances[f"lop-ep-{episode_id}"] = {"state": "running"}
    teardown._schedules[f"lop-ttl-{episode_id}"] = {"state": "active"}

    class _Aws:
        @staticmethod
        def for_teardown(credentials: Any, **kw: Any) -> FakeProvider:
            captured.append({"credentials": credentials, **kw})
            return teardown

    monkeypatch.setattr(aws_mod, "AwsProvider", _Aws)
    await adapter.begin_rescue(
        BeginRescueParams(
            operation_id="rescue-begin",
            descriptor=descriptor,
            descriptor_id=descriptor.descriptor_id,
            episode_id=episode_id,
            cleanup_plan_id=descriptor.cleanup_plan.cleanup_plan_id,
            selector_digest=canonical_digest("adapter-rescue-selector-v1", descriptor.selector),
            handshake_digest=canonical_digest("adapter-rescue-handshake-v1", descriptor.handshake),
            secrets=_AWS_SECRETS,
        )
    )
    assert captured == [
        {
            "credentials": captured[0]["credentials"],
            "region": "test-AWS_REGION",
            "lease_ref": f"lop-ttl-{episode_id}",
        }
    ]
    assert captured[0]["credentials"].secret_access_key == "marker-aws-secret"

    result = await adapter.cleanup(
        CleanupParams(
            operation_id="rescue-cleanup",
            cleanup_plan=descriptor.cleanup_plan,
            action_ids=tuple(a.action_id for a in descriptor.cleanup_plan.actions),
        )
    )
    by_id = {o.action_id: (o.status, o.evidence_code) for o in result.outcomes}
    assert by_id["release-instance"] == ("succeeded", "instance-terminated")
    assert by_id["revoke-ttl-lease"] == ("succeeded", "schedule-deleted")
    assert by_id["close-session"] == ("succeeded", "session-closed")
    assert teardown.terminated_refs == [f"lop-ep-{episode_id}"]


@pytest.mark.asyncio
async def test_rescue_without_aws_secrets_fails_closed(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path, {"task_plain": fixtures.PLAIN}, provider=None)
    adapter = OSWorldV2Adapter(workspace_root=workspace)
    descriptor = _descriptor(tmp_path, workspace, adapter, episode_id)
    with pytest.raises(AdapterStateError, match="AWS_ACCESS_KEY_ID"):
        await adapter.begin_rescue(
            BeginRescueParams(
                operation_id="rescue-begin",
                descriptor=descriptor,
                descriptor_id=descriptor.descriptor_id,
                episode_id=episode_id,
                cleanup_plan_id=descriptor.cleanup_plan.cleanup_plan_id,
                selector_digest=canonical_digest("adapter-rescue-selector-v1", descriptor.selector),
                handshake_digest=canonical_digest(
                    "adapter-rescue-handshake-v1", descriptor.handshake
                ),
            )
        )


# ---------------------------------------------------------------------------
# schema pins
# ---------------------------------------------------------------------------


def test_schema_is_1_2_and_secrets_live_only_on_reset_and_rescue() -> None:
    assert ADAPTER_SCHEMA_VERSION == "1.3"
    assert "secrets" in ResetStartParams.model_fields
    assert "secrets" in BeginRescueParams.model_fields
    assert "secrets" not in PrepareParams.model_fields
    assert "secrets" not in RescueDescriptor.model_fields
    # A 1.1-shaped reset (no secrets) still validates: the field defaults.
    params = ResetStartParams.model_validate(
        {
            "operation_id": "r",
            "task_id": "t",
            "episode_id": "e",
            "artifact_root": "/tmp/x",
        },
        strict=True,
    )
    assert params.secrets == ()
