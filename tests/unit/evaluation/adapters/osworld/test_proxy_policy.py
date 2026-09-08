"""System proxy policy is explicit infra, never an edit to a benchmark task."""

from pathlib import Path

import pytest
from lop_osworld_v2_adapter import provisioning, requirements, taskfile
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

from local_operator.evaluation.adapters.api import (
    InspectRequirementsParams,
    PrepareParams,
    ScopedInfraValue,
)
from tests.unit.evaluation.adapters.osworld import fixtures
from tests.unit.evaluation.adapters.osworld.test_provisioning import _INFRA


def _policy(value: str, purpose: str = "benchmark_compute") -> ScopedInfraValue:
    return ScopedInfraValue.model_validate(
        {"name": "OSWORLD_ENABLE_PROXY", "purpose": purpose, "value": value}
    )


def _task(proxy: bool) -> taskfile.TaskDescriptor:
    return taskfile.TaskDescriptor(
        task_id="synthetic", instruction="", source_sha256="0" * 64, proxy=proxy
    )


@pytest.mark.parametrize("hint", [False, True])
@pytest.mark.parametrize("value", [None, "false", "true"])
def test_policy_plan_and_requirements_agree(hint: bool, value: str | None) -> None:
    infra = _INFRA + (() if value is None else (_policy(value),))
    plan = provisioning.resolve(_task(hint), episode_id="ep-policy", infra_values=infra)
    expected = hint if value is None else value == "true"
    assert plan.enable_proxy is expected
    reqs = {r.name: r for r in requirements.derive_requirements(_task(hint), infra_values=infra)}
    assert reqs["OSWORLD_ENABLE_PROXY"].required is False
    assert reqs["OSWORLD_ENABLE_PROXY"].kind == "infra"
    # PROXY_CONFIG_FILE is the input upstream actually reads, so it appears --
    # and appears REQUIRED -- exactly when the episode will really use a proxy.
    needs_proxy = hint and expected
    assert ("PROXY_CONFIG_FILE" in reqs) is needs_proxy
    if needs_proxy:
        assert reqs["PROXY_CONFIG_FILE"].required is True
    # The legacy pair no longer gates a run: CREDENTIALS is gone entirely
    # (nothing consumed it) and ENDPOINT survives as optional so an existing
    # invocation that still supplies it is accepted unchanged.
    assert "OSWORLD_PROXY_CREDENTIALS" not in reqs
    assert ("OSWORLD_PROXY_ENDPOINT" in reqs) is needs_proxy
    if needs_proxy:
        assert reqs["OSWORLD_PROXY_ENDPOINT"].required is False


@pytest.mark.asyncio
@pytest.mark.parametrize("task_known", [False, True])
@pytest.mark.parametrize("bad", ["False", "TRUE", "0", "1", " false", "false ", "secret-canary"])
async def test_malformed_policy_fails_before_allocation(
    tmp_path: Path, task_known: bool, bad: str
) -> None:
    def forbidden_provider():
        pytest.fail("policy validation must precede provider construction")

    adapter = OSWorldV2Adapter(provider_factory=forbidden_provider, workspace_root=tmp_path)
    if task_known:
        adapter._task = _task(True)
    with pytest.raises(provisioning.ProvisioningError) as error:
        await adapter.prepare(
            PrepareParams(
                operation_id="prepare-policy",
                episode_id="ep-policy",
                secret_refs=(),
                infra_values=_INFRA + (_policy(bad),),
            )
        )
    assert str(error.value) == (
        "OSWORLD_ENABLE_PROXY requires benchmark_compute scope and exactly true or false"
    )
    assert adapter._refs is None and adapter._plan is None
    with pytest.raises(provisioning.ProvisioningError):
        requirements.derive_requirements(_task(True), infra_values=(_policy(bad),))


def test_wrong_scope_and_conflicting_policy_are_rejected() -> None:
    for infra in (
        (_policy("false", "benchmark_user_simulator"),),
        (_policy("true"), _policy("false")),
    ):
        with pytest.raises(provisioning.ProvisioningError):
            provisioning.resolve_proxy_policy(infra)


@pytest.mark.asyncio
async def test_post_prepare_requirements_respect_disabled_policy(
    tmp_path: Path,
) -> None:
    adapter = OSWorldV2Adapter(workspace_root=tmp_path)
    baseline = await adapter.inspect_requirements(InspectRequirementsParams())
    assert any(r.name == "OSWORLD_ENABLE_PROXY" and not r.required for r in baseline.requirements)
    await adapter.prepare(
        PrepareParams(
            operation_id="prepare-policy",
            episode_id="ep-policy",
            secret_refs=(),
            infra_values=_INFRA + (_policy("false"),),
        )
    )
    # The runner names the task after prepare; policy must survive that order.
    adapter._task = _task(True)
    post = await adapter.inspect_requirements(InspectRequirementsParams())
    assert not any(r.name.startswith("OSWORLD_PROXY_") for r in post.requirements)


@pytest.mark.asyncio
async def test_a_proxy_task_without_a_pool_config_is_refused_before_allocation(
    tmp_path: Path,
) -> None:
    """The refusal must fire for a TASK that needs a proxy, not just an override.

    This is the case the whole fix exists for, and it cannot be checked in
    ``prepare``: ``PrepareParams`` carries no ``task_id``, so ``self._task`` is
    still None there and the task's own ``proxy = True`` hint is unknowable.
    A guard placed in prepare reads ``task_proxy=False`` for every task and
    silently passes exactly the episodes it was written to protect -- which
    then allocate a VM and die at upstream's empty pool with the instance
    billed. So the guard lives in ``reset_start``, after the task loads and
    before any provider is constructed, and this test pins that placement by
    failing the moment a provider is built.
    """
    from local_operator.evaluation.adapters.api import ResetStartParams
    from tests.unit.evaluation.adapters.osworld.test_cache_dir import _write_workspace

    def forbidden_provider():
        pytest.fail("the proxy refusal must precede provider construction")

    workspace = _write_workspace(tmp_path, {"task_proxy": fixtures.PROXY})
    adapter = OSWorldV2Adapter(provider_factory=forbidden_provider, workspace_root=workspace)
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)

    # No PROXY_CONFIG_FILE anywhere in the infra values.
    await adapter.prepare(
        PrepareParams(
            operation_id="prepare-proxy",
            episode_id="ep-proxy",
            secret_refs=(),
            infra_values=_INFRA,
        )
    )
    with pytest.raises(provisioning.ProvisioningError) as error:
        await adapter.reset_start(
            ResetStartParams(
                operation_id="reset-proxy",
                task_id="task_proxy",
                episode_id="ep-proxy",
                artifact_root=str(artifacts),
                secrets=(),
            )
        )
    assert "PROXY_CONFIG_FILE" in str(error.value)
    assert "OSWORLD_ENABLE_PROXY=false" in str(error.value)
    # Nothing was provisioned: the refusal is free.
    assert adapter._plan is None


@pytest.mark.asyncio
async def test_a_non_proxy_task_needs_no_pool_config(tmp_path: Path) -> None:
    """The guard must not become a new required input for ordinary tasks."""
    from local_operator.evaluation.adapters.api import ResetStartParams
    from tests.unit.evaluation.adapters.osworld.test_cache_dir import _write_workspace

    built: list[str] = []

    def provider_factory():
        built.append("yes")
        raise RuntimeError("stop after the guard, before real work")

    workspace = _write_workspace(tmp_path, {"task_plain": fixtures.PLAIN})
    adapter = OSWorldV2Adapter(provider_factory=provider_factory, workspace_root=workspace)
    artifacts = tmp_path / "run" / "artifacts"
    artifacts.mkdir(parents=True)

    await adapter.prepare(
        PrepareParams(
            operation_id="prepare-plain",
            episode_id="ep-plain",
            secret_refs=(),
            infra_values=_INFRA,
        )
    )
    with pytest.raises(RuntimeError, match="stop after the guard"):
        await adapter.reset_start(
            ResetStartParams(
                operation_id="reset-plain",
                task_id="task_plain",
                episode_id="ep-plain",
                artifact_root=str(artifacts),
                secrets=(),
            )
        )
    # Reaching provider construction proves the proxy guard did not fire.
    assert built == ["yes"]
