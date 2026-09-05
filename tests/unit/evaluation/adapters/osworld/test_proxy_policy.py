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
    for name in ("OSWORLD_PROXY_CREDENTIALS", "OSWORLD_PROXY_ENDPOINT"):
        assert (name in reqs) is (hint and expected)


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
