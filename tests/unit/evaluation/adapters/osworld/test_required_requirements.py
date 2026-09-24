"""A required-and-absent declared requirement is refused BEFORE allocation.

THE CONTRACT these tests pin: declaring a requirement means refusing when it is
missing. A ``required=True`` row says the task cannot run without that value, so
an episode that proceeds without it does not run degraded -- it dies, and it
dies after a guest has been built and billed.

The measured failure this closes: 40 of the release's 108 task modules reach a
controller that reads its value out of the environment when the guest's task
object is instantiated, and nothing in the pipeline supplied it. ``task_016``
therefore allocated an EC2 desktop and only then failed inside vendor code::

    ValueError: WEBSITE_HOST_SUFFIX must be set in environment variables
      threads.py:25 to_thread <- aws.py:863 _start_desktop_env
      <- vendor_bridge.py:202 instantiate_task <- task_016.py:7 <- website.py:22

-- a traceback naming no adapter, for a requirement the task's own source
declares. GOOGLE_ACCOUNT_CREDENTIALS, OSWORLD_USER_SIM_API_KEY and
GITLAB_PRIVATE_TOKEN/GITLAB_URL fail the same way.

Four claims, each with its own test:

1. the refusal fires, names the absent REF (never a value), and lands BEFORE
   the provider is constructed -- watching the provider factory, which is what
   actually allocates, rather than reading a comment;
2. it is derived from the table, so it covers every family a task's own fields
   introduce instead of one hardcoded name (the agreement test removes each
   declared name in turn and requires exactly that one to be reported);
3. it does not over-refuse: a supplied value runs, an absent OPTIONAL
   declaration is benign, a task-conditional value given to a task that does
   not need it is accepted;
4. the always-on baseline is deliberately NOT gated here, which is the reason
   this check can live at the task seam at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter import provisioning, requirements, taskfile
from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

from local_operator.evaluation.adapters.api import (
    PrepareParams,
    ResetStartParams,
    ResolvedSecret,
    ScopedInfraValue,
)
from tests.unit.evaluation.adapters.osworld import fixtures
from tests.unit.evaluation.adapters.osworld.test_cache_dir import _write_workspace
from tests.unit.evaluation.adapters.osworld.test_provisioning import _INFRA

ProvisioningError = provisioning.ProvisioningError

# The four families a task's own fields introduce. Each entry is
# (task id, fixture source): the DECLARED set is read from the table rather
# than restated here, so adding a row to the table covers the new name without
# touching this file.
_CONDITIONAL_FAMILIES = (
    pytest.param("task_website", fixtures.WEBSITE, id="website-controller"),
    pytest.param("task_gitlab", fixtures.GITLAB, id="gitlab-controller"),
    pytest.param("task_gdrive", fixtures.GOOGLEDRIVE, id="googledrive-config"),
    pytest.param("task_llmsim", fixtures.LLM_SIMULATOR, id="llm-user-simulator"),
)


def _infra(name: str, value: str = "supplied") -> ScopedInfraValue:
    return ScopedInfraValue(name=name, purpose="benchmark_compute", value=value)


def _secret(name: str, value: str = "supplied") -> ResolvedSecret:
    return ResolvedSecret(name=name, value=value)


async def _seam(
    tmp_path: Path,
    episode_id: str,
    task_id: str,
    source: str,
    *,
    infra: tuple[ScopedInfraValue, ...] = _INFRA,
    secrets: tuple[ResolvedSecret, ...] = (),
) -> tuple[BaseException | None, list[str]]:
    """Drive ``prepare`` then ``reset_start`` against a recording provider factory.

    The factory raises when it is reached: reaching it means the refusal did not
    fire, and its recorded call is the observable proof that the provider -- the
    object that allocates the guest -- was never constructed.
    """

    built: list[str] = []

    def provider_factory() -> Any:
        built.append("provider")
        raise RuntimeError("reached provider construction after the refusal")

    workspace = _write_workspace(tmp_path, {task_id: source})
    adapter = OSWorldV2Adapter(provider_factory=provider_factory, workspace_root=workspace)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    await adapter.prepare(
        PrepareParams(
            operation_id=f"prepare-{episode_id}",
            episode_id=episode_id,
            secret_refs=(),
            infra_values=infra,
        )
    )
    error: BaseException | None = None
    try:
        await adapter.reset_start(
            ResetStartParams(
                operation_id=f"reset-{episode_id}",
                task_id=task_id,
                episode_id=episode_id,
                artifact_root=str(artifacts),
                secrets=secrets,
            )
        )
    except Exception as failure:  # the seam's refusal IS the subject under test
        error = failure
    return error, built


def _declared(source: str) -> tuple[requirements.Requirement, ...]:
    """The required rows of the TASK-DERIVED layer for one task source."""

    descriptor = taskfile.load_static(source.encode(), module_name="tasks/task.py")
    return tuple(
        requirement
        for requirement in requirements.derive_task_requirements(descriptor)
        if requirement.required
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_id", "source"), _CONDITIONAL_FAMILIES)
async def test_a_required_absent_requirement_is_refused_before_the_provider(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """The refusal is named, and it lands on the free side of the boundary.

    Nothing conditional is supplied, so every required row of the task layer is
    absent -- the exact shape of the measured failure, minus the allocated
    guest.
    """

    expected = {requirement.name for requirement in _declared(source)}
    assert expected, "the fixture family must declare something to be interesting"

    error, built = await _seam(tmp_path, episode_id, task_id, source)

    assert isinstance(error, requirements.MissingRequiredRequirements), error
    message = str(error)
    for name in expected:
        assert name in message
    # The side-effect boundary as an ORDER, not a claim: the provider is what
    # allocates, and it was never constructed.
    assert built == []
    # Names only, never values. Every value in the fixture set is spelled
    # ``value-<NAME>``, so a value leaking into the refusal shows up here.
    assert "value-" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_id", "source"), _CONDITIONAL_FAMILIES)
async def test_a_supplied_conditional_value_runs(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """The inverse direction, so the gate cannot pass by refusing everything.

    Every declared value is supplied -- each through the channel its ``kind``
    names, which is why the table's own kind decides where the test puts it.
    """

    declared = _declared(source)
    error, built = await _seam(
        tmp_path,
        episode_id,
        task_id,
        source,
        infra=_INFRA + tuple(_infra(row.name) for row in declared if row.kind == "infra"),
        secrets=tuple(_secret(row.name) for row in declared if row.kind == "secret"),
    )

    assert built == ["provider"], error


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_id", "source"), _CONDITIONAL_FAMILIES)
async def test_enforcement_and_the_declaration_table_agree(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """Enforcement may not demand anything other than the table's required rows.

    This is the drift the proxy guard already had to be fixed for: a check that
    demands a value the table calls OPTIONAL refuses a run the adapter told an
    operator was runnable, and that is unactionable from the outside. So each
    declared row is dropped in turn and the refusal must name exactly that one.
    """

    declared = _declared(source)
    full_infra = tuple(_infra(row.name) for row in declared if row.kind == "infra")
    full_secrets = tuple(_secret(row.name) for row in declared if row.kind == "secret")

    for row in declared:
        error, built = await _seam(
            tmp_path / row.name,
            episode_id,
            task_id,
            source,
            infra=_INFRA + tuple(item for item in full_infra if item.name != row.name),
            secrets=tuple(item for item in full_secrets if item.name != row.name),
        )
        assert isinstance(error, requirements.MissingRequiredRequirements), (row.name, error)
        message = str(error)
        assert row.name in message
        # Exactly that one: the rows that WERE supplied are not reported, so a
        # check that demanded the whole set would fail this line.
        for other in declared:
            if other.name != row.name:
                assert other.name not in message
        assert built == []


@pytest.mark.asyncio
async def test_an_absent_optional_declaration_stays_benign(tmp_path: Path, episode_id: str) -> None:
    """``required=False`` is not a requirement, whatever it is conditioned on.

    A relative-time evaluator wants its episode clock pinned, and the table
    declares ``OSWORLD_TASK_DATE`` OPTIONAL for it: an unpinned clock is a
    degraded comparison, not a crash, so refusing here would train an operator
    to invent a value.
    """

    optional = {
        requirement.name
        for requirement in requirements.derive_task_requirements(
            taskfile.load_static(fixtures.CLOCK.encode(), module_name="tasks/task_clock.py")
        )
        if not requirement.required
    }
    assert "OSWORLD_TASK_DATE" in optional, "the fixture must exercise the optional path"

    error, built = await _seam(tmp_path, episode_id, "task_clock", fixtures.CLOCK)

    assert built == ["provider"], error


@pytest.mark.asyncio
async def test_a_conditional_value_for_a_task_that_does_not_need_it_is_accepted(
    tmp_path: Path, episode_id: str
) -> None:
    """Supplied-but-unneeded is not this check's problem, and must stay legal.

    The gate reads the TASK's declared set. A value a different task would have
    needed is simply an extra input here, and refusing it would break the
    operator habit of supplying one consistent set for a whole release.
    """

    error, built = await _seam(
        tmp_path,
        episode_id,
        "task_plain",
        fixtures.PLAIN,
        infra=_INFRA + (_infra("WEBSITE_HOST_SUFFIX"),),
    )

    assert built == ["provider"], error


@pytest.mark.asyncio
async def test_the_always_on_baseline_is_not_gated_here(tmp_path: Path, episode_id: str) -> None:
    """Pins the SCOPE of the check, which is why it can live at this seam.

    The baseline rows are properties of the worker environment and of the
    SELECTED provider rather than of the task, so a task whose own fields
    declare nothing must still run with none of the AWS SECRETS supplied: the
    cloud-free build (``adapter-provider.json`` selecting the fake) supplies
    none of them, and the AWS credentials are refused by name when the
    provider is actually built (``adapter._aws_credentials``, still before
    allocation). Demanding them from the task layer would refuse a run this
    adapter can complete.
    """

    error, built = await _seam(tmp_path, episode_id, "task_plain", fixtures.PLAIN, infra=_INFRA)

    assert built == ["provider"], error


@pytest.mark.asyncio
async def test_the_always_on_infra_is_refused_by_its_consumer_by_name(
    tmp_path: Path, episode_id: str
) -> None:
    """The other half of the scope: unenforced here is not unchecked.

    ``provisioning.resolve`` reads each account fact it needs and refuses the
    absent one by name, still before the provider exists -- which is why the
    baseline needs no second check at this seam, and why the refusal that
    fires here is the consumer's ``ProvisioningError`` rather than this
    module's.
    """

    error, built = await _seam(tmp_path, episode_id, "task_plain", fixtures.PLAIN, infra=())

    assert not isinstance(error, requirements.MissingRequiredRequirements)
    assert isinstance(error, ProvisioningError) and "AWS_SUBNET_ID" in str(error)
    assert built == []
