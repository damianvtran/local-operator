"""A declared requirement that cannot be satisfied is refused BEFORE allocation.

THE CONTRACT these tests pin: declaring a requirement means refusing when it is
missing -- and never offering a remedy that cannot be performed. That second half
is not decoration: the failure it prevents is an operator complying with the
refusal and the episode still allocating a guest and dying in vendor code.

The measured failure this closes: 40 of the release's 108 task modules reach a
controller that reads its value out of the environment when the guest's task
object is instantiated, and nothing supplied it. ``task_016`` therefore allocated
an EC2 desktop and only then failed inside vendor code::

    ValueError: WEBSITE_HOST_SUFFIX must be set in environment variables
      threads.py:25 to_thread <- aws.py:863 _start_desktop_env
      <- vendor_bridge.py:202 instantiate_task <- task_016.py:7 <- website.py:22

-- a traceback naming no adapter, for a requirement the task's own source
declares. GOOGLE_ACCOUNT_CREDENTIALS, OSWORLD_USER_SIM_API_KEY and
GITLAB_PRIVATE_TOKEN/GITLAB_URL are the other four families.

FIVE claims, each with its own test:

1. a required-and-absent name that CAN be delivered is refused, names the absent
   REF (never a value), says which flag supplies it, and lands BEFORE the
   provider is constructed -- watching the provider factory, which is what
   actually allocates, rather than reading a comment;
2. a required name this build cannot deliver is refused even when it IS
   supplied, because the remedy would be inert (``GITLAB_PRIVATE_TOKEN``: the
   controller reads it from ``os.getenv`` at import and nothing here puts it
   there);
3. it is derived from the table, so it covers every family a task's own fields
   introduce instead of one hardcoded name (the agreement test removes each
   deliverable row in turn and requires exactly that one back);
4. it does not over-refuse: a supplied value runs, an absent OPTIONAL
   declaration is benign, a documented substitute satisfies what it stands in
   for, and a task-conditional value given to a task that does not need it is
   accepted;
5. the always-on baseline is deliberately NOT gated here, which is the reason
   this check can live at the task seam at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from lop_osworld_v2_adapter import provisioning, requirements, taskfile, vendor_bridge
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

# Families whose declared names this build CAN deliver: the value reaches the
# controller, so an operator who supplies it gets a run.
_MISSING_FAMILIES = (
    pytest.param("task_website", fixtures.WEBSITE, id="website-controller"),
    pytest.param("task_llmsim", fixtures.LLM_SIMULATOR, id="llm-user-simulator"),
)

# Families whose declared name has NO channel to its consumer in this build. The
# declared set is read from the table rather than restated here, so adding a row
# to the table covers the new name without touching this file.
_UNDELIVERABLE_FAMILIES = (pytest.param("task_gitlab", fixtures.GITLAB, id="gitlab-controller"),)


def _infra(name: str, value: str = "supplied") -> ScopedInfraValue:
    return ScopedInfraValue(name=name, purpose="benchmark_compute", value=value)


def _secret(name: str, value: str = "supplied") -> ResolvedSecret:
    return ResolvedSecret(name=name, value=value)


def _declared(source: str) -> tuple[Any, ...]:
    descriptor = taskfile.load_static(source.encode(), module_name="tasks/t.py")
    return requirements.derive_task_requirements(descriptor)


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

    The factory raises when it is reached: reaching it means no refusal fired,
    and its recorded call is the observable proof that the provider -- the object
    that allocates the guest -- was constructed.
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


def _required(source: str) -> tuple[Any, ...]:
    return tuple(row for row in _declared(source) if row.required)


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_id", "source"), _MISSING_FAMILIES)
async def test_a_required_absent_requirement_is_refused_before_the_provider(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """The deliverable half: refused, by name, with the flag that supplies it."""

    expected = _required(source)
    assert expected, "the fixture family must declare something to be interesting"

    error, built = await _seam(tmp_path, episode_id, task_id, source)

    assert isinstance(error, requirements.MissingRequiredRequirements), error
    message = str(error)
    for row in expected:
        assert row.name in message
        # The remedy is one step: the KIND decides the flag, and the flag is named.
        assert f"--{row.kind} {row.name}" in message
    # The side-effect boundary as an ORDER, not a claim: the provider is what
    # allocates, and it was never constructed.
    assert built == []
    # Names only, never values. Every value in the fixture set is spelled
    # ``value-<NAME>``, so a value leaking into the refusal shows up here.
    assert "value-" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_id", "source"), _UNDELIVERABLE_FAMILIES)
async def test_a_required_name_no_channel_can_carry_is_refused_even_when_supplied(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """QA round 1's reproduction, as a test: the inert remedy.

    ``GITLAB_PRIVATE_TOKEN`` is declared by the task, matched by name -- and
    unreachable: ``desktop_env/controllers/gitlab.py:19-23`` reads it from
    ``os.getenv`` at import, and this build hands the vendor only the env-delivery
    allowlist plus the provider's own credentials. So an operator who complies
    with a "supply this" refusal still gets the old failure: a guest allocated,
    then a vendor ``ValueError`` mid-setup. Refusing a value the run cannot use is
    the only honest answer, and it must hold in BOTH directions -- unsupplied and
    supplied.
    """

    declared = _required(source)
    # Secrets only: an infra value is deliverable by construction, so it is never
    # the reason this refusal fires (the message must agree with that split).
    undeliverable = {
        row.name
        for row in declared
        if row.kind == "secret" and row.name not in requirements._DELIVERED_SECRETS
    }
    assert (
        "GITLAB_PRIVATE_TOKEN" in undeliverable
    ), "the fixture family must declare the name with no channel"

    # (a) nothing supplied
    bare, bare_built = await _seam(tmp_path / "bare", episode_id, task_id, source)
    assert isinstance(bare, requirements.UndeliverableRequirement), bare
    for name in undeliverable:
        assert name in str(bare)
    assert bare_built == []

    # (b) the operator COMPLIED: every declared name supplied, the undeliverable
    # one included. The remedy still cannot be performed, so the refusal must not
    # disappear -- this is QA round 1's reproduction, and the reason the check
    # cannot be a supply-vs-missing test alone.
    complied, complied_built = await _seam(
        tmp_path / "complied",
        episode_id,
        task_id,
        source,
        infra=_INFRA + tuple(_infra(row.name) for row in declared if row.kind == "infra"),
        secrets=tuple(_secret(row.name) for row in declared if row.kind == "secret"),
    )
    assert isinstance(complied, requirements.UndeliverableRequirement), complied
    assert complied_built == []
    assert "change nothing" in str(complied)


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_id", "source"), _MISSING_FAMILIES)
async def test_a_supplied_conditional_value_runs(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """The inverse direction: supply what the table asks for and the run proceeds.

    Without this the gate could pass by refusing everything, which would break the
    very family it exists to make runnable.
    """

    declared = _required(source)
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
async def test_the_documented_substitute_satisfies_the_simulator_key(
    tmp_path: Path, episode_id: str
) -> None:
    """The vendor's own fallback chain is part of the contract this check reads.

    ``user_simulator.respond`` hands its options to ``model_client.generate_chat``
    WITHOUT a key when the simulator has none configured, and that client resolves
    one through ``default_api_key_env`` -- the judge key. Refusing the simulator
    key on its absence alone would newly block 6 release tasks that ran before
    (task_007/024/026/034/095/098), i.e. a coverage regression dressed as safety.
    """

    error, built = await _seam(
        tmp_path,
        episode_id,
        "task_llmsim",
        fixtures.LLM_SIMULATOR,
        secrets=(_secret(requirements._JUDGE_SECRET),),
    )

    assert built == ["provider"], error


@pytest.mark.asyncio
async def test_the_substitute_is_named_in_the_refusal(tmp_path: Path, episode_id: str) -> None:
    """And when neither is present, the refusal says what else would satisfy it."""

    error, built = await _seam(tmp_path, episode_id, "task_llmsim", fixtures.LLM_SIMULATOR)

    assert isinstance(error, requirements.MissingRequiredRequirements), error
    assert requirements._JUDGE_SECRET in str(error)
    assert built == []


@pytest.mark.asyncio
async def test_an_absent_optional_declaration_stays_benign(tmp_path: Path, episode_id: str) -> None:
    """``required=False`` is not a requirement, whatever it is conditioned on.

    Two cases, because the two ways to be optional are different facts. A
    relative-time evaluator wants its episode clock pinned and the table declares
    ``OSWORLD_TASK_DATE`` optional for it: an unpinned clock is a degraded
    comparison, not a crash. And ``GOOGLE_ACCOUNT_CREDENTIALS`` is optional
    because the pinned vendor reads it NOWHERE -- declaring it binding would demand
    a credential nothing consumes, the anti-pattern this table already condemns
    for ``OSWORLD_PROXY_CREDENTIALS``.
    """

    optional = {
        row.name
        for row in requirements.derive_task_requirements(
            taskfile.load_static(fixtures.CLOCK.encode(), module_name="tasks/task_clock.py")
        )
        if not row.required
    }
    assert "OSWORLD_TASK_DATE" in optional, "the fixture must exercise the optional path"
    clock_error, clock_built = await _seam(
        tmp_path / "clock", episode_id, "task_clock", fixtures.CLOCK
    )
    assert clock_built == ["provider"], clock_error

    gdrive = {row.name: row for row in _declared(fixtures.GOOGLEDRIVE)}
    assert gdrive["GOOGLE_ACCOUNT_CREDENTIALS"].required is False
    drive_error, drive_built = await _seam(
        tmp_path / "gdrive", episode_id, "task_gdrive", fixtures.GOOGLEDRIVE
    )
    assert drive_built == ["provider"], drive_error


@pytest.mark.asyncio
async def test_a_conditional_value_for_a_task_that_does_not_need_it_is_accepted(
    tmp_path: Path, episode_id: str
) -> None:
    """Supplied-but-unneeded is not this check's problem, and must stay legal.

    The gate reads the TASK's declared set. A value a different task would have
    needed is simply an extra input here, and refusing it would break the operator
    habit of supplying one consistent set for a whole release.
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
@pytest.mark.parametrize(("task_id", "source"), _MISSING_FAMILIES)
async def test_enforcement_and_the_declaration_table_agree(
    tmp_path: Path, episode_id: str, task_id: str, source: str
) -> None:
    """Enforcement may not demand anything other than the table's required rows.

    This is the drift the proxy guard already had to be fixed for: a check that
    demands a value the table calls OPTIONAL refuses a run the adapter told an
    operator was runnable, and that is unactionable from the outside. So each
    deliverable row is dropped in turn and the refusal must name exactly that one.
    """

    declared = _required(source)
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


def test_the_delivery_table_is_derived_not_restated() -> None:
    """Anti-drift: the delivery allowlist has ONE definition, in vendor_bridge.

    The check refuses a name because no channel can carry it, so the check's
    notion of "carriable" must be the delivery code's own list. A second copy here
    would let the two disagree silently -- the exact mistake the infra-disclosure
    table warns about -- and the symptom would be a refusal for a value that in
    fact reaches its consumer, or a run that proceeds on one that never can.
    """

    assert (
        requirements._DELIVERED_SECRETS
        == frozenset({"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"})
        | vendor_bridge.SECRET_ENV_NAMES
    )


@pytest.mark.asyncio
async def test_the_always_on_baseline_is_not_gated_here(tmp_path: Path, episode_id: str) -> None:
    """Pins the SCOPE of the check, which is why it can live at this seam.

    The baseline rows are properties of the worker environment and of the
    SELECTED provider rather than of the task, so a task whose own fields declare
    nothing must still run with none of the AWS SECRETS supplied: the cloud-free
    build (``adapter-provider.json`` selecting the fake) supplies none of them,
    and the AWS credentials are refused by name when the provider is actually
    built (``adapter._aws_credentials``, still before allocation). Demanding them
    from the task layer would refuse a run this adapter can complete.
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
    baseline needs no second check at this seam, and why the refusal that fires
    here is the consumer's ``ProvisioningError`` rather than this module's.
    """

    error, built = await _seam(tmp_path, episode_id, "task_plain", fixtures.PLAIN, infra=())

    assert not isinstance(error, requirements.MissingRequiredRequirements)
    assert isinstance(error, ProvisioningError) and "AWS_SUBNET_ID" in str(error)
    assert built == []
