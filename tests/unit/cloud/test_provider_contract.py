"""The seam itself: what ``ComputeProvider`` is, and who satisfies it.

Two properties are load-bearing here. The Protocol is STRUCTURAL, so the ledger
can hold any backend and never ask which (§3.2) — that is what makes
``FakeComputeProvider`` a real test of the path ``FargateProvider`` walks rather
than a mock beside it. And ``submit`` is the ONLY method whose call can create
anything, which the fake proves by construction: its ``describe()`` reading moves
when a submit happens and for no other method at all.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.cloud.fake import FakeComputeProvider
from local_operator.cloud.provider import ComputeProvider
from local_operator.cloud.types import PoolRequest

#: The seam's whole surface. Widening it is a design change, not a refactor, so
#: the set is pinned rather than described.
PROTOCOL_METHODS = frozenset({"describe", "submit", "status", "cancel", "drain"})

#: Every provider class this build ships, by name. Slice 2 adds ``fargate`` and
#: slice 4 adds ``k8s``; a new backend that is not listed here is a backend
#: nothing checks against the seam.
PROVIDERS: dict[str, Callable[[], Any]] = {"fake": FakeComputeProvider}


def _request() -> PoolRequest:
    return PoolRequest(request_id="req_0f0f0f0f", budget_usd=1.0, mock=True)


def test_the_seam_is_exactly_the_five_async_methods() -> None:
    """A sixth verb, or a sync one, changes what every backend must implement."""
    members = {name for name in dir(ComputeProvider) if not name.startswith("_")}
    assert members == PROTOCOL_METHODS
    # ``name`` is an ANNOTATION on the Protocol rather than a class attribute, so
    # it is pinned on the annotations instead: a backend without it cannot be
    # named in a ``--json`` body or in the ledger's ``provider`` field.
    assert "name" in ComputeProvider.__annotations__
    for method in sorted(PROTOCOL_METHODS):
        assert inspect.iscoroutinefunction(getattr(ComputeProvider, method)), method


@pytest.mark.parametrize("provider_name", sorted(PROVIDERS))
def test_every_provider_class_satisfies_the_runtime_checkable_protocol(
    provider_name: str,
) -> None:
    """``isinstance`` against a ``runtime_checkable`` Protocol: the seam is structural."""
    provider = PROVIDERS[provider_name]()
    assert isinstance(provider, ComputeProvider), provider_name
    assert provider.name == provider_name
    for method in sorted(PROTOCOL_METHODS):
        assert inspect.iscoroutinefunction(getattr(provider, method)), method


@pytest.mark.asyncio
async def test_submit_is_the_only_fake_method_that_changes_what_describe_reports() -> None:
    """The fake's own proof that "create" is a singular verb on this seam.

    ``describe()`` is a capacity reading (for Fargate it carries the cluster and
    the task definition the stack points at), so if a cancel or a status call
    moved it, the reading would depend on traffic rather than on capacity.
    """
    provider = FakeComputeProvider()
    baseline = await provider.describe()
    assert baseline.provider == "fake"

    ref = await provider.submit(_request())
    after_submit = await provider.describe()
    assert after_submit != baseline, "submit creates a workload, so describe() must see it"

    await provider.status(ref)
    await provider.cancel(ref, reason="a unit test")
    await provider.drain(ref, deadline_s=0.0)
    assert await provider.describe() == after_submit


@pytest.mark.asyncio
async def test_the_fake_refuses_a_reference_it_does_not_know() -> None:
    """A reading for an unknown reference is a refusal, never a guess."""
    from local_operator.cloud.provider import CloudRefused

    provider = FakeComputeProvider()
    with pytest.raises(CloudRefused) as refusal:
        await provider.status("fake-9999")
    assert refusal.value.code == "provider_ref_unknown"


def test_the_cloud_package_imports_with_boto3_absent() -> None:
    """A default install must be able to import ``local_operator.cloud`` (§3.7).

    Run in a SUBPROCESS: the simulation has to be a genuinely fresh interpreter
    import of the whole package graph (``sys.modules["boto3"] = None`` makes any
    ``import boto3`` raise), and doing it here would leave this interpreter with a
    half-poisoned module cache for every test that follows.
    """
    repo_root = Path(__file__).resolve().parents[3]
    script = textwrap.dedent("""
        import sys

        sys.modules["boto3"] = None
        import local_operator.cloud as cloud

        print("imported", cloud.CloudLedger.__name__, cloud.FakeComputeProvider.name)
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "imported CloudLedger fake" in completed.stdout
