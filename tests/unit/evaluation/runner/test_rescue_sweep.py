"""``sweep_rescue_root``: rescue every descriptor, retire only the confirmed.

The script-level tests (``adapters/osworld/test_build_and_scripts.py``) cover
``scripts/osworld_rescue_sweep.py``'s argument handling over this module;
these pin the module's own contract, which the in-process runner and the
script both rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import (
    ADAPTER_SCHEMA_VERSION,
    RescueDescriptor,
    ResolvedSecret,
    SecretRef,
)
from local_operator.evaluation.adapters.supervisor import persist_rescue
from local_operator.evaluation.lifecycle import record_cleanup
from local_operator.evaluation.runner.rescue_sweep import sweep_rescue_root
from local_operator.evaluation.runner.secrets import MissingSecret, StaticSecretResolver
from tests.unit.evaluation.runner.conftest import cleanup_plan, handshake, selector


def _descriptor(tmp_path: Path, episode_id: str, *refs: str) -> RescueDescriptor:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return RescueDescriptor(
        schema_version=ADAPTER_SCHEMA_VERSION,
        selector=selector(tmp_path),
        handshake=handshake(tmp_path),
        episode_id=episode_id,
        cleanup_plan=cleanup_plan(episode_id),
        secret_refs=tuple(SecretRef(name=name) for name in refs),
        infra_values=(),
        artifact_root=str(tmp_path),
    )


class _Aggregate:
    def __init__(self, complete: bool, receipts: tuple[Any, ...]) -> None:
        self.complete = complete
        self.receipts = receipts


def _aggregate(descriptor: RescueDescriptor, *, complete: bool, code: str) -> _Aggregate:
    action = descriptor.cleanup_plan.actions[0]
    receipt = record_cleanup(
        descriptor.cleanup_plan,
        action.action_id,
        status="succeeded" if complete else "attempted",
        evidence_code=code,
        duration_ms=1,
    )
    return _Aggregate(complete, (receipt,))


@pytest.mark.asyncio
async def test_sweep_rescues_each_descriptor_with_its_secrets_and_retires_only_complete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rescue"
    done = _descriptor(tmp_path / "a", "ep-done", "AWS_SECRET_ACCESS_KEY")
    stuck = _descriptor(tmp_path / "b", "ep-stuck", "AWS_SECRET_ACCESS_KEY")
    persist_rescue(root / "ep-done", done)
    persist_rescue(root / "ep-stuck", stuck)
    (root / "ep-empty").mkdir()  # no descriptor: skipped, not an error
    delivered: list[tuple[str, tuple[ResolvedSecret, ...]]] = []
    launches: list[Any] = []

    async def rescue(descriptor: Any, *, secrets: Any, launch: Any) -> Any:
        delivered.append((descriptor.episode_id, secrets))
        launches.append(launch)
        if descriptor.episode_id == "ep-done":
            return _aggregate(descriptor, complete=True, code="instance-terminated")
        return _aggregate(descriptor, complete=False, code="terminate-unconfirmed")

    sentinel_launch = object()
    entries = await sweep_rescue_root(
        root,
        StaticSecretResolver({"AWS_SECRET_ACCESS_KEY": "value-for-rescue"}),
        launch=sentinel_launch,
        rescue=rescue,
    )

    assert [(e.episode_id, e.complete, e.codes, e.error) for e in entries] == [
        ("ep-done", True, ("instance-terminated",), None),
        ("ep-stuck", False, ("terminate-unconfirmed",), None),
    ]
    assert delivered == [
        ("ep-done", (ResolvedSecret(name="AWS_SECRET_ACCESS_KEY", value="value-for-rescue"),)),
        ("ep-stuck", (ResolvedSecret(name="AWS_SECRET_ACCESS_KEY", value="value-for-rescue"),)),
    ]
    assert launches == [sentinel_launch, sentinel_launch]
    assert not (root / "ep-done" / "rescue.json").exists()
    assert (root / "ep-stuck" / "rescue.json").exists()


@pytest.mark.asyncio
async def test_sweep_reports_missing_secret_by_name_and_never_launches(tmp_path: Path) -> None:
    root = tmp_path / "rescue"
    persist_rescue(root / "ep-x", _descriptor(tmp_path / "a", "ep-x", "AWS_SECRET_ACCESS_KEY"))

    async def never(descriptor: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("rescue must not run without its secrets")

    entries = await sweep_rescue_root(root, StaticSecretResolver({}), rescue=never)
    assert len(entries) == 1
    assert entries[0].complete is False
    assert entries[0].error == str(MissingSecret("AWS_SECRET_ACCESS_KEY"))
    assert (root / "ep-x" / "rescue.json").exists()


@pytest.mark.asyncio
async def test_sweep_reports_a_failed_rescue_and_keeps_going(tmp_path: Path) -> None:
    root = tmp_path / "rescue"
    persist_rescue(root / "ep-1", _descriptor(tmp_path / "a", "ep-1"))
    persist_rescue(root / "ep-2", _descriptor(tmp_path / "b", "ep-2"))

    async def rescue(descriptor: Any, **kwargs: Any) -> Any:
        if descriptor.episode_id == "ep-1":
            raise RuntimeError("handshake differs")
        return _aggregate(descriptor, complete=True, code="instance-absent")

    entries = await sweep_rescue_root(root, StaticSecretResolver({}), rescue=rescue)
    assert [(e.episode_id, e.complete, e.error) for e in entries] == [
        ("ep-1", False, "rescue failed: handshake differs"),
        ("ep-2", True, None),
    ]
    assert (root / "ep-1" / "rescue.json").exists()
    assert not (root / "ep-2" / "rescue.json").exists()


@pytest.mark.asyncio
async def test_empty_or_missing_root_sweeps_nothing(tmp_path: Path) -> None:
    assert await sweep_rescue_root(tmp_path / "absent", StaticSecretResolver({})) == ()
    (tmp_path / "empty").mkdir()
    assert await sweep_rescue_root(tmp_path / "empty", StaticSecretResolver({})) == ()
