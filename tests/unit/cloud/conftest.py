"""Shared fixtures for the cloud seam's unit tests.

Every test builds its own record and its own config root: the ledger takes an
explicit ``root`` precisely so a test never has to monkeypatch ``HOME``, and the
one place that matters is isolation — a test that wrote into the operator's real
``~/.local-operator/cloud`` would be a defect wearing a green tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.cloud.types import PoolRequest


@pytest.fixture
def request_factory() -> Callable[..., PoolRequest]:
    """A ``PoolRequest`` builder whose every field is settable in one call.

    The defaults are deliberately NOT all equal to the dataclass defaults: a
    round-trip test built on a default-constructed record proves nothing about the
    fields whose default is also the value a lenient reader would fall back to.
    """

    def build(**overrides: Any) -> PoolRequest:
        values: dict[str, Any] = {
            "version": 2,
            "request_id": "req_1a2b3c4d",
            "network_id": "",
            "size": "m",
            "max_hours": 2.0,
            "requested_at": 1789399800.5,
            "requested_by": "d_6c1f",
            "tier": "fargate",
            "budget_usd": 5.0,
            "spot_ok": False,
            "repo_url": "https://example.invalid/fixture.git",
            "repo_sha": "69db7e55fc14f918cccdf2fea62894fc37f1f642",
            "prompt": "make the failing test `test_add` pass",
            "hosting": "hosted",
            "model": "anthropic/claude-sonnet-4-5",
            "labels": {"purpose": "unit-test"},
            "max_sessions": 1,
            "include_local": False,
            "mock": True,
        }
        values.update(overrides)
        return PoolRequest(**values)

    return build


@pytest.fixture(autouse=True)
def hermetic_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ambient AWS configuration may reach these tests.

    A developer's shell may export ``AWS_PROFILE`` or a default region, and botocore
    reads both at client construction. Pointing the config and credentials files at
    empty paths and dropping the ambient variables makes the tests hermetic, and it
    also pins the property the providers rely on: they are built from explicit
    values, never from the environment.
    """

    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_DEFAULT_REGION", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
