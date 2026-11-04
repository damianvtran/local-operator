"""Pure filesystem contracts for same-PID session fixture discovery.

These records implement the idle discovery protocol; they are not live Runtime
or PeerRecord instances. No server, admission fence, or TUI is started here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.sidebar_registry import rekey_session_registry


@dataclass
class IdleRecord:
    pid: int
    heartbeat_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"pid": self.pid, "heartbeat_at": self.heartbeat_at}


@dataclass
class IdleSession(IdleRecord):
    session_id: str = ""

    def to_json(self) -> dict[str, Any]:
        return {**super().to_json(), "session_id": self.session_id}


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    from local_operator.session.runtime import registry, restart_fence

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("idle registry contract must not acquire an admission fence")

    monkeypatch.setattr(restart_fence, "shared", forbidden)
    return registry


@pytest.mark.parametrize("dirname", ["run/peers", "run/other-discovery"])
def test_non_session_records_keep_real_publication_and_cleanup(
    registry: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dirname: str
) -> None:
    rekey_session_registry(monkeypatch)
    record = IdleRecord(43210)
    assert not hasattr(record, "session_id")
    target = registry.publish(record, tmp_path, dirname=dirname)
    assert target == tmp_path / dirname / "43210.json"
    assert target.stat().st_mode & 0o777 == 0o600
    assert record.heartbeat_at > 0
    assert json.loads(target.read_text()) == record.to_json()
    assert list(target.parent.iterdir()) == [target]
    assert registry.unpublish(record.pid, tmp_path, dirname=dirname) is None
    assert not target.exists()


def test_non_session_unpublish_removes_an_existing_peer_record(
    registry: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Seed through the original publisher so a broken publish override cannot
    # hide the independent cleanup regression behind an earlier exception.
    target = registry.publish(IdleRecord(43210), tmp_path, dirname="run/peers")
    rekey_session_registry(monkeypatch)
    registry.unpublish(43210, tmp_path, dirname="run/peers")
    assert not target.exists()


def test_same_pid_sessions_remain_distinct_until_fixture_cleanup(
    registry: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rekey_session_registry(monkeypatch)
    records = [IdleSession(43210, session_id=sid) for sid in ("first", "second")]
    paths = [registry.publish(record, tmp_path) for record in records]
    assert paths == [
        tmp_path / registry.RUN_DIRNAME / f"43210-{sid}.json" for sid in ("first", "second")
    ]
    assert [json.loads(path.read_text()) for path in paths] == [r.to_json() for r in records]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in paths)
    registry.unpublish(43210, tmp_path)
    assert all(path.exists() for path in paths)


def test_delegation_preserves_arguments_results_and_errors(
    registry: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = IdleRecord(43210)
    sentinel = tmp_path / "original-result"
    calls: list[tuple[Any, ...]] = []
    error = ValueError("synthetic delegation error")

    def publish(record: Any, root: Path | None = None, dirname: str = registry.RUN_DIRNAME) -> Path:
        calls.append(("publish", record, root, dirname))
        if dirname == "run/error":
            raise error
        return sentinel

    def unpublish(pid: int, root: Path | None = None, dirname: str = registry.RUN_DIRNAME) -> Any:
        calls.append(("unpublish", pid, root, dirname))
        if dirname == "run/error":
            raise error
        return sentinel

    monkeypatch.setattr(registry, "publish", publish)
    monkeypatch.setattr(registry, "unpublish", unpublish)
    rekey_session_registry(monkeypatch)
    assert registry.publish(record, tmp_path, dirname="custom") is sentinel
    assert registry.unpublish(43210, None, dirname="custom") is sentinel
    assert calls == [("publish", record, tmp_path, "custom"), ("unpublish", 43210, None, "custom")]
    with pytest.raises(ValueError) as published:
        registry.publish(record, None, dirname="run/error")
    assert published.value is error
    with pytest.raises(ValueError) as unpublished:
        registry.unpublish(43210, tmp_path, dirname="run/error")
    assert unpublished.value is error
    assert calls[-2:] == [
        ("publish", record, None, "run/error"),
        ("unpublish", 43210, tmp_path, "run/error"),
    ]
