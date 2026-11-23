"""A receiver that keeps failing to persist the same batch reports a jam.

The 2026-11-17 incident: ``record_feed_batch`` refused with
``ref_migration_malformed`` on every pass for four days, the cursor never
advanced, and nothing but a peer's ``not_admitted`` said so. ``run_forever``
now counts consecutive identical persistence refusals and ``receiver status``
reports them as ``jammed``.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge.connector import jam as jam_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.store import StoreRefused
from local_operator.team_bridge.types import OwnerSnapshot

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://api.radienthq.com/v1",
    enrollment_generation=1,
)


def _drive(tmp_path: Path, failures: list[BaseException | None], monkeypatch) -> None:
    """Run the real ``run_forever`` through ``failures`` (``None`` = a clean pass)."""

    stop = asyncio.Event()
    queue = list(failures)

    async def no_pause(stop_event: Any, seconds: float) -> None:
        return None

    monkeypatch.setattr(loop_mod, "_pause", no_pause)

    async def access_provider() -> Any:
        return object()

    async def run_iteration(*a: Any, **kw: Any) -> dict[str, Any]:
        failure = queue.pop(0)
        if not queue:
            stop.set()
        if failure is not None:
            raise failure
        return {}

    monkeypatch.setattr(loop_mod, "run_iteration", run_iteration)

    asyncio.run(
        loop_mod.run_forever(
            OWNER,
            config_dir=tmp_path,
            auth_store=None,
            access_provider=access_provider,
            owner_source=lambda: OWNER,
            stop=stop,
        )
    )


def _refusal() -> StoreRefused:
    return StoreRefused("ref_migration_malformed", "malformed conversation delivery key")


def test_the_incident_shape_is_reported_as_a_jam(tmp_path: Path, monkeypatch) -> None:
    _drive(tmp_path, [_refusal() for _ in range(jam_mod.JAM_THRESHOLD)], monkeypatch)

    jam = jam_mod.read_jam(tmp_path)
    assert jam is not None
    assert jam["code"] == "ref_migration_malformed"
    assert jam["count"] == jam_mod.JAM_THRESHOLD
    # Content law: closed code and numbers only, never the refusal's message.
    raw = jam_mod.jam_path(tmp_path).read_text(encoding="utf-8")
    assert "malformed conversation delivery key" not in raw
    assert json.loads(raw)["pid"] == os.getpid()


def test_one_short_of_the_threshold_is_not_a_jam(tmp_path: Path, monkeypatch) -> None:
    _drive(tmp_path, [_refusal() for _ in range(jam_mod.JAM_THRESHOLD - 1)], monkeypatch)
    assert jam_mod.read_jam(tmp_path) is None


def test_a_clean_pass_clears_the_jam(tmp_path: Path, monkeypatch) -> None:
    failures: list[BaseException | None] = [_refusal() for _ in range(jam_mod.JAM_THRESHOLD)]
    _drive(tmp_path, [*failures, None], monkeypatch)
    assert not jam_mod.jam_path(tmp_path).exists()


def test_network_failures_neither_count_nor_reset(tmp_path: Path, monkeypatch) -> None:
    """An offline laptop is not a jam; and it is not evidence the jam ended."""

    half = jam_mod.JAM_THRESHOLD // 2
    failures: list[BaseException | None] = [
        *[_refusal() for _ in range(half)],
        *[OSError("offline") for _ in range(jam_mod.JAM_THRESHOLD)],
        *[_refusal() for _ in range(jam_mod.JAM_THRESHOLD - half)],
    ]
    _drive(tmp_path, failures, monkeypatch)
    jam = jam_mod.read_jam(tmp_path)
    assert jam is not None and jam["count"] == jam_mod.JAM_THRESHOLD

    other = tmp_path / "offline-only"
    _drive(other, [OSError("offline") for _ in range(3 * jam_mod.JAM_THRESHOLD)], monkeypatch)
    assert jam_mod.read_jam(other) is None


def test_a_different_code_restarts_the_count(tmp_path: Path) -> None:
    tracker = jam_mod.JamTracker(tmp_path, threshold=3)
    tracker.note_failure(_refusal())
    tracker.note_failure(_refusal())
    tracker.note_failure(StoreRefused("schema_damaged", "x"))
    tracker.note_failure(StoreRefused("schema_damaged", "x"))
    assert jam_mod.read_jam(tmp_path) is None
    tracker.note_failure(StoreRefused("schema_damaged", "x"))
    jam = jam_mod.read_jam(tmp_path)
    assert jam is not None
    assert jam["code"] == "schema_damaged"


def test_an_unsafe_code_and_a_dead_pid_are_not_reported(tmp_path: Path, monkeypatch) -> None:
    tracker = jam_mod.JamTracker(tmp_path, threshold=1)
    tracker.note_failure(StoreRefused("Has Spaces: and content", "x"))
    assert not jam_mod.jam_path(tmp_path).exists()

    tracker.note_failure(_refusal())
    from local_operator import procstate

    monkeypatch.setattr(procstate, "pid_liveness", lambda pid: False)
    assert jam_mod.read_jam(tmp_path) is None


def test_status_json_reports_jammed(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from local_operator.team_bridge.connector import cli, install

    monkeypatch.setattr(install, "connector_state", lambda root: install.STATE_INSTALLED)
    tracker = jam_mod.JamTracker(tmp_path, threshold=1)
    tracker.note_failure(_refusal())

    assert cli._status(tmp_path, as_json=True) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["jammed"]["code"] == "ref_migration_malformed"
    assert payload["jammed"]["count"] == 1

    tracker.note_success()
    assert cli._status(tmp_path, as_json=True) == 0
    assert json.loads(capsys.readouterr().out)["jammed"] is None
