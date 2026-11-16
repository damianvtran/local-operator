"""The receiver records the build it is running; ``receiver status`` reports it.

Remote rollout confirmation: a launchd/systemd receiver keeps the code it was
exec'd from across ``lop`` updates, so ``status`` must name the running
version, generation, consent set, start and pid, and say ``Restart needed``
when that differs from the installed version (see ``runtime_record``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator import update
from local_operator.team_bridge.connector import runtime_record as rr
from local_operator.team_bridge.consent_versions import DEFAULT_ON_CONSENT_VERSIONS


@pytest.fixture
def facts(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Pin the version both sides read and the generation probe (no ``ps``)."""

    state: dict[str, Any] = {"version": "0.102.4", "generation": None, "stale": None}
    monkeypatch.setattr(update, "installed_version", lambda: state["version"])
    monkeypatch.setattr(
        update,
        "generation_of_process",
        lambda pid: Path("/gen") / state["generation"] if state["generation"] else None,
    )
    monkeypatch.setattr(update, "generation_staleness_of_process", lambda pid: state["stale"])
    return state


def _record(root: Path) -> dict[str, Any]:
    return json.loads(rr.record_path(root).read_text(encoding="utf-8"))


def test_start_records_the_running_build(tmp_path: Path, facts: dict[str, Any]) -> None:
    facts["generation"] = "20261115T090000Z-0.102.4"
    rr.record_start(tmp_path, started_at=1_700_000_000.0)
    record = _record(tmp_path)
    assert record["pid"] == os.getpid()
    assert record["version"] == "0.102.4"
    assert record["generation"] == "20261115T090000Z-0.102.4"
    assert record["consent_versions"] == sorted(DEFAULT_ON_CONSENT_VERSIONS)
    assert "bridge_consent_v3" in record["consent_versions"]
    assert record["started_at"] == 1_700_000_000.0
    assert record["singleton"] == rr.SINGLETON_WAITING

    rr.record_singleton_held(tmp_path, started_at=1_700_000_000.0)
    assert _record(tmp_path)["singleton"] == rr.SINGLETON_HELD

    rr.clear_own_record(tmp_path)
    assert not rr.record_path(tmp_path).exists()


def test_start_never_hides_a_live_peer(
    tmp_path: Path, facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A newer receiver refused by an older live holder must leave the holder named."""

    peer = {"schema": 1, "pid": 999_999, "version": "0.86.1", "singleton": "held"}
    rr.record_path(tmp_path).parent.mkdir(parents=True)
    rr.record_path(tmp_path).write_text(json.dumps(peer), encoding="utf-8")

    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: True)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == 999_999, "a live peer's record was clobbered"
    rr.clear_own_record(tmp_path)
    assert _record(tmp_path)["pid"] == 999_999, "clear removed a record that is not ours"

    # Unsure (``None``) is treated as alive: never clobber on doubt.
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: None)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == 999_999

    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: False)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == os.getpid(), "a dead peer's record must be replaced"


def test_restart_needed_only_for_a_live_receiver_on_other_code(
    tmp_path: Path, facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert rr.running_receiver(tmp_path) is None, "nothing recorded reads as None"

    rr.record_start(tmp_path, started_at=1.0)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: True)
    same = rr.running_receiver(tmp_path)
    assert same is not None and same["restart_needed"] is False
    assert same["installed_version"] == same["version"] == "0.102.4"

    # An update moved the installed version under the running process.
    facts["version"] = "0.102.5"
    moved = rr.running_receiver(tmp_path)
    assert moved is not None
    assert moved["version"] == "0.102.4" and moved["installed_version"] == "0.102.5"
    assert moved["version_mismatch"] is True and moved["restart_needed"] is True

    # Same version, but a generation provably not ``current``.
    facts["version"] = "0.102.4"
    facts["stale"] = True
    stale = rr.running_receiver(tmp_path)
    assert stale is not None and stale["restart_needed"] is True

    # A process that is gone is "exited", never "restart needed".
    facts["version"] = "0.102.5"
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: False)
    gone = rr.running_receiver(tmp_path)
    assert gone is not None and gone["alive"] is False and gone["restart_needed"] is False
    assert gone["generation_stale"] is None, "no generation probe for a dead pid"

    # An unknown installed version earns no verdict.
    facts["version"] = ""
    facts["stale"] = None
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: True)
    unknown = rr.running_receiver(tmp_path)
    assert unknown is not None and unknown["restart_needed"] is False


def test_a_reused_pid_is_not_a_running_receiver(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.team_bridge.connector import reap

    monkeypatch.setattr(reap, "_is_posix", lambda: True)
    monkeypatch.setattr(
        reap,
        "_read_process",
        lambda pid: reap._Process(pid=pid, uid=0, start="x", cpu_s=0.0, argv="/usr/bin/vim notes"),
    )
    assert rr._receiver_alive(os.getpid()) is False
    monkeypatch.setattr(
        reap,
        "_read_process",
        lambda pid: reap._Process(
            pid=pid,
            uid=0,
            start="x",
            cpu_s=0.0,
            argv="/x/python -m local_operator.team_bridge.connector --config-dir /c",
        ),
    )
    assert rr._receiver_alive(os.getpid()) is True
    assert rr._receiver_alive(0) is False


def _status(argv: list[str]) -> Any:
    from local_operator.cli import build_cli_parser

    return build_cli_parser().parse_args(argv)


def test_status_prints_the_running_build_and_the_mismatch(
    tmp_path: Path,
    facts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    assert "Running receiver: none recorded" in capsys.readouterr().out
    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["running"] is None

    facts["generation"] = "20261101T000000Z-0.102.4"
    rr.record_start(tmp_path, started_at=1_700_000_000.0)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: True)
    facts["version"] = "0.103.0"

    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    out = capsys.readouterr().out
    assert f"Running receiver: pid {os.getpid()}, version 0.102.4, started " in out
    assert (
        "Running build: 20261101T000000Z-0.102.4; "
        "consent versions it accepts: bridge_consent_v2, bridge_consent_v3" in out
    )
    assert "Restart needed: the running receiver is not the installed version (0.103.0)" in out

    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    running = json.loads(capsys.readouterr().out)["running"]
    assert running["pid"] == os.getpid() and running["restart_needed"] is True

    monkeypatch.setattr(rr, "_receiver_alive", lambda pid: False)
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    out = capsys.readouterr().out
    assert f"(pid {os.getpid()}, version 0.102.4) has exited" in out
    assert "Restart needed" not in out


def test_the_connector_records_itself_before_waiting_and_clears_on_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring: an idle (not enrolled) receiver is recorded too, and a stop clears it."""

    from local_operator.team_bridge.connector import __main__ as connector_main

    calls: list[str] = []
    monkeypatch.setattr(rr, "record_start", lambda root, **kw: calls.append("start"))
    monkeypatch.setattr(rr, "clear_own_record", lambda root: calls.append("clear"))
    monkeypatch.setattr(connector_main, "_configure_logging", lambda root: None)
    monkeypatch.setattr(connector_main, "_activate_operator_locale", lambda root: None)
    stop = asyncio.Event()
    stop.set()
    args = argparse.Namespace(
        config_dir=str(tmp_path),
        base_url=None,
        credential_id=None,
        account_id=None,
        tenant_id=None,
        endpoint_id=None,
    )
    assert asyncio.run(connector_main._main_async(args, stop)) == 0
    assert calls == ["start", "clear"]
