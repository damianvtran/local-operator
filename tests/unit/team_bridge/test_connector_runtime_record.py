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

    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: True)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == 999_999, "a live peer's record was clobbered"
    rr.clear_own_record(tmp_path)
    assert _record(tmp_path)["pid"] == 999_999, "clear removed a record that is not ours"

    # Unsure (``None``) is treated as alive: never clobber on doubt.
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: None)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == 999_999

    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: False)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == os.getpid(), "a dead peer's record must be replaced"


def test_restart_needed_only_for_a_live_receiver_on_other_code(
    tmp_path: Path, facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert rr.running_receiver(tmp_path) is None, "nothing recorded reads as None"

    rr.record_start(tmp_path, started_at=1.0)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: True)
    same = rr.running_receiver(tmp_path)
    assert same is not None and same["restart_needed"] is False
    assert same["installed_version"] == same["version"] == "0.102.4"

    # An update moved the installed version under the running process.
    facts["version"] = "0.102.5"
    moved = rr.running_receiver(tmp_path)
    assert moved is not None
    assert moved["version"] == "0.102.4" and moved["installed_version"] == "0.102.5"
    assert moved["version_mismatch"] is True and moved["restart_needed"] is True
    assert moved["restart_reason"] == "version_mismatch"

    # Same version, but a generation provably not ``current``.
    facts["version"] = "0.102.4"
    facts["stale"] = True
    stale = rr.running_receiver(tmp_path)
    assert stale is not None and stale["restart_needed"] is True
    assert stale["restart_reason"] == "replaced_build"

    # Liveness unproven (no ``ps``, Windows): a cause is "unknown", never True.
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: None)
    unsure = rr.running_receiver(tmp_path)
    assert unsure is not None and unsure["restart_needed"] is None
    assert unsure["restart_reason"] == "replaced_build"

    # A process that is gone is "exited", never "restart needed".
    facts["version"] = "0.102.5"
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: False)
    gone = rr.running_receiver(tmp_path)
    assert gone is not None and gone["alive"] is False and gone["restart_needed"] is False
    assert gone["restart_reason"] is None
    assert gone["generation_stale"] is None, "no generation probe for a dead pid"

    # An unknown installed version earns no verdict.
    facts["version"] = ""
    facts["stale"] = None
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: True)
    unknown = rr.running_receiver(tmp_path)
    assert unknown is not None and unknown["restart_needed"] is False


def test_a_reused_pid_is_not_a_running_receiver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator import procstate
    from local_operator.team_bridge.connector import reap

    monkeypatch.setattr(procstate, "pid_liveness", lambda pid: True)
    monkeypatch.setattr(reap, "_is_posix", lambda: True)
    argv: dict[str, str] = {"value": "/usr/bin/vim notes"}
    monkeypatch.setattr(
        reap,
        "_read_process",
        lambda pid: reap._Process(pid=pid, uid=0, start="x", cpu_s=0.0, argv=argv["value"]),
    )
    assert rr._receiver_alive(os.getpid(), tmp_path) is False

    argv["value"] = f"/x/python -m local_operator.team_bridge.connector --config-dir {tmp_path}"
    assert rr._receiver_alive(os.getpid(), tmp_path) is True
    assert rr._receiver_alive(0, tmp_path) is False

    # Review m4(b): a receiver serving ANOTHER config dir is not this record's.
    other = tmp_path / "other"
    argv["value"] = f"/x/python -m local_operator.team_bridge.connector --config-dir {other}"
    assert rr._receiver_alive(os.getpid(), tmp_path) is False

    # A config path with a space cannot be split back out of ``ps``: unsure.
    spaced = tmp_path / "John Smith"
    argv["value"] = "/x/python -m local_operator.team_bridge.connector --config-dir /else/x"
    assert rr._receiver_alive(os.getpid(), spaced) is None
    argv["value"] = f"/x/python -m local_operator.team_bridge.connector --config-dir {spaced}"
    assert rr._receiver_alive(os.getpid(), spaced) is True

    # Review m4(a): no ``ps`` (Windows) or no answer is unproven, never True.
    monkeypatch.setattr(reap, "_read_process", lambda pid: None)
    assert rr._receiver_alive(os.getpid(), tmp_path) is None
    monkeypatch.setattr(reap, "_is_posix", lambda: False)
    assert rr._receiver_alive(os.getpid(), tmp_path) is None


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
    from local_operator.team_bridge.connector import install

    # Never ask the real service manager: the service state is the test input.
    monkeypatch.setattr(install, "connector_state", lambda root: install.STATE_INSTALLED)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    assert "Running receiver: none recorded" in capsys.readouterr().out
    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is None
    assert payload["restart_needed"] is False and payload["restart_reason"] is None

    facts["generation"] = "20261101T000000Z-0.102.4"
    rr.record_start(tmp_path, started_at=1_700_000_000.0)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: True)
    facts["version"] = "0.103.0"

    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    out = capsys.readouterr().out
    assert f"Running receiver: pid {os.getpid()}, version 0.102.4, started " in out
    assert (
        "Running build: 20261101T000000Z-0.102.4; "
        "consent versions it accepts: bridge_consent_v2, bridge_consent_v3" in out
    )
    assert (
        "Restart needed: the receiver is running a different version from the installed "
        "one (0.103.0). Restart it to load the installed version." in out
    )

    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    running = payload["running"]
    assert running["pid"] == os.getpid() and running["restart_needed"] is True
    assert payload["restart_needed"] is True and payload["restart_reason"] == "version_mismatch"

    # Review C1/E1: same version, replaced build — never "a different version".
    facts["version"] = "0.102.4"
    facts["stale"] = True
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    out = capsys.readouterr().out
    assert (
        "Restart needed: the receiver is running code from before the last update "
        "(installed: 0.102.4). Restart it to load the installed code." in out
    )
    assert "different version" not in out
    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["restart_reason"] == "replaced_build"

    # Unproven liveness prints no verdict.
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: None)
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    assert "Restart needed" not in capsys.readouterr().out
    facts["version"] = "0.103.0"
    facts["stale"] = None

    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: False)
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


def test_status_says_restart_needed_for_a_loaded_receiver_with_no_record(
    tmp_path: Path,
    facts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review M1/C2: a receiver started by an older version records nothing.

    The service manager reports it loaded; without a record ``status`` must say
    a restart is needed (``no_record``), not "none recorded".
    """

    from local_operator.team_bridge.connector import cli as tb_cli
    from local_operator.team_bridge.connector import install

    monkeypatch.setattr(install, "connector_state", lambda root: install.STATE_LOADED)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    out = capsys.readouterr().out
    assert (
        "Running receiver: no details recorded. Restart needed: a receiver started by an "
        "older version records none; restart it to see them." in out
    )
    assert "Running receiver: none recorded" not in out
    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is None
    assert payload["restart_needed"] is True and payload["restart_reason"] == "no_record"

    # A record naming an exited pid while launchd holds a receiver: the loaded
    # one never recorded itself either.
    rr.record_start(tmp_path, started_at=1.0)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: False)
    assert tb_cli.main(_status(["team-bridge", "receiver", "status", "--json"])) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"]["alive"] is False
    assert payload["restart_reason"] == "no_record"

    # A live recorded receiver on the installed code: no verdict.
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: True)
    assert tb_cli.main(_status(["team-bridge", "receiver", "status"])) == 0
    out = capsys.readouterr().out
    assert "Restart needed" not in out and "no details recorded" not in out


_WINNER = 424_242


def _held_by_winner(root: Path) -> None:
    rr.record_path(root).parent.mkdir(parents=True, exist_ok=True)
    rr.record_path(root).write_text(
        json.dumps({"schema": 1, "pid": _WINNER, "version": "0.102.4", "singleton": "held"}),
        encoding="utf-8",
    )


def test_a_start_racing_the_winner_does_not_overwrite_it(
    tmp_path: Path, facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m1: the winner writes ``held`` while the loser forks ``ps`` for its facts.

    The facts are now computed BEFORE the read, so the loser sees the winner.
    """

    real_facts = rr._own_facts

    def facts_while_winner_takes_the_lock(**kw: Any) -> dict[str, Any]:
        _held_by_winner(tmp_path)  # the interleaving: W writes mid-``_own_facts``
        return real_facts(**kw)

    monkeypatch.setattr(rr, "_own_facts", facts_while_winner_takes_the_lock)
    monkeypatch.setattr(rr, "_receiver_alive", lambda pid, root: pid == _WINNER)
    rr.record_start(tmp_path, started_at=1.0)
    assert _record(tmp_path)["pid"] == _WINNER, "the loser overwrote the winner's record"
    rr.clear_own_record(tmp_path)
    assert _record(tmp_path)["pid"] == _WINNER


def test_the_holder_rewrites_a_missing_or_foreign_record(
    tmp_path: Path, facts: dict[str, Any]
) -> None:
    """Review m1: whatever window remains, the holder re-asserts its record."""

    held = rr.record_singleton_held(tmp_path, started_at=1.0)
    assert held is not None
    assert rr.ensure_own_record(tmp_path, held) is False, "ours already: no rewrite"

    # The residual interleaving: a loser overwrote us, then removed its own.
    _held_by_winner(tmp_path)
    assert rr.ensure_own_record(tmp_path, held) is True
    assert _record(tmp_path)["pid"] == os.getpid()
    assert _record(tmp_path)["singleton"] == rr.SINGLETON_HELD

    rr.record_path(tmp_path).unlink()
    assert rr.ensure_own_record(tmp_path, held) is True
    assert _record(tmp_path)["pid"] == os.getpid()

    # The background keeper does the same on its own period.
    rr.record_path(tmp_path).unlink()

    async def run_keeper() -> None:
        task = asyncio.create_task(rr.keep_own_record(tmp_path, held, interval=0.01))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if rr.record_path(tmp_path).exists():
                break
        task.cancel()

    asyncio.run(run_keeper())
    assert _record(tmp_path)["pid"] == os.getpid()


def _wire_owned_receiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lock_busy: bool,
    reap_kind: str = "refused",
) -> tuple[Any, list[str]]:
    """``_main_async`` with an owner present and every collaborator stubbed."""

    from local_operator.providers import auth_store as auth_store_module
    from local_operator.team_bridge import client as client_module
    from local_operator.team_bridge import store as store_module
    from local_operator.team_bridge.connector import __main__ as connector_main
    from local_operator.team_bridge.connector import lock as lock_module
    from local_operator.team_bridge.connector import loop as loop_module
    from local_operator.team_bridge.connector import reap

    calls: list[str] = []
    monkeypatch.setattr(rr, "record_start", lambda root, **kw: calls.append("start"))

    def held(root: Any, **kw: Any) -> dict[str, Any]:
        calls.append("held")
        return {"pid": os.getpid()}

    monkeypatch.setattr(rr, "record_singleton_held", held)
    monkeypatch.setattr(rr, "clear_own_record", lambda root: calls.append("clear"))

    async def keeper(root: Any, facts: Any, **kw: Any) -> None:
        calls.append("keeper")
        await asyncio.Event().wait()

    monkeypatch.setattr(rr, "keep_own_record", keeper)
    monkeypatch.setattr(connector_main, "_configure_logging", lambda root: None)
    monkeypatch.setattr(connector_main, "_activate_operator_locale", lambda root: None)

    class FakeStore:
        def __init__(self, root: Any) -> None:
            pass

        def active_owner(self) -> Any:
            return argparse.Namespace(credential_id="cred")

    class FakeLock:
        def __init__(self, root: Any) -> None:
            pass

        def acquire(self) -> None:
            if lock_busy:
                raise lock_module.ConnectorLockBusy("held by another receiver")

        def release(self) -> None:
            calls.append("release")

    async def refused(root: Any) -> Any:
        return argparse.Namespace(kind=reap_kind)

    async def serve_once(owner: Any, **kw: Any) -> None:
        await asyncio.sleep(0)  # let the keeper task start
        calls.append("serve")

    class FakeAuthStore:
        def __init__(self, **kw: Any) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(store_module, "Store", FakeStore)
    monkeypatch.setattr(lock_module, "ConnectorLock", FakeLock)
    monkeypatch.setattr(reap, "reap_stale_holder", refused)
    monkeypatch.setattr(loop_module, "run_forever", serve_once)
    monkeypatch.setattr(auth_store_module, "AuthStore", FakeAuthStore)
    monkeypatch.setattr(client_module, "TeamBridgeClient", lambda *a, **kw: object())
    args = argparse.Namespace(
        config_dir=str(tmp_path),
        base_url=None,
        credential_id=None,
        account_id=None,
        tenant_id=None,
        endpoint_id=None,
    )
    return connector_main._main_async(args, None), calls


def test_the_holder_records_held_keeps_it_and_clears_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m2: fails if the post-lock ``held`` write or the ``finally`` clear is removed."""

    coroutine, calls = _wire_owned_receiver(tmp_path, monkeypatch, lock_busy=False)
    assert asyncio.run(coroutine) == 0
    assert calls == ["start", "held", "keeper", "serve", "clear", "release"]


def test_a_receiver_refused_the_singleton_clears_its_record_and_exits_75(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m2: fails if either loser clear before ``return 75`` is removed."""

    coroutine, calls = _wire_owned_receiver(tmp_path, monkeypatch, lock_busy=True)
    assert asyncio.run(coroutine) == 75
    assert calls == ["start", "clear"]

    # Reaped, but the lock was taken again in the window.
    coroutine, calls = _wire_owned_receiver(
        tmp_path, monkeypatch, lock_busy=True, reap_kind="reaped"
    )
    assert asyncio.run(coroutine) == 75
    assert calls == ["start", "clear"]
