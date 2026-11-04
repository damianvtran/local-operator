"""Failure paths keep stop-marker cleanup tied to the original synthetic unit root."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.network import cli, relay
from tests.unit.network import test_relay_supervision_linux as linux_fakes

# Register the existing fixture rather than copying its host-global probe
# isolation; the two service test files must keep that safety contract together.
linux_host = linux_fakes.linux_host


@pytest.fixture()
def isolated_service(
    linux_host: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, Path, Path]]:
    forbidden_calls: list[str] = []

    # Reuse the existing no-listener Linux service fake, and fail shut at every
    # actual host-control route. Popen stays a class for import-time typing.
    class ForbiddenPopen(subprocess.Popen):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            forbidden_calls.append("Popen")
            raise AssertionError("service cleanup test attempted to start a host process")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        forbidden_calls.append("host control")
        raise AssertionError("service cleanup test attempted host service/signal control")

    monkeypatch.setattr(subprocess, "Popen", ForbiddenPopen)
    monkeypatch.setattr(os, "kill", forbidden)
    monkeypatch.setattr(os, "killpg", forbidden)
    monkeypatch.setattr(supervisors, "systemctl_user", forbidden)
    monkeypatch.setattr(supervisors, "schtasks", forbidden)
    monkeypatch.setattr(relay, "_launchctl", forbidden)
    # Negative control: the interlock must reject a host dispatch before the
    # individual cell installs its narrowly recorded synthetic systemctl arm.
    with pytest.raises(AssertionError, match="host service/signal control"):
        supervisors.systemctl_user("stop", "synthetic-never-a-live-unit")
    assert forbidden_calls == ["host control"]
    forbidden_calls.clear()
    print("negative control: host dispatch refused before synthetic arm installed")
    config, other = tmp_path / "original-root", tmp_path / "other-root"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    linux_fakes._write_unit_pinning(linux_host, config)
    linux_fakes._pin_supervision(
        monkeypatch, managed=linux_fakes._Script(424242), record=None, answering=True
    )
    yield linux_host, config, other
    # A broad production exception handler must not hide a blocked host attempt.
    assert forbidden_calls == []
    print(f"forbidden_calls after negative control: {forbidden_calls}")


def _cli(*words: str) -> int:
    parser = argparse.ArgumentParser(prog="lop")
    # Locale detection is outside this service-control test and may execute
    # macOS defaults even while the unit fake claims Linux.
    cli.add_parser(parser.add_subparsers(dest="subcommand"), help_locale="en")
    return cli.main(parser.parse_args(["network", *words, "--json"]))


def test_failed_stop_clears_original_root_not_mutated_unit_root(
    isolated_service: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, original, other = isolated_service
    calls: list[tuple[str, ...]] = []

    def fake_systemctl(*args: str, **kwargs: Any) -> Any:
        calls.append(args)
        if args[0] == "stop":
            assert relay.deliberate_stop_requested(original)
            relay.mark_deliberate_stop("other synthetic operation", other)
            linux_fakes._write_unit_pinning(home, other)
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic stop failure")
        return SimpleNamespace(returncode=0, stdout="enabled", stderr="")

    monkeypatch.setattr(supervisors, "systemctl_user", fake_systemctl)
    assert _cli("stop") == 1
    assert ("stop", relay.SYSTEMD_UNIT) in calls
    assert not relay.deliberate_stop_marker_path(original).exists()
    assert relay.deliberate_stop_requested(other), "another root's marker must survive"
    print(f"failed stop: argv={calls}; original marker removed; other marker preserved")


def test_throwing_stop_arm_clears_original_marker_and_preserves_exception(
    isolated_service: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, original, other = isolated_service
    failure = subprocess.TimeoutExpired(["synthetic-systemctl", "stop"], 30)

    def fake_systemctl(*args: str, **kwargs: Any) -> Any:
        if args[0] == "stop":
            assert relay.deliberate_stop_requested(original)
            relay.mark_deliberate_stop("other synthetic operation", other)
            linux_fakes._write_unit_pinning(home, other)
            raise failure
        return SimpleNamespace(returncode=0, stdout="enabled", stderr="")

    monkeypatch.setattr(supervisors, "systemctl_user", fake_systemctl)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        _cli("stop")
    assert raised.value is failure
    assert not relay.deliberate_stop_marker_path(original).exists()
    assert relay.deliberate_stop_requested(other)
    print(
        "throwing arm: original TimeoutExpired propagated; original marker removed; other preserved"
    )


def test_failed_disable_does_not_remove_unit_or_claim_complete(
    isolated_service: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home, original, other = isolated_service
    unit = linux_fakes._unit(home)
    calls: list[tuple[str, ...]] = []

    def fake_systemctl(*args: str, **kwargs: Any) -> Any:
        calls.append(args)
        if args[0] == "disable":
            assert relay.deliberate_stop_requested(original)
            relay.mark_deliberate_stop("other synthetic operation", other)
            linux_fakes._write_unit_pinning(home, other)
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic disable failure")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(supervisors, "systemctl_user", fake_systemctl)
    assert _cli("uninstall") == 1
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "ok": False,
        "code": "service_failed",
        "message": "synthetic disable failure",
    }
    assert unit.exists(), "do not remove the unit while its manager refused disable"
    assert calls == [("disable", "--now", relay.SYSTEMD_UNIT)]
    assert not relay.deliberate_stop_marker_path(original).exists()
    assert relay.deliberate_stop_requested(other)
    print(
        f"failed uninstall: argv={calls}; rc=1; unit kept; original marker removed; other preserved"
    )


@pytest.mark.parametrize("action", ["stop", "uninstall"])
def test_successful_service_action_retains_marker_for_relay_exit(
    isolated_service: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    home, original, _ = isolated_service
    calls: list[tuple[str, ...]] = []
    linux_fakes._pin_supervision(
        monkeypatch, managed=linux_fakes._Script(424242, None), record=None, answering=False
    )

    def fake_systemctl(*args: str, **kwargs: Any) -> Any:
        calls.append(args)
        assert relay.deliberate_stop_requested(original)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(supervisors, "systemctl_user", fake_systemctl)
    assert _cli(action) == 0
    assert relay.deliberate_stop_requested(original), "the synthetic relay has not consumed it"
    if action == "uninstall":
        assert calls == [("disable", "--now", relay.SYSTEMD_UNIT), ("daemon-reload",)]
        assert not linux_fakes._unit(home).exists()
    else:
        assert calls == [("stop", relay.SYSTEMD_UNIT)]
    print(f"successful {action}: argv={calls}; rc=0; marker retained for relay exit")
