"""`lop tunnel` and the RUNNING build (field finding 2026-11-18, review round 1).

An upgrade left the tunnel connector and the mobile relay on old code while
`lop tunnel install` answered success. These cells pin the tunnel side of the
fix without touching a real supervisor: the Linux install wiring (R1-1/R1-2),
the mobile relay `_prepare_mobile` restarts on a proven-stale build and what it
tells the operator (R1-3/R1-4), and the `--json` status key (R1-5).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_operator import launchd, services, supervisors, update
from local_operator.tunnels import cli, config
from local_operator.tunnels import install as tunnel_install
from local_operator.tunnels import report, request_log

# --------------------------------------------------------------------- Linux install


def _linux_tunnel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, check: launchd.InstallBuildCheck
) -> list[dict[str, object]]:
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SYSTEMCTL)
    monkeypatch.setattr(tunnel_install, "service_path", lambda: tmp_path / "lop-tunnel.service")
    monkeypatch.setattr(tunnel_install, "render_systemd", lambda: "[Unit]\n")
    monkeypatch.setattr(tunnel_install, "_run", lambda cmd, **kw: None)
    asked: list[dict[str, object]] = []

    def build_line(**kwargs: object) -> launchd.InstallBuildCheck:
        asked.append(kwargs)
        return check

    monkeypatch.setattr(update, "systemd_install_build_line", build_line)
    return asked


def test_linux_tunnel_install_names_an_unknown_build_and_does_not_bounce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    check = launchd.InstallBuildCheck(kind="unconfirmed", recovery="lop tunnel restart")
    asked = _linux_tunnel(monkeypatch, tmp_path, check)

    note = tunnel_install.install()

    assert note == check.step()
    assert "lop tunnel restart" in note
    assert asked == [
        {
            "name": "tunnel",
            "unit": "lop-tunnel.service",
            "recovery": "lop tunnel restart",
            "restart_when_unproven": False,
        }
    ]


def test_linux_tunnel_install_fails_on_a_failed_build_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    check = launchd.InstallBuildCheck(kind="failed", recovery="lop tunnel restart")
    _linux_tunnel(monkeypatch, tmp_path, check)

    with pytest.raises(ValueError, match="lop tunnel restart"):
        tunnel_install.install()


# --------------------------------------------------------------------- mobile relay


def _mobile_rig(
    monkeypatch: pytest.MonkeyPatch,
    *,
    serving: bool,
    pid: int | None,
    verdict: bool | None,
    result: dict[str, object],
) -> list[int]:
    from local_operator.mobile import install as mobile_install

    monkeypatch.setattr(mobile_install, "health", lambda port: serving)
    monkeypatch.setattr(mobile_install, "gate_closed", lambda port: serving)
    monkeypatch.setattr(mobile_install, "_supervised_pid", lambda: pid)
    monkeypatch.setattr(update, "generation_staleness_of_process", lambda p: verdict)
    installs: list[int] = []

    def fake_install(port: int) -> dict[str, object]:
        installs.append(port)
        return result

    monkeypatch.setattr(mobile_install, "install", fake_install)
    return installs


def _value() -> dict[str, Any]:
    return {"record": {"harnesses": [{"id": "local-operator", "enabled": True, "port": 4111}]}}


@pytest.mark.parametrize(
    ("pid", "verdict", "stale"),
    [(None, True, False), (77, None, False), (77, False, False), (77, True, True)],
)
def test_only_a_proven_stale_mobile_build_counts_as_stale(
    monkeypatch: pytest.MonkeyPatch, pid: int | None, verdict: bool | None, stale: bool
) -> None:
    _mobile_rig(monkeypatch, serving=True, pid=pid, verdict=verdict, result={})
    assert cli._mobile_build_is_stale() is stale


def test_a_proven_stale_mobile_relay_is_reinstalled_and_the_step_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installs = _mobile_rig(
        monkeypatch,
        serving=True,
        pid=77,
        verdict=True,
        result={"ok": True, "steps": ["a", "RESTARTED"], "build": "RESTARTED"},
    )

    assert cli._prepare_mobile(_value()) == "RESTARTED"
    assert installs == [4111]


@pytest.mark.parametrize("verdict", [None, False])
def test_an_unproven_or_current_mobile_relay_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, verdict: bool | None
) -> None:
    installs = _mobile_rig(monkeypatch, serving=True, pid=77, verdict=verdict, result={"ok": True})

    assert cli._prepare_mobile(_value()) == ""
    assert installs == []


def test_a_refused_mobile_build_restart_names_lop_mobile_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = launchd.InstallBuildCheck(kind="failed", recovery="lop mobile restart").step()
    _mobile_rig(
        monkeypatch,
        serving=True,
        pid=77,
        verdict=True,
        result={"ok": False, "steps": [], "error": failed},
    )

    with pytest.raises(ValueError) as raised:
        cli._prepare_mobile(_value())
    assert str(raised.value) == failed
    assert "lop mobile restart" in str(raised.value)


def test_a_down_mobile_relay_keeps_the_generic_install_advice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mobile_rig(
        monkeypatch,
        serving=False,
        pid=None,
        verdict=None,
        result={"ok": False, "steps": [], "error": "daemon did not come up"},
    )

    with pytest.raises(ValueError, match="Mobile relay installation failed"):
        cli._prepare_mobile(_value())


# --------------------------------------------------------------------- dispatch


@pytest.mark.asyncio
async def test_tunnel_install_prints_the_mobile_and_tunnel_build_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "load", lambda: {"credential_id": 7, "record": {}})
    monkeypatch.setattr(config, "save", lambda value: None)
    monkeypatch.setattr(cli, "cloudflared_binary", lambda path: "/trusted/cloudflared")
    monkeypatch.setattr(cli, "_prepare_mobile", lambda value: "MOBILE-STEP")
    monkeypatch.setattr(tunnel_install, "install", lambda: "TUNNEL-STEP")

    receipt = await cli.dispatch(argparse.Namespace(tunnel_command="install"))

    lines = receipt.splitlines()
    assert lines[:2] == ["mobile: MOBILE-STEP", "TUNNEL-STEP"]
    assert lines[2].startswith("Tunnel service started.")


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                services.StaleDaemon(
                    name="tunnel", pid=12, generation="20261113T090000Z-old", reason="r"
                )
            ],
            "20261113T090000Z-old",
        ),
        (
            [
                services.StaleDaemon(
                    name="mobile", pid=13, generation="20261113T090000Z-old", reason="r"
                )
            ],
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_tunnel_status_json_carries_the_proven_stale_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: list[services.StaleDaemon],
    expected: str | None,
) -> None:
    import json

    monkeypatch.setattr(
        config, "load", lambda: {"credential_id": 7, "tunnel_id": "t1", "record": {}}
    )
    monkeypatch.setattr(config, "directory", lambda: tmp_path)
    monkeypatch.setattr(tunnel_install, "rearm_if_parked", lambda **kwargs: None)
    api = AsyncMock()
    api.request.return_value = {"id": "t1"}
    monkeypatch.setattr(cli, "RadientTunnels", lambda *args: api)
    monkeypatch.setattr(report, "connector_state", AsyncMock(return_value={}))
    monkeypatch.setattr(report, "login_verdict", AsyncMock(return_value={}))
    monkeypatch.setattr(report, "payload", lambda *args, **kwargs: {"tunnel": "t1"})
    monkeypatch.setattr(request_log, "read_tail", lambda path: [])
    monkeypatch.setattr(services, "stale_background_daemons", lambda: rows)

    out = await cli.dispatch(argparse.Namespace(tunnel_command="status", json=True))

    assert json.loads(out)["running_build_stale"] == expected
