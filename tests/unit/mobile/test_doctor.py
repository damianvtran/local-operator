"""`lop mobile doctor` and `GET /v1/desktop/mobile/doctor`: read-only, redacted, calm.

The doctor exists because a teammate's phone said "Asleep / Not answering"
while his Mac looked healthy, and the shell diagnostics that would have found
out why can neither be run by him nor relayed over Team Bridge. These tests pin
what it may say (no secret, no home path, no email), what it concludes for each
daemon state, and the descriptor-limit raise the daemon now performs at start.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.mobile import doctor, fdlimit
from local_operator.mobile import install as mobile_install

TOKEN = "desktop-doctor-test-token"
SECRETS = {
    "password": "pw-SEEDED-portal-0f9e",
    "access": "acc-SEEDED-token-77aa",
    "refresh": "ref-SEEDED-token-88bb",
    "api_key": "sk-SEEDED-config-key-1234",
    "email": "seeded.person@example.org",
    "cookie": "lop_mobile=SEEDED-cookie-value",
}


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An isolated machine: its own config/home, no daemon, no network."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOP_TUNNEL_NO_REARM", "1")
    monkeypatch.setattr(mobile_install, "registration_present", lambda: False)
    monkeypatch.setattr(mobile_install, "health", lambda port, timeout=3.0: None)
    monkeypatch.setattr(mobile_install, "gate_closed", lambda port, timeout=3.0: True)
    monkeypatch.setattr(mobile_install, "_supervised_pid", lambda: None)
    monkeypatch.setattr(mobile_install, "_listening_pids", lambda port: set())
    (tmp_path / ".local-operator" / "logs").mkdir(parents=True)
    return tmp_path


def _run(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(doctor.collect(**kwargs))


def _by_id(report: dict[str, Any]) -> dict[str, Any]:
    return {check["id"]: check for check in report["checks"]}


def _seed_everything(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Secrets in every place the doctor reads near: env, config, store, logs."""
    from local_operator.providers.auth_store import AuthStore
    from local_operator.tunnels import api, config

    config_root = home / ".local-operator"
    monkeypatch.setenv("LOP_MOBILE_PASSWORD", SECRETS["password"])
    monkeypatch.setenv("RADIENT_API_KEY", SECRETS["api_key"])
    (config_root / "config.yml").write_text(f"values:\n  api_key: {SECRETS['api_key']}\n")
    with closing(AuthStore()) as store:
        row = store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": SECRETS["email"],
                "email": SECRETS["email"],
                "api_endpoint": api.TUNNEL_HUB,
                "access": SECRETS["access"],
                "refresh": SECRETS["refresh"],
                "expires": int(time.time() * 1000) + 3_600_000,
            },
        )
    config.save(
        {
            "tunnel_id": "tun-1",
            "credential_id": row.id,
            "gateway_port": 4100,
            "mobile_password": SECRETS["password"],
            "record": {
                "id": "tun-1",
                "status": "active",
                "device_id": "dev-1",
                "name": SECRETS["email"],
                "owner_account_id": SECRETS["email"],
                "harnesses": [
                    {"id": "local-operator", "enabled": True, "hostname": "abc.example.test"}
                ],
            },
        }
    )
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    secret_lines = " ".join(SECRETS.values()) + f" {home}/private/notes.txt"
    (config_root / "logs" / "mobile.log").write_text(
        f"{stamp},001 [ERROR] asyncio: socket.accept() out of system resource {secret_lines}\n"
        "OSError: [Errno 24] Too many open files\n"
        f"{stamp},002 [ERROR] attention: unable to open database file {secret_lines}\n"
    )


def test_nothing_secret_or_personal_leaves_the_doctor(rig, monkeypatch):
    """REQUIRED: seeded secrets in env, config, store and logs never appear."""
    from local_operator.tunnels.api import RadientTunnels

    _seed_everything(rig, monkeypatch)
    monkeypatch.setattr(mobile_install, "health", lambda port, timeout=3.0: {"ok": True})
    monkeypatch.setattr(mobile_install, "_listening_pids", lambda port: {"999999"})

    async def listing(self, method, path="", **kwargs):
        assert method == "GET", "the doctor only ever reads the tunnel list"
        return [{"id": "tun-1", "device_id": "dev-1", "name": SECRETS["email"]}]

    monkeypatch.setattr(RadientTunnels, "request", listing)
    report = _run(desktop_app="1.2.3")
    rendered = json.dumps(report) + doctor.text(report)
    for name, value in SECRETS.items():
        assert value not in rendered, f"{name} leaked into the doctor output"
    assert "SEEDED" not in rendered
    assert str(rig) not in rendered, "the home path must be written ~"
    checks = _by_id(report)
    # The facts still arrive: counts, not text.
    assert checks["logs"]["data"]["emfile"] == 1  # the traceback line, once
    assert checks["logs"]["data"]["sqlite_unable_to_open"] == 1
    assert checks["login"]["data"] == {"signed_in": True}
    assert checks["tunnel_duplicates"]["data"]["count"] == 1


def test_a_never_installed_daemon_is_not_running_not_an_open_gate(rig):
    checks = _by_id(_run(remote=False))
    daemon = checks["daemon"]
    assert daemon["status"] == "warn"
    assert daemon["data"]["state"] == "not running"
    assert daemon["data"]["gate"] == "not running"
    assert "OPEN" not in json.dumps(daemon)
    assert checks["files"]["status"] == "pass"


def test_an_installed_daemon_that_is_down_fails(rig, monkeypatch):
    monkeypatch.setattr(mobile_install, "registration_present", lambda: True)
    daemon = _by_id(_run(remote=False))["daemon"]
    assert daemon["status"] == "fail" and daemon["data"]["state"] == "not running"


def test_a_process_holding_the_port_without_answering_is_not_answering(rig, monkeypatch):
    """The reported symptom: kernel accepts, the exhausted event loop never answers."""
    monkeypatch.setattr(mobile_install, "_listening_pids", lambda port: {"4242"})
    monkeypatch.setattr(
        doctor, "open_files", lambda pid: {"open": 256, "soft": 256, "hard": 256, "source": "proc"}
    )
    checks = _by_id(_run(remote=False))
    assert checks["daemon"]["status"] == "fail"
    assert checks["daemon"]["data"]["state"] == "not answering"
    assert checks["files"]["status"] == "fail", "at the limit is a failure"


@pytest.mark.parametrize(
    "used,soft,status", [(10, 4096, "pass"), (210, 256, "warn"), (256, 256, "fail")]
)
def test_file_headroom_thresholds(rig, monkeypatch, used, soft, status):
    monkeypatch.setattr(mobile_install, "health", lambda port, timeout=3.0: {"ok": True})
    monkeypatch.setattr(
        doctor, "open_files", lambda pid: {"open": used, "soft": soft, "hard": None, "source": "x"}
    )
    assert _by_id(_run(remote=False))["files"]["status"] == status


def test_only_the_last_day_of_logs_counts(rig):
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 3 * 86400))
    new = time.strftime("%Y-%m-%d %H:%M:%S")
    (rig / ".local-operator" / "logs" / "mobile.log").write_text(
        f"{old},0 [ERROR] x\nOSError: [Errno 24] Too many open files\n"
        f"{new},0 [ERROR] y\nOSError: [Errno 24] Too many open files\n"
        "sqlite3.OperationalError: unable to open database file\n"
    )
    logs = doctor.check_logs()
    assert logs["data"]["emfile"] == 1
    assert logs["data"]["sqlite_unable_to_open"] == 1
    assert logs["status"] == "warn"


def test_duplicate_tunnels_for_this_device_fail(rig, monkeypatch):
    from local_operator.tunnels.api import RadientTunnels

    _seed_everything(rig, monkeypatch)

    async def listing(self, method, path="", **kwargs):
        return {
            "tunnels": [
                {"id": "tun-1", "device_id": "dev-1"},
                {"id": "tun-0", "device_id": "dev-1"},
                {"id": "other", "device_id": "dev-2"},
            ]
        }

    monkeypatch.setattr(RadientTunnels, "request", listing)
    check = _by_id(_run())["tunnel_duplicates"]
    assert check["status"] == "fail" and check["data"]["count"] == 2


def test_an_unreadable_tunnel_list_is_unchecked_not_a_guess(rig, monkeypatch):
    from local_operator.tunnels.api import RadientTunnels

    _seed_everything(rig, monkeypatch)

    async def refused(self, method, path="", **kwargs):
        raise ValueError("Radient tunnel request failed (HTTP 503).")

    monkeypatch.setattr(RadientTunnels, "request", refused)
    check = _by_id(_run())["tunnel_duplicates"]
    assert check["status"] == "warn" and check["data"] == {"checked": False}


def test_a_hostile_desktop_version_header_is_dropped(rig):
    versions = _by_id(_run(remote=False, desktop_app="1.0<script>"))["versions"]
    assert versions["data"]["desktop_app"] == "unknown"


def test_the_cli_exits_zero_and_prints_json(rig, capsys):
    from local_operator.cli import mobile_command

    code = mobile_command(argparse.Namespace(mobile_command="doctor", json=True))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == doctor.SCHEMA and payload["checks"]


def test_status_says_not_running_and_exits_zero_when_never_installed(rig, capsys, monkeypatch):
    from local_operator.cli import mobile_command

    monkeypatch.setattr(mobile_install, "gate_closed", lambda port, timeout=3.0: False)
    code = mobile_command(argparse.Namespace(mobile_command="status"))
    out = capsys.readouterr().out
    assert "auth gate:    not running" in out and "OPEN" not in out
    assert code == 0


# --- the descriptor limit -----------------------------------------------------------


class _FakeResource:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft: int, hard: int) -> None:
        self.limits = (soft, hard)

    def getrlimit(self, which):
        return self.limits

    def setrlimit(self, which, value):
        self.limits = value


@pytest.mark.parametrize(
    "soft,hard,expected",
    [
        (256, 10240, 4096),  # launchd's default: raised to the cap
        (256, -1, 4096),  # unlimited hard: the cap, never "infinity"
        (64, 64, 64),  # a hard limit below the cap is the ceiling
        (8192, 65535, 8192),  # never lowered
    ],
)
def test_the_daemon_raises_its_soft_limit_but_never_lowers_it(monkeypatch, soft, hard, expected):
    import sys

    fake = _FakeResource(soft, hard)
    monkeypatch.setitem(sys.modules, "resource", fake)
    result = fdlimit.raise_soft_limit()
    assert result["soft"] == expected and fake.limits[0] == expected
    assert fake.limits[1] == hard, "the hard limit is never touched"


def test_a_stale_limits_record_is_not_reported(rig):
    fdlimit.record({"before": 256, "soft": 4096, "hard": 10240}, pid=111)
    kept = fdlimit.recorded(111)
    assert kept is not None and kept["soft"] == 4096
    assert fdlimit.recorded(222) is None


# --- the desktop route --------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_route_needs_the_desktop_bearer_and_serves_the_report(rig, monkeypatch):
    from local_operator.server.features import static_feature_flags
    from local_operator.server.routes import desktop_mobile_doctor

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    app = FastAPI()
    app.include_router(desktop_mobile_doctor.router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as c:
        refused = await c.get("/v1/desktop/mobile/doctor")
        wrong = await c.get("/v1/desktop/mobile/doctor", headers={"Authorization": "Bearer nope"})
        served = await c.get(
            "/v1/desktop/mobile/doctor",
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "X-Local-Operator-Desktop-Version": "1.42.0",
            },
        )
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()
    assert refused.status_code == 401 and wrong.status_code == 401
    assert served.status_code == 200
    result = served.json()["result"]
    assert result["schema"] == doctor.SCHEMA
    assert _by_id(result)["versions"]["data"]["desktop_app"] == "1.42.0"
    assert static_feature_flags()["mobile_doctor"] == 1


def test_the_log_scan_adds_back_what_the_collapse_suppressed(rig):
    """A collapsed accept failure is ONE written traceback plus a count; the
    doctor reports every occurrence, not one line a minute."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (rig / ".local-operator" / "logs" / "mobile.log").write_text(
        f"{stamp},001 [ERROR] asyncio: socket.accept() out of system resource\n"
        "socket: <asyncio.TransportSocket fd=6> [41 similar suppressed in the last 60s]\n"
        "Traceback (most recent call last):\n"
        "OSError: [Errno 24] Too many open files\n"
        f"{stamp},002 [INFO] other: a different record [9 similar suppressed in the last 60s]\n"
    )
    logs = _by_id(_run(remote=False))["logs"]
    assert logs["data"]["emfile"] == 42  # the written copy + 41 suppressed; not the other 9
