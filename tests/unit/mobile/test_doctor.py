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


class _FakeRadient:
    """Every HTTP call the doctor makes, answered in-process.

    Patches `httpx.AsyncClient` with a mock transport, so the doctor's real
    request code runs (URL, bearer, per-request timeout) with no socket. A
    loopback request (the tunnel gateway's health probe) is refused like a
    closed port; everything else is recorded and answered by ``handler``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, handler=None) -> None:
        import httpx

        self.calls: list[httpx.Request] = []
        self.handler = handler or (lambda request: httpx.Response(200, json={"result": []}))
        real = httpx.AsyncClient
        fake = self

        async def route(request: httpx.Request) -> httpx.Response:
            if request.url.host in ("127.0.0.1", "localhost"):
                raise httpx.ConnectError("refused", request=request)
            fake.calls.append(request)
            answer = fake.handler(request)
            if asyncio.iscoroutine(answer):
                answer = await answer
            return answer

        def client(*args: Any, **kwargs: Any) -> Any:
            kwargs["transport"] = httpx.MockTransport(route)
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)


def _tree(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its sha256: the read-only proof."""
    import hashlib

    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _seed_everything(
    home: Path, monkeypatch: pytest.MonkeyPatch, *, expires_in_ms: int = 3_600_000
) -> None:
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
                "expires": int(time.time() * 1000) + expires_in_ms,
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
    import httpx

    _seed_everything(rig, monkeypatch)
    monkeypatch.setattr(mobile_install, "health", lambda port, timeout=3.0: {"ok": True})
    monkeypatch.setattr(mobile_install, "_listening_pids", lambda port: {"999999"})
    radient = _FakeRadient(
        monkeypatch,
        lambda request: httpx.Response(
            200, json={"result": [{"id": "tun-1", "device_id": "dev-1", "name": SECRETS["email"]}]}
        ),
    )
    report = _run(desktop_app="1.2.3")
    assert [r.method for r in radient.calls] == ["GET"], "the doctor only ever reads the list"
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
    # R1-7: the portal hostname is shortened to its first label.
    assert checks["tunnel"]["data"]["hostname"] == "abc\u2026"


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
    import httpx

    _seed_everything(rig, monkeypatch)
    listing = {
        "tunnels": [
            {"id": "tun-1", "device_id": "dev-1"},
            {"id": "tun-0", "device_id": "dev-1"},
            {"id": "other", "device_id": "dev-2"},
        ]
    }
    radient = _FakeRadient(monkeypatch, lambda r: httpx.Response(200, json={"result": listing}))
    check = _by_id(_run())["tunnel_duplicates"]
    (request,) = radient.calls
    assert request.headers["authorization"] == f"Bearer {SECRETS['access']}"
    assert request.url.path == "/v1/tunnels"
    assert check["status"] == "fail" and check["data"]["count"] == 2


def test_an_unreadable_tunnel_list_is_unchecked_not_a_guess(rig, monkeypatch):
    import httpx

    _seed_everything(rig, monkeypatch)
    _FakeRadient(monkeypatch, lambda r: httpx.Response(503, text=SECRETS["access"]))
    check = _by_id(_run())["tunnel_duplicates"]
    assert check["status"] == "warn"
    assert check["data"] == {"checked": False, "reason": "unreachable"}
    assert SECRETS["access"] not in json.dumps(check)


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


# --- round 1 remediation: strictly read-only, bounded, a fix on every problem --------


def test_an_expired_token_is_never_refreshed_and_the_store_is_untouched(rig, monkeypatch):
    """R1-1 + Q1: an expired bearer makes the duplicate check "not checked".
    No refresh POST, no list GET, and not one byte of the config tree moves."""
    from local_operator.tunnels import report

    _seed_everything(rig, monkeypatch, expires_in_ms=-3_600_000)

    async def refreshing_payload(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("local_payload computes the login verdict, which refreshes")

    monkeypatch.setattr(report, "local_payload", refreshing_payload)
    radient = _FakeRadient(monkeypatch)
    before = _tree(rig / ".local-operator")
    check = _by_id(_run())["tunnel_duplicates"]
    assert radient.calls == [], "no refresh POST and no list GET on an expired token"
    assert check["status"] == "warn"
    assert check["data"] == {"checked": False, "reason": "login_stale"}
    assert check["summary"] == doctor._t(doctor._md.tunnel_list_skipped())
    assert _tree(rig / ".local-operator") == before


def test_a_tunnel_with_no_credential_store_creates_no_files(rig, monkeypatch):
    """Q1's repro: a tunnel is configured but there is no auth.db. The doctor
    used to create auth.db and its bootstrap lock; now the tree is unchanged."""
    from local_operator.tunnels import config

    config.save(
        {
            "tunnel_id": "tun-1",
            "credential_id": 7,
            "gateway_port": 4100,
            "record": {"id": "tun-1", "status": "active", "device_id": "dev-1"},
        }
    )
    radient = _FakeRadient(monkeypatch)
    before = _tree(rig)
    report = _run()
    assert _tree(rig) == before, "the doctor created or changed a file"
    assert not (rig / ".local-operator" / "auth.db").exists()
    assert radient.calls == []
    checks = _by_id(report)
    assert checks["tunnel_duplicates"]["data"] == {"checked": False, "reason": "no_login"}


def test_a_hung_radient_is_not_checked_within_the_doctors_bound(rig, monkeypatch):
    """R1-6 + Q2: the request carries the doctor's own timeout (no 30 s client
    default can override it), and the whole check is bounded."""
    _seed_everything(rig, monkeypatch)
    seen: list[dict[str, Any]] = []

    async def hang(request: Any) -> Any:
        seen.append(dict(request.extensions["timeout"]))
        await asyncio.sleep(30)

    monkeypatch.setattr(doctor, "DUPLICATE_CHECK_TIMEOUT_S", 0.3)
    _FakeRadient(monkeypatch, hang)
    started = time.monotonic()
    check = _by_id(_run())["tunnel_duplicates"]
    elapsed = time.monotonic() - started
    assert check["data"] == {"checked": False, "reason": "timeout"}
    assert elapsed < 5, f"the duplicate check took {elapsed:.1f}s"
    assert seen and all(value == 0.3 for value in seen[0].values())


def test_the_log_scan_counts_only_emfile_copies_as_emfile(rig):
    """R1-2: an ENFILE accept record's suppressed count is not "too many open
    files" for this process, so it is not added to the EMFILE total."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (rig / ".local-operator" / "logs" / "mobile.log").write_text(
        f"{stamp},001 [ERROR] asyncio: socket.accept() out of system resource\n"
        "socket: <asyncio.TransportSocket fd=6> [41 similar suppressed in the last 60s]\n"
        "OSError: [Errno 24] Too many open files\n"
        f"{stamp},002 [ERROR] asyncio: socket.accept() out of system resource\n"
        "socket: <asyncio.TransportSocket fd=6> [50 similar suppressed in the last 60s]\n"
        "OSError: [Errno 23] Too many open files in system\n"
    )
    logs = doctor.check_logs()
    assert logs["data"]["emfile"] == 42  # 1 written + 41 suppressed; the ENFILE 50 are not


@pytest.mark.parametrize(
    "connector,stopped,expected,status",
    [
        ({"state": "parked", "reason": "login_required"}, False, "tunnel_login_needed", "fail"),
        ({"state": "parked", "reason": "owner_missing"}, False, "tunnel_setup_again", "fail"),
        (
            {"state": "stopped", "reason": "control_plane_unreachable"},
            False,
            "tunnel_offline",
            "warn",
        ),
        ({"state": "connecting", "reason": ""}, False, "tunnel_connecting", "warn"),
        ({"state": "stopped", "reason": ""}, True, "tunnel_turned_off", "warn"),
        ({"state": "stopped", "reason": ""}, False, "tunnel_not_running", "warn"),
        ({"state": "parked", "reason": "some_future_reason"}, False, "tunnel_not_running", "fail"),
        ({"state": "connected", "reason": ""}, False, "tunnel_ok", "pass"),
    ],
)
def test_a_connector_state_is_a_plain_sentence_with_a_fix(
    rig, monkeypatch, connector, stopped, expected, status
):
    """R1-3 + Q3: no "parked", no snake_case reason, no "(-)" in the summary;
    the code itself stays in `data` for support."""
    from local_operator.tunnels import config, report

    config.save(
        {"tunnel_id": "tun-1", "credential_id": 7, "gateway_port": 4100, "stopped": stopped}
    )

    async def state(value: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return dict(connector)

    monkeypatch.setattr(report, "connector_state", state)
    tunnel = _by_id(_run(remote=False))["tunnel"]
    assert tunnel["status"] == status
    assert tunnel["summary"] == doctor._t(getattr(doctor._md, expected)())
    for code in ("parked", "connector", "(-)", connector["reason"] or "\0"):
        assert code not in tunnel["summary"]
    assert tunnel["data"]["connector_reason"] == connector["reason"]


def test_at_the_file_limit_says_so(rig, monkeypatch):
    """Q4: the fail arm has its own sentence, not "close to its limit"."""
    monkeypatch.setattr(mobile_install, "health", lambda port, timeout=3.0: {"ok": True})
    monkeypatch.setattr(
        doctor, "open_files", lambda pid: {"open": 64, "soft": 64, "hard": 64, "source": "x"}
    )
    files = _by_id(_run(remote=False))["files"]
    assert files["status"] == "fail"
    assert files["summary"] == doctor._t(doctor._md.files_at_limit(open="64", limit="64"))


def test_every_english_problem_sentence_names_a_fix_without_jargon():
    """R1-3: the words a non-technical reader cannot act on are gone, and every
    sentence used on a warn/fail line carries a second sentence: the fix."""
    catalogue = json.loads(
        (
            Path(doctor.__file__).parents[1] / "i18n" / "catalogues" / "en" / "mobile.doctor.json"
        ).read_text()
    )
    for key, text in catalogue.items():
        for jargon in ("connector", "parked", "out-of-files", "out of files"):
            assert jargon not in text.lower(), f"{key} says {jargon!r}"
    problems = [
        "daemon.not_installed", "daemon.not_running", "daemon.not_answering", "daemon.slow",
        "daemon.gate_open", "files.near", "files.at_limit", "files.unknown", "logs.errors",
        "tunnel.not_configured", "tunnel.login_needed", "tunnel.setup_again", "tunnel.offline",
        "tunnel.connecting", "tunnel.turned_off", "tunnel.not_running", "tunnel.duplicates",
        "tunnel.list_unchecked", "tunnel.list_skipped", "login.missing",
    ]  # fmt: skip
    for key in problems:
        text = catalogue[f"mobile.doctor.{key}"]
        assert text.count(". ") >= 1, f"{key} states a problem but names no fix: {text}"


def test_status_says_not_answering_when_a_process_holds_the_port(rig, capsys, monkeypatch):
    """R1-4 + Q5: a held-but-silent port is not "not running"."""
    from local_operator.cli import mobile_command

    monkeypatch.setattr(mobile_install, "_listening_pids", lambda port: {"4242"})
    mobile_command(argparse.Namespace(mobile_command="status"))
    out = capsys.readouterr().out
    assert "auth gate:    not answering" in out
