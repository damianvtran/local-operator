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


def test_the_capability_is_advertised_to_the_desktop_only(monkeypatch):
    """The relay republishes the feature map minus DESKTOP_ONLY_FEATURES; it
    serves no doctor route, so it must not advertise one (mirrors the
    `mobile_password_read` guard in test_desktop_mobile_password.py)."""
    from local_operator.mobile import daemon
    from local_operator.server.features import feature_flags

    assert feature_flags()["mobile_doctor"] == 1
    # Drop the process memo so the relay re-reads the map under test.
    monkeypatch.setattr(daemon, "_FEATURE_FLAGS_MEMO", None)
    assert "mobile_doctor" not in daemon._feature_flags()


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


#: `mobile/accept_backoff.py`'s recovery line, copied verbatim from its format
#: string (#3971's wording) so a change there is a visible change here.
_RECOVERY_FORMAT = (
    "accepting connections again after %.1f s paused for lack of system "
    "resources (EMFILE/ENFILE/ENOBUFS/ENOMEM) (%d retries)"
)


def _recovery_line(stamp: str) -> str:
    message = _RECOVERY_FORMAT % (12.4, 5)
    return f"{stamp},003 [WARNING] local_operator.mobile.accept_backoff: {message}\n"


def test_the_backoff_recovery_line_is_not_an_emfile_occurrence(rig):
    """R5-1: the recovery line lists every errno the backoff handles, so after
    an ENOBUFS episode it must not read as "too many open files"."""
    import inspect

    from local_operator.mobile import accept_backoff

    # The copy above is still the emitter's: both literal halves, as written there.
    source = inspect.getsource(accept_backoff)
    assert '"accepting connections again after %.1f s paused for lack of system "' in source
    assert '"resources (EMFILE/ENFILE/ENOBUFS/ENOMEM) (%d retries)"' in source
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (rig / ".local-operator" / "logs" / "mobile.log").write_text(
        f"{stamp},001 [ERROR] asyncio: socket.accept() out of system resource\n"
        "socket: <asyncio.TransportSocket fd=6>\n"
        "OSError: [Errno 55] No buffer space available\n" + _recovery_line(stamp)
    )
    logs = doctor.check_logs()
    assert logs["data"]["emfile"] == 0
    assert logs["status"] == "pass"


def test_a_real_emfile_episode_counts_once_beside_its_recovery_line(rig):
    """R5-1: the episode's own error record is the occurrence; its recovery
    line adds nothing, and the check still warns."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (rig / ".local-operator" / "logs" / "mobile.log").write_text(
        f"{stamp},001 [ERROR] asyncio: socket.accept() out of system resource\n"
        "socket: <asyncio.TransportSocket fd=6>\n"
        "OSError: [Errno 24] Too many open files\n" + _recovery_line(stamp)
    )
    logs = doctor.check_logs()
    assert logs["data"]["emfile"] == 1
    assert logs["status"] == "warn"


@pytest.mark.parametrize(
    "line",
    [
        "pausing accept() while EMFILE/ENFILE pressure clears",
        "retrying; last error was not EMFILE",
        "OSError: [Errno 55] No buffer space available (handled like EMFILE)",
    ],
)
def test_a_line_that_only_names_emfile_is_not_an_occurrence(rig, line):
    """R5-1: EMFILE is classified by the errno a line reports ("[Errno 24]" /
    "Too many open files"), not by the word, so ANY prose naming it (not
    just today's recovery line) leaves the check passing."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (rig / ".local-operator" / "logs" / "mobile.log").write_text(
        f"{stamp},001 [WARNING] local_operator.mobile.example: {line}\n"
    )
    logs = doctor.check_logs()
    assert logs["data"]["emfile"] == 0
    assert logs["status"] == "pass"


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
        "watches.near", "watches.at_limit", "watches.instances_near",
        "watches.instances_at_limit", "watches.unknown",
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


# --- file watches (Linux inotify) ---------------------------------------------------


def _fake_proc(
    root: Path,
    holders: dict[int, list[int]],
    limits: tuple[int, int] | None,
    files: dict[int, dict[str, str | bytes]] | None = None,
) -> Path:
    """A ``/proc`` stand-in: ``holders`` maps pid -> watches per inotify fd.

    Each inotify fd is a dangling symlink to ``anon_inode:inotify`` (what
    ``readlink`` answers for the real thing) beside an ``fdinfo`` file carrying
    one ``inotify wd:`` line per watch; one ordinary fd per process proves only
    inotify fds are counted.
    """
    proc = root / "proc"
    for pid, fds in holders.items():
        (proc / str(pid) / "fd").mkdir(parents=True)
        (proc / str(pid) / "fdinfo").mkdir()
        (proc / str(pid) / "fd" / "0").symlink_to("/dev/null")
        for fd, watches in enumerate(fds, start=3):
            (proc / str(pid) / "fd" / str(fd)).symlink_to("anon_inode:inotify")
            lines = "".join(
                f"inotify wd:{n} ino:1 sdev:1 mask:2 ignored_mask:0\n" for n in range(watches)
            )
            (proc / str(pid) / "fdinfo" / str(fd)).write_text(f"pos:\t0\nflags:\t02000000\n{lines}")
    # Per-pid comm/cmdline/cgroup, what the holder's name and restart command
    # are read from.
    for pid, entries in (files or {}).items():
        for entry, content in entries.items():
            target = proc / str(pid) / entry
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content)
    (proc / "self").mkdir(exist_ok=True)  # non-numeric entries are skipped
    limits_dir = root / "limits"
    limits_dir.mkdir()
    if limits is not None:
        (limits_dir / "max_user_watches").write_text(f"{limits[0]}\n")
        (limits_dir / "max_user_instances").write_text(f"{limits[1]}\n")
    return proc


@pytest.mark.parametrize(
    "holders,limits,status",
    [
        ({10: [5, 1], 11: [2]}, (100, 128), "pass"),
        ({10: [85]}, (100, 128), "warn"),  # watches near the budget
        ({10: [1]} | {pid: [0] for pid in range(20, 123)}, (100_000, 128), "warn"),  # instances
        ({10: [99], 11: [1]}, (100, 128), "fail"),  # node2's shape: spent
    ],
)
def test_the_watch_row_sums_this_users_inotify_against_the_limit(
    tmp_path, monkeypatch, holders, limits, status
):
    proc = _fake_proc(tmp_path, holders, limits)
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["id"] == "watches"
    assert check["status"] == status
    data = check["data"]
    assert data["watches"] == sum(sum(fds) for fds in holders.values())
    assert data["instances"] == sum(len(fds) for fds in holders.values())
    assert (data["limit"], data["instance_limit"]) == limits


_SERVE_ARGV = b"/x/bin/python3\0-P\0-m\0local_operator.cli\0serve\0--port\08977\0"
_SERVE_FILES: dict[str, str | bytes] = {
    "comm": "Local Operator\n",
    "cmdline": _SERVE_ARGV,
    "cgroup": "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
    "local-operator-serve.service\n",
}


def test_at_the_watch_limit_names_the_biggest_holder(tmp_path, monkeypatch):
    """node2's shape: ``lop serve`` spent max_user_watches (ENOSPC).

    Design D1/D3: the row names the watch limit only, the program by name, and
    the exact command that restarts it (the serve daemon reloads in place).
    """
    proc = _fake_proc(tmp_path, {3105: [90, 9], 77: [1]}, (100, 128), {3105: _SERVE_FILES})
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["status"] == "fail"
    data = check["data"]
    assert data["top_pid"] == 3105 and data["top_watches"] == 99
    assert (data["bottleneck"], data["top_name"]) == ("watches", "Local Operator")
    assert data["restart_command"] == "lop services restart"
    assert check["summary"] == doctor._t(
        doctor._md.watches_at_limit(
            watches="100",
            limit="100",
            name="Local Operator",
            pid="3105",
            how="command",
            command="lop services restart",
        )
    )
    summary = check["summary"]
    assert "No space left on device" in summary and "Too many open files" not in summary
    assert "fs.inotify.max_user_watches" in summary
    assert "max_user_instances" not in summary
    assert "Local Operator (process 3105)" in summary
    assert "restart it with lop services restart" in summary


def test_at_the_instance_limit_names_the_biggest_instance_holder(tmp_path, monkeypatch):
    """Design D2: when instances are the bottleneck, rank by instances, not watches.

    pid 3105 holds the most watches (one fd, 50 watches) but pid 77 holds the
    most inotify fds; restarting 3105 would not free a single instance.
    """
    holders = {3105: [50], 77: [1] * 6, 78: [1]}
    proc = _fake_proc(tmp_path, holders, (100_000, 8), {77: {"comm": "watchdog\n"}})
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["status"] == "fail"
    data = check["data"]
    assert (data["top_pid"], data["top_watches"]) == (3105, 50)
    assert (data["top_instance_pid"], data["top_instances"]) == (77, 6)
    assert data["bottleneck"] == "instances"
    assert check["summary"] == doctor._t(
        doctor._md.watches_instances_at_limit(
            instances="8", instance_limit="8", name="watchdog", pid="77", how="program", command=""
        )
    )


def test_at_the_instance_limit_the_row_says_too_many_open_files(tmp_path, monkeypatch):
    """Design D1 / review M3: a full max_user_instances is EMFILE.

    inotify_init1 fails with "Too many open files", not "No space left on
    device", and only fs.inotify.max_user_instances raises it.
    """
    cgroup = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/syncthing.service\n"
    files: dict[int, dict[str, str | bytes]] = {
        77: {"comm": "syncthing\n", "cmdline": b"/usr/bin/syncthing\0serve\0", "cgroup": cgroup}
    }
    proc = _fake_proc(tmp_path, {3105: [50], 77: [1] * 6, 78: [1]}, (100_000, 8), files)
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    monkeypatch.setattr(doctor, "_unit_main_pid", lambda unit, *, user: 77)  # its main process
    summary = doctor.check_watches(proc)["summary"]
    assert "Too many open files" in summary and "No space left on device" not in summary
    assert "fs.inotify.max_user_instances" in summary
    assert "max_user_watches" not in summary
    assert "used up its watch handles on this computer (8 of 8)" in summary
    # Design D2: the sentence opens with words, not a lowercase program name.
    assert "Most are held by syncthing (process 77): restart it with" in summary
    assert "restart it with systemctl --user restart syncthing.service" in summary


def test_near_the_instance_limit_warns_with_the_instance_wording(tmp_path, monkeypatch):
    holders = {10: [1, 0]} | {pid: [0] for pid in range(20, 123)}  # 105 of 128 instances
    proc = _fake_proc(tmp_path, holders, (100_000, 128), {10: {"comm": "code\n"}})
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["status"] == "warn"
    assert check["summary"] == doctor._t(
        doctor._md.watches_instances_near(
            instances="105", instance_limit="128", name="code", pid="10", how="program", command=""
        )
    )
    assert "fs.inotify.max_user_instances" in check["summary"]
    assert "Most are held by code (process 10): restart that program" in check["summary"]


def test_ok_row_calls_instances_watch_handles_not_watchers(tmp_path, monkeypatch):
    """Design D5: "file watches" and "watchers" differed by two letters."""
    proc = _fake_proc(tmp_path, {10: [5]}, (100, 128))
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    summary = doctor.check_watches(proc)["summary"]
    assert "1 of 128 watch handles" in summary
    assert "watchers" not in summary


_USER_UNIT = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
_WAKES = _USER_UNIT + "local-operator-wakes.service\n"
_SERVE_UNIT = _USER_UNIT + "local-operator-serve.service\n"


@pytest.mark.parametrize(
    "files,main_pid,command",
    [
        (_SERVE_FILES, 42, "lop services restart"),
        # Review n1: `lop serve` from a terminal keeps the console script's argv.
        (
            {
                "cmdline": b"/home/u/.local/bin/lop\0serve\0--port\08977\0",
                "cgroup": "0::/x.scope\n",
            },
            None,
            "lop services restart",
        ),
        (
            {"cmdline": b"/x/python3\0-m\0local_operator.mobile.service\0--port\04098\0"},
            None,
            "lop mobile restart",
        ),
        ({"cgroup": _USER_UNIT + "x.service\n"}, 42, "systemctl --user restart x.service"),
        (
            {"cgroup": "0::/system.slice/fluent-bit.service\n"},
            42,
            "sudo systemctl restart fluent-bit.service",
        ),
        (
            {"cgroup": "1:name=systemd:/system.slice/old.service\n"},
            42,
            "sudo systemctl restart old.service",
        ),
        # Review M1: a process in a unit's cgroup that is NOT its MainPID is a
        # child; restarting the unit would not replace it.
        ({"cgroup": _USER_UNIT + "x.service\n"}, 7, None),
        # ... and a tsc --watch from an ssh login on a host without logind, which
        # stays in sshd.service (KillMode=process would leave it running).
        (
            {"cmdline": b"node\0/x/tsc\0--watch\0", "cgroup": "0::/system.slice/sshd.service\n"},
            1200,
            None,
        ),
        # MainPID unreadable (no systemctl, no user bus, timeout): fail closed.
        ({"cgroup": "0::/system.slice/fluent-bit.service\n"}, None, None),
        # A dev server an agent started lives in the wakes unit: restarting it
        # would end every session. Never systemctl for a lop unit, even if
        # systemd were to call it the main process.
        ({"cmdline": b"node\0/x/vite\0", "cgroup": _WAKES}, 42, None),
        ({"cmdline": b"node\0/x/vite\0", "cgroup": _WAKES}, 3106, None),
        # A session runtime under the serve unit is not the serve itself.
        (
            {
                "cmdline": b"/x/python3\0-m\0local_operator.session.runtime.process\0",
                "cgroup": _SERVE_UNIT,
            },
            3105,
            None,
        ),
        # The user's own service manager: restarting it ends every user service.
        ({"cgroup": "0::/user.slice/user-1000.slice/user@1000.service/init.scope\n"}, 42, None),
        ({"cgroup": "0::/user.slice/user-1000.slice/user@1000.service\n"}, 42, None),
        ({"cgroup": "0::/user.slice/user-1000.slice/session-3.scope\n"}, 42, None),  # a terminal
        ({}, 42, None),  # exited: nothing readable
    ],
)
def test_the_restart_command_is_named_only_when_it_is_known(
    tmp_path, monkeypatch, files, main_pid, command
):
    """Design D3 / review M1: lop's daemons, and a systemd unit's MAIN process only."""
    asked: list[tuple[str, bool]] = []

    def main(unit: str, *, user: bool) -> int | None:
        asked.append((unit, user))
        return main_pid

    monkeypatch.setattr(doctor, "_unit_main_pid", main)
    proc = _fake_proc(tmp_path, {42: [1]}, None, {42: files})
    assert doctor.restart_command(proc, 42) == command
    if command and command.startswith("systemctl --user"):
        assert asked == [(command.rsplit(" ", 1)[1], True)]
    if command and command.startswith("sudo"):
        assert asked == [(command.rsplit(" ", 1)[1], False)]
    assert not any(unit.startswith("local-operator-") for unit, _ in asked)


class _Ran:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode, self.stdout = returncode, stdout


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (_Ran(0, "4242\n"), 4242),
        (_Ran(0, "0\n"), None),  # unit not running
        (_Ran(1, ""), None),  # no such unit / no user bus
        (_Ran(0, "garbage\n"), None),
        (FileNotFoundError("systemctl"), None),
        (doctor.subprocess.TimeoutExpired("systemctl", 2.0), None),
    ],
)
def test_unit_main_pid_fails_closed(monkeypatch, outcome, expected):
    """Review M1: anything but a clean MainPID answer offers no unit restart."""
    calls: list[tuple[list[str], float]] = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs["timeout"]))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(doctor.subprocess, "run", run)
    assert doctor._unit_main_pid("x.service", user=True) == expected
    assert calls == [
        (["systemctl", "--user", "show", "-p", "MainPID", "--value", "x.service"], 2.0)
    ]


def test_an_unknown_holder_drops_the_holder_clause(tmp_path, monkeypatch):
    """Review m1: no "? (process ?)" when the top holder is unknown.

    A zero max_user_instances has no holder at all; a holder that exited
    before its comm was read has a pid but no name to show.
    """
    proc = _fake_proc(tmp_path, {10: []}, (100, 0))  # no inotify fd at all
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["status"] == "fail"
    assert check["summary"].endswith(
        "“Too many open files”. Ask an administrator to raise fs.inotify.max_user_instances."
    )
    assert "?" not in check["summary"] and "process" not in check["summary"]
    assert (check["data"]["top_name"], check["data"]["restart_command"]) == (None, None)

    exited = _fake_proc(tmp_path / "exited", {10: [99]}, (100, 128))  # no comm file
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "exited" / "limits")
    summary = doctor.check_watches(exited)["summary"]
    assert summary.endswith("Ask an administrator to raise fs.inotify.max_user_watches.")
    assert "?" not in summary and "process" not in summary


def test_the_pass_row_has_the_problem_rows_json_shape(tmp_path, monkeypatch):
    """QA Q2: bottleneck/top_name/restart_command are present (null) on pass too."""
    proc = _fake_proc(tmp_path, {10: [5]}, (100, 128))
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["status"] == "pass"
    data = check["data"]
    assert {k: data[k] for k in ("bottleneck", "top_name", "restart_command")} == {
        "bottleneck": None,
        "top_name": None,
        "restart_command": None,
    }


def test_unreadable_watch_limits_are_unknown_not_a_guess(tmp_path, monkeypatch):
    proc = _fake_proc(tmp_path, {10: [1]}, None)
    monkeypatch.setattr(doctor, "_INOTIFY_LIMITS", tmp_path / "limits")
    check = doctor.check_watches(proc)
    assert check["status"] == "warn"
    assert check["summary"] == doctor._t(doctor._md.watches_unknown())


def test_the_watch_row_is_reported_on_linux_only(rig, monkeypatch):
    monkeypatch.setattr(doctor, "check_watches", lambda: doctor._check("watches", "pass", "ok"))
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    assert "watches" in _by_id(_run(remote=False))
    monkeypatch.setattr(doctor.sys, "platform", "darwin")
    assert "watches" not in _by_id(_run(remote=False))
