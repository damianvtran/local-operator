"""Cloud-free supervisor contracts and real loopback refusal/recovery paths.

The host has no Linux systemd daemon. Script tests execute the shipped Python
with a redirected /proc, /etc and /var and a fake manager, not a native-systemd
claim. Transport tests use actual sockets; no AWS clients make remote calls.
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import errno
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from lop_osworld_v2_adapter import guest_disk, guest_supervisor
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.base import (
    GuestServerRecovered,
    guest_deadline_for,
)

from tests.unit.evaluation.adapters.osworld.test_action_wait_order import (
    _adapter,
    _click,
    _params,
)
from tests.unit.evaluation.adapters.osworld.test_aws_provider import (
    _plan,
    _Stubs,
    _task,
)
from tests.unit.evaluation.adapters.osworld.test_guest_loss import (
    _closed_port,
    _NoConsoleEc2,
    _provider_over,
)

from tests.unit.evaluation.adapters.osworld.test_aws_provider import (  # noqa: F401  isort:skip
    _hermetic_aws,
)


def _envelope(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "records": records,
        "captured_exits": len(records),
        "baseline_nrestarts": 0,
        "baseline_active_enter": 100,
        "systemd_nrestarts": len(records),
        "systemd_active_enter": 200 if records else 100,
    }


def _script_guest(
    tmp_path: Path,
    *,
    props: dict[str, str] | None = None,
    effective_props: dict[str, str] | None = None,
    watchdog_props: dict[str, str] | None = None,
    watchdog_stop_effective: bool = True,
    ppid: int = 8000,
) -> tuple[Any, list[Any]]:
    """Run the exact install source; only OS-boundary modules are substituted."""
    proc = tmp_path / "proc/8000"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    (proc / "cgroup").write_text("0::/system.slice/action-server.service\n")
    # The install now anchors on the listening socket, not on the ancestry pid
    # being MainPID: the AMI's real layout has a second :5000 holder beside the
    # unit's MainPID, and that is the case the fixture must reproduce.
    _write_listeners(tmp_path / "proc", {8000: 4242})
    properties = {
        "MainPID": "8000",
        "ExecStart": "/usr/bin/python3 /home/user/server/main.py",
        "StandardError": "journal",
        "Type": "simple",
        "LoadState": "loaded",
        "ControlGroup": "/system.slice/action-server.service",
        "NRestarts": "0",
        "ActiveEnterTimestampMonotonic": "100",
    }
    properties.update(props or {})
    calls: list[Any] = []
    watchdog_properties = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "ControlGroup": "/system.slice/lop-action-watchdog.service",
    }
    watchdog_properties.update(watchdog_props or {})
    watchdog_state = {"stopped": False}

    def check_output(args: Any, **kwargs: Any) -> str:
        calls.append(args)
        if args[1] == "daemon-reload":
            return ""
        if args[1] == "stop" and args[2] == guest_supervisor.WATCHDOG_UNIT:
            if watchdog_stop_effective:
                watchdog_state["stopped"] = True
            return ""
        if args[1] == "kill":
            watchdog_state["stopped"] = True
            return ""
        if args[1] == "show" and args[2] == guest_supervisor.WATCHDOG_UNIT:
            if watchdog_state["stopped"]:
                return "ActiveState=inactive\nControlGroup="
            return "\n".join(k + "=" + v for k, v in watchdog_properties.items())
        result = properties
        if "Restart" in args:
            result = {
                **properties,
                "Restart": "always",
                "KillMode": "process",
                "ExecStopPost": str(tmp_path / "var/lib/lop-action-supervisor/capture.py")
                + " action-server.service",
                "ProtectSystem": "no",
                "StartLimitBurst": "3",
                "StartLimitIntervalUSec": "1h",
                "RestartUSec": "1s",
                "RestartPreventExitStatus": "",
                "ExitType": "main",
                **(effective_props or {}),
            }
        return "\n".join(k + "=" + v for k, v in result.items())

    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(getppid=lambda: ppid, readlink=os.readlink),
        "subprocess": SimpleNamespace(check_output=check_output),
        "sys": SimpleNamespace(
            argv=[
                "-c",
                guest_supervisor.CAPTURE_SOURCE,
                "5000",
                guest_supervisor.WATCHDOG_SOURCE,
                guest_supervisor.RESTART_SOURCE,
            ]
        ),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    def install() -> None:
        exec(
            guest_supervisor.INSTALL_SOURCE,
            {"__builtins__": {**vars(builtins), "__import__": importer}},
        )

    return install, calls


def test_install_is_idempotent_and_cannot_restart_or_kill_desktop(tmp_path: Path) -> None:
    install, calls = _script_guest(tmp_path)
    install()
    install()
    assert sum(args[1] == "daemon-reload" for args in calls) == 1
    assert sum("Restart" in args for args in calls) == 2
    assert all(
        "RestartPreventExitStatus" in args and "ExitType" in args
        for args in calls
        if "Restart" in args
    )
    assert not any(
        args[1] in {"restart", "start", "stop", "kill", "reset-failed"}
        and guest_supervisor.WATCHDOG_UNIT not in args
        for args in calls
    )
    # The ONLY unit the install starts is its own watchdog; the action server is
    # never touched by the install, which is the reviewed invariant.
    assert {args[2] for args in calls if args[1] == "start"} == {guest_supervisor.WATCHDOG_UNIT}
    policy = (
        tmp_path / "etc/systemd/system/action-server.service.d/90-lop-supervisor.conf"
    ).read_text()
    assert "Restart=always\nRestartSec=1\n" in policy
    assert "StartLimitBurst=3" in policy and "StartLimitIntervalSec=3600" in policy
    assert "KillMode=process" in policy
    # The AMI's server takes its bind from this variable (xlang-ai/osworld-server
    # src/server.py hands it to uvicorn); pinning it keeps every systemd restart
    # on the wildcard, the only bind the host's public-address path can reach.
    assert 'Environment="OSWORLD_SERVER_HOST=0.0.0.0"' in policy
    assert (
        "ExecStopPost=+/usr/bin/python3 "
        + str(tmp_path / "var/lib/lop-action-supervisor/capture.py")
        + " action-server.service"
    ) in policy
    assert "ExecStart=" not in policy
    for source in (guest_supervisor.INSTALL_SOURCE, guest_supervisor.CAPTURE_SOURCE):
        assert "/home/" not in source and "/Desktop" not in source and "/task" not in source
        assert "os.kill" not in source and "killpg" not in source


def test_a_refusal_stops_a_watchdog_this_attempt_started_even_if_it_existed(
    tmp_path: Path,
) -> None:
    """``created`` is not the test: we started it, so a refusal must stop it."""
    install, _ = _script_guest(tmp_path)
    install()
    # The same tree, a harness whose post-start verification will refuse: the
    # unit file already exists, but this attempt still started the process.
    shutil.rmtree(tmp_path / "proc")
    install_again, calls = _script_guest(
        tmp_path,
        watchdog_props={"ControlGroup": "/system.slice/action-server.service"},
    )
    with pytest.raises(RuntimeError, match="shares the action-server cgroup"):
        install_again()
    verbs = [
        args[1] for args in calls if len(args) > 2 and args[2] == guest_supervisor.WATCHDOG_UNIT
    ]
    assert "stop" in verbs and "disable" in verbs


def test_a_second_install_after_a_refused_first_succeeds(tmp_path: Path) -> None:
    """A refusal must leave no half-installation that blocks the next attempt."""
    refused, _ = _script_guest(
        tmp_path,
        watchdog_props={"ControlGroup": "/system.slice/action-server.service"},
    )
    with pytest.raises(RuntimeError, match="shares the action-server cgroup"):
        refused()
    root = tmp_path / "var/lib/lop-action-supervisor"
    for name in ("watchdog.py", "restart.py", "capture.py", "action-server.service.json"):
        assert not (root / name).exists(), name
    assert not (tmp_path / "etc/systemd/system/lop-action-watchdog.service").exists()

    shutil.rmtree(tmp_path / "proc")
    retry, calls = _script_guest(tmp_path)
    retry()
    assert (root / "watchdog.py").exists()
    assert (root / "restart.py").exists()
    assert ("systemctl", "start", guest_supervisor.WATCHDOG_UNIT) in calls


def test_a_watchdog_that_survives_stop_is_killed_on_rollback(tmp_path: Path) -> None:
    """The rollback asserts death, not the verbs it issued."""
    install, calls = _script_guest(
        tmp_path,
        watchdog_props={"ControlGroup": "/system.slice/action-server.service"},
        watchdog_stop_effective=False,
    )
    with pytest.raises(RuntimeError, match="watchdog unit shares the action-server cgroup"):
        install()
    assert (
        "systemctl",
        "kill",
        "--kill-whom=all",
        guest_supervisor.WATCHDOG_UNIT,
    ) in calls


@pytest.mark.parametrize(
    "props",
    [
        {"MainPID": "7"},
        {"ExecStop": "/bin/stop-desktop"},
        {"StandardError": "null"},
        {"ExecStartPost": "/bin/change-task"},
        {"Type": "oneshot"},
        {"LoadState": "error"},
    ],
)
def test_unsupported_units_are_unchanged(tmp_path: Path, props: dict[str, str]) -> None:
    install, calls = _script_guest(tmp_path, props=props)
    with pytest.raises(RuntimeError):
        install()
    assert not (tmp_path / "etc").exists()
    assert not any(args[1] == "daemon-reload" for args in calls)


def test_non_systemd_guest_is_unchanged(tmp_path: Path) -> None:
    install, _ = _script_guest(tmp_path)
    (tmp_path / "proc/8000/cgroup").write_text("0::/user.slice/session-1.scope\n")
    with pytest.raises(RuntimeError, match="no private service"):
        install()
    assert not (tmp_path / "etc").exists()


def test_capture_retains_exit_status_stderr_and_bounds_crash_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Derive a real signal exit rather than inventing a returncode. PID ownership
    # stays with this test, and the process is reaped even on assertion failure.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        child.kill()
        code = child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    assert code == -9
    root = tmp_path / "var/lib/lop-action-supervisor"
    root.mkdir(parents=True)
    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "subprocess": SimpleNamespace(
            run=lambda *a, **k: SimpleNamespace(stdout="fatal action-server crash\n")
        ),
        "sys": SimpleNamespace(argv=["capture.py", "action-server.service"]),
    }
    monkeypatch.setenv("EXIT_CODE", "killed")
    monkeypatch.setenv("EXIT_STATUS", "KILL")
    monkeypatch.setenv("SERVICE_RESULT", "signal")
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    for _ in range(7):
        exec(
            guest_supervisor.CAPTURE_SOURCE,
            {"__builtins__": {**vars(builtins), "__import__": importer}},
        )
    records = json.loads((root / "action-server.service.json").read_text())["records"]
    assert len(records) == 4
    assert records[-1]["exit_code"] == "killed"
    assert records[-1]["exit_status"] == "KILL"
    assert records[-1]["stderr_tail"] == "fatal action-server crash\n"


@pytest.mark.asyncio
async def test_setup_unavailable_is_loud_once_and_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        calls = []
        real_command = provider._run_guest_command

        def command(*args: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(args)
            return guest_disk.CommandResult(
                0,
                "escalation=agentless\n"
                + json.dumps(
                    {"installed": False, "reason": "existing server stderr is not journalled"}
                ),
                "",
            )

        monkeypatch.setattr(provider, "_run_guest_command", command)
        provider._install_guest_supervisor(_plan())
        provider._install_guest_supervisor(_plan())
        assert provider._supervisor_unit is None
        assert len(calls) == 1
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report == {
            "installed": False,
            "status": "supervision unavailable",
            "cause": "existing server stderr is not journalled",
        }
        monkeypatch.setattr(provider, "_run_guest_command", real_command)
        await provider.execute(["unchanged_episode_input()"])
        assert len(stubs.guest_posts) == 1
        argv = stubs.guest_posts[0]["command"]
        assert base64.b64decode("".join(argv[3:])).decode() == "unchanged_episode_input()"


@pytest.mark.asyncio
async def test_allocation_installs_once_before_task_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        order: list[str] = []
        monkeypatch.setattr(provider, "_run_instance", lambda plan: "i-synthetic")
        monkeypatch.setattr(provider, "_create_lease", lambda *args: None)
        monkeypatch.setattr(provider, "_wait_ready", lambda *args: "127.0.0.1")
        monkeypatch.setattr(provider, "_prepare_guest_disk", lambda *args: order.append("disk"))
        monkeypatch.setattr(provider, "_start_desktop_env", lambda *args: order.append("task"))

        def install(*args: Any, **kwargs: Any) -> guest_disk.CommandResult:
            order.append("supervisor")
            return guest_disk.CommandResult(
                0, 'escalation=agentless\n{"installed": true, "unit": "action-server.service"}', ""
            )

        monkeypatch.setattr(provider, "_run_guest_command", install)
        await provider.allocate(_plan(), _task(), cache_root=provider._cache_root)
        provider._install_guest_supervisor(_plan())
        assert order == ["disk", "supervisor", "task"]
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["installed"] is True
        assert report["status"] == "supervised"
        # The installed path is not a refusal: it carries no diagnostic layout.
        assert "layout" not in report


@pytest.mark.asyncio
async def test_recovered_server_stops_batch_and_rebinds_before_more_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, provider = _adapter(monkeypatch)

    async def recovered(*args: Any, **kwargs: Any) -> None:
        provider.calls.append(("server-restarted",))
        raise GuestServerRecovered()

    monkeypatch.setattr(provider, "execute", recovered)
    result = await adapter.execute(_params(_click()))
    assert provider.calls == [("server-restarted",), ("observe", None)]
    assert result.observation.observation_id == "obs-2"
    assert adapter._current_observation == result.observation


@pytest.mark.asyncio
async def test_refused_socket_recovers_without_replaying_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        real_post = aws_mod.build_clients(provider._credentials, provider._region).http_post_json
        provider._clients.http_post_json = real_post
        checks = iter(
            [
                guest_supervisor.ReceiptResult("verified_none"),
                guest_supervisor.ReceiptResult("new_restart"),
            ]
        )
        deliveries = []

        def recover(
            delivery: str = "pre_action", *, budget_s: float | None = None
        ) -> guest_supervisor.ReceiptResult:
            deliveries.append(delivery)
            return next(checks)

        monkeypatch.setattr(provider, "_recover_guest_supervisor", recover)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["never_replay()", "never_send()"])
        assert deliveries == ["pre_action", "not_sent"]


def test_recovery_record_contains_cause_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        record = {
            "restart": 123,
            "exit_code": "exited",
            "exit_status": "1",
            "service_result": "exit-code",
            "stderr_tail": "fatal crash",
        }

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document(receipt=_envelope([record]))
            return guest_disk.CommandResult(
                0,
                "escalation=agentless\n" + json.dumps(_envelope([record])),
                "",
            )

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        assert provider._recover_guest_supervisor("ambiguous").state == "new_restart"
        assert provider._recover_guest_supervisor().state == "verified_none"
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["records"] == [record]
        assert evidence["failed_action_delivery"] == "ambiguous"
        assert evidence["restart_verification"]["status"] == "verified"


@pytest.mark.asyncio
async def test_crash_loop_exhausts_wait_and_ends_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        provider._clients.http_post_json = aws_mod.build_clients(
            provider._credentials, provider._region
        ).http_post_json
        provider._clients.ec2 = _NoConsoleEc2()
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (False, {"answered": False}),
        )
        with pytest.raises(
            aws_mod.GuestExecutionError, match="supervisor recovery budget exhausted"
        ):
            await provider.execute(["not_applied()"])
        # The whole pre-action wall-clock wait (probes at t=0, 0.5, 1, 2, 3, 5,
        # 7, 9, 11, 13; no probe is started at t=15 with nothing left to answer
        # inside), then a clean end without sending input: a patched probe
        # reports no verdict, so the receipt is treated as genuinely
        # unavailable and the action is never attempted.
        assert sum(stubs.slept) == aws_mod._PRE_ACTION_RECOVERY_BUDGET_S
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is False
        assert evidence["final_guest_record"].startswith("not retrieved")
        # Every probe is accounted for, even when the probe itself is stubbed:
        # the list is index-aligned with the attempts, never padded.
        assert [entry["probe"] for entry in evidence["probe_outcomes"]] == list(range(1, 11))
        assert not any("not_applied()" in str(body) for body in stubs.guest_posts)


@pytest.mark.asyncio
async def test_an_observation_wait_is_short_even_when_the_server_is_unreachable(
    tmp_path: Path,
) -> None:
    """An observation read is not worth the pre-action wait.

    It consumes a restart generation and then reads a frame; when the server
    cannot be reached the frame read fails and is reported anyway, and the
    harness re-observes once the guest is back. Paying 15 s here would delay
    every retry for nothing.
    """

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        provider._clients.http_get = lambda url, timeout: 503
        observed = await provider.observe()

    assert observed["screenshot"] == b"png"
    assert aws_mod._OBSERVATION_RECOVERY_BUDGET_S < aws_mod._PRE_ACTION_RECOVERY_BUDGET_S
    assert sum(stubs.slept) == aws_mod._OBSERVATION_RECOVERY_BUDGET_S


def test_exhaustion_evidence_names_which_path_element_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six silent probes used to collapse to ``read_failed: true``.

    The decision is unchanged -- still ``read_failed``, still the full wall-clock
    wait, still no input sent -- but the evidence now names the element that
    failed. A refused port, a name that does not resolve and a connect timeout
    are three different faults, and only one of them is the action server.
    """

    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        # The REAL probe, over a real (closed) loopback port: the chain under
        # test is the one a dying guest produces, not a stand-in.
        provider._clients.http_get = aws_mod.build_clients(
            provider._credentials, provider._region
        ).http_get
        result = provider._recover_guest_supervisor()
        assert sum(stubs.slept) == aws_mod._PRE_ACTION_RECOVERY_BUDGET_S

    assert result.state == "read_failed"
    assert result.recovery_exhausted is True
    # Every probe failed on the CONNECT: the host never reached the guest, which
    # is a different fact from a receipt that could not be read over a live
    # server -- and the only one that lets a caller try the action.
    assert result.path_unreachable is True
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["read_failed"] is True
    outcomes = evidence["probe_outcomes"]
    assert len(outcomes) == 10, "one probe plus the backoff schedule"
    assert evidence["path_unreachable"] is True
    assert evidence["budget_s"] == aws_mod._PRE_ACTION_RECOVERY_BUDGET_S
    # A real-clock refused-port run ends before the budget (13.596 s measured):
    # the last 2 s wait does not fit, so equality is an artefact of the stub
    # clock. The invariant is that the wait never spends MORE than its budget.
    assert 0.0 < evidence["waited_s"] <= aws_mod._PRE_ACTION_RECOVERY_BUDGET_S
    assert {outcome["kind"] for outcome in outcomes} == {"refused"}
    assert {outcome["errno"] for outcome in outcomes} == {errno.ECONNREFUSED}
    assert all(outcome["answered"] is False for outcome in outcomes)
    # The other side of the distinction: this branch MAY say unreachable, and it
    # names the element. An answered 503 must never reach this wording.
    assert "unreachable" in evidence["cause"]
    assert "refused" in evidence["cause"]


#: How long the loopback server below holds ``/terminal``. Longer than the OLD
#: 0.5 s probe deadline, far shorter than the watchdog-parity 5 s one -- the
#: window where a busy action server was mistaken for an ambiguous path.
_BUSY_TERMINAL_DELAY_S = 1.5


def _start_action_server(
    *, terminal_delay_s: float
) -> tuple[ThreadingHTTPServer, list[dict[str, Any]], threading.Thread]:
    """A real loopback action server: delayed ``/terminal``, recording ``/execute``.

    Real sockets rather than a stubbed client, because the defect lives in the
    interaction between the probe's deadline and a server that is slow to
    answer; a stub cannot reproduce a read timeout. The receipt command is
    answered with a healthy baseline, so a probe that succeeds proceeds to a
    normal ``verified_none`` rather than a restart.
    """

    action_posts: list[dict[str, Any]] = []
    baseline = {
        "schema_version": 1,
        "records": [],
        "captured_exits": 0,
        "baseline_nrestarts": 0,
        "baseline_active_enter": 100,
        "systemd_nrestarts": 0,
        "systemd_active_enter": 100,
    }

    class Handler(BaseHTTPRequestHandler):
        # The stdlib's own signature: the parameter is literally ``format`` and
        # keyword-capable, and pyright rejects a ``*args``-only override as an
        # incompatible method override.
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _respond(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            # Accepted immediately, answered only after the simulated screenshot
            # or page load: exactly the shape a short probe calls ambiguous.
            time.sleep(terminal_delay_s)
            self._respond(b"")

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            script = " ".join(payload["command"])
            if "/var/lib/lop-action-supervisor" in script:
                output = "escalation=agentless\n" + json.dumps(baseline)
            else:
                action_posts.append(payload)
                output = ""
            self._respond(json.dumps({"returncode": 0, "output": output, "error": ""}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, action_posts, thread


def _stop_action_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _provider_with_real_transport(stubs: _Stubs, tmp_path: Path) -> aws_mod.AwsProvider:
    """The provider wired to the REAL loopback transport, with a supervisor unit."""

    provider = _provider_over(stubs, tmp_path)
    assert provider._cache_root is not None
    provider._supervisor_unit = "action-server.service"
    transport = aws_mod.build_clients(provider._credentials, provider._region)
    provider._clients.http_get = transport.http_get
    provider._clients.http_post_json = transport.http_post_json
    provider._clients.ec2 = _NoConsoleEc2()
    return provider


def test_recovery_probe_timeout_is_not_shorter_than_the_watchdogs() -> None:
    """Probe-timeout parity, pinned at the source of the other constant.

    The watchdog states its 5 s bound inside its guest source, derived from the
    same fact: the probe classifies, it does not kill, so a mislabel costs
    evidence fidelity and nothing else. The host probe had no such parity --
    0.5 s against a server whose control endpoints answer in seconds -- and
    that asymmetry refused a live episode. Parsed rather than duplicated so an
    edit to either constant fails here instead of in production.
    """

    watchdog = re.search(
        r"^PROBE_TIMEOUT_S = ([\d.]+)$", guest_supervisor.WATCHDOG_SOURCE, re.MULTILINE
    )
    assert watchdog is not None, "the watchdog's probe bound moved or was renamed"
    assert aws_mod._RECOVERY_PROBE_TIMEOUT_S >= float(watchdog.group(1))


@pytest.mark.asyncio
async def test_a_busy_server_is_not_mistaken_for_a_dead_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Busy is not dead: one real busy server, one constant apart.

    The action server is single-threaded around its control endpoints, so a
    1280x720 screenshot or a page load blocks ``/terminal`` for seconds. At the
    old 0.5 s probe deadline that reads as ``ambiguous`` -- connected, never
    answered -- which leaves ``path_unreachable`` false and refuses an action
    the SAME server serves once the probe is allowed the watchdog's 5 s. The
    negative control runs the old constant against this server and must send
    nothing; the positive control runs the new one and must attempt and deliver.
    """

    server, action_posts, thread = _start_action_server(terminal_delay_s=_BUSY_TERMINAL_DELAY_S)
    real_probe_timeout = aws_mod._RECOVERY_PROBE_TIMEOUT_S
    try:
        monkeypatch.setattr(aws_mod, "GUEST_PORT", server.server_address[1])
        with _Stubs() as stubs:
            provider = _provider_with_real_transport(stubs, tmp_path)
            assert provider._cache_root is not None

            # Negative control: the OLD constant on the SAME server. The budget
            # is shortened only because ten 0.5 s timeouts would spend 5 s of
            # real test time; no budget makes a 0.5 s probe outlast a 1.5 s
            # answer, which is the point.
            monkeypatch.setattr(aws_mod, "_RECOVERY_PROBE_TIMEOUT_S", 0.5)
            monkeypatch.setattr(aws_mod, "_PRE_ACTION_RECOVERY_BUDGET_S", 2.0)
            with pytest.raises(aws_mod.GuestExecutionError, match="not attempted"):
                await provider.execute(["busy()"], settle=False)
            assert action_posts == [], "the old probe must refuse without sending input"
            evidence = json.loads(
                (provider._cache_root / "guest-supervisor-recovery.json").read_text()
            )
            assert evidence["path_unreachable"] is False
            assert "unreachable" not in evidence["cause"]
            assert {entry["error_class"] for entry in evidence["probe_outcomes"]} == {"ReadTimeout"}
            assert evidence["probe_timeout_s"] == 0.5
            assert evidence["budget_s"] == 2.0
            assert evidence["waited_s"] == 2.0
            assert all(entry["timeout_s"] == 0.5 for entry in evidence["probe_outcomes"])

            # Positive control: the watchdog-parity constant, same server, same
            # delay. The probe answers, the receipt reads, the action lands.
            monkeypatch.setattr(aws_mod, "_RECOVERY_PROBE_TIMEOUT_S", real_probe_timeout)
            monkeypatch.setattr(aws_mod, "_PRE_ACTION_RECOVERY_BUDGET_S", 15.0)
            await provider.execute(["busy()"], settle=False)
            assert len(action_posts) == 1, "the patient probe must attempt and deliver"
    finally:
        _stop_action_server(server, thread)


@pytest.mark.asyncio
async def test_a_zero_budget_refuses_instead_of_licensing_the_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``all([])`` is True, and that must never become permission to send.

    ``path_unreachable`` is the one verdict that lets ``execute`` attempt an
    action over an unread receipt. A non-positive budget makes ZERO probes; on
    an empty list ``all(...)`` used to evaluate True, so the wait granted the
    very licence it had learned nothing to support. The real server here would
    accept and record the action, which is exactly what must not happen.
    """

    server, action_posts, thread = _start_action_server(terminal_delay_s=0.0)
    try:
        monkeypatch.setattr(aws_mod, "GUEST_PORT", server.server_address[1])
        with _Stubs() as stubs:
            provider = _provider_with_real_transport(stubs, tmp_path)
            assert provider._cache_root is not None
            monkeypatch.setattr(aws_mod, "_PRE_ACTION_RECOVERY_BUDGET_S", 0.0)
            with pytest.raises(aws_mod.GuestExecutionError, match="not attempted"):
                await provider.execute(["must_not_send()"], settle=False)
            assert action_posts == [], "a wait that probed nothing must not authorize a send"
            evidence = json.loads(
                (provider._cache_root / "guest-supervisor-recovery.json").read_text()
            )
            assert evidence["probe_outcomes"] == []
            assert evidence["path_unreachable"] is False
            assert evidence["budget_s"] == 0.0
            assert evidence["waited_s"] == 0.0
            assert "unreachable" not in evidence["cause"]
    finally:
        _stop_action_server(server, thread)


def test_a_budget_shorter_than_the_first_wait_does_not_index_an_empty_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sub-0.5 s budget still makes one probe, so the schedule cannot be empty.

    The failed probe indexes the wait that follows it; when the derivation
    returned ``()`` for a budget below the first 0.5 s wait, that lookup raised
    ``IndexError`` instead of letting the deadline end the wait.
    """

    assert aws_mod._recovery_probe_delays(0.0) == (0.5,)
    assert aws_mod._recovery_probe_delays(0.1) == (0.5,)
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    with _Stubs() as stubs:
        provider = _provider_with_real_transport(stubs, tmp_path)
        result = provider._recover_guest_supervisor(budget_s=0.1)
    assert result.state == "read_failed"
    assert result.recovery_exhausted is True
    assert result.path_unreachable is True


_LOOPBACK_SERVER = r"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
port, report, ledger = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if '/var/lib/lop-action-supervisor' in str(body['command']):
            baseline = json.dumps({'schema_version': 1, 'records': [], 'captured_exits': 0,
                'baseline_nrestarts': 0, 'baseline_active_enter': 100,
                'systemd_nrestarts': 0, 'systemd_active_enter': 100})
            output = 'escalation=agentless\n'
            output += report.read_text() if report.exists() else baseline
        else:
            with ledger.open('a') as out:
                out.write('input\n')
            output = ''
        payload = json.dumps({'returncode': 0, 'output': output, 'error': ''}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
HTTPServer.allow_reuse_address = True
server = HTTPServer(('127.0.0.1', port), Handler)
print('ready', flush=True)
print('server diagnostic before SIGKILL', file=sys.stderr, flush=True)
server.serve_forever()
"""


@pytest.mark.asyncio
async def test_real_server_process_dies_restarts_and_next_batch_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real process/socket path; restarting here EMULATES systemd, not native proof."""
    port = _closed_port()
    report = tmp_path / "loopback-record.json"
    ledger = tmp_path / "loopback-input.txt"
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    processes = []

    def start() -> Any:
        process = subprocess.Popen(
            [sys.executable, "-c", _LOOPBACK_SERVER, str(port), str(report), str(ledger)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        return process

    try:
        process = start()
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path)
            assert provider._cache_root is not None
            provider._supervisor_unit = "action-server.service"
            # The explicit restart is a privileged guest command a macOS host
            # cannot execute; the script tests own that surface. Intercept ONLY
            # that command and give its verification the same receipt the real
            # report command reads, so this test still exercises the real HTTP
            # transport for the refusal, the receipt and the rebind.
            real_command = provider._run_guest_command

            def command(argv: Any, timeout: float) -> guest_disk.CommandResult:
                if _is_restart_command(argv):
                    return guest_disk.CommandResult(0, "", "")
                if _is_restart_result_command(argv):
                    return _restart_document(receipt=json.loads(report.read_text()))
                return real_command(argv, timeout)

            monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
            monkeypatch.setattr(provider, "_run_guest_command", command)
            real = aws_mod.build_clients(provider._credentials, provider._region)
            provider._clients.http_get = real.http_get
            provider._clients.http_post_json = real.http_post_json
            # Healthy baseline goes through the real HTTP execute transport.
            await provider.execute(["baseline()"], settle=False)
            assert ledger.read_text() == "input\n"
            process.kill()
            status = process.wait(timeout=5)
            assert status < 0
            stderr = process.stderr.read()
            report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "captured_exits": 1,
                        "baseline_nrestarts": 0,
                        "baseline_active_enter": 100,
                        "systemd_nrestarts": 1,
                        "systemd_active_enter": 200,
                        "records": [
                            {
                                "restart": 999,
                                "exit_code": "killed",
                                "exit_status": str(-status),
                                "service_result": "signal",
                                "stderr_tail": stderr,
                            }
                        ],
                    }
                )
            )
            restarted = False

            def manager_tick(delay: float) -> None:
                nonlocal restarted
                if not restarted:
                    start()
                    restarted = True

            # On the first real refused health request the fixture manager
            # restarts ONLY its own child, exactly once. No input is replayed.
            provider._sleep = manager_tick
            adapter, _ = _adapter(monkeypatch)
            adapter._provider = provider
            result = await adapter.execute(_params(_click()))
            assert result.observation.observation_id == "obs-2"
            assert restarted
            assert ledger.read_text() == "input\n"
            # A fresh observation is the next adapter response; only the next
            # batch may apply input after the recovery receipt was consumed.
            await provider.execute(["new_binding_input()"], settle=False)
            assert ledger.read_text() == "input\ninput\n"
            evidence = json.loads(
                (provider._cache_root / "guest-supervisor-recovery.json").read_text()
            )
            assert evidence["records"][0]["exit_status"] == str(-status)
            assert "server diagnostic before SIGKILL" in evidence["records"][0]["stderr_tail"]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


@pytest.mark.parametrize(
    ("effective", "reason"),
    [
        ({"Type": "oneshot"}, "oneshot action-server service cannot be supervised"),
        ({"Restart": "no"}, "effective Restart is not always"),
        ({"LoadState": "error"}, "action-server service is not loaded"),
        ({"ProtectSystem": "strict"}, "service sandbox may block supervisor capture"),
        ({"KillMode": "control-group"}, "effective supervisor restart policy was overridden"),
        ({"RestartPreventExitStatus": "SIGKILL"}, "unit has RestartPreventExitStatus exclusions"),
        ({"RestartPreventExitStatus": "SIGSEGV"}, "unit has RestartPreventExitStatus exclusions"),
        ({"RestartPreventExitStatus": "9 11"}, "unit has RestartPreventExitStatus exclusions"),
        ({"ExitType": "cgroup"}, "unit uses ExitType=cgroup"),
    ],
)
def test_effective_configuration_is_verified_and_new_dropin_rolled_back(
    tmp_path: Path, effective: dict[str, str], reason: str
) -> None:
    install, calls = _script_guest(tmp_path, effective_props=effective)
    with pytest.raises(RuntimeError, match=reason):
        install()
    assert not (
        tmp_path / "etc/systemd/system/action-server.service.d/90-lop-supervisor.conf"
    ).exists()
    assert sum(args[1] == "daemon-reload" for args in calls) == 2


def test_idempotent_install_reverifies_effective_restart_without_rewriting(tmp_path: Path) -> None:
    effective: dict[str, str] = {}
    install, calls = _script_guest(tmp_path, effective_props=effective)
    install()
    path = tmp_path / "etc/systemd/system/action-server.service.d/90-lop-supervisor.conf"
    before = path.stat().st_mtime_ns
    effective["Restart"] = "no"
    with pytest.raises(RuntimeError, match="effective Restart is not always"):
        install()
    assert path.stat().st_mtime_ns == before
    assert sum(args[1] == "daemon-reload" for args in calls) == 1
    assert sum("Restart" in args for args in calls) == 2


def test_parse_output_reads_the_clean_single_line_payload() -> None:
    """The clean case must stay what the old fixed offset read."""

    captured = (
        'escalation=agentless>supplied\n{"installed": true, "unit": "action-server.service"}\n'
    )
    assert guest_supervisor.parse_output(
        captured, guest_supervisor.is_install_payload
    ) == json.loads(captured.split("\n", 1)[1])
    assert guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload) == {
        "installed": True,
        "unit": "action-server.service",
    }


def test_parse_output_survives_the_sudo_prompt_before_the_payload() -> None:
    """The captured form that broke the install: sudo's prompt is on stderr."""

    captured = (
        "escalation=agentless>supplied\n"
        "[sudo] password for user:\n"
        '{"unit": "action-server.service", "installed": true}\n'
    )
    assert guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload) == {
        "unit": "action-server.service",
        "installed": True,
    }


def test_parse_output_reads_a_payload_spanning_several_lines() -> None:
    captured = (
        "escalation=none\n"
        "{\n"
        '  "unit": "action-server.service",\n'
        '  "installed": true\n'
        "}\n"
    )
    assert guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload) == {
        "unit": "action-server.service",
        "installed": True,
    }


def test_parse_output_ignores_noise_after_the_payload() -> None:
    captured = (
        "escalation=agentless\n"
        '{"installed": true, "unit": "action-server.service"}\n'
        "systemd: unit reloaded\n"
    )
    assert guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload) == {
        "installed": True,
        "unit": "action-server.service",
    }


def test_parse_output_reads_a_payload_with_no_metadata_line() -> None:
    captured = '{"installed": true, "unit": "action-server.service"}\n'
    assert guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload) == {
        "installed": True,
        "unit": "action-server.service",
    }


def test_parse_output_without_json_raises_a_named_failure_that_quotes_no_guest_text() -> None:
    captured = (
        "escalation=agentless>supplied\n"
        "[sudo] password for user:\n"
        "sudo: 1 incorrect password attempt\n"
    )
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable) as failure:
        guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload)
    assert type(failure.value).__name__ == "SupervisorOutputUnparseable"
    assert str(failure.value) == "supervisor output contained no JSON object"
    # The captured text can carry a credential prompt; it is never quoted.
    assert "password" not in str(failure.value)
    assert "incorrect password attempt" not in str(failure.value)
    # A ValueError is what the json module raised here before, so every
    # caller's handler keeps its meaning.
    assert isinstance(failure.value, ValueError)


def test_parse_output_skips_a_shadow_envelope_before_the_real_receipt() -> None:
    """A preamble envelope must not displace the payload and report "no restart".

    The shadow here is a FULLY VALID empty envelope, so a parser that keyed on
    shape alone (or on the first decodable object) would accept it and read a
    real restart as "no restart" -- the silent loss this module exists to
    prevent.
    """

    captured = (
        "escalation=agentless>supplied\n"
        + json.dumps(_envelope([]))
        + "\n"
        + json.dumps(_envelope([_restart_record()]))
        + "\n"
    )
    records = guest_supervisor.receipt_records(captured)
    assert [record["restart"] for record in records] == [42]


def test_parse_output_skips_an_unexpected_object_before_the_real_payload() -> None:
    captured = (
        "escalation=agentless>supplied\n"
        '{"note": "hi"}\n'
        '{"installed": true, "unit": "action-server.service"}\n'
    )
    assert guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload) == {
        "installed": True,
        "unit": "action-server.service",
    }


def test_parse_output_refuses_a_truncated_payload_whose_inner_object_is_wrong_shape() -> None:
    """A truncated payload whose innermost object is the WRONG shape must not win.

    The inner object here (``{"n": 1}``) does not satisfy ``is_install_payload``,
    so it is skipped and the parse fails closed. A completed, shape-MATCHING
    nested object is refused by the document-boundary rule instead, which the
    two ``installed_unit`` tests below pin -- shape alone could never tell it
    apart from a real payload.
    """

    captured = (
        "escalation=agentless>supplied\n"
        '{"unit": "action-server.service", "meta": {"n": 1}, "installed": true\n'
    )
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable) as failure:
        guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload)
    assert str(failure.value) == "supervisor output contained no JSON object"


def test_installed_unit_accepts_the_real_same_line_sudo_capture() -> None:
    """Real sudo writes its prompt WITHOUT a newline; the payload shares the line.

    Every other sudo fixture in this file puts a newline after the prompt, a
    shape real sudo never produces when its input is a pipe (it prints
    ``[sudo] password for user:`` and the command's own output follows on the
    same line). A left-boundary rule therefore rejected the genuine capture and
    re-created the unsupervised-episode defect this module fixes.
    """

    captured = (
        "escalation=agentless>supplied\n"
        '[sudo] password for user: {"unit": "action-server.service", "installed": true}\n'
    )
    assert guest_supervisor.installed_unit(captured) == "action-server.service"


def test_the_real_same_line_sudo_capture_installs_the_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the provider records an installed unit for the real capture."""

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        capture = (
            "escalation=agentless>supplied\n"
            '[sudo] password for user: {"unit": "action-server.service", "installed": true}\n'
        )
        calls = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(argv)
            return guest_disk.CommandResult(0, capture, "")

        monkeypatch.setattr(provider, "_run_guest_command", command)
        provider._install_guest_supervisor(_plan())
        assert len(calls) == 1
        assert provider._supervisor_unit == "action-server.service"
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report == {
            "installed": True,
            "unit": "action-server.service",
            "status": "supervised",
        }


def test_installed_unit_refuses_an_array_wrapped_object() -> None:
    """An array is not the install payload, even when it wraps one.

    The pre-PR parser returned the LIST for this capture and ``installed_unit``
    refused it. Decoding the inner object would install a unit the GUEST chose
    for a service the action server does not own.
    """

    captured = 'escalation=x\n[{"installed": true, "unit": "evil.service"}]\n'
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable) as failure:
        guest_supervisor.installed_unit(captured)
    assert isinstance(failure.value, ValueError)


def test_installed_unit_refuses_an_object_nested_inside_a_larger_object() -> None:
    """A member of another object is not a top-level payload."""

    captured = 'escalation=x\n{"outer": {"installed": true, "unit": "evil.service"}}\n'
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable):
        guest_supervisor.installed_unit(captured)


def test_installed_unit_refuses_a_multiline_member_of_another_object() -> None:
    """A member of another object is never top level, however it is spelled.

    The outer object decodes whole and consumes the member, so the member is
    never a candidate; the end-boundary rule would refuse it too, because a
    ``}`` closes its container right after it.
    """

    captured = (
        "escalation=x\n"
        "{\n"
        '  "outer":\n'
        '  {"installed": true, "unit": "evil.service"}\n'
        "}\n"
    )
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable):
        guest_supervisor.installed_unit(captured)


def test_parse_output_fails_closed_past_the_candidate_bound() -> None:
    """Guest-controlled brace-led lines must not buy unbounded scan work."""

    captured = "escalation=agentless\n" + "{\n" * (guest_supervisor._MAX_PAYLOAD_CANDIDATES + 1)
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable) as failure:
        guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload)
    assert str(failure.value) == "supervisor output exceeded the parser candidate bound"


def test_parse_output_fails_closed_past_the_size_bound() -> None:
    captured = "escalation=agentless\n" + "x" * guest_supervisor._MAX_CAPTURE_CHARS
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable) as failure:
        guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload)
    assert str(failure.value) == "supervisor output exceeded the parser size bound"


def test_parse_output_wraps_decoder_recursion_in_the_named_failure() -> None:
    """Deep nesting must not escape as a bare RecursionError.

    The depth is past where the decoder provably recurses (measured: depth
    8,000 decodes, 12,000 raises), and the fixture asserts that premise rather
    than trusting it -- 4,000 decodes and would have passed for the wrong
    reason.
    """

    nested = '{"a":' * 12000 + "1" + "}" * 12000
    with pytest.raises(RecursionError):
        json.JSONDecoder().raw_decode(nested, 0)
    captured = "escalation=agentless\n" + nested + "\n"
    with pytest.raises(guest_supervisor.SupervisorOutputUnparseable) as failure:
        guest_supervisor.parse_output(captured, guest_supervisor.is_install_payload)
    assert type(failure.value).__name__ == "SupervisorOutputUnparseable"


def test_a_brace_heavy_journal_tail_still_parses() -> None:
    """A legitimate receipt with thousands of brace pairs must not trip the bound.

    Braces inside a decoded string value are not document candidates: the
    decoded document is skipped whole. Counting them would spend the 32-candidate
    budget on one 28 KB journal tail and refuse a perfectly good receipt.
    """

    record = dict(_restart_record())
    record["stderr_tail"] = '{"frame": 1} ' * 2000
    captured = (
        "escalation=agentless\n[sudo] password for user: " + json.dumps(_envelope([record])) + "\n"
    )
    assert len(captured) > 20000, "fixture must be brace-heavy enough to matter"
    records = guest_supervisor.receipt_records(captured)
    assert [item["restart"] for item in records] == [42]


def test_receipt_records_survive_the_sudo_prompt_before_the_payload() -> None:
    """The report path shares the parser, so it shares the tolerance."""

    captured = (
        "escalation=agentless>supplied\n"
        "[sudo] password for user:\n" + json.dumps(_envelope([_restart_record()])) + "\n"
    )
    records = guest_supervisor.receipt_records(captured)
    assert [record["exit_status"] for record in records] == ["KILL"]


def test_setup_evidence_names_the_unparseable_output_instead_of_a_json_decode_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The setup evidence must say WHAT failed, and never quote the guest."""

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None

        def command(*args: Any, **kwargs: Any) -> guest_disk.CommandResult:
            return guest_disk.CommandResult(
                0,
                "escalation=agentless>supplied\n" "[sudo] password for user:\n" "not json at all\n",
                "",
            )

        monkeypatch.setattr(provider, "_run_guest_command", command)
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["installed"] is False
        assert report["status"] == "supervision unavailable"
        assert report["cause"] == "SupervisorOutputUnparseable"
        assert report["cause"] != "JSONDecodeError"
        assert "password" not in json.dumps(report)


def test_capture_preserves_exit_status_when_journal_times_out(tmp_path: Path) -> None:
    root = tmp_path / "var/lib/lop-action-supervisor"
    root.mkdir(parents=True)
    record_path = root / "action-server.service.json"

    def timeout(*args: Any, **kwargs: Any) -> Any:
        # Status must ALREADY be durable when the slow journal is consulted.
        first = json.loads(record_path.read_text())["records"][-1]
        assert first["exit_status"] == "SEGV"
        assert first["service_result"] == "core-dump"
        raise subprocess.TimeoutExpired("journalctl", 2)

    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(
            environ={"EXIT_CODE": "dumped", "EXIT_STATUS": "SEGV", "SERVICE_RESULT": "core-dump"}
        ),
        "subprocess": SimpleNamespace(run=timeout),
        "sys": SimpleNamespace(argv=["capture.py", "action-server.service"]),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    exec(
        guest_supervisor.CAPTURE_SOURCE,
        {"__builtins__": {**vars(builtins), "__import__": importer}},
    )
    final = json.loads(record_path.read_text())["records"][-1]
    assert final["exit_code"] == "dumped"
    assert final["exit_status"] == "SEGV"
    assert final["journal_tail_status"] == "unavailable"


def _restart_record() -> dict[str, Any]:
    return {
        "restart": 42,
        "exit_code": "killed",
        "exit_status": "KILL",
        "service_result": "signal",
        "stderr_tail": "fatal server diagnostic",
    }


def _receipt(records: list[dict[str, Any]]) -> guest_disk.CommandResult:
    return guest_disk.CommandResult(
        0, "escalation=agentless\n" + json.dumps(_envelope(records)), ""
    )


def _is_restart_command(argv: Any) -> bool:
    return "--kill-whom=all" in " ".join(str(part) for part in argv)


def _is_restart_result_command(argv: Any) -> bool:
    script = " ".join(str(part) for part in argv)
    return guest_supervisor.RESTART_RESULT_PATH in script and "--kill-whom=all" not in script


def _restart_document(
    *,
    token: int = 777,
    pre: tuple[int, ...] = (1637, 1887),
    post: tuple[int, ...] = (1901,),
    main_pid: int = 1901,
    main_start_ticks: int = 101000,
    kill_ticks: int = 100000,
    owners: list[dict[str, Any]] | None = None,
    receipt: dict[str, Any] | None = None,
    pre_complete: bool = True,
    post_complete: bool = True,
) -> guest_disk.CommandResult:
    """One RESTART_RESULT_SOURCE payload, with the real AMI's two-holder shape."""
    if owners is None:
        owners = [
            {
                "pid": pid,
                "ppid": main_pid,
                "start_ticks": main_start_ticks,
                "ancestors": [main_pid],
            }
            for pid in post
        ]
    document = {
        "schema_version": 1,
        "token": token,
        "unit": "action-server.service",
        "pre_restart_pids": list(pre),
        "pre_restart_count": len(pre),
        "pre_scan_complete": pre_complete,
        "kill_uptime_ticks": kill_ticks,
        "post_restart_pids": list(post),
        "post_restart_count": len(post),
        "post_scan_complete": post_complete,
        "post_restart_owners": owners,
        "new_main_pid": main_pid,
        "new_main_start_ticks": main_start_ticks,
        "receipt": receipt if receipt is not None else _envelope([_restart_record()]),
    }
    return guest_disk.CommandResult(0, "escalation=agentless\n" + json.dumps(document), "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["warning", "empty", "empty_list", "nonzero", "exception", "scrub_exception"]
)
async def test_receipt_read_failure_retries_once_and_never_posts_stale_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        calls = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(argv)
            assert "/var/lib/lop-action-supervisor" in str(argv), "stale input was posted"
            if failure == "exception":
                raise OSError("synthetic read failure")
            if failure == "scrub_exception":
                return _receipt([_restart_record()])
            if failure == "nonzero":
                return guest_disk.CommandResult(1, "", "")
            if failure == "empty":
                return guest_disk.CommandResult(0, "", "")
            if failure == "empty_list":
                return guest_disk.CommandResult(0, "escalation=agentless\n[]", "")
            # A guest warning with NO receipt behind it: the read fails
            # closed, so a restart that really happened can never be read
            # as "no restart". A warning in FRONT of a real receipt is the
            # tolerated case below, not a read failure.
            return guest_disk.CommandResult(0, "sudo warning\nnot a receipt\n", "")

        monkeypatch.setattr(provider, "_run_guest_command", command)
        if failure == "scrub_exception":

            def scrub_failure(*args: Any) -> Any:
                raise ValueError("synthetic scrub failure")

            monkeypatch.setattr(aws_mod.guest_loss, "_scrub", scrub_failure)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()", "nor_this()"])
        assert len(calls) == 2
        assert provider._supervisor_restart is None
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["receipt_unavailable"] is True
        assert evidence["read_failed"] is True
        assert evidence["read_attempts"] == 2


@pytest.mark.asyncio
async def test_a_receipt_behind_a_guest_warning_is_read_and_its_restart_is_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preamble before the payload must not hide a restart that really happened.

    The old parser read a fixed offset after the metadata line, so a warning
    line in front of the receipt JSON made the whole read fail. The receipt
    then looked unreadable instead of reporting the restart it carried, which
    is the same blindness the supervisor was merged to remove.
    """

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        calls = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(argv)
            assert "/var/lib/lop-action-supervisor" in str(argv), "stale input was posted"
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            return guest_disk.CommandResult(
                0, "sudo warning\n" + _receipt([_restart_record()]).stdout, ""
            )

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()", "nor_this()"])
        assert len(calls) == 3
        # The planted restart was READ, never quietly reported as "no restart".
        assert provider._supervisor_restart == 42
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is True
        assert evidence["read_failed"] is False
        assert evidence["receipt_unavailable"] is False


@pytest.mark.asyncio
async def test_a_shadow_envelope_before_the_receipt_cannot_report_no_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid empty envelope ahead of the real receipt must not win.

    Measured consequence before this guard: the shadow was accepted, the real
    ``restart: 42`` was read as "no restart", the action was posted on the
    possibly-restarted binding, and no recovery evidence was written.
    """

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        calls = []
        shadow = json.dumps(_envelope([]))

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(argv)
            assert "/var/lib/lop-action-supervisor" in str(argv), "stale input was posted"
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            return guest_disk.CommandResult(
                0,
                "escalation=agentless>supplied\n"
                + shadow
                + "\n"
                + _receipt([_restart_record()]).stdout,
                "",
            )

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()", "nor_this()"])
        assert len(calls) == 3
        assert provider._supervisor_restart == 42
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is True
        assert evidence["read_failed"] is False


@pytest.mark.asyncio
async def test_unreadable_receipt_returns_a_fresh_adapter_binding_without_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        calls = []

        def invalid(*args: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(args)
            return guest_disk.CommandResult(0, "escalation=agentless\nmalformed receipt", "")

        monkeypatch.setattr(provider, "_run_guest_command", invalid)
        adapter, _ = _adapter(monkeypatch)
        adapter._provider = provider
        result = await adapter.execute(_params(_click(), _click()))
        assert result.observation.observation_id == "obs-2"
        assert adapter._current_observation == result.observation
        assert len(calls) == 4  # Two failed action reads, then two pre-frame reads.
        assert stubs.guest_posts == []
        assert provider._supervisor_restart is None


def test_transient_receipt_failure_retries_before_accepting_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        replies = iter(
            [
                guest_disk.CommandResult(1, "", ""),
                _receipt([_restart_record()]),
                guest_disk.CommandResult(0, "", ""),
                _restart_document(),
            ]
        )
        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", lambda *a, **k: next(replies))
        assert provider._recover_guest_supervisor().state == "new_restart"
        assert stubs.slept == [0.1]


def test_withheld_records_are_explicitly_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            return _receipt([_restart_record()])

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        monkeypatch.setattr(aws_mod.guest_loss, "_scrub", lambda *args: None)
        assert provider._recover_guest_supervisor().state == "new_restart"
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["records"] == []
        assert evidence["records_withheld"] is True
        assert evidence["read_failed"] is False


@pytest.mark.asyncio
async def test_false_readiness_does_not_invent_recovery_budget_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        provider._clients.http_post_json = aws_mod.build_clients(
            provider._credentials, provider._region
        ).http_post_json
        provider._clients.ec2 = _NoConsoleEc2()
        monkeypatch.setattr(
            provider,
            "_recover_guest_supervisor",
            lambda *a, **k: guest_supervisor.ReceiptResult("verified_none"),
        )
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (False, {"answered": False, "status": 503}),
        )
        with pytest.raises(aws_mod.GuestExecutionError) as raised:
            await provider.execute(["not_sent()"])
        assert "guest down" in str(raised.value)
        assert "recovery budget exhausted" not in str(raised.value)


@pytest.mark.asyncio
async def test_statement_three_restart_interrupts_batch_and_rebinds_before_next_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        reads = 0
        input_posts = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            nonlocal reads
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            if "/var/lib/lop-action-supervisor" in str(argv):
                reads += 1
                return _receipt([] if reads < 3 else [_restart_record()])
            input_posts.append(argv)
            return guest_disk.CommandResult(0, "", "")

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        adapter, _ = _adapter(monkeypatch)
        adapter._provider = provider
        result = await adapter.execute(_params(_click(), _click(), _click(), _click()))
        assert len(input_posts) == 2
        assert result.observation.observation_id == "obs-2"
        assert adapter._current_observation == result.observation
        assert reads == 4  # Three per-action reads, then one BEFORE the fresh frame.
        await provider.execute(["new_binding_input()"], settle=False)
        assert len(input_posts) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("write_errno", [errno.ENOSPC, errno.EROFS])
@pytest.mark.parametrize("nrestarts", [0, 1])
async def test_failed_first_capture_write_cannot_hide_generation_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_errno: int, nrestarts: int
) -> None:
    """Run the shipped capture AND report; the manager view outlives a failed write."""
    root = tmp_path / "var/lib/lop-action-supervisor"
    root.mkdir(parents=True)
    path = root / "action-server.service.json"
    baseline = _envelope([])
    baseline.update(baseline_nrestarts=7, systemd_nrestarts=7)
    path.write_text(json.dumps(baseline))
    native = {"NRestarts": 7, "ActiveEnterTimestampMonotonic": 100}
    events = []
    generation = 1
    real_write = Path.write_text

    def unwritable(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == root / "action-server.service.tmp":
            raise OSError(write_errno, "synthetic capture storage failure")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", unwritable)

    def execute_source(source: str) -> str:
        def show(*args: Any, **kwargs: Any) -> str:
            assert args[0][:3] == ["systemctl", "show", "action-server.service"]
            return "\n".join(f"{key}={value}" for key, value in native.items())

        modules = {
            "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
            "os": SimpleNamespace(
                environ={"EXIT_CODE": "killed", "EXIT_STATUS": "KILL", "SERVICE_RESULT": "signal"}
            ),
            "subprocess": SimpleNamespace(check_output=show),
            "sys": SimpleNamespace(argv=["-c", "action-server.service"]),
        }
        real_import = builtins.__import__

        def importer(name: str, *args: Any, **kwargs: Any) -> Any:
            return modules[name] if name in modules else real_import(name, *args, **kwargs)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(source, {"__builtins__": {**vars(builtins), "__import__": importer}})
        return output.getvalue()

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            nonlocal generation
            if "/var/lib/lop-action-supervisor" in str(argv):
                return guest_disk.CommandResult(
                    0, "escalation=agentless\n" + execute_source(guest_supervisor.REPORT_SOURCE), ""
                )
            events.append(("input", generation))
            if len(events) == 2:
                # The old process exited after input two; the helper runs but
                # cannot persist even its first record. The manager still sees
                # a new activation, with or without an automatic restart count.
                with pytest.raises(OSError) as raised:
                    execute_source(guest_supervisor.CAPTURE_SOURCE)
                assert raised.value.errno == write_errno
                assert json.loads(path.read_text())["records"] == []
                native.update(NRestarts=7 + nrestarts, ActiveEnterTimestampMonotonic=200)
                generation = 2
            return guest_disk.CommandResult(0, "", "")

        monkeypatch.setattr(provider, "_run_guest_command", command)
        old_frame = provider._env._get_obs

        def frame() -> Any:
            events.append(("screen", generation))
            return old_frame()

        monkeypatch.setattr(provider._env, "_get_obs", frame)
        adapter, _ = _adapter(monkeypatch)
        adapter._provider = provider
        result = await adapter.execute(_params(_click(111), _click(222), _click(333)))
        assert events == [("input", 1), ("input", 1), ("screen", 2)]
        assert result.observation.observation_id == "obs-2"
        assert provider._supervisor_restart is None
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["receipt_unavailable"] is True
        assert evidence["read_failed"] is True
        assert evidence["receipt_failure"] == "systemd_generation_disagreement"
        assert evidence["read_attempts"] == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"systemd_nrestarts": 2},  # One newer restart is missing, despite a valid older record.
        {"systemd_nrestarts": 0},  # Counter reset/manual start cannot be correlated.
        {"systemd_active_enter": 100},  # An exit record cannot describe the still-old activation.
    ],
)
def test_restart_history_must_agree_with_independent_systemd_view(changes: dict[str, int]) -> None:
    history = _envelope([_restart_record()])
    history.update(changes)
    with pytest.raises(guest_supervisor.GenerationMismatch):
        guest_supervisor.receipt_records("escalation=agentless\n" + json.dumps(history))


# ---------------------------------------------------------------------------
# Layout diagnostic on a unit-discovery refusal
# ---------------------------------------------------------------------------


def _layout_payload(**overrides: Any) -> dict[str, Any]:
    """The raw fields LAYOUT_SOURCE prints -- no derived claims."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "socket_pids": [9000],
        "socket_owners": [
            {
                "pid": 9000,
                "ppid": 8000,
                "comm": "python3",
                "cgroup": "/system.slice/action-server.service",
            }
        ],
        "server": {
            "pid": 9000,
            "ppid": 8000,
            "comm": "python3",
            "cgroup": "/system.slice/action-server.service",
        },
        "ancestry": [
            {"pid": 9000, "ppid": 8000, "comm": "python3"},
            {"pid": 8000, "ppid": 1, "comm": "systemd"},
        ],
        "units": [
            {
                "unit": "action-server.service",
                "main_pid": 8001,
                "exec_start": "/usr/bin/python3 /home/user/server/main.py",
                "type": "simple",
                "control_group": "/system.slice/action-server.service",
                "sub_state": "running",
                "load_state": "loaded",
            }
        ],
    }
    payload.update(overrides)
    # Completeness defaults for the common shape; a test that exercises a
    # truncated capture overrides them explicitly.
    payload.setdefault("socket_scan_complete", True)
    payload.setdefault("socket_owner_count", len(payload["socket_pids"]))
    return payload


def _install_refusal(reason: str) -> guest_disk.CommandResult:
    return guest_disk.CommandResult(
        0, "escalation=agentless\n" + json.dumps({"installed": False, "reason": reason}), ""
    )


def _scripted_commands(
    provider: Any,
    monkeypatch: pytest.MonkeyPatch,
    install: guest_disk.CommandResult,
    layout: guest_disk.CommandResult | Callable[[], guest_disk.CommandResult],
) -> list[float]:
    """Answer install and layout commands from one fake transport."""
    timeouts: list[float] = []

    def command(argv: Any, timeout: float) -> guest_disk.CommandResult:
        timeouts.append(timeout)
        if "socket_pids" in " ".join(argv):
            return layout if isinstance(layout, guest_disk.CommandResult) else layout()
        return install

    monkeypatch.setattr(provider, "_run_guest_command", command)
    return timeouts


def test_exclusive_ownership_refusal_records_the_guest_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        timeouts = _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("service does not exclusively own action server"),
            guest_disk.CommandResult(
                0, "escalation=agentless\n" + json.dumps(_layout_payload()), ""
            ),
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["installed"] is False
        assert report["status"] == "supervision unavailable"
        assert report["cause"] == "service does not exclusively own action server"
        layout = report["layout"]
        assert set(layout) == {
            "schema_version",
            "status",
            "port",
            "pid_source",
            "socket_pids",
            "socket_owners",
            "socket_owner_count",
            "socket_scan_complete",
            "server",
            "ancestry",
            "units",
            "socket_owned_by_unit",
            "other_socket_owners_share_unit_cgroup",
            "supervisor_dropin_kill_mode",
        }
        assert layout["status"] == "ok"
        assert layout["units"] == _layout_payload()["units"]
        assert layout["server"] == _layout_payload()["server"]
        assert layout["ancestry"] == _layout_payload()["ancestry"]
        assert layout["socket_pids"] == [9000]
        # The default fixture's single owner is in the unit's cgroup -- and the
        # owning unit is identified by cgroup here because its MainPID (8001)
        # is not the server: no other owner means the vacuous ``true``.
        assert layout["other_socket_owners_share_unit_cgroup"] is True
        # Completeness signals and the drop-in's KillMode are recorded beside
        # it: membership is not survival under KillMode=process.
        assert layout["socket_owner_count"] == 1
        assert layout["socket_scan_complete"] is True
        assert layout["supervisor_dropin_kill_mode"] == "process"
        # Derived host-side, never accepted from the guest.
        assert layout["port"] == guest_supervisor.ACTION_SERVER_PORT
        assert layout["pid_source"] == "socket"
        assert layout["socket_owned_by_unit"] is False
        # Install first; the capture uses a socket deadline DERIVED from the
        # guest's intended deadline, not the 4 s host budget (which would ask
        # the guest to finish in 1 s), and one bounded probe.
        assert timeouts[0] == 15.0
        assert guest_deadline_for(timeouts[1]) == guest_supervisor.LAYOUT_GUEST_DEADLINE_S


def test_a_loaded_guest_capture_completes_inside_the_host_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ~2.5 s guest (a loaded AMI) must return a layout, not a timeout.

    The guest deadline is derived from LAYOUT_GUEST_DEADLINE_S; with the old
    ``timeout=LAYOUT_BUDGET_S`` the body was 1 s and this capture was killed
    before it could print. The host budget stays 4 s.
    """

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None

        def layout() -> guest_disk.CommandResult:
            time.sleep(2.5)
            return guest_disk.CommandResult(0, json.dumps(_layout_payload()), "")

        _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("service does not exclusively own action server"),
            layout,
        )
        started = time.monotonic()
        provider._install_guest_supervisor(_plan())
        elapsed = time.monotonic() - started
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["cause"] == "service does not exclusively own action server"
        assert report["layout"]["status"] == "ok"
        assert report["layout"]["server"] == _layout_payload()["server"]
        assert elapsed < guest_supervisor.LAYOUT_BUDGET_S


def test_no_listening_socket_falls_back_to_the_ancestry_and_records_no_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        payload = _layout_payload(
            socket_pids=[],
            server={"pid": 8000, "ppid": 1, "comm": "python3", "cgroup": "/system.slice"},
            ancestry=[{"pid": 8000, "ppid": 1, "comm": "python3"}],
            units=[],
        )
        _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("action server has no private service"),
            guest_disk.CommandResult(0, "escalation=agentless\n" + json.dumps(payload), ""),
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["cause"] == "action server has no private service"
        assert report["layout"]["pid_source"] == "ancestry"
        assert report["layout"]["units"] == []
        assert report["layout"]["server"]["pid"] == 8000
        assert report["layout"]["socket_owned_by_unit"] is None


def test_a_socket_owner_outside_any_unit_is_still_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        payload = _layout_payload(
            server={"pid": 9000, "ppid": 8000, "comm": "python3", "cgroup": "/system.slice"},
            units=[],
        )
        _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("service does not exclusively own action server"),
            guest_disk.CommandResult(0, "escalation=agentless\n" + json.dumps(payload), ""),
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["layout"]["server"]["cgroup"] == "/system.slice"
        assert report["layout"]["units"] == []
        assert report["layout"]["socket_pids"] == [9000]
        assert report["layout"]["pid_source"] == "socket"
        assert report["layout"]["socket_owned_by_unit"] is None


def test_a_hung_layout_capture_is_bounded_and_leaves_the_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        monkeypatch.setattr(guest_supervisor, "LAYOUT_BUDGET_S", 0.05)

        def layout() -> guest_disk.CommandResult:
            time.sleep(0.5)
            return guest_disk.CommandResult(0, json.dumps(_layout_payload()), "")

        _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("service does not exclusively own action server"),
            layout,
        )
        started = time.monotonic()
        provider._install_guest_supervisor(_plan())
        elapsed = time.monotonic() - started
        assert elapsed < 0.75
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["installed"] is False
        assert report["cause"] == "service does not exclusively own action server"
        assert report["layout"] == {"schema_version": 1, "status": "timeout"}


def test_a_raising_layout_capture_never_changes_the_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None

        def layout() -> guest_disk.CommandResult:
            raise RuntimeError("transport exploded")

        _scripted_commands(
            provider, monkeypatch, _install_refusal("action server has no private service"), layout
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["status"] == "supervision unavailable"
        assert report["cause"] == "action server has no private service"
        assert report["layout"] == {"schema_version": 1, "status": "error"}


def test_a_malformed_layout_capture_is_unparseable_not_an_outcome_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("service does not exclusively own action server"),
            guest_disk.CommandResult(0, "not a json document at all", ""),
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["cause"] == "service does not exclusively own action server"
        assert report["layout"] == {"schema_version": 1, "status": "unparseable"}


def test_a_planted_secret_never_reaches_the_layout_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "canary" + "-value-" + "42"
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._known_secret_values = (*provider._known_secret_values, secret)
        payload = _layout_payload(
            server={
                "pid": 9000,
                "ppid": 8000,
                "comm": secret,
                "cgroup": "/system.slice/" + secret + ".service",
            },
            units=[
                {
                    **_layout_payload()["units"][0],
                    "exec_start": "/usr/bin/python3 /home/user/server/main.py --password=" + secret,
                }
            ],
        )
        _scripted_commands(
            provider,
            monkeypatch,
            _install_refusal("service does not exclusively own action server"),
            guest_disk.CommandResult(0, "escalation=agentless\n" + json.dumps(payload), ""),
        )
        provider._install_guest_supervisor(_plan())
        raw = (provider._cache_root / "guest-supervisor-setup.json").read_text()
        assert secret not in raw
        assert "[redacted]" in raw
        assert json.loads(raw)["layout"]["status"] == "ok"


def test_layout_record_recomputes_guest_derived_claims() -> None:
    """A payload's own port/pid_source/differs are ignored, not trusted."""
    payload = _layout_payload(
        port=22,
        pid_source="ancestry",
        socket_owned_by_unit=False,
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["port"] == guest_supervisor.ACTION_SERVER_PORT
    assert record["pid_source"] == "socket"
    # unit MainPID 8001 is known and does not own socket 9000.
    assert record["socket_owned_by_unit"] is False

    # The guest still claims a difference, but the only unit's MainPID is the
    # socket owner: the host must report agreement it can prove.
    payload["units"][0]["main_pid"] = 9000
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is True

    # A server proc that is not the socket owner is not "socket"-sourced.
    payload["server"]["pid"] = 7000
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["pid_source"] is None


def test_unknown_main_pid_is_undetermined_not_ownership() -> None:
    """A missing MainPID can never be read as ``false`` (not owned)."""
    payload = _layout_payload()
    payload["units"][0]["main_pid"] = None
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is None

    # One unknown unit beside a known non-matching one: still undetermined.
    second = {**payload["units"][0], "unit": "other.service", "main_pid": 8001}
    payload["units"].append(second)
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is None

    # A known match settles it regardless of the other unit's missing value.
    second["main_pid"] = 9000
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is True

    # No socket owner at all is also undetermined.
    payload["socket_pids"] = []
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is None


def test_layout_record_drops_unknown_keys_and_neutralizes_malformed_fields() -> None:
    payload = {
        "schema_version": 1,
        "status": "pretending-to-be-ok",
        "argv": ["--secret"],
        "socket_pids": [9000, 9000.0, "9001", True, 0, -1],
        "units": [
            {
                "unit": "action-server.service",
                "main_pid": 8001.0,
                "exec_start": "python3\nmain.py\x00",
                "type": "invented",
                "control_group": "/system.slice/action-server.service",
                "sub_state": "Not A State",
                "load_state": "loaded",
                "environment": {"PASSWORD": "x"},
            },
            {"unit": "../evil.service", "main_pid": 1},
            {"unit": "bool.service", "main_pid": True, "sub_state": "running"},
            "not a dict",
        ],
        "ancestry": [{"pid": 9000, "ppid": True, "comm": "python3\rmain\x00"}],
        "server": {"pid": 9000, "ppid": 8000, "comm": "python3", "cgroup": "/system.slice"},
    }
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["status"] == "ok"
    assert record["socket_pids"] == [9000]
    assert [unit["unit"] for unit in record["units"]] == ["action-server.service", "bool.service"]
    assert all(unit["main_pid"] is None for unit in record["units"])
    unit = record["units"][0]
    assert "environment" not in unit
    assert unit["exec_start"] == "python3main.py"
    assert unit["type"] == "other"
    assert unit["sub_state"] == "other"
    # True is not PID 1.
    assert record["ancestry"][0]["ppid"] is None
    assert record["ancestry"][0]["comm"] == "python3main"
    assert "argv" not in record
    assert record["socket_owned_by_unit"] is None


def _owner(pid: int, cgroup: str | None, comm: str = "python3") -> dict[str, Any]:
    return {"pid": pid, "ppid": 8000, "comm": comm, "cgroup": cgroup}


def test_every_other_socket_owner_in_the_unit_cgroup_is_true() -> None:
    """Cgroup membership only: the second holder is inside the unit cgroup.

    It still survives a ``KillMode=process`` restart; what the fact decides is
    whether an explicit cgroup-wide kill can reach it.
    """
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, "/system.slice/action-server.service"),
        ],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is True
    assert [owner["pid"] for owner in record["socket_owners"]] == [9000, 1887]


def test_one_socket_owner_in_a_different_cgroup_is_false() -> None:
    """The production question: the second holder is outside the unit cgroup.

    Shaped like the first real capture (server 1637 = osworld.service MainPID,
    an unexplained second holder 1887). Out-of-cgroup means a cgroup-wide kill
    cannot reach it, so the answer must be ``false`` -- even though a
    ``KillMode=process`` restart leaves an in-cgroup holder alive too.
    """
    unit = {
        "unit": "osworld.service",
        "main_pid": 1637,
        "exec_start": "{ path=/usr/bin/python ; argv[]=/usr/bin/python /home/user/server/main.py }",
        "type": "simple",
        "control_group": "/system.slice/osworld.service",
        "sub_state": "running",
        "load_state": "loaded",
    }
    payload = _layout_payload(
        socket_pids=[1637, 1887],
        server={"pid": 1637, "ppid": 1, "comm": "python", "cgroup": unit["control_group"]},
        socket_owners=[
            _owner(1637, unit["control_group"], "python"),
            _owner(1887, "/system.slice/zombie.service", "python"),
        ],
        units=[unit, {"unit": "init.scope", "main_pid": None, "control_group": ""}],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is True
    assert record["other_socket_owners_share_unit_cgroup"] is False

    # The same capture with the second owner inside the unit cgroup flips the
    # membership answer; survival is unchanged under KillMode=process.
    payload["socket_owners"][1]["cgroup"] = unit["control_group"]
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is True


def test_an_unreadable_owner_cgroup_is_undetermined() -> None:
    """A cgroup that could not be read is null, never agreement."""
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, None),
        ],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None
    assert record["socket_owners"][1]["cgroup"] is None

    # A missing owner entry is a gap in the evidence, not an absent process.
    payload["socket_owners"] = payload["socket_owners"][:1]
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_an_owner_outside_the_unit_is_false_even_when_another_is_unreadable() -> None:
    """A known difference refutes the universal claim on its own."""
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887, 1888],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, "/system.slice/zombie.service"),
            _owner(1888, None),
        ],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is False


def test_no_unit_cgroup_is_undetermined() -> None:
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000, "control_group": ""}],
        socket_pids=[9000, 1887],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, "/system.slice/action-server.service"),
        ],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None

    # No owning unit at all is the same undetermined answer.
    payload["units"] = []
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_a_single_socket_owner_is_vacuously_true() -> None:
    """No other owner means no sibling that could outlive a restart.

    The question quantifies over owners OTHER than the server, so an empty set
    is true -- provided the unit itself is identified. This is the only ``true``
    that does not compare a cgroup, and it is not derived from missing data:
    the capture observed exactly one listener.
    """
    payload = _layout_payload(units=[{**_layout_payload()["units"][0], "main_pid": 9000}])
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_pids"] == [9000]
    assert record["other_socket_owners_share_unit_cgroup"] is True


def test_no_observed_socket_owner_is_undetermined() -> None:
    payload = _layout_payload(socket_pids=[], socket_owners=[])
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_a_guest_claim_about_sharing_is_recomputed() -> None:
    """``other_socket_owners_share_unit_cgroup`` is host-derived, never read."""
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, "/system.slice/zombie.service"),
        ],
        other_socket_owners_share_unit_cgroup=True,
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is False


def test_the_recorded_kill_mode_matches_the_install_drop_in() -> None:
    """Membership sits beside KillMode=process, not beside a survival claim."""
    record = guest_supervisor.layout_record(json.dumps(_layout_payload()))
    assert (
        record["supervisor_dropin_kill_mode"] == guest_supervisor.SUPERVISOR_KILL_MODE == "process"
    )
    assert f"KillMode={guest_supervisor.SUPERVISOR_KILL_MODE}" in guest_supervisor.INSTALL_SOURCE


def test_a_truncated_socket_owner_list_is_undetermined() -> None:
    """Nine real holders, cap eight: the hidden one cannot be assumed away.

    The ninth (highest pid, dropped by the cap) is the one in zombie.service;
    the eight recorded entries all share the unit cgroup. Without the pre-cap
    count the host would answer ``true`` for a list it never saw in full.
    """
    pids = [8992 + index for index in range(9)]
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=pids,
        socket_owners=[
            _owner(
                pid,
                (
                    "/system.slice/zombie.service"
                    if pid == 9000
                    else "/system.slice/action-server.service"
                ),
            )
            for pid in pids
        ],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert len(record["socket_pids"]) == 8
    assert record["socket_owner_count"] == 9
    assert record["socket_scan_complete"] is True
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_an_incomplete_socket_scan_is_undetermined() -> None:
    """A /proc scan that stopped early did not see every owner."""
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, "/system.slice/action-server.service"),
        ],
        socket_scan_complete=False,
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_scan_complete"] is False
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_missing_completeness_signals_are_undetermined() -> None:
    """A capture that did not report completeness cannot assert ``true``."""
    payload = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887],
        socket_owners=[
            _owner(9000, "/system.slice/action-server.service"),
            _owner(1887, "/system.slice/action-server.service"),
        ],
    )
    assert (
        guest_supervisor.layout_record(json.dumps(payload))["other_socket_owners_share_unit_cgroup"]
        is True
    )
    del payload["socket_scan_complete"]
    assert (
        guest_supervisor.layout_record(json.dumps(payload))["other_socket_owners_share_unit_cgroup"]
        is None
    )
    payload["socket_scan_complete"] = True
    del payload["socket_owner_count"]
    assert (
        guest_supervisor.layout_record(json.dumps(payload))["other_socket_owners_share_unit_cgroup"]
        is None
    )


def test_a_pid_recorded_twice_with_different_cgroups_is_undetermined() -> None:
    """List order must not decide which cgroup a repeated pid has."""
    in_unit = "/system.slice/action-server.service"
    outside = "/system.slice/zombie.service"
    base = _layout_payload(
        units=[{**_layout_payload()["units"][0], "main_pid": 9000}],
        socket_pids=[9000, 1887],
        socket_owners=[_owner(9000, in_unit), _owner(1887, outside), _owner(1887, in_unit)],
    )
    assert (
        guest_supervisor.layout_record(json.dumps(base))["other_socket_owners_share_unit_cgroup"]
        is None
    )
    base["socket_owners"] = [
        _owner(9000, in_unit),
        _owner(1887, in_unit),
        _owner(1887, outside),
    ]
    assert (
        guest_supervisor.layout_record(json.dumps(base))["other_socket_owners_share_unit_cgroup"]
        is None
    )
    # Agreeing duplicates carry no contradiction.
    base["socket_owners"] = [
        _owner(9000, in_unit),
        _owner(1887, in_unit),
        _owner(1887, in_unit),
    ]
    assert (
        guest_supervisor.layout_record(json.dumps(base))["other_socket_owners_share_unit_cgroup"]
        is True
    )


def test_layout_record_caps_every_list() -> None:
    payload = _layout_payload(
        units=[
            {**_layout_payload()["units"][0], "unit": f"u{index}.service"} for index in range(20)
        ],
        ancestry=[{"pid": 1000 + index, "ppid": 1, "comm": "x"} for index in range(20)],
        socket_pids=[10000 + index for index in range(20)],
        socket_owners=[
            _owner(10000 + index, "/system.slice/action-server.service") for index in range(20)
        ],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert len(record["units"]) == 8
    assert len(record["ancestry"]) == 16
    assert len(record["socket_pids"]) == 8
    assert len(record["socket_owners"]) == 8
    # The pre-cap count survives the cap, so the derived answer can see that
    # the list it compared is incomplete.
    assert record["socket_owner_count"] == 20
    assert record["other_socket_owners_share_unit_cgroup"] is None


def _write_listeners(
    proc: Path, inodes_by_pid: dict[int, int], *, bind: str = "00000000"
) -> None:
    """Point each pid's fd at its own listening socket and publish the inodes.

    ``bind`` is the /proc local-address word, host byte order: the default is
    the wildcard every earlier test assumed, and ``0100007F`` is the loopback
    bind a restart must never accept as host-reachable. A 32-hex-digit value is
    a v6 address and goes into ``tcp6``, which is the table Linux reports a
    v4-mapped (``::ffff:a.b.c.d``) bind in.
    """
    for pid, inode in inodes_by_pid.items():
        directory = proc / str(pid) / "fd"
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / "3"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(f"socket:[{inode}]")
    net = proc / "net"
    net.mkdir(exist_ok=True)
    lines = ["  sl  local_address rem_address   st"]
    for inode in inodes_by_pid.values():
        lines.append(f"   0: {bind}:1388 00000000:0000 0A 00:00000000 00000000 0 0 1000 {inode}")
    header = "  sl\n"
    body = "\n".join(lines) + "\n"
    if len(bind) == 32:
        (net / "tcp").write_text(header)
        (net / "tcp6").write_text(body)
    else:
        (net / "tcp").write_text(body)
        (net / "tcp6").write_text(header)


def _execute_layout_script(
    tmp_path: Path, *, listening: bool, owner_cgroups: dict[int, str] | None = None
) -> dict[str, Any]:
    """Run the shipped LAYOUT_SOURCE against a redirected /proc and fake systemd.

    ``owner_cgroups`` is the observable ground truth for several socket owners:
    one :5000 listener per pid, each in the cgroup given. The default is the
    single-owner shape the merged diagnostic first shipped.
    """
    cgroup = "/system.slice/action-server.service"
    owners = owner_cgroups if owner_cgroups is not None else {9000: cgroup}
    if not listening:
        owners = {}
    for pid, (name, ppid) in {
        9000: ("python3", 8000),
        8000: ("python3", 1),
        1: ("systemd", 0),
    }.items():
        directory = tmp_path / "proc" / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text(f"Name:\t{name}\nPPid:\t{ppid}\n")
        (directory / "cgroup").write_text(f"0::{owners.get(pid, cgroup)}\n")
        (directory / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0--do-not-emit\0")
    for pid, group in owners.items():
        if pid == 9000:
            continue
        directory = tmp_path / "proc" / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text("Name:\tpython3\nPPid:\t8000\n")
        (directory / "cgroup").write_text(f"0::{group}\n")
        (directory / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    if owners:
        _write_listeners(tmp_path / "proc", {pid: 4242 + index for index, pid in enumerate(owners)})
    else:
        net = tmp_path / "proc" / "net"
        net.mkdir()
        (net / "tcp").write_text("  sl  local_address rem_address   st\n")
        (net / "tcp6").write_text("  sl\n")

    def check_output(args: Any, **kwargs: Any) -> str:
        return "\n".join(
            (
                "MainPID=8001",
                "ExecStart=/usr/bin/python3 /home/user/server/main.py",
                "Type=simple",
                "ControlGroup=/system.slice/action-server.service",
                "SubState=running",
                "LoadState=loaded",
            )
        )

    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(getppid=lambda: 8000, readlink=os.readlink),
        "subprocess": SimpleNamespace(check_output=check_output),
        "sys": SimpleNamespace(argv=["-c", "5000"]),
        "time": SimpleNamespace(monotonic=time.monotonic),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    output: io.StringIO = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(
            guest_supervisor.LAYOUT_SOURCE,
            {"__builtins__": {**vars(builtins), "__import__": importer}},
        )
    return json.loads(output.getvalue())


def test_layout_script_reports_the_socket_owner_and_never_argv(tmp_path: Path) -> None:
    payload = _execute_layout_script(tmp_path, listening=True)
    assert payload["socket_pids"] == [9000]
    assert payload["server"]["pid"] == 9000
    assert [member["pid"] for member in payload["ancestry"]] == [9000, 8000, 1]
    assert payload["units"][0]["main_pid"] == 8001
    assert payload["socket_owners"] == [
        {
            "pid": 9000,
            "ppid": 8000,
            "comm": "python3",
            "cgroup": "/system.slice/action-server.service",
        }
    ]
    # The script emits no derived claim and no argv.
    assert set(payload) == {
        "schema_version",
        "socket_pids",
        "socket_owners",
        "socket_owner_count",
        "socket_scan_complete",
        "server",
        "ancestry",
        "units",
    }
    assert payload["socket_owner_count"] == 1
    assert payload["socket_scan_complete"] is True
    assert "--do-not-emit" not in json.dumps(payload)
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "ok"
    assert record["pid_source"] == "socket"
    assert record["port"] == guest_supervisor.ACTION_SERVER_PORT
    assert record["socket_owned_by_unit"] is False


def test_layout_script_records_every_socket_owner(tmp_path: Path) -> None:
    """Several owners, one outside the unit cgroup: the derived answer is false."""
    payload = _execute_layout_script(
        tmp_path,
        listening=True,
        owner_cgroups={
            8900: "/system.slice/action-server.service",
            9000: "/system.slice/zombie.service",
        },
    )
    assert payload["socket_pids"] == [8900, 9000]
    # socket_pids[0] is the chosen server; the other owner is the differing one.
    assert payload["server"]["pid"] == 8900
    assert {owner["pid"]: owner["cgroup"] for owner in payload["socket_owners"]} == {
        8900: "/system.slice/action-server.service",
        9000: "/system.slice/zombie.service",
    }
    assert payload["socket_owner_count"] == 2
    assert payload["socket_scan_complete"] is True
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    # The fake unit's MainPID (8001) is not the server, so the older claim stays
    # false; the cgroup comparison still identifies the unit by ControlGroup.
    assert record["socket_owned_by_unit"] is False
    assert record["other_socket_owners_share_unit_cgroup"] is False


def test_layout_script_caps_socket_owner_detail(tmp_path: Path) -> None:
    """A huge owner set is cut by the same cap as ``socket_pids``."""
    payload = _execute_layout_script(
        tmp_path,
        listening=True,
        owner_cgroups={9000 - index: "/system.slice/action-server.service" for index in range(12)},
    )
    assert len(payload["socket_pids"]) == 8
    assert len(payload["socket_owners"]) == 8
    assert [owner["pid"] for owner in payload["socket_owners"]] == payload["socket_pids"]
    # The pre-cap count tells the host this list is incomplete.
    assert payload["socket_owner_count"] == 12
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_a_hidden_ninth_socket_owner_is_undetermined(tmp_path: Path) -> None:
    """R2: the ninth holder is outside the cap and outside the unit cgroup.

    The eight recorded owners all share the unit cgroup, so comparing only the
    printed list would say ``true``. The guest's pre-cap count is what keeps
    the hidden holder from being invisible.
    """
    payload = _execute_layout_script(
        tmp_path,
        listening=True,
        owner_cgroups={
            9000
            - index: (
                "/system.slice/zombie.service"
                if index == 0
                else "/system.slice/action-server.service"
            )
            for index in range(9)
        },
    )
    assert payload["socket_owner_count"] == 9
    assert payload["socket_scan_complete"] is True
    assert len(payload["socket_pids"]) == 8
    # 9000 is the highest pid and is the one cut by the cap.
    assert 9000 not in payload["socket_pids"]
    assert all(
        owner["cgroup"] == "/system.slice/action-server.service"
        for owner in payload["socket_owners"]
    )
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    assert record["other_socket_owners_share_unit_cgroup"] is None


def test_layout_script_falls_back_to_ancestry_when_nothing_listens(tmp_path: Path) -> None:
    payload = _execute_layout_script(tmp_path, listening=False)
    assert payload["socket_pids"] == []
    assert payload["server"]["pid"] == 8000
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    assert record["pid_source"] == "ancestry"
    assert record["socket_owned_by_unit"] is None


def test_ownership_matches_any_listening_socket() -> None:
    """R2-2: a MainPID owning ANY observed listener is ownership."""
    payload = _layout_payload(
        socket_pids=[9000, 9001],
        server={"pid": 9000, "ppid": 8000, "comm": "python3", "cgroup": "/system.slice/x"},
        units=[{**_layout_payload()["units"][0], "main_pid": 9001}],
    )
    record = guest_supervisor.layout_record(json.dumps(payload))
    assert record["socket_owned_by_unit"] is True
    assert record["pid_source"] == "socket"


def test_ownership_is_derived_from_the_capped_unit_list() -> None:
    """R2-3: the claim must be reproducible from the units the record carries."""
    unit = _layout_payload()["units"][0]
    units = [{**unit, "unit": f"u{index}.service", "main_pid": 8001} for index in range(8)]
    units.append({**unit, "unit": "ninth.service", "main_pid": 9000})
    record = guest_supervisor.layout_record(json.dumps(_layout_payload(units=units)))
    assert len(record["units"]) == 8
    assert all(item["unit"] != "ninth.service" for item in record["units"])
    assert record["socket_owned_by_unit"] is False


def test_pid_source_ancestry_needs_ancestry_evidence() -> None:
    """R2-4: a lone server record is not an observed ancestry."""
    server = {"pid": 7000, "ppid": 1, "comm": "python3", "cgroup": "/system.slice"}
    record = guest_supervisor.layout_record(
        json.dumps(_layout_payload(socket_pids=[], server=server, ancestry=[]))
    )
    assert record["pid_source"] is None

    # An ancestry whose head is a different process proves nothing either.
    record = guest_supervisor.layout_record(
        json.dumps(
            _layout_payload(
                socket_pids=[],
                server=server,
                ancestry=[{"pid": 8000, "ppid": 1, "comm": "python3"}],
            )
        )
    )
    assert record["pid_source"] is None

    # The truthful shape: ancestry starts at the recorded server proc.
    record = guest_supervisor.layout_record(
        json.dumps(
            _layout_payload(
                socket_pids=[],
                server=server,
                ancestry=[{"pid": 7000, "ppid": 1, "comm": "python3"}],
            )
        )
    )
    assert record["pid_source"] == "ancestry"


def _execute_slow_layout_script(
    tmp_path: Path, *, unit_count: int, delay: float, owner_count: int = 1
) -> tuple[dict[str, Any], float]:
    """Run LAYOUT_SOURCE with a slow systemctl and a distinct unit per ancestry hop.

    ``owner_count`` adds that many :5000 listeners beside the unit processes,
    all in the unit cgroup, so the per-owner reads run while the unit loop is
    already under deadline pressure -- the budget claim covers both.
    """
    pids = [9000 - index for index in range(unit_count)]
    for index, pid in enumerate(pids):
        directory = tmp_path / "proc" / str(pid)
        directory.mkdir(parents=True)
        ppid = pids[index + 1] if index + 1 < len(pids) else 1
        (directory / "status").write_text(f"Name:\tpython3\nPPid:\t{ppid}\n")
        (directory / "cgroup").write_text(f"0::/system.slice/u{index}.service\n")
        (directory / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    root = tmp_path / "proc" / "1"
    root.mkdir()
    (root / "status").write_text("Name:\tsystemd\nPPid:\t0\n")
    (root / "cgroup").write_text("0::/init.scope\n")
    owner_pids = [pids[0]] + [9500 + index for index in range(owner_count - 1)]
    for pid in owner_pids[1:]:
        directory = tmp_path / "proc" / str(pid)
        directory.mkdir(parents=True)
        (directory / "status").write_text("Name:\tpython3\nPPid:\t8000\n")
        (directory / "cgroup").write_text("0::/system.slice/u0.service\n")
        (directory / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    _write_listeners(tmp_path / "proc", {pid: 4242 + index for index, pid in enumerate(owner_pids)})

    def check_output(args: Any, **kwargs: Any) -> str:
        # Honour the script's own per-call timeout: an in-flight call can only
        # consume the time left before DEADLINE.
        time.sleep(min(delay, kwargs.get("timeout", delay)))
        return "\n".join(
            (
                "MainPID=8001",
                "ExecStart=/usr/bin/python3 /home/user/server/main.py",
                "Type=simple",
                f"ControlGroup=/system.slice/{args[2]}",
                "SubState=running",
                "LoadState=loaded",
            )
        )

    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(getppid=lambda: pids[0], readlink=os.readlink),
        "subprocess": SimpleNamespace(check_output=check_output),
        "sys": SimpleNamespace(argv=["-c", "5000"]),
        "time": SimpleNamespace(monotonic=time.monotonic),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    started = time.monotonic()
    output: io.StringIO = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(
            guest_supervisor.LAYOUT_SOURCE,
            {"__builtins__": {**vars(builtins), "__import__": importer}},
        )
    return json.loads(output.getvalue()), time.monotonic() - started


def test_a_slow_systemctl_still_prints_a_complete_payload(tmp_path: Path) -> None:
    """R2-1: the unit loop must stop at the deadline and still print JSON.

    With eight 0.8 s units the ungated loop runs ~6.4 s and is killed by the
    3.5 s body timeout with NO output. Gated, it prints what it gathered just
    after 3 s. This test asserts the payload, not just a fast return.
    """
    payload, elapsed = _execute_slow_layout_script(tmp_path, unit_count=8, delay=0.8)
    assert payload["schema_version"] == 1
    assert 1 <= len(payload["units"]) <= 8
    # The fixture must actually put the loop under pressure: with fewer than two
    # units the gate never fires and this test would prove nothing.
    assert 2 <= len(payload["units"]) < 8
    assert elapsed < guest_supervisor.LAYOUT_GUEST_DEADLINE_S
    # A real layout record can be built from the printed payload.
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "ok"
    assert record["units"] == payload["units"]


def test_several_socket_owners_still_print_inside_the_guest_deadline(
    tmp_path: Path,
) -> None:
    """Per-owner reads must not push the capture past the shared deadline.

    Four socket owners beside the same eight 0.8 s unit queries as the bounded
    test above: the payload still prints before LAYOUT_GUEST_DEADLINE_S, with
    an entry per owner, and the host derives the agreement from that payload.
    """
    payload, elapsed = _execute_slow_layout_script(tmp_path, unit_count=8, delay=0.8, owner_count=4)
    assert payload["socket_pids"] == [9000, 9500, 9501, 9502]
    assert [owner["pid"] for owner in payload["socket_owners"]] == [9000, 9500, 9501, 9502]
    assert all(owner["cgroup"] == "/system.slice/u0.service" for owner in payload["socket_owners"])
    assert elapsed < guest_supervisor.LAYOUT_GUEST_DEADLINE_S
    record = guest_supervisor.layout_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "ok"
    assert record["other_socket_owners_share_unit_cgroup"] is True


def _add_socket_holder(tmp_path: Path, pid: int, inode: int, cgroup: str) -> None:
    """Add one more :5000 holder beside the fixture's MainPID 8000."""
    directory = tmp_path / "proc" / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    (directory / "cgroup").write_text(f"0::{cgroup}\n")
    _write_listeners(tmp_path / "proc", {8000: 4242, pid: inode})


def test_install_accepts_the_real_two_holder_layout_when_the_mainpid_owns_one(
    tmp_path: Path,
) -> None:
    """The AMI's socket_pids [1637, 1887] must not be read as "not ours".

    The fixed check anchors on the unit's MainPID being ONE of the socket's
    owners. The second holder is a forked server in the same cgroup, so the
    explicit cgroup-wide kill in the restart path can still reach it.
    """
    install, calls = _script_guest(tmp_path, ppid=8100)
    _add_socket_holder(tmp_path, 8100, 4243, "/system.slice/action-server.service")
    install()
    policy = (
        tmp_path / "etc/systemd/system/action-server.service.d/90-lop-supervisor.conf"
    ).read_text()
    assert "KillMode=process" in policy
    # No restart at install time, even with a second holder present. The
    # watchdog is the only unit the install starts, and it is ours.
    assert not any(
        args[1] in {"restart", "start", "stop", "kill"}
        and guest_supervisor.WATCHDOG_UNIT not in args
        for args in calls
    )


def test_install_refuses_a_socket_owner_outside_the_unit_control_group(
    tmp_path: Path,
) -> None:
    """A holder outside the cgroup survives even an explicit all-process kill."""
    install, calls = _script_guest(tmp_path)
    _add_socket_holder(tmp_path, 8100, 4243, "/system.slice/zombie.service")
    with pytest.raises(RuntimeError, match=guest_supervisor.SOCKET_OWNER_UNREACHABLE):
        install()
    assert not (tmp_path / "etc").exists()
    assert not any(args[1] == "daemon-reload" for args in calls)


def test_install_refuses_when_nothing_listens_on_the_action_port(tmp_path: Path) -> None:
    install, _ = _script_guest(tmp_path)
    _write_listeners(tmp_path / "proc", {})
    with pytest.raises(RuntimeError, match=guest_supervisor.NO_LISTENING_SOCKET):
        install()
    assert not (tmp_path / "etc").exists()


def _write_stat(proc: Path, pid: int, ppid: int, start_ticks: int) -> None:
    """Publish one /proc/<pid>/stat with the fields the restart reads."""
    directory = proc / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["S", str(ppid)] + ["0"] * 17 + [str(start_ticks)]
    (directory / "stat").write_text(f"{pid} (python3) " + " ".join(fields) + "\n")


def _write_uptime(proc: Path, seconds: float) -> None:
    (proc / "uptime").write_text(f"{seconds:.2f} 0.00\n")


def _installer_receipt() -> dict[str, Any]:
    """Exactly the five keys CAPTURE_SOURCE persists; no report-only fields."""
    return {
        "schema_version": 1,
        "records": [_restart_record()],
        "captured_exits": 1,
        "baseline_nrestarts": 0,
        "baseline_active_enter": 100,
    }


def _serve(
    proc: Path, pid: int, inode: int, ppid: int, start_ticks: int, *, bind: str = "00000000"
) -> None:
    _write_listeners(proc, {pid: inode}, bind=bind)
    _write_stat(proc, pid, ppid, start_ticks)


def _execute_restart_script(
    tmp_path: Path,
    *,
    on_kill: Callable[[Path], None] | None = None,
    on_start: Callable[[Path], None] | None = None,
    main_pid_after: int = 9100,
    hide_net: bool = False,
    unreadable_pre: bool = False,
    hang_kill: bool = False,
    dispatch_argv: list[str] | None = None,
    seed_receipt: bool = True,
    reset_ok: bool = True,
    unit_failed: bool = True,
) -> tuple[dict[str, Any], list[Any], float]:
    """Run the shipped RESTART_SOURCE against a redirected /proc and fake systemd.

    The verdict is read from RESTART_RESULT_PATH, never stdout: in production
    the process that would carry stdout dies with the target unit, which is why
    the script is dispatched as a transient service.

    The virtual clock makes the bounded waits deterministic and instant while
    still proving the script cannot outlive its budget. The fake manager owns
    the observable state: handlers mutate the /proc tree exactly as a real
    kill/start would.
    """
    proc = tmp_path / "proc"
    _write_listeners(proc, {9000: 4242})
    _write_stat(proc, 9000, 1, 90000)
    _write_stat(proc, 1, 0, 1)
    _write_uptime(proc, 1000.0)
    if hide_net:
        (proc / "net" / "tcp").unlink()
        (proc / "net" / "tcp6").unlink()
    if unreadable_pre:
        (proc / "9000" / "fd").chmod(0o000)
    root = tmp_path / "var/lib/lop-action-supervisor"
    root.mkdir(parents=True, exist_ok=True)
    if seed_receipt:
        (root / "action-server.service.json").write_text(json.dumps(_installer_receipt()))
    calls: list[Any] = []
    # The fake manager owns the observable systemd state, including the two
    # facts the real one measured on 255: reset-failed ZEROES NRestarts, and a
    # start moves ActiveEnterTimestampMonotonic to a new value.
    state = {"main": 9000, "now": 0.0, "nrestarts": 1, "enter": 200}

    def check_output(args: Any, **kwargs: Any) -> str:
        calls.append(args)
        if args[1] == "reset-failed" and not reset_ok:
            raise OSError("start limit not cleared")
        if args[1] == "show":
            props = {}
            if "NRestarts" in args:
                props["NRestarts"] = str(state["nrestarts"])
            if "ActiveEnterTimestampMonotonic" in args:
                props["ActiveEnterTimestampMonotonic"] = str(state["enter"])
            if "ActiveState" in args:
                props["ActiveState"] = "failed" if unit_failed else "active"
            if not props:
                props["MainPID"] = str(state["main"])
            return "\n".join(k + "=" + v for k, v in props.items())
        if args[1] == "reset-failed":
            state["nrestarts"] = 0
            return ""
        if args[1] == "kill":
            if hang_kill:
                # Consume the whole per-call bound, then fail as a hung
                # subprocess would: the script must still produce its verdict.
                allowed = float(kwargs.get("timeout") or 0.0)
                state["now"] += allowed
                raise subprocess.TimeoutExpired(args, allowed)
            if on_kill is not None:
                on_kill(proc)
        if args[1] == "start":
            if on_start is not None:
                on_start(proc)
            state["main"] = main_pid_after
            state["enter"] += 1000
        return ""

    def sleep(seconds: float) -> None:
        state["now"] += seconds

    # MODEL THE REAL INTERPRETER, not the old seam. ``python3 -c <source> a b``
    # gives sys.argv == ['-c', 'a', 'b']: the source is NEVER an argv element.
    if dispatch_argv is not None:
        assert dispatch_argv[0] == "python3" and dispatch_argv[1] == "-c"
        source = dispatch_argv[2]
        argv = ["-c", *dispatch_argv[3:]]
    else:
        source = guest_supervisor.RESTART_SOURCE
        argv = [
            "-c",
            "action-server.service",
            "5000",
            guest_supervisor.RESTART_RESULT_PATH,
            "777",
        ]
    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(readlink=os.readlink, sysconf=lambda name: 100),
        "subprocess": SimpleNamespace(check_output=check_output),
        "sys": SimpleNamespace(argv=argv),
        "time": SimpleNamespace(monotonic=lambda: state["now"], sleep=sleep),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    exec(
        source,
        {"__builtins__": {**vars(builtins), "__import__": importer}},
    )
    result_path = tmp_path / str(guest_supervisor.RESTART_RESULT_PATH).lstrip("/")
    return json.loads(result_path.read_text()), calls, state["now"]


def test_a_live_unit_is_not_reset_and_keeps_its_restart_counters(
    tmp_path: Path,
) -> None:
    """reset-failed zeroes NRestarts on a LIVE unit too (measured on 255).

    The host reconciles ``systemd_nrestarts - baseline == captured_exits`` on
    every ordinary restart, so clearing counters there would break every
    host-initiated recovery. Only a failed unit has a latch to clear.
    """
    payload, calls, _ = _execute_restart_script(
        tmp_path,
        unit_failed=False,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    order = [args[1] for args in calls if args[1] in {"reset-failed", "kill", "start"}]
    assert order == ["kill", "start"], "no reset on a unit without a latch"
    assert payload["start_limit_cleared"] is True
    receipt = payload["receipt"]
    assert receipt["systemd_nrestarts"] - receipt["baseline_nrestarts"] == receipt["captured_exits"]


def test_the_restart_clears_the_start_limit_between_the_scan_and_the_kill(
    tmp_path: Path,
) -> None:
    """A unit latched ``start-limit-hit`` refuses ``start`` until reset-failed.

    QA measured this on systemd 255: the kill succeeded, the start was refused
    with "Start request repeated too quickly", and the server stayed down. The
    order is the contract: scan the holders, clear the latch, kill, start.
    """
    payload, calls, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    order = [args[1] for args in calls if args[1] in {"reset-failed", "kill", "start"}]
    assert order == ["reset-failed", "kill", "start"], "the latch clears before the start"
    assert payload["start_limit_cleared"] is True
    assert payload["new_main_pid"] == 9100
    # The manager fake ZEROED NRestarts on reset, exactly as systemd 255 does,
    # so the receipt must have been rebased in the same breath: counters and
    # records cleared, activation baseline moved to the new entry stamp.
    receipt = payload["receipt"]
    assert receipt["captured_exits"] == 0 and receipt["records"] == []
    assert receipt["baseline_nrestarts"] == 0
    assert receipt["systemd_nrestarts"] == 0
    assert receipt["systemd_active_enter"] == receipt["baseline_active_enter"] == 1200
    # The host's REPORT path must consume this generation without a mismatch;
    # that is the difference between "the unit is alive" and "input can post".
    assert guest_supervisor._receipt_records_value(receipt) == []


def test_restart_script_kills_all_then_starts_and_reports_a_fresh_mainpid(
    tmp_path: Path,
) -> None:
    payload, calls, elapsed = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    order = [args[1] for args in calls if args[1] in {"kill", "start"}]
    assert order == ["kill", "start"], "the explicit kill must precede the start"
    kill = next(args for args in calls if args[1] == "kill")
    assert "--kill-whom=all" in kill and "action-server.service" in kill
    assert payload["pre_restart_pids"] == [9000]
    assert payload["post_restart_pids"] == [9100]
    assert payload["new_main_pid"] == 9100
    assert payload["new_main_start_ticks"] == 101000
    assert payload["kill_uptime_ticks"] == 100000
    assert payload["token"] == 777
    assert elapsed < guest_supervisor.RESTART_SCRIPT_DEADLINE_S
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "verified"
    assert record["fresh"] is True
    assert record["listener_is_new_main_pid"] is True
    assert record["token"] == 777


def _installer_written_receipt(tmp_path: Path) -> Path:
    """Run the REAL install, then one REAL capture, and return the receipt file.

    The installer writes the five-key empty envelope; only the capture hook
    appends a record. Hand-building that file is how the report-only fields
    slipped into a fixture the guest never produces.
    """
    install, _ = _script_guest(tmp_path)
    install()
    path = tmp_path / "var/lib/lop-action-supervisor/action-server.service.json"
    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(environ={}),
        "subprocess": SimpleNamespace(
            run=lambda *a, **k: (_ for _ in ()).throw(OSError("journal unavailable"))
        ),
        "sys": SimpleNamespace(argv=["capture.py", "action-server.service"]),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    exec(
        guest_supervisor.CAPTURE_SOURCE,
        {"__builtins__": {**vars(builtins), "__import__": importer}},
    )
    return path


def test_restart_result_validates_the_receipt_the_installer_actually_writes(
    tmp_path: Path,
) -> None:
    """BLOCKER 2's fixture is the installer's file, not a hand-written one.

    The installer persists five keys; ``systemd_nrestarts`` and
    ``systemd_active_enter`` are report-only, added in memory. The restart
    must add them itself or the host can never validate the envelope.
    """
    path = _installer_written_receipt(tmp_path)
    on_disk = json.loads(path.read_text())
    assert set(on_disk) == {
        "schema_version",
        "records",
        "captured_exits",
        "baseline_nrestarts",
        "baseline_active_enter",
    }
    assert on_disk["captured_exits"] == 1
    # A LIVE unit: the host's ordinary bounce never resets, so the installer's
    # crash accounting survives and the restart only adds the report-only keys.
    payload, _, _ = _execute_restart_script(
        tmp_path,
        seed_receipt=False,
        unit_failed=False,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    assert payload["receipt"]["systemd_nrestarts"] == 1
    assert payload["receipt"]["systemd_active_enter"] == 1200
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "verified"
    assert record["latest_restart"] == on_disk["records"][-1]["restart"]


def test_restart_argument_order_is_one_contract() -> None:
    service = guest_supervisor.restart_service_argv("action-server.service", 5000, 777)
    assert guest_supervisor.RESTART_ARGUMENT_ORDER == (
        "unit",
        "port",
        "result_path",
        "token",
    )
    assert service[-4:] == [
        "action-server.service",
        "5000",
        guest_supervisor.RESTART_RESULT_PATH,
        "777",
    ]


def test_the_built_dispatch_executes_with_the_real_python_argv(tmp_path: Path) -> None:
    """BLOCKER: run the argv the REAL builder produces, through an exec stand-in.

    ``python3 -c <source> a b c`` gives sys.argv == ['-c', 'a', 'b', 'c']; the
    source is not an argv element. The old seam injected it at argv[1], which
    is why 135 green tests could not see the off-by-one.
    """
    built = guest_supervisor.restart_command("action-server.service", "pw", 777)
    shell = built[-1]
    service = guest_supervisor.restart_service_argv("action-server.service", 5000, 777)
    assert "systemd-run" in shell
    assert "--collect" in shell
    assert "--scope" not in shell, "a scope would put the restart back in the target cgroup"
    assert "--unit=lop-supervisor-restart-777" in shell
    assert "action-server.service 5000 %s 777" % guest_supervisor.RESTART_RESULT_PATH in shell
    payload, _, _ = _execute_restart_script(
        tmp_path,
        dispatch_argv=service[4:],
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "verified"
    assert record["token"] == 777
    assert record["fresh"] is True


def test_restart_script_reports_the_survivor_and_the_host_refuses_it(
    tmp_path: Path,
) -> None:
    """The zombie case, constructed: SIGTERM does not reap the old holder."""
    payload, _, _ = _execute_restart_script(
        tmp_path,
        on_kill=None,
        on_start=lambda proc: _write_stat(proc, 9100, 1, 101000),
        main_pid_after=9100,
    )
    assert payload["pre_restart_pids"] == [9000]
    assert payload["post_restart_pids"] == [9000]
    assert payload["new_main_pid"] == 9100
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False
    assert record["listener_is_new_main_pid"] is False


def test_restart_script_refuses_a_worker_forked_after_the_kill(tmp_path: Path) -> None:
    """A new pid is not fresh when a dying holder produced it.

    pre [9000]; the kill lands; the holder forks 9105 before it dies; systemd
    starts MainPID 9100. 9105 is absent from the pre set and started after the
    kill, but it is not a descendant of 9100 -- it serves pre-restart state.
    """
    payload, _, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _serve(proc, 9105, 4345, 9000, 100500),
        on_start=lambda proc: _write_stat(proc, 9100, 1, 101000),
        main_pid_after=9100,
    )
    assert payload["post_restart_pids"] == [9105]
    assert payload["post_restart_owners"][0]["ancestors"] == [9000]
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False


def test_restart_script_reports_a_forked_listener_not_the_mainpid(tmp_path: Path) -> None:
    payload, _, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: (
            _serve(proc, 9101, 4343, 9100, 101500),
            _write_stat(proc, 9100, 1, 101000),
        )[-1],
        main_pid_after=9100,
    )
    assert payload["post_restart_pids"] == [9101]
    assert payload["new_main_pid"] == 9100
    assert payload["post_restart_owners"][0]["ancestors"] == [9100]
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "verified"
    assert record["fresh"] is True
    assert record["listener_is_new_main_pid"] is False


def test_restart_script_is_bounded_when_no_listener_appears(tmp_path: Path) -> None:
    payload, calls, elapsed = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _write_stat(proc, 9100, 1, 101000),
        main_pid_after=9100,
    )
    assert payload["post_restart_pids"] == []
    assert elapsed <= guest_supervisor.RESTART_SCRIPT_DEADLINE_S
    assert any(args[1] == "start" for args in calls)
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "verification_timeout"
    assert record["fresh"] is False


def test_an_incomplete_pre_scan_refuses_before_killing_anything(tmp_path: Path) -> None:
    payload, calls, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        hide_net=True,
    )
    assert not any(args[1] in {"kill", "start"} for args in calls)
    assert payload["pre_scan_complete"] is False
    assert payload["post_scan_complete"] is False
    assert payload["kill_uptime_ticks"] == 0
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "incomplete_scan"


def _restart_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "token": 777,
        "unit": "action-server.service",
        "pre_restart_pids": [1637, 1887],
        "pre_restart_count": 2,
        "pre_scan_complete": True,
        "kill_uptime_ticks": 100000,
        "post_restart_pids": [1901],
        "post_restart_count": 1,
        "post_scan_complete": True,
        "post_restart_owners": [
            {"pid": 1901, "ppid": 1900, "start_ticks": 101000, "ancestors": [1900]}
        ],
        "new_main_pid": 1900,
        "new_main_start_ticks": 100500,
        "receipt": _envelope([_restart_record()]),
    }
    payload.update(overrides)
    return payload


def test_restart_record_refuses_a_stale_listener() -> None:
    record = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                post_restart_pids=[1887],
                post_restart_owners=[
                    {"pid": 1887, "ppid": 1, "start_ticks": 90000, "ancestors": []}
                ],
            )
        )
    )
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False
    assert record["listener_is_new_main_pid"] is False
    assert record["post_restart_owner"] == 1887


def test_restart_record_refuses_a_worker_forked_after_the_kill() -> None:
    """The MAJOR: postdates the kill but is not the new MainPID's descendant."""
    record = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                post_restart_pids=[1905],
                post_restart_owners=[
                    {"pid": 1905, "ppid": 1887, "start_ticks": 100500, "ancestors": [1887]}
                ],
            )
        )
    )
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False


def test_restart_record_accepts_a_fresh_mainpid() -> None:
    record = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                post_restart_pids=[1901],
                post_restart_owners=[
                    {"pid": 1901, "ppid": 1, "start_ticks": 101000, "ancestors": [1]}
                ],
                new_main_pid=1901,
                new_main_start_ticks=101000,
            )
        )
    )
    assert record["status"] == "verified"
    assert record["fresh"] is True
    assert record["listener_is_new_main_pid"] is True
    assert record["latest_restart"] == 42


def test_restart_record_accepts_a_forked_child_with_the_claim_recorded() -> None:
    record = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                post_restart_pids=[1901],
                post_restart_owners=[
                    {"pid": 1901, "ppid": 1900, "start_ticks": 101000, "ancestors": [1900]}
                ],
                new_main_pid=1900,
                new_main_start_ticks=100500,
            )
        )
    )
    assert record["status"] == "verified"
    assert record["fresh"] is True
    assert record["listener_is_new_main_pid"] is False


def test_restart_record_refuses_a_descendant_started_before_the_kill() -> None:
    record = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                post_restart_owners=[
                    {"pid": 1901, "ppid": 1900, "start_ticks": 99999, "ancestors": [1900]}
                ],
            )
        )
    )
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False


def test_listener_postdates_restart_covers_absent_and_duplicate_pids() -> None:
    owner = {"pid": 1901, "ppid": 1900, "start_ticks": 101000, "ancestors": [1900]}
    assert guest_supervisor.listener_postdates_restart([], 1900, 100500, 100000) is False
    assert guest_supervisor.listener_postdates_restart([owner], 0, 0, 100000) is False
    assert guest_supervisor.listener_postdates_restart([owner], 1901, 101000, 100000) is True
    assert guest_supervisor.listener_postdates_restart([owner], 1901, 99000, 100000) is False
    assert guest_supervisor.listener_postdates_restart([owner], 1900, 100500, 100000) is True
    assert guest_supervisor.listener_postdates_restart([owner, owner], 1900, 100500, 100000) is True
    stale = {"pid": 1887, "ppid": 1, "start_ticks": 90000, "ancestors": []}
    assert (
        guest_supervisor.listener_postdates_restart([owner, stale], 1900, 100500, 100000) is False
    )


def test_restart_record_fails_closed_on_an_incomplete_owner_set() -> None:
    record = guest_supervisor.restart_record(
        json.dumps(_restart_payload(pre_restart_pids=list(range(1, 9)), pre_restart_count=9))
    )
    assert record["status"] == "incomplete_scan"
    assert record["fresh"] is False


def test_restart_record_ignores_guest_claims_about_freshness() -> None:
    payload = _restart_payload(
        post_restart_pids=[1887],
        post_restart_owners=[{"pid": 1887, "ppid": 1, "start_ticks": 90000, "ancestors": []}],
        fresh=True,
        status="verified",
    )
    record = guest_supervisor.restart_record(json.dumps(payload))
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False


def test_restart_record_requires_a_reconcilable_receipt() -> None:
    broken = _installer_receipt()
    record = guest_supervisor.restart_record(json.dumps(_restart_payload(receipt=broken)))
    assert record["status"] == "receipt_unavailable"
    assert record["fresh"] is True
    assert record["latest_restart"] is None


def _recovering_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: guest_disk.CommandResult,
    *,
    restart_error: Exception | None = None,
    results: list[guest_disk.CommandResult] | None = None,
) -> tuple[Any, list[str], list[str]]:
    """A provider whose receipt shows one restart and whose bounce is scripted.

    ``events`` records the ORDER of the two-channel restart: a dispatch, then
    one or more result reads. The handler dying mid-request is modelled by
    ``restart_error``; the restart must still be recovered from the file.
    """
    stubs = _Stubs()
    provider = _provider_over(stubs, tmp_path)
    provider._supervisor_unit = "action-server.service"
    monkeypatch.setattr(
        provider,
        "_probe_control_server_outcome",
        lambda timeout: (True, {"answered": True}),
    )
    monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
    monkeypatch.setattr(aws_mod.time, "monotonic", stubs.monotonic)
    events: list[str] = []
    posts: list[str] = []
    pending = list(results) if results is not None else None

    def command(argv: Any, timeout: float) -> guest_disk.CommandResult:
        script = " ".join(str(part) for part in argv)
        if _is_restart_command(argv):
            events.append("dispatch")
            if restart_error is not None:
                raise restart_error
            return guest_disk.CommandResult(0, "", "")
        if _is_restart_result_command(argv):
            events.append("result")
            if pending is not None:
                return pending.pop(0) if pending else guest_disk.CommandResult(1, "", "")
            return result
        if "/var/lib/lop-action-supervisor" in script:
            return _receipt([_restart_record()])
        posts.append(script)
        return guest_disk.CommandResult(0, "", "")

    monkeypatch.setattr(provider, "_run_guest_command", command)
    return provider, posts, events


@pytest.mark.asyncio
async def test_a_verified_restart_posts_the_next_action_only_after_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, posts, events = _recovering_provider(tmp_path, monkeypatch, _restart_document())
    with pytest.raises(GuestServerRecovered):
        await provider.execute(["must_not_post()"])
    assert posts == [], "the refused batch must not be replayed"
    await provider.execute(["after_rebind()"])
    assert len(posts) == 1
    encoded = posts[0].split()[-1]
    assert base64.b64decode(encoded).decode() == "after_rebind()"
    assert provider._supervisor_restart == 42
    assert events == ["dispatch", "result"]
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["recovered"] is True
    assert evidence["restart_verification"]["status"] == "verified"


@pytest.mark.asyncio
async def test_the_restart_survives_the_handler_dying_mid_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKER 1: the kill closes the connection that carried the request.

    The dispatch raises (the Flask process serving it was signalled); the
    verdict still arrives through the detached result file and recovery is
    claimed only from that. A design that needed the dispatch response would
    return ``command_failed`` here.
    """
    provider, posts, events = _recovering_provider(
        tmp_path,
        monkeypatch,
        _restart_document(),
        restart_error=ConnectionResetError("handler killed"),
    )
    with pytest.raises(GuestServerRecovered):
        await provider.execute(["must_not_post()"])
    assert posts == []
    assert events == ["dispatch", "result"]
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["recovered"] is True
    assert evidence["restart_verification"]["status"] == "verified"


@pytest.mark.asyncio
async def test_a_stale_result_file_is_not_read_as_the_current_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, posts, events = _recovering_provider(
        tmp_path,
        monkeypatch,
        _restart_document(),
        results=[_restart_document(token=1), _restart_document()],
    )
    with pytest.raises(GuestServerRecovered):
        await provider.execute(["must_not_post()"])
    assert posts == []
    assert events == ["dispatch", "result", "result"]
    assert provider._supervisor_restart == 42


@pytest.mark.asyncio
async def test_a_surviving_stale_holder_fails_closed_and_never_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The zombie: a receipt says restarted, the old holder still answers."""
    stale = _restart_document(
        post=(1887,),
        main_pid=1900,
        main_start_ticks=101000,
        owners=[{"pid": 1887, "ppid": 1, "start_ticks": 90000, "ancestors": []}],
    )
    provider, posts, _ = _recovering_provider(tmp_path, monkeypatch, stale)
    with pytest.raises(aws_mod.GuestExecutionError):
        await provider.execute(["must_not_post()"])
    assert posts == [], "input must not reach an unverified listener"
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["recovered"] is False
    assert evidence["cause"] == "post-restart listener predates the restart"
    assert evidence["cause_status"] == "stale_listener"
    assert evidence["restart_verification"]["post_restart_owner"] == 1887
    assert evidence["remaining_batch"] == "not_sent"


@pytest.mark.asyncio
async def test_a_worker_forked_after_the_kill_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forked_late = _restart_document(
        post=(1905,),
        main_pid=1900,
        main_start_ticks=101000,
        owners=[{"pid": 1905, "ppid": 1887, "start_ticks": 100500, "ancestors": [1887]}],
    )
    provider, posts, _ = _recovering_provider(tmp_path, monkeypatch, forked_late)
    with pytest.raises(aws_mod.GuestExecutionError):
        await provider.execute(["must_not_post()"])
    assert posts == []
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["restart_verification"]["status"] == "stale_listener"


@pytest.mark.asyncio
async def test_a_forked_listener_is_accepted_after_an_explicit_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forked = _restart_document(
        post=(1901,),
        main_pid=1900,
        main_start_ticks=100500,
        owners=[{"pid": 1901, "ppid": 1900, "start_ticks": 101000, "ancestors": [1900]}],
    )
    provider, posts, _ = _recovering_provider(tmp_path, monkeypatch, forked)
    with pytest.raises(GuestServerRecovered):
        await provider.execute(["must_not_post()"])
    await provider.execute(["after_rebind()"])
    assert len(posts) == 1
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["restart_verification"]["status"] == "verified"
    assert evidence["restart_verification"]["listener_is_new_main_pid"] is False


@pytest.mark.asyncio
async def test_restart_result_timeout_is_bounded_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, posts, events = _recovering_provider(
        tmp_path, monkeypatch, _restart_document(), results=[]
    )
    with pytest.raises(aws_mod.GuestExecutionError):
        await provider.execute(["must_not_post()"])
    assert posts == []
    assert events[0] == "dispatch"
    assert events.count("result") > 1
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["restart_verification"]["status"] == "result_unavailable"
    assert evidence["cause"] == "restart result was not readable"
    assert evidence["recovered"] is False


@pytest.mark.asyncio
async def test_restart_dispatch_failure_fails_closed_without_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        events: list[str] = []
        posts: list[str] = []

        def command(argv: Any, timeout: float) -> guest_disk.CommandResult:
            if _is_restart_command(argv):
                events.append("dispatch")
                return guest_disk.CommandResult(1, "", "")
            if _is_restart_result_command(argv):
                events.append("result")
                return _restart_document()
            if "/var/lib/lop-action-supervisor" in str(argv):
                return _receipt([_restart_record()])
            posts.append(" ".join(str(part) for part in argv))
            return guest_disk.CommandResult(0, "", "")

        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(aws_mod.GuestExecutionError):
            await provider.execute(["must_not_post()"])
        assert posts == []
        assert events == ["dispatch"]
        assert provider._cache_root is not None
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["restart_verification"]["status"] == "command_failed"
        assert evidence["cause"] == "restart dispatch failed"


def test_the_restart_script_budgets_match_the_host_constants() -> None:
    """The guest literals and the host's bounds are one fact.

    The script deadline is deliberately inside the guest's subprocess deadline
    (measured from launch, not from script start), and the install scan is
    bounded so it cannot spend the install command's whole 5 s budget.
    """
    source = guest_supervisor.RESTART_SOURCE
    assert "DEADLINE = time.monotonic() + %r" % guest_supervisor.RESTART_SCRIPT_DEADLINE_S in source
    assert "KILL_GRACE = %r" % guest_supervisor.RESTART_KILL_GRACE_S in source
    assert "VERIFY = %r" % guest_supervisor.RESTART_VERIFY_S in source
    assert (
        "SCAN_DEADLINE = time.monotonic() + %r" % guest_supervisor.INSTALL_SCAN_DEADLINE_S
        in guest_supervisor.INSTALL_SOURCE
    )
    assert guest_supervisor.RESTART_SCRIPT_DEADLINE_S < (guest_supervisor.RESTART_GUEST_DEADLINE_S)
    assert guest_supervisor.INSTALL_SCAN_DEADLINE_S < guest_deadline_for(15.0)


def test_install_refuses_an_unreadable_socket_holder(tmp_path: Path) -> None:
    """A live holder we cannot inspect makes the scan incomplete, not absent.

    The old scanner swallowed the per-process OSError and returned
    ``complete=True``, so an EACCES fd table simply disappeared from the owner
    set -- and the install authorized a restart that could not reap it.
    """
    install, _ = _script_guest(tmp_path)
    _add_socket_holder(tmp_path, 8100, 4243, "/system.slice/action-server.service")
    fd = tmp_path / "proc/8100/fd"
    fd.chmod(0o000)
    try:
        with pytest.raises(RuntimeError, match=guest_supervisor.SOCKET_SCAN_INCOMPLETE):
            install()
    finally:
        fd.chmod(0o700)
    assert not (tmp_path / "etc").exists()


def test_restart_script_refuses_an_unreadable_pre_holder_without_killing(
    tmp_path: Path,
) -> None:
    payload, calls, _ = _execute_restart_script(tmp_path, unreadable_pre=True)
    assert not any(args[1] in {"kill", "start"} for args in calls)
    assert payload["pre_scan_complete"] is False
    assert payload["kill_uptime_ticks"] == 0
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "incomplete_scan"
    assert record["fresh"] is False


def test_restart_script_marks_an_unreadable_post_holder_incomplete(tmp_path: Path) -> None:
    def start(proc: Path) -> None:
        _serve(proc, 9100, 4343, 1, 101000)
        (proc / "9100" / "fd").chmod(0o000)

    try:
        payload, _, _ = _execute_restart_script(
            tmp_path,
            on_kill=lambda proc: _write_listeners(proc, {}),
            on_start=start,
            main_pid_after=9100,
        )
    finally:
        (tmp_path / "proc/9100/fd").chmod(0o700)
    assert payload["post_restart_pids"] == []
    assert payload["post_scan_complete"] is False
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "incomplete_scan"
    assert record["fresh"] is False


def test_a_hung_kill_still_writes_its_verdict_inside_the_guest_budget(
    tmp_path: Path,
) -> None:
    """QA measured 7.1 s against a 7.0 s guest deadline on the old script.

    The kill consumes its whole per-call bound and times out. The script must
    still reach the result file before the guest's subprocess deadline, with
    the designed stale_listener verdict rather than a lost command.
    """
    payload, _, elapsed = _execute_restart_script(
        tmp_path,
        hang_kill=True,
        on_start=lambda proc: _write_stat(proc, 9100, 1, 101000),
        main_pid_after=9100,
    )
    assert elapsed < guest_supervisor.RESTART_GUEST_DEADLINE_S
    assert payload["post_restart_pids"] == [9000]
    record = guest_supervisor.restart_record("escalation=agentless\n" + json.dumps(payload))
    assert record["status"] == "stale_listener"
    assert record["fresh"] is False


def test_an_empty_pre_set_is_proved_by_start_time_and_ancestry() -> None:
    """An empty pre set is not automatically fresh, and not automatically lost.

    A dead server has no pre holder; the restart must still be able to start
    it. An unseen holder is refused by the postdate/ancestry rule instead:
    its old start time, or its descent from a dying holder, fails the check.
    """
    empty = {"pre_restart_pids": [], "pre_restart_count": 0}
    ok = guest_supervisor.restart_record(json.dumps(_restart_payload(**empty)))
    assert ok["status"] == "verified"
    stale = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                **empty,
                post_restart_pids=[1887],
                post_restart_owners=[
                    {"pid": 1887, "ppid": 1, "start_ticks": 90000, "ancestors": []}
                ],
            )
        )
    )
    assert stale["status"] == "stale_listener"
    late = guest_supervisor.restart_record(
        json.dumps(
            _restart_payload(
                **empty,
                post_restart_pids=[1905],
                post_restart_owners=[
                    {"pid": 1905, "ppid": 1887, "start_ticks": 100500, "ancestors": [1887]}
                ],
            )
        )
    )
    assert late["status"] == "stale_listener"


# ---------------------------------------------------------------------------
# The guest-local watchdog: its own unit, its own cgroup, its own recovery path.
# ---------------------------------------------------------------------------


class _StopWatchdog(Exception):
    """Ends the script's infinite loop from the virtual clock, never from code."""


def _execute_watchdog(
    tmp_path: Path,
    *,
    active_states: list[str],
    listening: list[bool],
    cycles: int,
    answering: list[bool] | None = None,
    failures: dict[int, Exception] | None = None,
    failures_on: dict[str, Exception] | None = None,
    seed: dict[str, Any] | None = None,
    restart_effect: str = "success",
    v6: bool = False,
    reachable_host: str = "127.0.0.1",
    bind_address: str = "00000000",
    restart_bind: str = "00000000",
    clock_start: float = 0.0,
    target_unit: str = "action-server.service",
    probe_log: list[tuple[str, int, float]] | None = None,
    probe_status: bytes = b"HTTP/1.0 200 OK\r\n",
    probe_timeouts: bool = False,
) -> tuple[dict[str, Any], list[Any], float]:
    """Run the shipped WATCHDOG_SOURCE against a redirected /proc and fake manager.

    The fake clock advances one ``INTERVAL_S`` per cycle, so the rate limits are
    exercised as real elapsed time instead of patched thresholds. ``answering``
    drives the one-shot HTTP liveness probe (a wedged listener accepts at the
    kernel level and then returns nothing); ``probe_timeouts`` makes that probe
    raise the ``TimeoutError`` a BUSY server produces, which is the shape that
    must never be mistaken for death. ``failures`` makes the named cycle's
    ``systemctl`` raise; ``failures_on`` targets one verb.

    ``restart_effect`` chooses what the transient restart does on the fake
    guest; the verdict written to the result file is always produced by the
    REAL ``RESTART_SOURCE`` (``_real_restart_verdict``), never hand-built: a
    seam that invents a field the producer does not write is exactly the class
    of defect this work keeps removing.
    """
    root = tmp_path / "var/lib/lop-action-supervisor"
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "watchdog.json"
    result_path = root / "restart-result.json"
    restart_script = root / "restart.py"
    restart_script.write_text(guest_supervisor.RESTART_SOURCE)
    if seed is not None:
        record_path.write_text(json.dumps(seed))
    calls: list[Any] = []
    probes: list[tuple[str, int, float]] = []
    clock = {"now": clock_start, "cycle": 0}
    failures = failures or {}
    failures_on = failures_on or {}
    answers = answering if answering is not None else [True] * cycles

    def publish(listening_now: bool, answers_now: bool) -> None:
        net = tmp_path / "proc/net"
        net.mkdir(parents=True, exist_ok=True)
        header = "  sl  local_address rem_address   st\n"
        row = f"   0: {bind_address}:1388 00000000:0000 0A 00:00000000 00000000 0 0 1000 4242\n"
        if v6:
            (net / "tcp").write_text(header)
            (net / "tcp6").write_text(header + row)
        else:
            (net / "tcp").write_text(header + row if listening_now else header)
            (net / "tcp6").write_text(header)

    class _Connection:
        def __init__(self, ok: bool, slow: bool = False) -> None:
            self.ok = ok
            self.slow = slow

        def sendall(self, payload: bytes) -> None:
            pass

        def recv(self, size: int) -> bytes:
            if self.slow:
                # A server busy with an action: the socket is connected and
                # nothing arrives before the probe bound. ``socket.timeout`` is
                # ``TimeoutError`` since 3.10 and is an ``OSError``, exactly
                # what ``answers()`` catches.
                raise TimeoutError("timed out")
            return probe_status if self.ok else b""

        def close(self) -> None:
            pass

    def create_connection(address: tuple[str, int], timeout: float) -> Any:
        probes.append((address[0], address[1], timeout))
        if address[0] != reachable_host:
            raise OSError("no route to host")
        return _Connection(answers[min(clock["cycle"], len(answers) - 1)], probe_timeouts)

    def check_output(args: Any, **kwargs: Any) -> str:
        calls.append(args)
        if args[0] == "systemd-run":
            # The transient service would have run RESTART_SOURCE. Produce the
            # verdict with the REAL script and write exactly what it writes.
            token = int(args[-1])
            if restart_effect == "missing":
                return ""
            producer_token = token + 1 if restart_effect == "foreign" else token
            payload = _real_restart_verdict(
                tmp_path / ("producer-%d" % token),
                token=producer_token,
                effect=restart_effect,
                bind=restart_bind,
            )
            result_path.write_text(json.dumps(payload))
            return ""
        if clock["cycle"] in failures:
            raise failures[clock["cycle"]]
        if args[1] in failures_on:
            raise failures_on[args[1]]
        if args[1] == "show":
            index = min(clock["cycle"], len(active_states) - 1)
            return "ActiveState=" + active_states[index]
        return ""

    def sleep(seconds: float) -> None:
        clock["now"] += seconds
        if seconds < guest_supervisor.WATCHDOG_INTERVAL_S:
            # A verification poll, not the loop boundary: advance the clock,
            # never the cycle.
            return
        clock["cycle"] += 1
        if clock["cycle"] >= cycles:
            raise _StopWatchdog()
        index = min(clock["cycle"], len(listening) - 1)
        publish(listening[index], answers[min(index, len(answers) - 1)])

    publish(listening[0], answers[0])
    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "socket": SimpleNamespace(
            create_connection=create_connection,
            AF_INET6=10,
            SOCK_STREAM=1,
        ),
        "subprocess": SimpleNamespace(check_output=check_output),
        "sys": SimpleNamespace(
            argv=[
                "watchdog.py",
                target_unit,
                "5000",
                str(guest_supervisor.WATCHDOG_RECORD_PATH),
                str(guest_supervisor.RESTART_SCRIPT_PATH),
                str(guest_supervisor.RESTART_RESULT_PATH),
            ]
        ),
        "time": SimpleNamespace(monotonic=lambda: clock["now"], sleep=sleep),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    with pytest.raises(_StopWatchdog):
        exec(
            guest_supervisor.WATCHDOG_SOURCE,
            {"__builtins__": {**vars(builtins), "__import__": importer}},
        )
    if probe_log is not None:
        probe_log.extend(probes)
    return json.loads(record_path.read_text()), calls, clock["now"]


def _real_restart_verdict(
    tmp_path: Path,
    *,
    token: int,
    effect: str = "success",
    unit: str = "action-server.service",
    unit_failed: bool = True,
    bind: str = "00000000",
) -> dict[str, Any]:
    """Produce the REAL RESTART_SOURCE document for the watchdog to consume.

    The only acceptable test for a cross-process contract runs the real
    producer: a hand-built ``{"status": ...}`` document is the invented field
    that made round 1's repair read as a failure on the real guest.
    """
    payload, _, _ = _execute_restart_script(
        tmp_path,
        dispatch_argv=[
            "python3",
            "-c",
            guest_supervisor.RESTART_SOURCE,
            unit,
            "5000",
            guest_supervisor.RESTART_RESULT_PATH,
            str(token),
        ],
        on_kill=(
            (lambda proc: _write_listeners(proc, {}))
            if effect in ("success", "no-effect")
            else None
        ),
        on_start=(
            (lambda proc: _serve(proc, 9100, 4343, 1, 101000, bind=bind))
            if effect == "success"
            else None
        ),
        reset_ok=effect != "refused-reset",
        unit_failed=unit_failed,
    )
    return payload


def _dispatches(calls: list[Any]) -> list[Any]:
    """Every repair is ONE dispatch of the shared restart script."""
    return [args for args in calls if args[0] == "systemd-run"]


def test_a_failed_unit_is_revived_by_the_shared_restart(tmp_path: Path) -> None:
    record, calls, _ = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[False], cycles=1
    )
    assert len(_dispatches(calls)) == 1
    assert record["attempts"] == 1
    assert record["intervened"] is True
    assert record["last_trigger"] == "failed"
    assert record["last_action"] == "verified-restart"
    assert record["last_outcome"] == "restarted"
    assert record["events"] == [
        {
            "us": record["last_attempt_us"],
            "trigger": "failed",
            "action": "verified-restart",
            "outcome": "restarted",
        }
    ]


def test_an_inactive_unit_is_revived_by_the_shared_restart(tmp_path: Path) -> None:
    record, calls, _ = _execute_watchdog(
        tmp_path, active_states=["inactive"], listening=[False], cycles=1
    )
    assert len(_dispatches(calls)) == 1
    assert record["attempts"] == 1
    assert record["last_trigger"] == "inactive"
    assert record["last_outcome"] == "restarted"


def test_a_healthy_server_is_never_touched(tmp_path: Path) -> None:
    """Zero interventions over twenty cycles of an active, listening server."""
    record, calls, elapsed = _execute_watchdog(
        tmp_path, active_states=["active"], listening=[True], cycles=20
    )
    assert _dispatches(calls) == []
    assert not any(args[0] == "systemd-run" for args in calls)
    assert record["attempts"] == 0
    assert record["intervened"] is False
    assert record["events"] == []
    # A serving port short-circuits even the unit query: a healthy episode
    # issues ZERO systemctl calls -- no show, no start, nothing.
    assert not any(args[1] == "show" for args in calls)
    assert record["last_observation"] == "serving"
    assert elapsed >= guest_supervisor.WATCHDOG_INTERVAL_S * 10


def test_a_listener_that_answers_is_left_alone(tmp_path: Path) -> None:
    """KillMode=process leaves live survivors; a serving one needs no repair."""
    record, calls, _ = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[True], cycles=5
    )
    assert _dispatches(calls) == []
    assert not any(args[0] == "systemd-run" for args in calls)
    assert record["attempts"] == 0
    assert record["last_observation"] == "serving"


def test_a_failed_unit_with_a_wedged_listener_takes_the_verified_restart(
    tmp_path: Path,
) -> None:
    """A listener that accepts but never answers is the host's dead-channel shape.

    ``start`` cannot fix it (the port is held) and cannot even run (the unit is
    failed); only the all-process restart can, so the watchdog dispatches the
    host's own RESTART_SOURCE as a transient service and records its verdict.
    """
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[True],
        answering=[False],
        cycles=1,
        clock_start=1000.0,
    )
    dispatch = next(args for args in calls if args[0] == "systemd-run")
    assert "--wait" in dispatch and "--collect" in dispatch
    assert "action-server.service" in dispatch and "5000" in dispatch
    assert dispatch[-1].isdigit(), "the dispatch carries its token"

    def mapped(guest: str) -> str:
        return str(tmp_path / guest.lstrip("/"))

    # The script path is passed through verbatim; the result path is opened
    # through the (redirected) pathlib, so it is the mapped path here.
    assert guest_supervisor.RESTART_SCRIPT_PATH in dispatch
    assert mapped(guest_supervisor.RESTART_RESULT_PATH) in dispatch
    assert record["attempts"] == 1
    assert record["last_trigger"] == "unresponsive"
    assert record["last_action"] == "verified-restart"
    # last_outcome reflects the most recent DECISION, and the cycle after a
    # repair is legitimately throttled; the dispatch itself is the event.
    assert record["events"][0]["outcome"] == "restarted"
    assert record["last_observation"] == "unresponsive"
    # No bare start or reset is ever issued beside the holder.
    assert not any(len(args) > 1 and args[1] in {"start", "reset-failed"} for args in calls)


def test_a_verified_restart_that_fails_is_recorded_as_restart_failed(
    tmp_path: Path,
) -> None:
    """The REAL producer's stale report: the holder survived and was not replaced."""
    record, _, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[True],
        answering=[False],
        cycles=1,
        clock_start=1000.0,
        restart_effect="stale",
    )
    assert record["attempts"] == 1
    assert record["last_action"] == "verified-restart"
    assert record["last_outcome"] == "restart-failed"


def test_a_refused_reset_is_a_distinct_outcome_not_a_failed_restart(
    tmp_path: Path,
) -> None:
    """The latch never cleared, so the start could not have been accepted."""
    record, _, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[True],
        answering=[False],
        cycles=1,
        clock_start=1000.0,
        restart_effect="refused-reset",
    )
    assert record["last_outcome"] == "reset-failed-error"
    assert record["attempts"] == 1


def test_a_verdict_for_another_dispatch_is_not_accepted(tmp_path: Path) -> None:
    """The token ties the verdict to THIS dispatch; a stale file cannot pass."""
    record, _, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[True],
        answering=[False],
        cycles=1,
        clock_start=1000.0,
        restart_effect="foreign",
    )
    assert record["last_outcome"] == "restart-failed"


def test_one_unresponsive_observation_is_not_death(tmp_path: Path) -> None:
    """A single slow response must never restart a live server."""
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["active"],
        listening=[True],
        answering=[False],
        cycles=1,
    )
    assert not any(args[0] == "systemd-run" for args in calls)
    assert record["attempts"] == 0
    assert record["last_observation"] == "unresponsive"


def test_a_busy_but_active_server_is_never_touched(tmp_path: Path) -> None:
    """A HEALTHY server too slow for any probe must never be killed.

    Production shape: a 1280x720 screenshot or a long page load outlives the
    probe bound, and the retired two-consecutive-silence rule killed exactly
    such a server mid-action -- the adapter's in-flight request then died with
    a transport error at step 2 and the episode was lost. ``ActiveState=active``
    is systemd's statement that the process is still there, so silence is an
    OBSERVATION: over a sustained window no kill, start, reset, stop or
    dispatch may be issued, and no attempt may be spent.

    ACCEPTED COST (the reason this is a rule and not a heuristic): a listener
    that accepts and never answers while the unit stays ``active`` -- main.py
    alive, worker hung, socket held -- is now repaired by neither the guest nor
    the host, so that episode is lost. The alternative kills the identical
    shape a busy server presents, which is one episode against every episode.
    Revisiting it needs a record of ``unresponsive`` with ``attempts: 0`` over
    many cycles WHILE the host sees ``active`` for the same window -- and then
    a discriminator stronger than silence (socket-owner identity), never a
    shorter probe.
    """
    record, calls, elapsed = _execute_watchdog(
        tmp_path / "busy",
        active_states=["active"],
        listening=[True],
        answering=[False],
        probe_timeouts=True,
        cycles=30,
    )
    assert _dispatches(calls) == []
    assert not any(
        len(args) > 1 and args[1] in {"kill", "start", "stop", "restart", "reset-failed"}
        for args in calls
    ), "the watchdog issued a destructive verb against a live unit"
    assert record["attempts"] == 0
    assert record["intervened"] is False
    assert record["events"] == []
    assert record["last_observation"] == "unresponsive"
    # SUSTAINED, and proved by ITERATION COUNT rather than the wall of a
    # seeded clock: the fake clock starts at 0 (the harness returns it
    # absolutely), and every silent cycle issues exactly one unit query, so a
    # window that collapsed to a cycle or two cannot satisfy this.
    shows = [args for args in calls if len(args) > 1 and args[1] == "show"]
    assert len(shows) >= 25, f"only {len(shows)} cycles ran; the window is not sustained"
    assert elapsed >= guest_supervisor.WATCHDOG_INTERVAL_S * 25

    # NEGATIVE CONTROL: the SAME silence with the unit not running must
    # still dispatch, so the test above cannot pass on a dead harness -- and
    # the discriminating variable is the unit state, not the probe.
    control, control_calls, _ = _execute_watchdog(
        tmp_path / "control",
        active_states=["failed"],
        listening=[True],
        answering=[False],
        probe_timeouts=True,
        cycles=1,
    )
    assert len(_dispatches(control_calls)) == 1
    assert control["attempts"] == 1
    assert control["last_trigger"] == "unresponsive"


def test_an_active_unit_with_no_listener_is_observed_not_touched(
    tmp_path: Path,
) -> None:
    """A slow or wedged start is observed; ``start`` on an active unit is a no-op."""
    record, calls, _ = _execute_watchdog(
        tmp_path, active_states=["active"], listening=[False], cycles=5
    )
    # t=0/10/20 stay inside the 30 s grace; t=30 records the observation.
    assert record["attempts"] == 0
    assert record["last_observation"] == "no-listener"
    assert record["events"] == []
    assert _dispatches(calls) == []


def test_an_activating_unit_is_never_merged_into(tmp_path: Path) -> None:
    record, calls, _ = _execute_watchdog(
        tmp_path, active_states=["activating"], listening=[False], cycles=10
    )
    assert record["attempts"] == 0
    assert _dispatches(calls) == []


def test_a_unit_that_exits_immediately_is_never_recorded_as_started(
    tmp_path: Path,
) -> None:
    """A start whose server exits at once must never read as a revival.

    Measured on real systemd 255: ``systemctl start`` exits 0 when the JOB
    completes, even while the unit stays failed and nothing ever listens. The
    shared restart's payload reports no post owner, so the watchdog records
    ``no-effect`` -- and a no-effect dispatch draws on a SMALLER budget than a
    real repair, so a permanently broken unit cannot starve a later real
    failure of its five attempts.
    """
    record, calls, elapsed = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[False],
        cycles=20,
        restart_effect="no-effect",
    )
    assert record["attempts"] == guest_supervisor.WATCHDOG_MAX_INEFFECTIVE
    assert record["capped"] is True
    assert {event["outcome"] for event in record["events"]} == {"no-effect"}
    assert "started" not in {event["outcome"] for event in record["events"]}
    assert len(_dispatches(calls)) == guest_supervisor.WATCHDOG_MAX_INEFFECTIVE
    # Two dispatches spaced by the 30 s throttle.
    assert elapsed >= 30.0


def test_a_refused_reset_is_never_recorded_as_a_restart(
    tmp_path: Path,
) -> None:
    """The latch never cleared: a distinct outcome from a restart that failed."""
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[False],
        cycles=1,
        restart_effect="refused-reset",
    )
    assert record["last_outcome"] == "reset-failed-error"
    assert record["attempts"] == 1
    assert len(_dispatches(calls)) == 1


def test_a_start_that_takes_effect_is_recorded_as_restarted(tmp_path: Path) -> None:
    """The success word requires the postcondition, and it is still reported."""
    record, _, _ = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[False], cycles=1
    )
    assert record["last_action"] == "verified-restart"
    assert record["last_outcome"] == "restarted"


def _execute_report(
    tmp_path: Path,
    *,
    watchdog_text: str | None,
    receipt: dict[str, Any] | None = None,
    nrestarts: int = 0,
    active_enter: int = 100,
) -> str:
    """Run the shipped REPORT_SOURCE against a redirected /var and fake systemd."""
    root = tmp_path / "var/lib/lop-action-supervisor"
    root.mkdir(parents=True, exist_ok=True)
    (root / "action-server.service.json").write_text(
        json.dumps(
            receipt
            or {
                "schema_version": 1,
                "records": [],
                "captured_exits": 0,
                "baseline_nrestarts": 0,
                "baseline_active_enter": 100,
            }
        )
    )
    if watchdog_text is not None:
        (root / "watchdog.json").write_text(watchdog_text)
    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "subprocess": SimpleNamespace(
            check_output=lambda *args, **kwargs: (
                f"NRestarts={nrestarts}\n" f"ActiveEnterTimestampMonotonic={active_enter}"
            )
        ),
        "sys": SimpleNamespace(argv=["-c", "action-server.service"]),
    }
    real_import = builtins.__import__

    def importer(name: str, *args: Any, **kwargs: Any) -> Any:
        return modules[name] if name in modules else real_import(name, *args, **kwargs)

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(
            guest_supervisor.REPORT_SOURCE,
            {"__builtins__": {**vars(builtins), "__import__": importer}},
        )
    return output.getvalue()


def test_a_corrupt_watchdog_file_is_unparseable_while_no_file_is_absent(
    tmp_path: Path,
) -> None:
    """``absent`` means no file; a file we cannot read is its own answer."""
    corrupt = _execute_report(tmp_path, watchdog_text="{not json")
    _, watchdog = guest_supervisor.receipt_report(corrupt)
    assert watchdog["status"] == "unparseable"
    literal_null = _execute_report(tmp_path, watchdog_text="null")
    _, watchdog = guest_supervisor.receipt_report(literal_null)
    assert watchdog["status"] == "unparseable"
    (tmp_path / "var/lib/lop-action-supervisor/watchdog.json").unlink()
    missing = _execute_report(tmp_path, watchdog_text=None)
    _, watchdog = guest_supervisor.receipt_report(missing)
    assert watchdog["status"] == "absent"


def test_the_rate_limit_stops_a_spin(tmp_path: Path) -> None:
    """One intervention per MIN_INTERVAL; the suppressed cycles record throttled."""
    record, calls, elapsed = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[False], cycles=3
    )
    # t=0 intervenes, t=10/20 are inside the 30 s minimum interval.
    assert elapsed >= 20.0
    assert record["attempts"] == 1
    assert record["last_outcome"] == "throttled"
    assert len(_dispatches(calls)) == 1


def test_the_cap_is_reached_and_recorded(tmp_path: Path) -> None:
    """At most MAX_ATTEMPTS per rolling window; then the cap itself is evidence."""
    record, calls, _ = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[False], cycles=20
    )
    assert record["attempts"] == guest_supervisor.WATCHDOG_MAX_ATTEMPTS
    assert record["capped"] is True
    assert record["last_outcome"] == "capped"
    assert len(_dispatches(calls)) == guest_supervisor.WATCHDOG_MAX_ATTEMPTS
    assert len(record["events"]) <= guest_supervisor.WATCHDOG_EVENT_CAP


def test_the_event_ring_is_bounded_across_windows(tmp_path: Path) -> None:
    """Ten attempts over two rolling hours still leave at most eight events."""
    record, _, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[False],
        cycles=760,
    )
    assert record["attempts"] >= 2 * guest_supervisor.WATCHDOG_MAX_ATTEMPTS
    assert len(record["events"]) == guest_supervisor.WATCHDOG_EVENT_CAP
    assert record["events"][-1]["outcome"] == "restarted"


def test_a_watchdog_error_does_not_kill_the_loop(tmp_path: Path) -> None:
    """A raising systemctl is swallowed; a later cycle still repairs the unit."""
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[False],
        cycles=4,
        failures={0: OSError("transient manager failure")},
    )
    assert record["attempts"] == 1
    assert record["events"][0]["outcome"] == "restarted"
    assert len(_dispatches(calls)) == 1


def test_a_v6_only_listener_is_observed_and_probed_on_v6(tmp_path: Path) -> None:
    """The tcp6 parse path is the AMI's own, so the probe must try ::1 too."""
    probes: list[tuple[str, int, float]] = []
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["active"],
        listening=[True],
        cycles=2,
        v6=True,
        reachable_host="::1",
        probe_log=probes,
    )
    assert record["attempts"] == 0
    assert record["last_observation"] == "serving"
    assert _dispatches(calls) == []
    assert probes[0][0] == "127.0.0.1" and probes[1][0] == "::1"


def test_any_complete_http_status_counts_as_answering(tmp_path: Path) -> None:
    """A 404/500 is a live app; only silence is "unresponsive"."""
    record, _, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[True],
        cycles=1,
        probe_status=b"HTTP/1.0 500 Internal Server Error\r\n",
    )
    assert record["attempts"] == 0
    assert record["last_observation"] == "serving"


def test_a_restarted_watchdog_seeds_its_counter_and_cap(tmp_path: Path) -> None:
    """The unit's own Restart=always must not hand a crash loop a fresh cap."""
    seed = {
        "schema_version": 1,
        "attempts": 5,
        "intervened": True,
        "capped": True,
        "last_attempt_us": 1,
        "last_trigger": "failed",
        "last_action": "reset-failed+start",
        "last_outcome": "started",
        "last_observation": "no-listener",
        "events": [
            {"us": n, "trigger": "failed", "action": "reset-failed+start", "outcome": "started"}
            for n in (1, 2, 3, 4, 5)
        ],
    }
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[False],
        cycles=2,
        seed=seed,
        clock_start=1000.0,
    )
    # The restarted process inherits the spent cap instead of earning a new one.
    assert record["attempts"] == 5
    assert record["capped"] is True
    assert record["last_outcome"] == "capped"
    assert len(record["events"]) == 5
    assert _dispatches(calls) == []


def test_the_cap_clears_when_the_window_rolls(tmp_path: Path) -> None:
    """``capped`` is window-scoped evidence, not a latch."""
    stale = {
        "schema_version": 1,
        "attempts": 5,
        "intervened": True,
        "capped": True,
        "last_attempt_us": 1,
        "last_trigger": "failed",
        "last_action": "reset-failed+start",
        "last_outcome": "started",
        "last_observation": "no-listener",
        "events": [
            {"us": n, "trigger": "failed", "action": "reset-failed+start", "outcome": "started"}
            for n in (1, 2, 3, 4, 5)
        ],
    }
    # The clock starts 27 h after the seed's stamps, so every one of them is
    # outside the rolling hour and ``capped`` must clear on the first cycle.
    record, _, _ = _execute_watchdog(
        tmp_path,
        active_states=["active"],
        listening=[True],
        cycles=2,
        seed=stale,
        clock_start=100_000.0,
    )
    assert record["capped"] is False
    assert record["attempts"] == 5
    # Past interventions stay in the bounded ring as evidence even after the
    # window that gated them has rolled.
    assert len(record["events"]) == 5


def test_the_watchdog_record_carries_no_guest_text(tmp_path: Path) -> None:
    record, _, _ = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[False], cycles=1
    )
    assert set(record) == {
        "schema_version",
        "attempts",
        "intervened",
        "capped",
        "last_attempt_us",
        "last_trigger",
        "last_action",
        "last_outcome",
        "last_observation",
        "events",
    }
    assert set(record["events"][0]) == {"us", "trigger", "action", "outcome"}
    text = json.dumps(record)
    assert "action-server.service" not in text
    assert "lop-action-supervisor" not in text


def test_the_watchdog_script_never_writes_argv_or_environment(tmp_path: Path) -> None:
    record, _, _ = _execute_watchdog(
        tmp_path, active_states=["failed"], listening=[False], cycles=1
    )
    assert "os.environ" not in guest_supervisor.WATCHDOG_SOURCE
    # Everything the script received on argv is absent from the record: the
    # only string values are the fixed vocabulary words.
    values = {word for word in record.values() if isinstance(word, str)}
    assert values <= {"none", "failed", "no-listener", "verified-restart", "restarted"}


def test_the_watchdog_unit_is_installed_beside_the_drop_in(tmp_path: Path) -> None:
    install, calls = _script_guest(tmp_path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        install()
    payload = json.loads(output.getvalue())
    assert payload["installed"] is True
    assert payload["watchdog"] == {
        "installed": True,
        "status": "active",
        "cgroup_distinct": True,
    }
    unit = (tmp_path / "etc/systemd/system/lop-action-watchdog.service").read_text()
    script = tmp_path / "var/lib/lop-action-supervisor/watchdog.py"
    assert "Type=simple" in unit
    # The watchdog is its own rate limit authority: latching it failed is the
    # one state nothing else repairs.
    assert "Restart=always" in unit and "StartLimitIntervalSec=0" in unit
    assert (
        "ExecStart=/usr/bin/python3 "
        + str(script)
        + " action-server.service 5000 "
        + str(tmp_path / "var/lib/lop-action-supervisor/watchdog.json")
        + " "
        + str(tmp_path / "var/lib/lop-action-supervisor/restart.py")
        + " "
        + str(tmp_path / "var/lib/lop-action-supervisor/restart-result.json")
    ) in unit
    assert script.read_text() == guest_supervisor.WATCHDOG_SOURCE
    # The watchdog dispatches the host's own restart implementation verbatim.
    assert (
        tmp_path / "var/lib/lop-action-supervisor/restart.py"
    ).read_text() == guest_supervisor.RESTART_SOURCE
    # Enabled for boot, started now, and never aimed at the action server.
    assert ("systemctl", "enable", guest_supervisor.WATCHDOG_UNIT) in calls
    assert ("systemctl", "start", guest_supervisor.WATCHDOG_UNIT) in calls
    assert not any(
        args[1] in {"restart", "start", "reset-failed"}
        and guest_supervisor.WATCHDOG_UNIT not in args
        for args in calls
    )
    # A second install is byte-idempotent and does not re-write the unit.
    mtime = (tmp_path / "etc/systemd/system/lop-action-watchdog.service").stat().st_mtime_ns
    with contextlib.redirect_stdout(io.StringIO()):
        install()
    assert (tmp_path / "etc/systemd/system/lop-action-watchdog.service").stat().st_mtime_ns == mtime


def test_install_refuses_a_watchdog_that_shares_the_action_server_cgroup(
    tmp_path: Path,
) -> None:
    """A watchdog inside the unit it repairs dies with it: refuse, don't decorate."""
    install, calls = _script_guest(
        tmp_path, watchdog_props={"ControlGroup": "/system.slice/action-server.service"}
    )
    with pytest.raises(RuntimeError, match="watchdog unit shares the action-server cgroup"):
        install()
    # The started watchdog is stopped and disabled BEFORE its file is removed:
    # a live watchdog beside ``installed: false`` would keep repairing a server
    # the host believes unsupervised.
    verbs = [
        args[1] for args in calls if len(args) > 2 and args[2] == guest_supervisor.WATCHDOG_UNIT
    ]
    assert "stop" in verbs and "disable" in verbs
    assert not (tmp_path / "etc/systemd/system/lop-action-watchdog.service").exists()
    assert not (
        tmp_path / "etc/systemd/system/action-server.service.d/90-lop-supervisor.conf"
    ).exists()


def test_install_refuses_a_watchdog_in_a_child_of_the_action_server_cgroup(
    tmp_path: Path,
) -> None:
    install, _ = _script_guest(
        tmp_path,
        watchdog_props={"ControlGroup": "/system.slice/action-server.service/watchdog"},
    )
    with pytest.raises(RuntimeError, match="watchdog unit shares the action-server cgroup"):
        install()


def test_install_refuses_an_inactive_watchdog(tmp_path: Path) -> None:
    install, _ = _script_guest(tmp_path, watchdog_props={"ActiveState": "inactive"})
    with pytest.raises(RuntimeError, match="watchdog unit is not active"):
        install()
    assert not (tmp_path / "etc/systemd/system/lop-action-watchdog.service").exists()


def test_install_refuses_a_watchdog_without_a_control_group(tmp_path: Path) -> None:
    install, _ = _script_guest(tmp_path, watchdog_props={"ControlGroup": ""})
    with pytest.raises(RuntimeError, match="watchdog unit has no control group"):
        install()


# ---------------------------------------------------------------------------
# Host side: the watchdog summary is evidence, sanitized like every other
# guest-derived field, and it never changes a recovery outcome.
# ---------------------------------------------------------------------------


def _watchdog_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "attempts": 3,
        "intervened": True,
        "capped": False,
        "last_attempt_us": 12_345_678,
        "last_trigger": "failed",
        "last_action": "reset-failed+start",
        "last_outcome": "started",
        "last_observation": "serving",
        "events": [
            {
                "us": 12_345_678,
                "trigger": "failed",
                "action": "reset-failed+start",
                "outcome": "started",
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_the_watchdog_summary_is_sanitized_onto_the_receipt() -> None:
    envelope = _envelope([])
    envelope["watchdog"] = _watchdog_payload(
        unit="action-server.service", argv=["/usr/bin/python3"], note="guest text"
    )
    records, watchdog = guest_supervisor.receipt_report(
        "escalation=agentless\n" + json.dumps(envelope)
    )
    assert records == []
    assert watchdog["status"] == "ok"
    assert watchdog["attempts"] == 3
    assert watchdog["intervened"] is True
    assert watchdog["capped"] is False
    assert watchdog["last_trigger"] == "failed"
    assert watchdog["last_action"] == "reset-failed+start"
    assert watchdog["last_outcome"] == "started"
    assert watchdog["events"] == _watchdog_payload()["events"]
    # Unknown keys are dropped whole: no unit name, argv or free text advances.
    assert set(watchdog) == {
        "schema_version",
        "status",
        "attempts",
        "intervened",
        "capped",
        "last_attempt_us",
        "last_trigger",
        "last_action",
        "last_outcome",
        "last_observation",
        "events",
    }
    assert "action-server.service" not in json.dumps(watchdog)


@pytest.mark.parametrize(
    ("field", "hostile", "neutral"),
    [
        ("attempts", True, 0),
        ("attempts", -1, 0),
        ("attempts", "3", 0),
        ("last_attempt_us", 1.5, 0),
        ("last_trigger", "arbitrary guest text", "none"),
        ("last_action", "systemctl start", "none"),
        ("last_outcome", "oops", "none"),
        ("last_observation", "arbitrary guest text", "none"),
    ],
)
def test_hostile_watchdog_fields_become_neutral(
    field: str, hostile: object, neutral: object
) -> None:
    envelope = _envelope([])
    envelope["watchdog"] = _watchdog_payload(**{field: hostile})
    _, watchdog = guest_supervisor.receipt_report(json.dumps(envelope))
    assert watchdog[field] == neutral


def test_a_no_effect_start_outcome_survives_sanitization() -> None:
    """The honest failure word must reach the evidence, not be neutralized."""
    envelope = _envelope([])
    envelope["watchdog"] = _watchdog_payload(
        last_outcome="no-effect",
        events=[
            {
                "us": 5,
                "trigger": "failed",
                "action": "reset-failed+start",
                "outcome": "no-effect",
            }
        ],
    )
    _, watchdog = guest_supervisor.receipt_report(json.dumps(envelope))
    assert watchdog["last_outcome"] == "no-effect"
    assert watchdog["events"][0]["outcome"] == "no-effect"


def test_the_watchdog_event_ring_is_bounded_host_side() -> None:
    envelope = _envelope([])
    envelope["watchdog"] = _watchdog_payload(
        events=[
            {"us": n, "trigger": "failed", "action": "reset-failed+start", "outcome": "started"}
            for n in range(20)
        ]
    )
    _, watchdog = guest_supervisor.receipt_report(json.dumps(envelope))
    assert len(watchdog["events"]) == guest_supervisor.WATCHDOG_EVENT_CAP
    assert watchdog["events"][-1]["us"] == 19


def test_a_corrupt_watchdog_record_is_absent_not_a_failed_receipt() -> None:
    envelope = _envelope([])
    envelope["watchdog"] = {"schema_version": 99, "attempts": 4}
    records, watchdog = guest_supervisor.receipt_report(json.dumps(envelope))
    assert records == []
    assert watchdog == guest_supervisor.empty_watchdog("unparseable")
    envelope.pop("watchdog")
    _, missing = guest_supervisor.receipt_report(json.dumps(envelope))
    assert missing == guest_supervisor.empty_watchdog("absent")
    # An explicit JSON null is a record that EXISTS but cannot be used.
    envelope["watchdog"] = None
    _, null = guest_supervisor.receipt_report(json.dumps(envelope))
    assert null == guest_supervisor.empty_watchdog("unparseable")


def test_install_evidence_carries_the_watchdog_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        payload = {
            "installed": True,
            "unit": "action-server.service",
            "watchdog": {"installed": True, "status": "active", "cgroup_distinct": True},
        }
        monkeypatch.setattr(
            provider,
            "_run_guest_command",
            lambda *args, **kwargs: guest_disk.CommandResult(
                0, "escalation=agentless\n" + json.dumps(payload), ""
            ),
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["watchdog"] == {
            "installed": True,
            "status": "active",
            "cgroup_distinct": True,
        }


def test_install_evidence_does_not_fabricate_a_watchdog_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        monkeypatch.setattr(
            provider,
            "_run_guest_command",
            lambda *args, **kwargs: guest_disk.CommandResult(
                0,
                'escalation=agentless\n{"installed": true, "unit": "action-server.service"}',
                "",
            ),
        )
        provider._install_guest_supervisor(_plan())
        report = json.loads((provider._cache_root / "guest-supervisor-setup.json").read_text())
        assert report["installed"] is True
        assert "watchdog" not in report


@pytest.mark.asyncio
async def test_a_restart_that_needed_the_watchdog_is_visible_in_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recovery the watchdog performed must not read as a clean episode."""
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )

        envelope = _envelope([_restart_record()])
        envelope["watchdog"] = _watchdog_payload(attempts=2, capped=True)

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            return guest_disk.CommandResult(0, json.dumps(envelope), "")

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()"])
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is True
        assert evidence["watchdog"]["attempts"] == 2
        assert evidence["watchdog"]["capped"] is True
        assert evidence["watchdog"]["last_trigger"] == "failed"


@pytest.mark.asyncio
async def test_a_changed_watchdog_summary_is_written_without_a_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verified_none still surfaces interventions, and only when they change."""
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        writes: list[str] = []
        real_write = provider._write_supervisor_evidence

        def write(name: str, value: object) -> None:
            writes.append(name)
            real_write(name, value)

        monkeypatch.setattr(provider, "_write_supervisor_evidence", write)
        envelope = _envelope([])
        envelope["watchdog"] = _watchdog_payload(attempts=4)
        monkeypatch.setattr(
            provider,
            "_run_guest_command",
            lambda *args, **kwargs: guest_disk.CommandResult(0, json.dumps(envelope), ""),
        )
        # The FIRST read is a silent baseline: a healthy episode writes no
        # recovery file at all.
        assert provider._recover_guest_supervisor().state == "verified_none"
        assert writes == []
        # A CHANGE is evidence, even without a recovery.
        envelope["watchdog"] = _watchdog_payload(attempts=5)
        assert provider._recover_guest_supervisor().state == "verified_none"
        assert writes == ["guest-supervisor-recovery.json"]
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["observed"] == "verified_none"
        assert evidence["watchdog"]["attempts"] == 5
        # The next action sees the same summary and writes nothing new.
        assert provider._recover_guest_supervisor().state == "verified_none"
        assert writes == ["guest-supervisor-recovery.json"]


@pytest.mark.asyncio
async def test_a_latched_recovery_leaves_a_receipt_the_host_can_post_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blocker's full chain, at the seams that carried it.

    A real restart-script run against a latched unit rebases the receipt; the
    real REPORT_SOURCE reads that receipt on the guest; the host's recovery
    must then return ``verified_none`` (NOT ``read_failed``, which becomes
    GuestServerRecovered and silently drops every later batch) and its next
    execute must actually reach the guest.
    """
    payload, _, _ = _execute_restart_script(
        tmp_path / "guest",
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    receipt = payload["receipt"]
    assert receipt["captured_exits"] == 0 and receipt["records"] == []
    report = _execute_report(
        tmp_path / "report",
        watchdog_text=None,
        receipt=receipt,
        nrestarts=receipt["systemd_nrestarts"],
        active_enter=receipt["systemd_active_enter"],
    )
    # The host's report parser must not raise GenerationMismatch here.
    assert guest_supervisor.receipt_records(report) == []

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        posted: list[str] = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            text = " ".join(str(part) for part in argv)
            if "systemctl" in text:
                return guest_disk.CommandResult(0, report, "")
            posted.append(text)
            return guest_disk.CommandResult(0, "", "")

        monkeypatch.setattr(provider, "_run_guest_command", command)
        assert provider._recover_guest_supervisor().state == "verified_none"
        # FOUR batches, not one: the blocker made every later batch vanish, so
        # a single post would not have caught it.
        for index in range(4):
            await provider.execute([f"rebased_action_{index}()"])
        assert len(posted) == 4, "an action was silently dropped after the recovery"
        for index in range(4):
            # The statement rides base64-encoded in the argv, exactly as
            # upstream's controller sends it.
            encoded = base64.b64encode(f"rebased_action_{index}()".encode()).decode()
            assert encoded in posted[index]


@pytest.mark.asyncio
async def test_a_watchdog_only_change_never_overwrites_a_verified_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the episode really restarted, its evidence outranks later chatter.

    The first read is a silent baseline, so only a CHANGE writes; and a change
    that arrives after ``recovered: true`` must not replace that snapshot with
    a ``verified_none`` observation.
    """
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )
        envelope = _envelope([_restart_record()])
        envelope["watchdog"] = _watchdog_payload(attempts=1)

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            return guest_disk.CommandResult(0, json.dumps(envelope), "")

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()"])
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is True

        # The next action sees a NEWER watchdog summary and the already-consumed
        # generation: verified_none, watchdog changed -- and no rewrite.
        envelope["watchdog"] = _watchdog_payload(attempts=3)
        assert provider._recover_guest_supervisor().state == "verified_none"
        after = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert after["recovered"] is True
        assert after["watchdog"]["attempts"] == 1


@pytest.mark.asyncio
async def test_a_corrupt_watchdog_record_never_changes_a_recovery_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert provider._cache_root is not None
        provider._supervisor_unit = "action-server.service"
        monkeypatch.setattr(
            provider,
            "_probe_control_server_outcome",
            lambda timeout: (True, {"answered": True}),
        )

        envelope = _envelope([_restart_record()])
        envelope["watchdog"] = "not a record"

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            if _is_restart_command(argv):
                return guest_disk.CommandResult(0, "", "")
            if _is_restart_result_command(argv):
                return _restart_document()
            return guest_disk.CommandResult(0, json.dumps(envelope), "")

        monkeypatch.setattr(aws_mod.time, "time_ns", lambda: 777)
        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()"])
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is True
        assert evidence["watchdog"]["status"] == "unparseable"


def test_the_bind_decoder_reads_proc_net_addresses_per_family() -> None:
    """The rig's split, decoded: 127.0.0.1 and ::1 are not "listening".

    /proc prints the local address in host byte order, one 32-bit word for v4
    and four for v6, so the decode is endianness-aware rather than spelled
    out; a big-endian guest must not read its loopback as a public address.
    """
    namespace: dict[str, Any] = {"sys": sys}
    exec(guest_supervisor.LISTENER_BIND_SOURCE, namespace)
    bind_kind = namespace["bind_kind"]
    assert bind_kind("00000000", 4) == "wildcard"
    assert bind_kind("0100007F", 4) == "loopback"
    assert bind_kind("0A051FAC", 4) == "addressed"
    assert bind_kind("00000000000000000000000000000000", 6) == "wildcard"
    assert bind_kind("00000000000000000000000001000000", 6) == "loopback"
    assert bind_kind("0000000000000000FFFF00000A051FAC", 6) == "addressed"
    assert bind_kind("not-hex", 4) == "unknown"
    # IPv4-MAPPED addresses are classified by the IPv4 address they carry: a
    # v4 loopback bind the kernel reported in the v6 table must not read as a
    # public one, which is the same failure one representation away.
    assert bind_kind("0000000000000000FFFF00000100007F", 6) == "loopback"
    assert bind_kind("0000000000000000FFFF000000000000", 6) == "wildcard"
    # The other byte order is advertised, so exercise it: on a big-endian guest
    # /proc prints the words in network order and nothing is reversed.
    big: dict[str, Any] = {"sys": SimpleNamespace(byteorder="big")}
    exec(guest_supervisor.LISTENER_BIND_SOURCE, big)
    big_kind = big["bind_kind"]
    assert big_kind("00000000", 4) == "wildcard"
    assert big_kind("7F000001", 4) == "loopback"
    assert big_kind("00000000000000000000000000000001", 6) == "loopback"
    assert big_kind("00000000000000000000FFFF7F000001", 6) == "loopback"


def test_a_restart_whose_listener_loopback_alone_can_reach_is_refused(
    tmp_path: Path,
) -> None:
    """A fresh, answering, loopback-only listener is NOT a revival.

    This is the measured production split at the seat that declares the guest
    ready: the host posts to the instance's public address, a 127.0.0.1 bind
    refuses it with errno 61, and every guest-side probe still answers.
    """
    payload, _, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000, bind="0100007F"),
    )
    assert payload["post_restart_pids"] == [9100]
    assert payload["post_restart_bind"] == ["loopback"]
    record = guest_supervisor.restart_record(json.dumps(payload))
    assert record["status"] == "unreachable_listener"
    assert record["post_restart_bind"] == ["loopback"]
    assert record["fresh"] is True
    assert "loopback" in guest_supervisor.restart_cause(record["status"])


def test_a_restart_whose_listener_binds_the_wildcard_is_verified(tmp_path: Path) -> None:
    payload, _, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000),
    )
    assert payload["post_restart_bind"] == ["wildcard"]
    assert guest_supervisor.restart_record(json.dumps(payload))["status"] == "verified"


def test_the_watchdog_names_a_loopback_only_listener_and_never_restarts_it(
    tmp_path: Path,
) -> None:
    """Observed, never repaired: a restart cannot change an env-derived bind.

    The 2026-10 arm's four refused episodes reported ``serving`` here, because
    the port listened and the loopback probe answered. The record must say
    which address the listener bound, and the watchdog must not spend an
    attempt killing a server that still serves the guest's own state.
    """
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["active", "active"],
        listening=[True, True],
        answering=[True, True],
        cycles=2,
        bind_address="0100007F",
    )
    assert record["last_observation"] == "loopback-only"
    assert record["attempts"] == 0 and record["intervened"] is False
    assert not [call for call in calls if call[0] == "systemctl" and call[1] in {"start", "kill"}]


def test_a_repair_that_only_loopback_can_reach_is_not_a_revival(tmp_path: Path) -> None:
    """The guest's own verdict word, produced and consumed for real.

    ``restart_status`` is the watchdog's mirror of the host rule; this drives
    it end-to-end (the REAL RESTART_SOURCE writes the verdict file, the REAL
    watchdog reads it) so the two copies cannot drift into disagreeing about a
    loopback bind. It also counts as ineffective: two such repairs and the
    watchdog stops instead of re-killing a server for the same environment.
    """
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["failed"],
        listening=[False],
        cycles=1,
        restart_bind="0100007F",
    )
    assert len(_dispatches(calls)) == 1
    assert record["last_outcome"] == "unreachable-listener"
    assert record["intervened"] is True


MAPPED_LOOPBACK = "0000000000000000FFFF00000100007F"  # ::ffff:127.0.0.1, LE words


def test_a_restart_bound_to_a_v4_mapped_loopback_is_refused(tmp_path: Path) -> None:
    """The same split, one representation away: ::ffff:127.0.0.1.

    The kernel reports a v4-mapped bind in the v6 table; a decoder that read it
    as a plain v6 address called it ``addressed`` and blessed a listener only
    loopback can reach. This is the gate that has to refuse it.
    """
    payload, _, _ = _execute_restart_script(
        tmp_path,
        on_kill=lambda proc: _write_listeners(proc, {}),
        on_start=lambda proc: _serve(proc, 9100, 4343, 1, 101000, bind=MAPPED_LOOPBACK),
    )
    assert payload["post_restart_pids"] == [9100]
    assert payload["post_restart_bind"] == ["loopback"]
    record = guest_supervisor.restart_record(json.dumps(payload))
    assert record["status"] == "unreachable_listener"
    assert record["fresh"] is True


def test_the_watchdog_names_a_v4_mapped_loopback_listener(tmp_path: Path) -> None:
    record, calls, _ = _execute_watchdog(
        tmp_path,
        active_states=["active", "active"],
        listening=[True, True],
        answering=[True, True],
        cycles=2,
        v6=True,
        bind_address=MAPPED_LOOPBACK,
    )
    assert record["last_observation"] == "loopback-only"
    assert record["attempts"] == 0 and record["intervened"] is False
    assert not [call for call in calls if call[0] == "systemctl" and call[1] in {"start", "kill"}]
