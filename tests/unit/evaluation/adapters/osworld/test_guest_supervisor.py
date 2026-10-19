"""Cloud-free supervisor contracts and real loopback refusal/recovery paths.

The host has no Linux systemd daemon. Script tests execute the shipped Python
with a redirected /proc, /etc and /var and a fake manager, not a native-systemd
claim. Transport tests use actual sockets; no AWS clients make remote calls.
"""

from __future__ import annotations

import base64
import builtins
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from lop_osworld_v2_adapter import guest_disk, guest_supervisor
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.base import GuestServerRecovered

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


def _script_guest(tmp_path: Path, *, props: dict[str, str] | None = None) -> tuple[Any, list[Any]]:
    """Run the exact install source; only OS-boundary modules are substituted."""
    proc = tmp_path / "proc/8000"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    (proc / "cgroup").write_text("0::/system.slice/action-server.service\n")
    properties = {
        "MainPID": "8000",
        "ExecStart": "/usr/bin/python3 /home/user/server/main.py",
        "StandardError": "journal",
    }
    properties.update(props or {})
    calls: list[Any] = []

    def check_output(args: Any, **kwargs: Any) -> str:
        calls.append(args)
        if args[1] == "daemon-reload":
            return ""
        return "\n".join(k + "=" + v for k, v in properties.items())

    modules = {
        "pathlib": SimpleNamespace(Path=lambda p: tmp_path / str(p).lstrip("/")),
        "os": SimpleNamespace(getppid=lambda: 8000),
        "subprocess": SimpleNamespace(check_output=check_output),
        "sys": SimpleNamespace(argv=["-c", guest_supervisor.CAPTURE_SOURCE]),
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
    assert not any(
        args[1] in {"restart", "start", "stop", "kill", "reset-failed"} for args in calls
    )
    policy = (
        tmp_path / "etc/systemd/system/action-server.service.d/90-lop-supervisor.conf"
    ).read_text()
    assert "Restart=always\nRestartSec=1\n" in policy
    assert "StartLimitBurst=3" in policy and "StartLimitIntervalSec=3600" in policy
    assert "KillMode=process" in policy
    assert (
        "ExecStopPost=+/usr/bin/python3 "
        + str(tmp_path / "var/lib/lop-action-supervisor/capture.py")
        + " action-server.service"
    ) in policy
    assert "ExecStart=" not in policy
    for source in (guest_supervisor.INSTALL_SOURCE, guest_supervisor.CAPTURE_SOURCE):
        assert "/home/" not in source and "/Desktop" not in source and "/task" not in source
        assert "os.kill" not in source and "killpg" not in source


@pytest.mark.parametrize(
    "props",
    [
        {"MainPID": "7"},
        {"ExecStop": "/bin/stop-desktop"},
        {"StandardError": "null"},
        {"ExecStartPost": "/bin/change-task"},
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
    records = json.loads((root / "action-server.service.json").read_text())
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
                    {"installed": False, "reason": "action server has no private service"}
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
            "cause": "action server has no private service",
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
        checks = iter([None, True])
        deliveries = []

        def recover(delivery: str = "pre_action") -> bool | None:
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
        record = {
            "restart": 123,
            "exit_code": "exited",
            "exit_status": "1",
            "stderr_tail": "fatal crash",
        }
        monkeypatch.setattr(
            provider,
            "_run_guest_command",
            lambda *a, **k: guest_disk.CommandResult(
                0, "escalation=agentless\n" + json.dumps([record]), ""
            ),
        )
        assert provider._recover_guest_supervisor("ambiguous") is True
        assert provider._recover_guest_supervisor() is None
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["records"] == [record]
        assert evidence["failed_action_delivery"] == "ambiguous"


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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: False)
        with pytest.raises(
            aws_mod.GuestExecutionError, match="supervisor recovery budget exhausted"
        ):
            await provider.execute(["not_applied()"])
        # One fixed wait, then a clean end without sending stale-frame input.
        assert stubs.slept.count(0.5) == 6
        evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
        assert evidence["recovered"] is False
        assert evidence["final_guest_record"].startswith("not retrieved")
        assert not any("not_applied()" in str(body) for body in stubs.guest_posts)


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
            output = 'escalation=agentless\n' + (report.read_text() if report.exists() else '[]')
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
                    [
                        {
                            "restart": 999,
                            "exit_code": "killed",
                            "exit_status": str(-status),
                            "stderr_tail": stderr,
                        }
                    ]
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
