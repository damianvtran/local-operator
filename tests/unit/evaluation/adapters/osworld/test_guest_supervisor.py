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
import subprocess
import sys
import time
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
) -> tuple[Any, list[Any]]:
    """Run the exact install source; only OS-boundary modules are substituted."""
    proc = tmp_path / "proc/8000"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(b"python3\0/home/user/server/main.py\0")
    (proc / "cgroup").write_text("0::/system.slice/action-server.service\n")
    properties = {
        "MainPID": "8000",
        "ExecStart": "/usr/bin/python3 /home/user/server/main.py",
        "StandardError": "journal",
        "Type": "simple",
        "LoadState": "loaded",
        "NRestarts": "0",
        "ActiveEnterTimestampMonotonic": "100",
    }
    properties.update(props or {})
    calls: list[Any] = []

    def check_output(args: Any, **kwargs: Any) -> str:
        calls.append(args)
        if args[1] == "daemon-reload":
            return ""
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
    assert sum("Restart" in args for args in calls) == 2
    assert all(
        "RestartPreventExitStatus" in args and "ExitType" in args
        for args in calls
        if "Restart" in args
    )
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

        def recover(delivery: str = "pre_action") -> guest_supervisor.ReceiptResult:
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
            "service_result": "exit-code",
            "stderr_tail": "fatal crash",
        }
        monkeypatch.setattr(
            provider,
            "_run_guest_command",
            lambda *a, **k: guest_disk.CommandResult(
                0,
                "escalation=agentless\n" + json.dumps(_envelope([record])),
                "",
            ),
        )
        assert provider._recover_guest_supervisor("ambiguous").state == "new_restart"
        assert provider._recover_guest_supervisor().state == "verified_none"
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
        calls = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(argv)
            assert "/var/lib/lop-action-supervisor" in str(argv), "stale input was posted"
            return guest_disk.CommandResult(
                0, "sudo warning\n" + _receipt([_restart_record()]).stdout, ""
            )

        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()", "nor_this()"])
        assert len(calls) == 1
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
        calls = []
        shadow = json.dumps(_envelope([]))

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            calls.append(argv)
            assert "/var/lib/lop-action-supervisor" in str(argv), "stale input was posted"
            return guest_disk.CommandResult(
                0,
                "escalation=agentless>supplied\n"
                + shadow
                + "\n"
                + _receipt([_restart_record()]).stdout,
                "",
            )

        monkeypatch.setattr(provider, "_run_guest_command", command)
        with pytest.raises(GuestServerRecovered):
            await provider.execute(["must_not_post()", "nor_this()"])
        assert len(calls) == 1
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
        replies = iter([guest_disk.CommandResult(1, "", ""), _receipt([_restart_record()])])
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
        monkeypatch.setattr(
            provider, "_run_guest_command", lambda *a, **k: _receipt([_restart_record()])
        )
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
            lambda *a: guest_supervisor.ReceiptResult("verified_none"),
        )
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: False)
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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)
        reads = 0
        input_posts = []

        def command(argv: Any, **kwargs: Any) -> guest_disk.CommandResult:
            nonlocal reads
            if "/var/lib/lop-action-supervisor" in str(argv):
                reads += 1
                return _receipt([] if reads < 3 else [_restart_record()])
            input_posts.append(argv)
            return guest_disk.CommandResult(0, "", "")

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
        monkeypatch.setattr(provider, "_probe_control_server", lambda timeout: True)

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


def _write_listeners(proc: Path, inodes_by_pid: dict[int, int]) -> None:
    """Point each pid's fd at its own listening socket and publish the inodes."""
    for pid, inode in inodes_by_pid.items():
        directory = proc / str(pid) / "fd"
        directory.mkdir(parents=True)
        (directory / "3").symlink_to(f"socket:[{inode}]")
    net = proc / "net"
    net.mkdir()
    lines = ["  sl  local_address rem_address   st"]
    for inode in inodes_by_pid.values():
        lines.append(f"   0: 00000000:1388 00000000:0000 0A 00:00000000 00000000 0 0 1000 {inode}")
    (net / "tcp").write_text("\n".join(lines) + "\n")
    (net / "tcp6").write_text("  sl\n")


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
