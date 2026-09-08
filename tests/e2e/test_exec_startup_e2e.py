"""Installed-style CLI/worker tests against a real loopback provider.

Only the remote model is scripted. CLI parsing, config, session construction,
team attachment, runtime discovery, transcript writes and worker disposal are
production code. Every process uses an isolated HOME and no ambient cmux IDs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry


@pytest.fixture
def exec_server(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    root = home / ".local-operator"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    for key in list(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    requests: list[dict[str, Any]] = []
    release = threading.Event()
    worker_ids: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            body = json.dumps(
                {"object": "list", "data": [{"id": "exec-fixture", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            wire = json.dumps(body.get("messages", []))
            if "FAIL_FIXTURE" in wire:
                error = json.dumps(
                    {
                        "error": {
                            "message": "fixture invalid request",
                            "type": "invalid_request_error",
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(error)))
                self.end_headers()
                self.wfile.write(error)
                return
            if "HOLD_FIXTURE" in wire:
                release.wait(timeout=45)
            delta = {"role": "assistant", "content": "Fixture completed"}
            finish_reason = "stop"
            called = {
                call["function"]["name"]
                for message in body.get("messages", [])
                for call in message.get("tool_calls", [])
            }
            # Request real tools: the dispatcher and delegated child are not mocked.
            tool = None
            if "DELEGATE_FIXTURE" in wire and "task" not in called:
                tool = (
                    "task",
                    {
                        "i": "Delegating fixture work",
                        "label": "fixture-child",
                        "prompt": "CHILD_TASK: verify the project brief",
                        "agent": "coder",
                    },
                )
            elif "DELEGATE_FIXTURE" in wire and "wait" not in called:
                tool = (
                    "wait",
                    {"i": "Awaiting fixture child", "job_id": "fixture-child", "wait_ms": 10000},
                )
            elif "WRITE_FIXTURE" in wire and "write" not in called:
                tool = (
                    "write",
                    {
                        "i": "Writing fixture evidence",
                        "path": str(tmp_path / "written.txt"),
                        "content": "side-effect",
                    },
                )
            if tool:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_" + tool[0],
                            "type": "function",
                            "function": {"name": tool[0], "arguments": json.dumps(tool[1])},
                        }
                    ],
                }
                finish_reason = "tool_calls"
            elif "You are judging" in wire:
                delta = {"role": "assistant", "content": "VERDICT: ACHIEVED\nFixture verified"}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in [(delta, None), ({}, finish_reason)]:
                chunk = {
                    "id": "fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "exec-fixture",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    config = ConfigManager(root)
    config.update_config(
        {
            "hosting": "openai-compatible",
            "model_name": "exec-fixture",
            "providers": {
                "openai-compatible": {
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "models": {"exec-fixture": {"context_window": 100000, "supports_tools": True}},
                }
            },
        }
    )
    TeamRegistry(root).create_team(
        TeamEditFields(
            name="release",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="COLLABORATION_SENTINEL: review before shipping",
            project="PROJECT_SENTINEL: verify headless teams",
        )
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_") and k != "NO_COLOR"}
    env["TERM"] = "xterm-256color"

    def run(*args, stdin=None):
        result = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", *args],
            env=env,
            input=stdin,
            text=True,
            capture_output=True,
            timeout=60,
            cwd=tmp_path,
        )
        print("COMMAND", "lop", *args, "EXIT", result.returncode)
        print("STDOUT", result.stdout)
        print("STDERR", result.stderr)
        if "Background job " in result.stderr:
            worker_ids.append(result.stderr.split("Background job ", 1)[1].split(":", 1)[0])
        return result

    setattr(run, "release", release)
    setattr(run, "env", env)
    try:
        yield run, requests, root
    finally:
        import signal

        from local_operator.exec_mode import job_status
        from local_operator.tools.group_reaper import _owner_is_dead

        release.set()
        # Assertions may fail while a supervisor gate is parked. Clean only
        # this fixture's generation-verified workers, never ambient discovery.
        for job_id in worker_ids:
            state = job_status(job_id)
            generation = state.get("process_generation")
            if (
                state.get("status") in ("starting", "running")
                and generation
                and _owner_is_dead(state["pid"], generation) is False
            ):
                os.kill(state["pid"], signal.SIGTERM)
        server.shutdown()
        server.server_close()
        serving.join(timeout=5)


def test_exec_team_count_loop_and_resume(exec_server):
    run, requests, root = exec_server
    result = run(
        "exec",
        "Initial task",
        "--team",
        "release",
        "--profile",
        "reviewer",
        "--goal",
        "Ship safely",
        "--loop",
        "2",
        "--name",
        "Headless audit",
        "--json",
        stdin="",
    )
    assert result.returncode == 0
    assert len(requests) == 3
    wire = json.dumps(requests[0])
    for expected in ("COLLABORATION_SENTINEL", "PROJECT_SENTINEL", "manager", "coder", "reviewer"):
        assert expected in wire
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows
    session_id = result.stderr.split("session_id=", 1)[1].split()[0]
    directory = root / "sessions" / session_id
    assert directory.is_dir()
    resumed = run(
        "exec", "Resume without replay", "--resume", session_id, "--clear-goal", "--json", stdin=""
    )
    assert resumed.returncode == 0
    assert len(requests) == 4
    missing_goal = run("exec", "--resume", session_id, "--loop", "1", stdin="")
    assert missing_goal.returncode != 0
    assert len(requests) == 4
    assert "COLLABORATION_SENTINEL" in json.dumps(requests[-1])
    persisted = (directory / "transcript.jsonl").read_text()
    assert "Headless audit" in persisted
    assert '"status": "completed"' in json.dumps(
        [json.loads(line) for line in persisted.splitlines()]
    )


def test_exec_unknown_and_goal_only_do_not_call_provider(exec_server):
    run, requests, _ = exec_server
    for args in [
        ("exec", "task", "--team", "missing", "--background"),
        ("exec", "task", "--profile", "missing"),
        ("exec", "--goal", "alone"),
    ]:
        assert run(*args, stdin="").returncode != 0
    assert not requests


def test_exec_agent_profile_stdin_and_numeric_goal(exec_server):
    from local_operator.agents import AgentRegistry

    run, requests, root = exec_server
    legacy = run(
        "exec", "--agent", "legacy-worker", "--name", "Legacy fixture", stdin="Piped legacy task\n"
    )
    assert legacy.returncode == 0
    agent = AgentRegistry(root).get_agent_by_name("legacy-worker")
    assert agent is not None
    by_id = run(
        "exec", "-", "--agent-id", agent.id, "--name", "Exact fixture", stdin="Exact agent task\n"
    )
    assert by_id.returncode == 0
    goal = run("exec", "--loop-goal", "123", "--name", "Goal fixture", stdin="")
    assert goal.returncode == 0
    assert "You are judging" in json.dumps(requests[-1])
    invalid = run("exec", "No model turn", "--effort", "not-a-level", stdin="")
    assert invalid.returncode != 0
    assert len(requests) == 4


def test_exec_delegates_actual_team_child(exec_server):
    run, requests, _ = exec_server
    result = run(
        "exec",
        "DELEGATE_FIXTURE",
        "--team",
        "release",
        "--yolo",
        "--name",
        "Delegation fixture",
        stdin="",
    )
    assert result.returncode == 0
    child_requests = [
        request
        for request in requests
        if "CHILD_TASK" in json.dumps(request) and "DELEGATE_FIXTURE" not in json.dumps(request)
    ]
    assert child_requests, "The actual delegated child must reach the provider"
    child_wire = json.dumps(child_requests[0])
    assert "COLLABORATION_SENTINEL" in child_wire
    assert "PROJECT_SENTINEL" in child_wire
    assert "coder" in child_wire


def test_exec_default_non_tty_denies_and_explicit_yolo_writes(exec_server, tmp_path):
    run, requests, _ = exec_server
    denied = run("exec", "WRITE_FIXTURE", "--name", "Deny fixture", stdin="")
    assert denied.returncode == 0
    assert not (tmp_path / "written.txt").exists()
    assert "denied" in json.dumps(requests).lower()
    allowed = run("exec", "WRITE_FIXTURE", "--yolo", "--name", "Allow fixture", stdin="")
    assert allowed.returncode == 0
    assert (tmp_path / "written.txt").read_text() == "side-effect"


@pytest.mark.asyncio
@pytest.mark.parametrize("team", [None, "release"])
async def test_exec_live_tui_attachment_and_settled_frames(exec_server, tmp_path, team):
    import asyncio
    from argparse import Namespace

    from local_operator.agents import AgentRegistry
    from local_operator.credentials import CredentialManager
    from local_operator.exec_mode import ExecArgs, job_status
    from local_operator.session.remote import RemoteSession
    from local_operator.session_factory import create_session
    from local_operator.tui.app import OperatorApp
    from tests.e2e.harness import wait_for_adoption

    run, requests, root = exec_server
    options = ["--team", team] if team else []
    result = run(
        "exec", "HOLD_FIXTURE", "--background", "--name", "Headless audit", *options, stdin=""
    )
    assert result.returncode == 0
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]
    status = json.loads(run("exec", "--status", job_id, stdin="").stdout)
    session_id = status["session_id"]

    async def factory():
        return await create_session(
            Namespace(**vars(ExecArgs(resume=session_id))),
            ConfigManager(root),
            CredentialManager(root),
            AgentRegistry(root),
            has_ui=True,
            cwd=str(tmp_path),
        )

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(110, 34)) as pilot:
            await wait_for_adoption(app, pilot)
            assert isinstance(app._session, RemoteSession)
            assert app._session.session_id == session_id
            assert app._session.active_team_name == (team or "")
            await pilot.pause()
            destination = os.environ.get("EXEC_EVIDENCE_DIR")
            if destination:
                out = Path(destination)
                out.mkdir(parents=True, exist_ok=True)
                label = "after-team" if team else "before-no-team"
                app.save_screenshot(str(out / f"{label}-first.svg"))
                await pilot.pause()
                app.save_screenshot(str(out / f"{label}-settled.svg"))
                geometry = {
                    widget.id: {
                        "region": str(widget.region),
                        "size": str(widget.size),
                        "virtual_size": str(widget.virtual_size),
                    }
                    for widget in app.query("#transcript, #editor, #status-bar")
                }
                (out / f"{label}-geometry.json").write_text(json.dumps(geometry, indent=2))
            # A viewer detaching must leave the exec owner running.
        assert json.loads(run("exec", "--status", job_id, stdin="").stdout)["status"] == "running"
    finally:
        getattr(run, "release").set()
        for _ in range(100):
            await asyncio.sleep(0.05)
            if job_status(job_id)["status"] not in ("starting", "running"):
                break
    assert requests
    assert job_status(job_id)["status"] == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("approve", [False, True])
async def test_exec_supervisor_approval_ui(exec_server, tmp_path, approve):
    import asyncio
    from argparse import Namespace

    from local_operator.agents import AgentRegistry
    from local_operator.credentials import CredentialManager
    from local_operator.exec_mode import ExecArgs, job_status
    from local_operator.session_factory import create_session
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.approval import ApprovalPrompt
    from tests.e2e.harness import wait_for_adoption

    run, requests, root = exec_server
    result = run(
        "exec",
        "WRITE_FIXTURE",
        "--control",
        "--background",
        "--name",
        "Supervised fixture",
        stdin="",
    )
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]
    state = job_status(job_id)
    assert state["status"] == "running"
    assert not (tmp_path / "written.txt").exists()

    async def factory():
        return await create_session(
            Namespace(**vars(ExecArgs(resume=state["session_id"]))),
            ConfigManager(root),
            CredentialManager(root),
            AgentRegistry(root),
            has_ui=True,
            cwd=str(tmp_path),
        )

    app = OperatorApp(factory)
    async with app.run_test(size=(110, 34)) as pilot:
        await wait_for_adoption(app, pilot)
        for _ in range(100):
            await pilot.pause()
            if app.query(ApprovalPrompt):
                break
        assert app.query(ApprovalPrompt), "An explicit --control run must expose its parked gate"
        destination = os.environ.get("EXEC_EVIDENCE_DIR")
        if destination:
            app.save_screenshot(
                str(Path(destination) / f"supervised-{'allow' if approve else 'deny'}-pending.svg")
            )
        await pilot.press("y" if approve else "n")
        for _ in range(100):
            await asyncio.sleep(0.05)
            await pilot.pause()
            if job_status(job_id)["status"] not in ("starting", "running"):
                break
        assert job_status(job_id)["status"] == "succeeded"
        if destination:
            app.save_screenshot(
                str(Path(destination) / f"supervised-{'allow' if approve else 'deny'}-settled.svg")
            )
    assert (tmp_path / "written.txt").exists() is approve
    assert requests


@pytest.mark.parametrize(
    "termination,expected", [("term", "cancelled"), ("kill", "interrupted"), ("stop", "cancelled")]
)
def test_exec_loop_lifecycle_outcomes(exec_server, termination, expected):
    import signal
    import time

    from local_operator.exec_mode import job_status

    run, requests, root = exec_server
    result = run(
        "exec",
        "--goal",
        "HOLD_FIXTURE",
        "--loop",
        "2",
        "--name",
        "Lifecycle fixture",
        "--background",
        stdin="",
    )
    assert result.returncode == 0
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]
    status = job_status(job_id)
    assert status["status"] == "running"
    assert status["process_generation"]
    if termination == "stop":
        response = run("stop", "--session", status["session_id"], "--yes", stdin="")
        assert response.returncode == 0
    else:
        # Only the worker created by THIS fixture, under its private config,
        # is signalled. Never target ambient session discovery or cmux state.
        os.kill(status["pid"], signal.SIGTERM if termination == "term" else signal.SIGKILL)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = job_status(job_id)
        if status["status"] not in ("starting", "running"):
            break
        time.sleep(0.05)
    assert status["status"] == expected
    before = len(requests)
    assert json.loads(run("exec", "--status", job_id, stdin="").stdout)["status"] == expected
    assert len(requests) == before, "Status/reconciliation must never restart iterations"
    if termination != "kill":
        assert not Path(status["runtime_path"]).exists()


def test_detached_worker_outliving_its_launcher_stays_running(exec_server):
    """Launcher exit is NOT evidence about the worker.

    A detached run deliberately survives the process that spawned it (the group
    reaper excludes exec workers for this reason), so reconciliation must fence
    on the WORKER's own pid+generation and never infer a terminal state from the
    launcher being gone. Getting this wrong would close a browser scope out from
    under live work — the hazard the browser owner asked us to rule out.
    """
    import time

    from local_operator.exec_mode import job_status
    from local_operator.tools.group_reaper import _owner_is_dead

    run, _requests, _root = exec_server
    # The launcher is `run(...)`: a subprocess that has already exited by the
    # time it returns, while the worker it spawned keeps holding the model.
    result = run("exec", "HOLD_FIXTURE", "--background", "--name", "Survivor fixture", stdin="")
    assert result.returncode == 0
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]

    status = job_status(job_id)
    assert status["status"] == "running"
    # The worker's generation is live; the launcher is provably gone.
    assert _owner_is_dead(status["pid"], status["process_generation"]) is False

    # Reconciliation re-run while the launcher is dead must NOT move it.
    for _ in range(5):
        time.sleep(0.05)
        assert job_status(job_id)["status"] == "running"

    getattr(run, "release").set()
    deadline = time.monotonic() + 15
    status = job_status(job_id)
    while time.monotonic() < deadline and status["status"] in ("starting", "running"):
        time.sleep(0.05)
        status = job_status(job_id)
    assert status["status"] == "succeeded"


def test_exec_loop_failure_is_durable_and_stops_iterations(exec_server):
    import time

    from local_operator.exec_mode import job_status

    run, requests, _ = exec_server
    result = run(
        "exec",
        "--goal",
        "FAIL_FIXTURE",
        "--loop",
        "2",
        "--name",
        "Failure fixture",
        "--background",
        stdin="",
    )
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]
    deadline = time.monotonic() + 15
    status = job_status(job_id)
    while time.monotonic() < deadline and status["status"] in ("starting", "running"):
        time.sleep(0.05)
        status = job_status(job_id)
    assert status["status"] == "failed"
    assert status["exit_code"] == 1
    assert len(requests) == 1
    transcript = Path(status["session_directory"]) / "transcript.jsonl"
    assert '"status": "failed"' in json.dumps(
        [json.loads(line) for line in transcript.read_text().splitlines()]
    )


def test_exec_background_receipt_and_durable_status(exec_server):
    import time

    run, requests, root = exec_server
    result = run(
        "exec",
        "Detached task",
        "--team",
        "release",
        "--background",
        "--name",
        "Detached fixture",
        stdin="",
    )
    assert result.returncode == 0
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]
    deadline = time.monotonic() + 30
    status = {}
    while time.monotonic() < deadline:
        response = run("exec", "--status", job_id, stdin="")
        assert response.returncode == 0
        status = json.loads(response.stdout)
        if status["status"] not in ("starting", "running"):
            break
    assert status["status"] == "succeeded"
    assert status["session_id"] != job_id
    assert status["process_generation"]
    assert Path(status["log"]).is_file()
    assert Path(status["session_directory"]).is_dir()
    assert not Path(status["runtime_path"]).exists()
    assert len(requests) == 1
