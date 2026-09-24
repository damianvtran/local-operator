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
from tests.e2e.harness import NO_NOTIFY_ENV, provider_call_kinds, user_turns


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
    # A real ``lop`` CLI child. Re-asserted rather than inherited: the strip
    # above removes the pane families only, and this mapping is handed to a
    # process that can park a gate (see the ``--team`` cell below, which drives
    # a real runtime) — with no gate it would put a genuine macOS banner on the
    # operator's screen from a test.
    env.update(NO_NOTIFY_ENV)

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
    setattr(run, "cwd", tmp_path)
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


def _running_status(job_status: Any, job_id: str, timeout: float = 20.0) -> dict[str, Any]:
    """The job's status once its detached worker has left ``starting``.

    ``--background`` returns as soon as the job record exists, and the worker
    marks it ``running`` from its own process some time later. A cell that
    asserts ``running`` on the next line is asserting a wall-clock race: it held
    on a quiet host and failed under load (QA round 2 on #1475, Q5, which saw
    ``'starting' == 'running'`` in three cells with a different set failing on
    each of three runs). This is the deadline-bounded wait
    ``test_dash_leading_values_reach_the_detached_worker`` already used, so a
    worker that never boots still fails, on the caller's own assertion.
    """
    import time

    deadline = time.monotonic() + timeout
    state = job_status(job_id)
    while time.monotonic() < deadline and state.get("status") == "starting":
        time.sleep(0.05)
        state = job_status(job_id)
    return state


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
    # A CENSUS, not a total. This command starts ONE user turn, and the standing
    # goal's judge rides beside it: the judge is armed by ``--goal`` and NOT by
    # ``--loop`` (a count loop judges nothing — ``GoalLoop.run``'s ``if goal:``
    # guards that), so it fires ONCE, at the first turn end, and then stops
    # because its verdict settled the goal. The loop then contributes its own two
    # iterations. A bare ``== 3`` was written before the judge existed and cannot
    # tell this from a command that re-submitted its argument, which is what the
    # census below is for (see ``provider_call_kinds``).
    kinds = provider_call_kinds(requests, goal="Ship safely")
    assert user_turns(kinds) == 1, "one user-authored turn, submitted once"
    assert kinds.count("judge") == 1, "the goal judge's forked aside, exactly once"
    assert kinds.count("loop") == 2, "the count loop's own two iterations"
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
    # One more user turn and NOTHING else: ``--clear-goal`` is what stops a
    # judge riding this turn end, so the goal's aside is still the single call
    # the first command made.
    kinds = provider_call_kinds(requests, goal="Ship safely")
    assert user_turns(kinds) == 2, "the exec turn and the resume's own turn"
    assert kinds.count("judge") == 1, "a cleared goal is not judged"
    assert kinds.count("loop") == 2
    before = len(requests)
    missing_goal = run("exec", "--resume", session_id, "--loop", "1", stdin="")
    assert missing_goal.returncode != 0
    # A REFUSED run reaches no provider at all, and this is the refusal the
    # cleared goal must produce: ``--loop`` needs a goal and this resume has
    # none. It is the regression the cell exists for — clearing only the
    # attachment left the text in ``goal.json``, so this resume read the goal
    # back out of the sidecar and STARTED a loop instead of refusing.
    assert len(requests) == before
    assert "COLLABORATION_SENTINEL" in json.dumps(requests[-1])
    persisted = (directory / "transcript.jsonl").read_text()
    assert "Headless audit" in persisted
    assert '"status": "completed"' in json.dumps(
        [json.loads(line) for line in persisted.splitlines()]
    )


def test_exec_goal_over_a_settled_goal_is_judged_again(exec_server):
    """Agent review round 2, MAJOR-5, end to end through the real CLI.

    The unit half is ``tests/unit/test_exec_startup.py``'s ``apply_startup`` pin.
    What only the real thing can show is that a SECOND ``--goal`` over a goal the
    judge already settled is actually JUDGED: the judge's own provider call, the
    ``<goal>`` block on the wire, and the two durable halves still naming the same
    objective. Before the fix the new objective kept the settled goal's ``done``,
    so the judge's aside never came and the block was withheld — a goal silently
    inert on a documented CLI path.
    """
    run, requests, root = exec_server
    first = run("exec", "Ship the first objective", "--goal", "Ship safely", "--json", stdin="")
    assert first.returncode == 0
    kinds = provider_call_kinds(requests, goal="Ship safely")
    assert kinds.count("judge") == 1
    session_id = first.stderr.split("session_id=", 1)[1].split()[0]
    directory = root / "sessions" / session_id
    # The state MAJOR-5 is about, asserted rather than assumed: the goal this run
    # set is SETTLED, so the next objective lands on top of a settled record.
    settled = json.loads((directory / "goal.json").read_text())
    assert settled["goal"] == "Ship safely"
    assert settled["status"] == "done"

    before = len(requests)
    second = run(
        "exec",
        "Land the new objective",
        "--resume",
        session_id,
        "--goal",
        "Land the new billing migration",
        "--json",
        stdin="",
    )
    assert second.returncode == 0
    second_requests = requests[before:]
    kinds = provider_call_kinds(second_requests, goal="Land the new billing migration")
    assert kinds.count("judge") == 1, "the new objective gets its own verdict"
    wire = json.dumps(second_requests[0])
    assert "Land the new billing migration" in wire, "the new objective is the one in the prompt"
    assert "The user's standing objective" in wire, "the <goal> block is not withheld"
    record = json.loads((directory / "goal.json").read_text())
    # The judge's ACHIEVED verdict settled the NEW goal, and the goal it replaced
    # is kept as history — the two halves of the record agree about which goal is
    # current, which is what the attachment used to contradict.
    assert record["goal"] == "Land the new billing migration"
    assert record["status"] == "done"
    assert [row["text"] for row in record["history"]] == [
        "Land the new billing migration",
        "Ship safely",
    ]
    attachment = json.loads((directory / "attachment.json").read_text())
    assert attachment["goal"] == "Land the new billing migration"


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
    from local_operator.exec_mode import ExecArgs, job_status
    from local_operator.session.attached import AttachedSession
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
            AgentRegistry(root),
            has_ui=True,
            cwd=str(tmp_path),
        )

    app = OperatorApp(factory)
    try:
        async with app.run_test(size=(110, 34)) as pilot:
            await wait_for_adoption(app, pilot)
            assert isinstance(app._session, AttachedSession)
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
    """A supervisor answers a parked ``--control`` gate — DENYING it, not allowing it.

    The supervisor here is an ``OperatorApp`` in the TEST process and the run
    whose gate it answers was started by a SEPARATE ``lop exec --control
    --background`` process. Under issue #1310 that distinction is load-bearing:
    `/approvals auto` and an APPROVED card remove the gate that constrains the
    caller, so they additionally require the per-session operator capability. The
    supervisor did not start this one, so its ``y`` is refused — while its ``n``
    (deny) is deliberately ordinary and must keep working, since a deny settles
    the card in the safe direction.

    WHAT THIS CELL IS *NOT* (UX round 6, U9). Its docstring used to state the
    refusal as the general rule for supervisor approval — "a run that must be
    approved interactively has to be started where the approver is" — and that is
    the model revision 2 replaces, not the rule it ships. The rule now has a
    stated exception with a flag: ``--supervisor-fd`` hands the run's capability UP
    to the supervisor, which is exactly how a supervisor in another process DOES
    approve. This cell keeps the ``--background`` shape (no such flag can be
    passed to it) and the sibling below drives the exception.
    """
    import asyncio
    from argparse import Namespace

    from local_operator.agents import AgentRegistry
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
    state = _running_status(job_status, job_id)
    assert state["status"] == "running"
    assert not (tmp_path / "written.txt").exists()

    async def factory():
        return await create_session(
            Namespace(**vars(ExecArgs(resume=state["session_id"]))),
            ConfigManager(root),
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
        if approve:
            # REFUSED, and the refusal is not silent at the wire: the run stays
            # parked with its gate unanswered, so the tool never runs and the
            # job stays running. Asserted in both directions because "the job is
            # still running" would also be true of a supervisor that never saw
            # the card at all — the parked prompt above is what rules that out.
            assert (
                job_status(job_id)["status"] == "running"
            ), "a supervisor that did not start this run settled its gate by approval"
            # AND THE OPERATOR IS TOLD, on this surface too (item 8 of the round-3
            # remediation). "The job is still running" is also true of a refusal
            # nobody ever saw — which is what the round-2 defect was, on the pane
            # that pressed the key — so the notice is asserted rather than
            # assumed, and it is the CARD's sentence rather than a command's.
            from local_operator.tui.widgets.transcript import (
                NoticeBlock,
                TranscriptView,
            )

            # THE SENTENCE HAS TWO HOST FORMS NOW (agent review round 6, U1/U2):
            # with an anchor installed the card refusal names the two surfaces that
            # can sign, and on a host whose anchor is not installed it names
            # `lop operator install` instead — because on that host neither surface
            # can act. This cell is about the refusal REACHING an operator at all,
            # which is the half both forms share, so it asserts the shared opening
            # rather than pinning one host's form. The two forms are pinned
            # individually in the seam suite, where the anchor state is controlled.
            opening = "this approval is still waiting: only the operator can allow it"
            notices: list[str] = []
            for _ in range(60):
                await pilot.pause()
                notices = [
                    block._text
                    for block in app.query_one(TranscriptView).blocks()
                    if isinstance(block, NoticeBlock)
                ]
                if any(opening in text for text in notices):
                    break
                await asyncio.sleep(0.05)
            assert any(
                opening in text for text in notices
            ), f"the refusal never reached the operator: {notices}"
        else:
            assert job_status(job_id)["status"] == "succeeded"
        if destination:
            app.save_screenshot(
                str(Path(destination) / f"supervised-{'allow' if approve else 'deny'}-settled.svg")
            )
    # NEITHER direction lets the write tool run: a deny denies it, and an
    # approval from a process that did not start the run is refused.
    assert not (tmp_path / "written.txt").exists(), "the gated tool ran anyway"
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
    status = _running_status(job_status, job_id)
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


def test_dash_leading_values_reach_the_detached_worker(exec_server):
    """A value starting with ``-`` must survive the launcher/worker argv hop.

    Forwarded as two argv items, ``--name -nightly`` reads as an OPTION to the
    worker's argparse and dies at ``parse_args`` — before ``--job-id`` is
    honoured, so no terminal ledger row is written and reconciliation reports
    ``interrupted``: the word reserved for a worker killed mid-flight, for a
    run that never started. The values most likely to lead with a dash are the
    free-text ones this feature adds.
    """
    import time

    from local_operator.exec_mode import job_status

    run, _requests, _root = exec_server
    # The `=` form is what a user must type for a dash-leading value; the
    # launcher's own argparse rejects the two-token form just as the worker's
    # does. The bug was that the launcher ACCEPTED this and then re-emitted it
    # to the worker in the two-token form the worker cannot parse.
    result = run(
        "exec", "HOLD_FIXTURE", "--background", "--name=-nightly", "--goal=-- ship it", stdin=""
    )
    assert result.returncode == 0
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]

    deadline = time.monotonic() + 20
    state = job_status(job_id)
    while time.monotonic() < deadline and state.get("status") == "starting":
        time.sleep(0.05)
        state = job_status(job_id)
    # The worker actually booted, rather than dying at parse_args and being
    # mislabelled `interrupted` by reconciliation.
    assert state["status"] == "running", state
    assert state["session_id"]

    getattr(run, "release").set()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and state["status"] in ("starting", "running"):
        time.sleep(0.05)
        state = job_status(job_id)
    assert state["status"] == "succeeded", state


def test_parked_supervised_run_is_distinguishable_from_working(exec_server):
    """``--status`` must tell "waiting for you" apart from "working".

    A supervised run parked on an approval sits at ``running`` indefinitely,
    and that is the one state a user has to notice. ``lop sessions`` already
    computes it; this asserts ``--status`` reports the same thing rather than
    collapsing both states into ``running``.
    """
    import time

    from local_operator.exec_mode import job_status

    run, _requests, _root = exec_server
    result = run(
        "exec", "WRITE_FIXTURE", "--control", "--background", "--name", "Parked fixture", stdin=""
    )
    assert result.returncode == 0
    # The receipt must name the command that shows what a run needs.
    assert "lop sessions" in result.stderr
    job_id = result.stderr.split("Background job ", 1)[1].split(":", 1)[0]

    # The gate parks asynchronously; wait for the state rather than sleeping.
    deadline = time.monotonic() + 30
    state = job_status(job_id)
    while time.monotonic() < deadline and not state.get("pending"):
        time.sleep(0.1)
        state = job_status(job_id)
    assert state["status"] == "running"
    assert state["pending"] == "approval", state

    # Live state only: it must never be frozen into the durable ledger, where
    # every row is a fact that stays true.
    persisted = (Path(state["log"]).parent / "exec-jobs.jsonl").read_text()
    assert '"pending"' not in persisted


def test_loop_only_run_does_not_block_on_an_inherited_open_pipe(exec_server):
    """A loop-only run must not read a stdin that never closes.

    Every other case here passes ``stdin=""`` through ``subprocess.run``, which
    closes the pipe immediately — so the suite structurally could not see this.
    A supervisor (CI runner, cmux surface, ``Popen(stdin=PIPE)``) hands its
    child an inherited pipe and keeps the write end OPEN, so ``sys.stdin.read()``
    never sees EOF and the documented ``exec --goal X --loop N`` hangs forever
    with no output. This test therefore opens the pipe and deliberately never
    writes to or closes it.
    """
    import subprocess as sp

    run, _requests, _root = exec_server
    env = getattr(run, "env")
    process = sp.Popen(
        [
            sys.executable,
            "-m",
            "local_operator.cli",
            "exec",
            "--goal",
            "Ship safely",
            "--loop",
            "1",
        ],
        env=env,
        text=True,
        stdin=sp.PIPE,  # held open on purpose; nothing is ever written or closed
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        cwd=getattr(run, "cwd"),
    )
    try:
        # Generous relative to the run itself (the fixture answers instantly),
        # but finite: before the fix this waited forever.
        process.wait(timeout=45)
    except sp.TimeoutExpired:
        process.kill()
        process.wait()
        raise AssertionError("loop-only exec blocked on an inherited open stdin")
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    stdout = process.stdout.read() if process.stdout else ""
    stderr = process.stderr.read() if process.stderr else ""
    assert process.returncode == 0, (stdout, stderr)


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

    status = _running_status(job_status, job_id)
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


def test_a_supervised_run_is_approved_through_the_handoff(exec_server, tmp_path):
    """Stage E, driven through the REAL CLI: the supervisor's ``y`` is ACCEPTED.

    WHY THIS CELL EXISTS (agent review round 6, R6-5). Stage E's positive path was
    proven only in-process, and ``open_supervisor_cap_channel`` had no caller
    outside tests — so the one thing the stage claims, that a supervisor in another
    process can approve a card this run parks, was never exercised against the
    shipped CLI. This drives it exactly as the helper documents it:

    1. one channel, created before the child exists;
    2. the run launched with ``channel.argv`` appended and ``pass_fds`` handed to the
       spawn, so only the descriptor NUMBER rides in argv;
    3. the run's endpoint line, read off stderr (where it belongs: stdout is the
       machine-readable payload stream), which carries the pid;
    4. ``remember_operator_cap(pid, channel.read())`` — after which the supervisor's
       own attach presents the proof with no further work.

    Then the operator-visible outcome: the TUI supervisor presses ``y``, the card
    settles APPROVED, the gated ``write`` runs, and the run exits 0. The sibling cell
    above asserts the opposite for the shape with no descriptor (a ``--background``
    run, whose card nobody may approve), so the pair states the rule and its
    exception rather than one of them.

    The supervisor here is this test process, which is also where the TUI attaches
    from — the same split the sibling uses, with the descriptor being the only
    difference. No provider is reached but the fixture's, and every process runs
    under the fixture's stripped environment (no ``CMUX_*``, notifications off).
    """
    import asyncio
    import queue
    import threading
    from argparse import Namespace

    from local_operator.agents import AgentRegistry
    from local_operator.exec_mode import ExecArgs
    from local_operator.harness.approval import (
        OPERATOR_CAP_BYTES,
        open_supervisor_cap_channel,
        operator_cap_for,
        remember_operator_cap,
    )
    from local_operator.session_factory import create_session
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.approval import ApprovalPrompt
    from tests.e2e.harness import wait_for_adoption

    run, _requests, root = exec_server
    # THE DOCUMENTED ESCAPE HATCH, and the reason this cell needs it while its
    # sibling does not: `subprocess.run` in the fixture inherits the test process's
    # environment, and an agent's own shell carries `LOCAL_OPERATOR_AGENT_SHELL`,
    # which refuses `lop exec` a session the operator did not open. That refusal is
    # about SESSIONS, and the sibling cell dodges it by taking `--background`.
    # A real-CLI e2e run from inside an agent's shell is exactly what
    # `LOCAL_OPERATOR_ALLOW_NESTED_SESSION` exists for, so it is set here rather
    # than the marker being scrubbed: scrubbing would run the CLI in an environment
    # no operator has.
    env = dict(run.env)
    env["LOCAL_OPERATOR_ALLOW_NESTED_SESSION"] = "1"
    channel = open_supervisor_cap_channel()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "local_operator.cli",
            "exec",
            "WRITE_FIXTURE",
            "--control",
            "--name",
            "Supervised fixture",
            *channel.argv,
        ],
        env=env,
        cwd=run.cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=channel.pass_fds,
        close_fds=channel.close_fds,
    )
    lines: queue.Queue[str] = queue.Queue()

    def pump(stream: Any, label: str) -> None:
        for line in stream:
            lines.put(f"{label} {line}")

    # BOTH streams drained, and both kept: the endpoint line belongs to stderr, but
    # a run that fails before reaching it explains itself on either — and a cell
    # that reads only one of them turns an early exit into a 90-second timeout with
    # nothing to say about why.
    for stream, label in ((process.stdout, "OUT"), (process.stderr, "ERR")):
        threading.Thread(target=pump, args=(stream, label), daemon=True).start()
    try:
        # STEP 4's credential first, because a supervisor that attaches before it
        # holds the capability is exactly the shape the sibling cell refuses: the
        # run writes it up the descriptor at startup and then closes it.
        capability = channel.read(timeout_s=90.0)
        if capability is None:
            captured = []
            while not lines.empty():
                captured.append(lines.get_nowait().rstrip())
            raise AssertionError(
                "the run did not write its capability upward; child said: "
                + " | ".join(captured[-14:])
            )
        assert len(capability) == OPERATOR_CAP_BYTES

        endpoint = ""
        for _ in range(600):
            try:
                line = lines.get(timeout=30)
            except queue.Empty:  # pragma: no cover — a run that never announced itself
                break
            print("RUN", line.rstrip())
            if "lop exec control:" in line:
                endpoint = line
                break
        assert "lop exec control:" in endpoint, endpoint
        pid = int(endpoint.split("pid=", 1)[1].split()[0])
        session_id = endpoint.split("session_id=", 1)[1].split()[0]

        # THE SUPERVISOR'S HALF, asserted rather than assumed: the value is filed
        # under the pid the endpoint line already printed, which is what makes an
        # attached client's proof arrive without any wiring at the attach site.
        remember_operator_cap(pid, capability)
        assert operator_cap_for(pid) == capability

        async def factory():
            return await create_session(
                Namespace(**vars(ExecArgs(resume=session_id))),
                ConfigManager(root),
                AgentRegistry(root),
                has_ui=True,
                cwd=str(tmp_path),
            )

        async def supervise() -> None:
            app = OperatorApp(factory)
            async with app.run_test(size=(110, 34)) as pilot:
                await wait_for_adoption(app, pilot)
                for _ in range(200):
                    await pilot.pause()
                    if app.query(ApprovalPrompt):
                        break
                assert app.query(ApprovalPrompt), "the supervised run must park its gate"
                await pilot.press("y")
                for _ in range(200):
                    await asyncio.sleep(0.05)
                    await pilot.pause()
                    if process.poll() is not None:
                        break

        asyncio.run(supervise())
        assert process.wait(timeout=90) == 0, process.returncode
        # THE EFFECT, not a flag: the gated tool really ran.
        assert (tmp_path / "written.txt").exists(), "the approved tool never ran"
    finally:
        channel.close()
        if process.poll() is None:  # pragma: no cover — only on a failed cell
            process.kill()
            process.wait(timeout=30)
