"""#2803, end to end: a RESUMED headless turn gets its conversation's ask queue.

The report was a session that had asked something, been stopped, and then been
resumed headless — where the turn ran with no ``ask`` tool at all and no queue
behind it, so the ask it had left open could never be settled and a deadline that
passed while the process was down reached the model only if some later turn
happened to start.

Three properties are asserted here, and all three are acceptance criteria of the
slice plan's PR-2 (SA-D2):

1. ``exec --background --resume`` runs its turn with ``ask`` and ``ask_withdraw``
   on the wire, and a withdrawal issued from that turn settles the index row.
2. A deadline that passed before the resume is delivered AT that turn's start —
   the timeout row is in the transcript, and in the request, ahead of the prompt.
3. ``sessions(op='resume')`` from another session reaches the same surface: the
   tool spawns the same ``exec --resume``, so the child's request carries the
   tools too.

Only the model is scripted: a loopback openai-compatible server answers per
cell marker. CLI parsing, ``exec``'s composition root, the runtime handle, the
real ask queue and ``asks.jsonl``, the real wake/turn-start reconcile and the
``sessions`` tool are all production code.

Every child gets an isolated ``HOME``/``LOCAL_OPERATOR_CONFIG_DIR`` with
``CMUX_*``, ``LOP_*`` and ``NO_COLOR`` stripped (an inherited ``CMUX_WORKSPACE_ID``
would rename the operator's real workspaces; an inherited ``LOP_*`` would make the
fixture's provider selection a lie), and ``NO_NOTIFY_ENV`` is re-asserted because
a real ``lop`` child announces its completion. Sessions are synthetic ids.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import store as ask_store
from local_operator.config import ConfigManager
from tests.e2e.harness import NO_NOTIFY_ENV

pytestmark = pytest.mark.e2e

#: The escape the QA/bench harnesses use, set for the CHILD env AND the test
#: process: a cell run from inside a delegated session is refused by the same
#: agent-shell guard the CLI children are.
ALLOW_NESTED_ENV = "LOCAL_OPERATOR_ALLOW_NESTED_SESSION"

#: Synthetic session ids, one per cell direction.
SESSION_A = "resumeask00001"
SESSION_B = "resumeask00002"

#: The ask the resumed turn settles.
ASK_ID = "a-resumeask00001"
#: The ask whose deadline has already passed when the resume runs.
LATE_ASK_ID = "a-resumeask00002"

#: Prompt markers. Each cell's scripted reply is keyed on the marker in the
#: request, exactly as ``test_exec_output_modes_e2e`` scripts its cells — never
#: on wall-clock, which would make the census load-dependent.
MARK_WITHDRAW = "RESUMEQ1"
MARK_DEADLINE = "RESUMEQ2"
MARK_SESSIONS = "RESUMES"

TITLE_SYSTEM_PROMPT = "Name this conversation"


def _is_title_call(body: dict[str, Any]) -> bool:
    """Whether this request is the harness's own conversation-naming call."""
    messages = body.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return False
    return str(messages[0].get("content", "") or "").startswith(TITLE_SYSTEM_PROMPT)


def _tool_names(body: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in body.get("tools") or ():
        function = tool.get("function") if isinstance(tool, dict) else None
        name = (function or {}).get("name") or (tool.get("name") if isinstance(tool, dict) else "")
        if name:
            names.append(str(name))
    return names


def _asked_tools(body: dict[str, Any]) -> set[str]:
    """Every tool this conversation has already called, from the wire history.

    The scripted server decides "call this once, then answer" from the history
    rather than from a per-process counter: two children share this server, and a
    counter would make the second child's first call look like a retry.
    """
    return {
        str(call.get("function", {}).get("name") or "")
        for message in body.get("messages") or ()
        for call in message.get("tool_calls") or ()
    }


def _wire_text(body: dict[str, Any]) -> str:
    return json.dumps(body.get("messages") or ())


@pytest.fixture
def resume_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated home + config + a scripted loopback provider; returns tools."""
    home = tmp_path / "home"
    home.mkdir()
    root = home / ".local-operator"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv(ALLOW_NESTED_ENV, "1")
    for key in list(os.environ):
        if key.startswith("CMUX_") or key.startswith("LOP_") or key == "NO_COLOR":
            monkeypatch.delenv(key)
    requests: list[dict[str, Any]] = []
    worker_ids: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler's name
            pass

        def do_GET(self):
            body = json.dumps(
                {"object": "list", "data": [{"id": "resume-fixture", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            wire = _wire_text(body)
            if _is_title_call(body):
                delta: dict[str, Any] = {
                    "role": "assistant",
                    "content": "<title>Resumed ask queue</title>",
                }
                finish_reason = "stop"
            elif MARK_SESSIONS in wire and "sessions" not in _asked_tools(body):
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_sessions_resume",
                            "type": "function",
                            "function": {
                                "name": "sessions",
                                "arguments": json.dumps(
                                    {
                                        "i": "Resuming a paused conversation",
                                        "op": "resume",
                                        "session": SESSION_A,
                                        "prompt": f"{MARK_WITHDRAW} status?",
                                        "background": True,
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
            elif MARK_WITHDRAW in wire and "ask_withdraw" not in _asked_tools(body):
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_ask_withdraw",
                            "type": "function",
                            "function": {
                                "name": "ask_withdraw",
                                "arguments": json.dumps(
                                    {
                                        "i": "Settling the open ask",
                                        "ask_id": ASK_ID,
                                        "reason": "answered_in_chat",
                                        "answers": {"q1": ["ship it"]},
                                    }
                                ),
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
            else:
                delta = {"role": "assistant", "content": "Fixture completed"}
                finish_reason = "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece, finish in [(delta, None), ({}, finish_reason)]:
                chunk = {
                    "id": "fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "resume-fixture",
                    "choices": [{"index": 0, "delta": piece, "finish_reason": finish}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    # ONE classification switch, for the reason the exec e2e records: the
    # knowledge layer's advisory fires its own model call per user message
    # against this same server and would consume a scripted reply.
    config = ConfigManager(root)
    config.update_config(
        {
            "hosting": "openai-compatible",
            "model_name": "resume-fixture",
            "classification": {"auto": False},
            "providers": {
                "openai-compatible": {
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "models": {
                        "resume-fixture": {"context_window": 100000, "supports_tools": True}
                    },
                }
            },
        }
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("CMUX_") and not k.startswith("LOP_") and k != "NO_COLOR"
    }
    env["TERM"] = "xterm-256color"
    env[ALLOW_NESTED_ENV] = "1"
    env.update(NO_NOTIFY_ENV)

    def run(*args: str):
        # Per-child budget: startup imports of the assembled CLI run 10-30 s on a
        # loaded dev host, so 90 s would red a merely slow machine.
        result = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", *args],
            env=env,
            text=True,
            capture_output=True,
            timeout=300,
            cwd=tmp_path,
        )
        print("COMMAND", "lop", *args, "EXIT", result.returncode)
        print("STDOUT", result.stdout)
        print("STDERR", result.stderr)
        if "Background job " in result.stderr:
            worker_ids.append(result.stderr.split("Background job ", 1)[1].split(":", 1)[0])
        return result

    try:
        yield run, requests, root, worker_ids
    finally:
        from local_operator.exec_mode import job_status
        from local_operator.tools.group_reaper import _owner_is_dead

        # Only this fixture's recorded workers, generation-checked — never ambient
        # discovery (see the startup e2e fixture for the reasoning).
        for job_id in list(worker_ids):
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


def _seed_session(root: Path, session_id: str, *, ask_id: str = "", expires_in_ms: int = 0) -> Path:
    """A stored conversation with one prior message, plus an OPEN ask if asked.

    The transcript is seeded because a resume reads it: a session directory with
    no history is a different (and refused) shape.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    # Written straight to the store, the shape `test_cold_wake_e2e` seeds: the
    # resume needs a real history on disk, and the transcript writer is async.
    (directory / "transcript.jsonl").write_text(
        json.dumps(
            {
                "id": "seed-user",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "earlier turn"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if ask_id:
        now = int(time.time() * 1000)
        ask_store.append_event(
            directory,
            {
                "v": ask_store.EVENT_SCHEMA,
                "kind": ask_store.EVENT_QUEUED,
                "ask_id": ask_id,
                "at": now - 600_000,
                "created_at": now - 600_000,
                "expires_at": now + expires_in_ms,
                "timeout_s": max(120, abs(expires_in_ms) // 1000),
                "urgent": False,
                "tool_call_id": "",
                "questions": [
                    {
                        "id": "q1",
                        "question": "Do you still want this shipped?",
                        "options": [
                            {"label": "ship it", "description": ""},
                            {"label": "drop it", "description": ""},
                        ],
                        "multi": False,
                        "secret": False,
                        "persist": False,
                    }
                ],
            },
        )
    return directory


def _rows(directory: Path) -> list[dict[str, Any]]:
    path = directory / "transcript.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _row_ids(directory: Path) -> list[str]:
    return [str(row.get("id") or "") for row in _rows(directory)]


def _row_text(directory: Path) -> str:
    return json.dumps([row.get("payload") or {} for row in _rows(directory)])


def _await_terminal(job_id: str, timeout: float = 240.0) -> dict[str, Any]:
    from local_operator.exec_mode import job_status

    deadline = time.monotonic() + timeout
    state = job_status(job_id)
    while time.monotonic() < deadline and state.get("status") in ("starting", "running"):
        time.sleep(0.1)
        state = job_status(job_id)
    return state


def _job_id_from(text: str) -> str:
    """The job id a launcher receipt names, in either of its two shapes.

    ``lop exec --background`` prints ``Background job <id>: running``; the
    ``sessions`` tool's receipt is prose (``reopened \"…\" (session …, job …)``),
    so both are read here. ``job_id`` is not matched: the underscore is a word
    character, so the boundary in the pattern keeps the JSON key out.
    """
    match = re.search(r"\bjob ([0-9a-f]{6,})\b", text)
    return match.group(1) if match else ""


def _turn_requests(requests: list[dict[str, Any]], marker: str) -> list[dict[str, Any]]:
    return [body for body in requests if marker in _wire_text(body) and not _is_title_call(body)]


def _poll(predicate, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


def test_a_resumed_headless_turn_can_settle_its_open_ask(resume_server) -> None:
    """Acceptance 1: the resumed turn carries `ask`/`ask_withdraw`, and a
    withdrawal issued from it settles the index row."""
    run, requests, root, _workers = resume_server
    directory = _seed_session(root, SESSION_A, ask_id=ASK_ID, expires_in_ms=120_000)

    launched = run("exec", "--background", "--resume", SESSION_A, "--", f"{MARK_WITHDRAW} status?")
    assert launched.returncode == 0, launched.stderr
    job_id = _job_id_from(launched.stderr)
    assert job_id, launched.stderr
    assert _await_terminal(job_id)["status"] == "succeeded"

    turns = _turn_requests(requests, MARK_WITHDRAW)
    assert turns, "the resumed turn never reached the provider"
    names = _tool_names(turns[0])
    assert (
        "ask" in names and "ask_withdraw" in names
    ), f"a resumed headless turn did not advertise the queued-ask tools (#2803); got {names}"
    # Acceptance 5: the prompt speaks the same mode the tool list does.
    system_text = " ".join(
        str(message.get("content") or "")
        for message in turns[0]["messages"]
        if message.get("role") == "system"
    )
    assert (
        "Treat every `ask` as QUEUED" in system_text
    ), "the resumed prompt does not describe the queued ask its tools advertise"
    assert (
        "If the user answers nothing, take your own recommendation" not in system_text
    ), "the resumed prompt shipped the blocking-ask paragraph beside a queued tool"

    events = ask_store.read_events(directory)
    records = ask_store.fold(events, int(time.time() * 1000))
    assert [record["status"] for record in records] == [ask_store.STATUS_ANSWERED], records
    answered = [
        event
        for event in events
        if event.get("kind") == ask_store.EVENT_ANSWERED and event.get("ask_id") == ASK_ID
    ]
    assert answered, "the withdrawal recorded no answer row in asks.jsonl"
    assert (answered[-1].get("by") or {}).get("surface") == "chat", (
        "the settlement did not name the chat door, so it did not come from the "
        f"agent's `ask_withdraw`; got {answered[-1]}"
    )
    # And the model was TOLD: the standard response row lands in the transcript.
    assert _poll(
        lambda: ask_store.response_row_id(ASK_ID) in _row_ids(directory), 30.0
    ), "the settled ask's response row never reached the transcript"


def test_a_deadline_that_passed_is_delivered_inside_the_resumed_run(resume_server) -> None:
    """Acceptance 2: the overdue deadline reaches the model in the resumed RUN.

    What the criterion is for: a deadline that passed while nothing was running
    must not wait for some later turn that may never come (#2803). What the run
    actually does, measured on the wire, is deliver it as its own turn: the
    turn-start reconcile hands the row to ``deliver_ask_messages`` while the
    prompt turn is streaming, so it rides the steering queue and a follow-up turn
    carries it — the design's §2.3 delivery shape ("an IDLE session runs ONE turn
    carrying the whole batch"), not a line appended to the prompt's first request.

    Asserted as three facts, all on the wire or the durable transcript: the row
    is written, a request in THIS run carried it, and no second process or peer
    turn was needed to get it there.
    """
    run, requests, root, _workers = resume_server
    directory = _seed_session(root, SESSION_A, ask_id=LATE_ASK_ID, expires_in_ms=-60_000)

    launched = run("exec", "--background", "--resume", SESSION_A, "--", f"{MARK_DEADLINE} status?")
    assert launched.returncode == 0, launched.stderr
    job_id = _job_id_from(launched.stderr)
    assert job_id, launched.stderr
    assert _await_terminal(job_id)["status"] == "succeeded"

    timeout_row = ask_store.timeout_row_id(LATE_ASK_ID)
    ids = _row_ids(directory)
    assert timeout_row in ids, (
        "the deadline that passed while the session was stopped was not delivered "
        f"on resume; transcript ids: {ids}"
    )
    # Inside the turn, before the model's reply: the notice rode that turn, not
    # a later one.
    reply_index = next(
        index
        for index, row in enumerate(_rows(directory))
        if (row.get("payload") or {}).get("role") == "assistant"
        and "Fixture completed" in json.dumps(row.get("payload") or {})
    )
    assert ids.index(timeout_row) < reply_index, (
        "the deadline row landed after the turn's own reply; " f"ids: {ids}"
    )

    turns = _turn_requests(requests, MARK_DEADLINE)
    assert turns, "the resumed turn never reached the provider"
    carrying = [body for body in turns if "Ask timed out" in _wire_text(body)]
    assert carrying, (
        "no request in the resumed run carried the lapse, so the model was never "
        f"told inside the run; requests: {[_wire_text(b)[-300:] for b in turns]}"
    )
    # And it is the RUN's own follow-up: the first request preceded it, so the
    # row was owed from before this process existed rather than produced by it.
    assert len(turns) > 1 and "Ask timed out" not in _wire_text(turns[0]), (
        "the row appeared in the prompt turn itself; this cell no longer proves "
        "the delivery path for a row owed BEFORE the resume"
    )


@pytest.mark.asyncio
async def test_sessions_resume_from_another_session_gets_the_same_surface(
    resume_server,
) -> None:
    """Acceptance 3: the `sessions` tool's resume op reaches the same surface.

    The tool is the door the operator's other sessions use, and it launches
    ``lop exec --background --resume`` — the same composition root. Session B is
    a REAL session with the REAL tool; only its model is scripted.
    """
    from local_operator.tools.registry import create_tools
    from tests.e2e.harness import (
        ScriptedStream,
        build_session,
        dispose_quietly,
        text_turn,
        tool_call_turn,
    )

    _run, requests, root, _workers = resume_server
    _seed_session(root, SESSION_A, ask_id=ASK_ID, expires_in_ms=120_000)

    stream = ScriptedStream(
        [
            tool_call_turn(
                text="resuming",
                tool_name="sessions",
                tool_call_id="call_sessions_resume",
                arguments={
                    "i": "Resuming a paused conversation",
                    "op": "resume",
                    "session": SESSION_A,
                    "prompt": f"{MARK_WITHDRAW} status?",
                    "background": True,
                },
            ),
            text_turn("done"),
        ]
    )
    directory = root / "sessions" / SESSION_B
    session = build_session(directory, stream, cwd=root)
    child_job = ""
    try:
        session.refresh_tools(create_tools(session._build_tool_context()))

        await session.prompt("resume the paused one")

        # The launcher's receipt is in the tool result the session wrote, which
        # is also how a model learns the job id.
        child_job = _job_id_from(_row_text(directory))
        assert child_job, (
            "the sessions tool reported no job id in its result; transcript: "
            f"{_row_text(directory)[-1200:]}"
        )
        assert _await_terminal(child_job)["status"] == "succeeded"

        assert _poll(
            lambda: bool(_turn_requests(requests, MARK_WITHDRAW)), 30.0
        ), "the `sessions` resume never produced a provider turn"
        names = _tool_names(_turn_requests(requests, MARK_WITHDRAW)[0])
        assert "ask" in names and "ask_withdraw" in names, (
            "`sessions(op='resume')` reached a worker without the queued-ask tools; " f"got {names}"
        )
    finally:
        await dispose_quietly(session)
        if child_job:
            from local_operator.exec_mode import job_status
            from local_operator.tools.group_reaper import _owner_is_dead

            state = job_status(child_job)
            generation = state.get("process_generation")
            if (
                state.get("status") in ("starting", "running")
                and generation
                and _owner_is_dead(state["pid"], generation) is False
            ):
                os.kill(state["pid"], signal.SIGTERM)
