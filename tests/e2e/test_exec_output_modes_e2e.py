"""``lop exec`` output modes, end to end against a scripted loopback provider.

Only the remote model is scripted. CLI parsing, config, session construction,
the harness gate at the turn's terminal seam, the retry, stdout/stderr
rendering, detached-worker disposal and the job ledger are production code.

Every CLI child runs with an isolated ``HOME``/``LOCAL_OPERATOR_CONFIG_DIR``
and with ``CMUX_*`` and ``LOP_*`` stripped (an inherited ``CMUX_WORKSPACE_ID``
would rename the operator's real workspaces; an inherited ``LOP_*`` would make
the fixture's provider/model selection a lie). ``NO_NOTIFY_ENV`` is re-asserted
because a real ``lop`` child with a mock model announces a completion banner
whose body is the mock's own sentence.

The handler is scripted per SCENARIO, keyed on the cell marker in the
request's messages and the per-cell request count — never on wall-clock. Each
test prints COMMAND/EXIT/STDOUT/STDERR so the raw run is the evidence.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.session.naming import TITLE_SYSTEM_PROMPT
from tests.e2e.harness import NO_NOTIFY_ENV

#: The escape the QA/bench harnesses use, set for the CHILD env AND the test
#: process: the SDK cell (in-process) is refused by the same agent-shell guard
#: the CLI children are, and the documented escape is what makes the cell
#: runnable from inside a delegated session.
ALLOW_NESTED_ENV = "LOCAL_OPERATOR_ALLOW_NESTED_SESSION"

CELL_RE = re.compile(r"CELL(\d+[A-Z]*)")

#: Prompt markers whose reply is valid JSON behind a fence with prose around
#: it: unenforced stdout is the raw message, enforced stdout is the payload
#: span — so the resume cell can tell enforcement from its absence.
FENCED_JSON_REPLY = 'Answer:\n```json\n{"cell": 11}\n```'


def _is_title_call(body: dict[str, Any]) -> bool:
    """Whether this request is the harness's own conversation-naming call.

    An UNNAMED exec run fires one ``Name this conversation …`` model call
    beside the turn (``session.naming``, whose prompt is the constant above);
    it carries the user message, so without this predicate the fixture would
    count it as a turn call — and, worse, let it consume a scripted reply
    from the cell's sequence. The cells count TURN calls.
    """
    messages = body.get("messages") or []
    if not messages or messages[0].get("role") != "system":
        return False
    content = str(messages[0].get("content", "") or "")
    return content.startswith(TITLE_SYSTEM_PROMPT.split("\n", 1)[0])


@pytest.fixture
def exec_server(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    root = home / ".local-operator"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv(ALLOW_NESTED_ENV, "1")
    for key in list(os.environ):
        if key.startswith("CMUX_") or key.startswith("LOP_"):
            monkeypatch.delenv(key)
    requests: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    counts_lock = threading.Lock()
    worker_ids: list[str] = []

    def _calls(cell: str) -> int:
        with counts_lock:
            counts[cell] = counts.get(cell, 0) + 1
            return counts[cell]

    def _reply(cell: str, calls: int) -> str:
        """The scripted model sentence for one cell and one call number."""
        if cell == "CELL2":
            return '{"cell": 2, "ok": true}'
        if cell == "CELL3Y":
            return "cell: 3\nok: true"
        if cell == "CELL3T":
            return "cell = 3\nok = true"
        if cell == "CELL4":
            return "# Summary\nDone.\n\n## Risks\nNone identified."
        if cell == "CELL4M":
            # Always missing the required sections: the cell asserts the
            # failure mode, so retrying must not rescue it by accident.
            return "# Summary\nDone."
        if cell == "CELL5":
            return "that is not json" if calls == 1 else '{"cell": 5}'
        if cell == "CELL6":
            return 'Here it is:\n```json\n{"cell": 6}\n```\nHope that helps.'
        if cell == "CELL7":
            return "never valid json, on any attempt"
        if cell == "CELL8":
            return '{"wrong": "shape"}'
        if cell == "CELL9":
            return "not json at all" if calls == 1 else '{"cell": 9, "ok": true}'
        if cell == "CELL10":
            return "never valid json (background)"
        if cell in ("CELL11A", "CELL11B", "CELL11C"):
            return FENCED_JSON_REPLY
        return "Fixture completed"

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
            if _is_title_call(body):
                # The harness's naming call: answer it with a well-formed title
                # (so the run is titled like any real one) and hand it to
                # neither the per-cell call counter nor the scripted replies.
                text = "<title>Output modes cell</title>"
            else:
                wire = json.dumps(body.get("messages", []))
                match = CELL_RE.search(wire)
                cell = match.group(0) if match else ""
                text = _reply(cell, _calls(cell))
            delta = {"role": "assistant", "content": text}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece, finish in [(delta, None), ({}, "stop")]:
                chunk = {
                    "id": "fixture",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "exec-fixture",
                    "choices": [{"index": 0, "delta": piece, "finish_reason": finish}],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    # The knowledge layer's advisory probe is switched OFF for these cells, by
    # its documented off-switch: with ``classification.auto`` off the wiring
    # returns before it imports the package and the prompt is byte-identical to
    # a harness without the layer. Left on, it fires one classification model
    # call per user message against the SAME loopback provider, racing the turn
    # on a 50 ms budget — which both consumes a scripted reply from the cell's
    # sequence and inflates the per-cell provider-call census non-
    # deterministically (observed: a 2-call retry cell counted 3). The cells are
    # about the output contract, not about the recommender; a count that varies
    # with the machine's load would be exactly the clock-keyed assertion the
    # harness's own testing rules forbid.
    config = ConfigManager(root)
    config.update_config(
        {
            "hosting": "openai-compatible",
            "model_name": "exec-fixture",
            "classification": {"auto": False},
            "providers": {
                "openai-compatible": {
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "models": {"exec-fixture": {"context_window": 100000, "supports_tools": True}},
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

    def run(*args, stdin=None):
        # Per-child budget: startup imports of the assembled CLI run 10-30 s
        # on a loaded dev host (this fleet's chronic state), so a 60-90 s
        # budget reds a merely slow machine. Generous on purpose; the raw
        # output printed around each run is the evidence either way.
        result = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", *args],
            env=env,
            input=stdin,
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

    setattr(run, "env", env)
    setattr(run, "cwd", tmp_path)
    try:
        yield run, requests, root
    finally:
        import signal

        from local_operator.exec_mode import job_status
        from local_operator.tools.group_reaper import _owner_is_dead

        # Clean only this fixture's generation-verified workers, never ambient
        # discovery (see the startup e2e fixture for the reasoning).
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


def _cells(requests: list[dict[str, Any]], cell: str) -> list[dict[str, Any]]:
    """The recorded TURN requests carrying this cell's prompt marker.

    The harness's own naming call is excluded: it carries the user message
    too, but it is not a turn call and counting it inflated every census by
    one (observed on the first e2e runs: a 2-call retry counted 3).
    """
    return [
        body
        for body in requests
        if cell in json.dumps(body.get("messages", [])) and not _is_title_call(body)
    ]


def _await_terminal(job_id: str, timeout: float = 240.0) -> dict[str, Any]:
    from local_operator.exec_mode import job_status

    deadline = time.monotonic() + timeout
    state = job_status(job_id)
    while time.monotonic() < deadline and state.get("status") in ("starting", "running"):
        time.sleep(0.1)
        state = job_status(job_id)
    return state


# --- cells 1-6: control, formats, markdown, retry, noise --------------------------


def test_default_control_is_unchanged_and_json_gains_no_events(exec_server):
    """Cell 1. No flag = today's behaviour: raw text on stdout, and a --json
    run emits ZERO ``output_validation`` lines (the byte-compat proof)."""
    run, requests, root = exec_server
    plain = run("exec", "CELL1A nothing enforced", stdin="")
    assert plain.returncode == 0
    assert plain.stdout.strip() == "Fixture completed"
    assert "output check" not in plain.stderr

    as_json = run("exec", "CELL1B json control", "--json", stdin="")
    assert as_json.returncode == 0
    rows = [json.loads(line) for line in as_json.stdout.splitlines() if line.strip()]
    assert rows, "a --json run emits its event stream"
    assert [r for r in rows if r.get("type") == "output_validation"] == []
    assert [r for r in rows if r.get("type") == "agent_end"][-1].get("error") is None


def test_enforced_json_yaml_and_toml_print_the_payload_span(exec_server):
    """Cells 2-3. stdout is the validated payload, machine-consumable in each
    format (the tOML and YAML runs are `| python -c`-able by construction)."""
    run, requests, root = exec_server
    as_json = run("exec", "CELL2 structured json", "--output-format", "json", stdin="")
    assert as_json.returncode == 0
    assert json.loads(as_json.stdout) == {"cell": 2, "ok": True}

    as_yaml = run("exec", "CELL3Y structured yaml", "--output-format", "yaml", stdin="")
    assert as_yaml.returncode == 0
    assert as_yaml.stdout.strip() == "cell: 3\nok: true"

    as_toml = run("exec", "CELL3T structured toml", "--output-format", "toml", stdin="")
    assert as_toml.returncode == 0
    assert as_toml.stdout.strip() == "cell = 3\nok = true"


def test_enforced_markdown_with_required_sections(exec_server, tmp_path):
    """Cell 4. Markdown + ``{"required_sections": [...]}``; the missing-section
    variant fails after its retries with the pinned reason."""
    run, requests, root = exec_server
    sections = tmp_path / "sections.json"
    sections.write_text(json.dumps({"required_sections": ["Summary", "Risks"]}))
    ok = run(
        "exec",
        "CELL4 markdown with sections",
        "--output-format",
        "markdown",
        "--output-schema",
        str(sections),
        stdin="",
    )
    assert ok.returncode == 0
    assert ok.stdout.startswith("# Summary")

    missing = run(
        "exec",
        "CELL4M markdown missing section",
        "--output-format",
        "markdown",
        "--output-schema",
        str(sections),
        stdin="",
    )
    assert missing.returncode == 1
    assert "missing required section 'Risks'" in missing.stderr
    assert missing.stdout == ""
    assert len(_cells(requests, "CELL4M")) == 3  # retried to the default budget


def test_retry_feeds_the_contract_back_on_the_wire(exec_server):
    """Cell 5. First completion invalid, second valid: two provider calls, and
    the second request carries the retry instruction as a user message."""
    run, requests, root = exec_server
    result = run("exec", "CELL5 retry me", "--output-format", "json", stdin="")
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"cell": 5}
    calls = _cells(requests, "CELL5")
    assert len(calls) == 2, "one retry, not a second session"
    second_wire = json.dumps(calls[1]["messages"])
    assert "Harness output check:" in second_wire
    assert "(attempt 2 of 3)" in second_wire
    assert "output check failed (json, attempt 1/3)" in result.stderr


def test_prose_and_fence_noise_still_yields_the_bare_payload(exec_server):
    """Cell 6. Payload fenced with prose around it: stdout is the bare span."""
    run, requests, root = exec_server
    result = run("exec", "CELL6 noisy", "--output-format", "json", stdin="")
    assert result.returncode == 0
    assert result.stdout.strip() == '{"cell": 6}'


# --- cells 7-8: exhaustion and schema failures ------------------------------------


def test_exhaustion_fails_loudly_and_prints_nothing(exec_server):
    """Cell 7. Three attempts, exit 1, the exact stderr line, empty stdout."""
    run, requests, root = exec_server
    result = run("exec", "CELL7 never valid", "--output-format", "json", stdin="")
    assert result.returncode == 1
    assert (
        "final response did not satisfy the output contract (json) after 3 attempts"
        in result.stderr
    )
    assert result.stdout == ""
    assert len(_cells(requests, "CELL7")) == 3
    assert "output check failed (json, attempt 3/3)" in result.stderr


def test_schema_failure_is_retried_then_reported(exec_server, tmp_path):
    """Cell 8. Valid JSON of the wrong SHAPE against a schema file: retried,
    then exit 1 with the schema reason — proving the file is read and enforced."""
    run, requests, root = exec_server
    schema = tmp_path / "report.schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["cell"],
                "properties": {"cell": {"type": "integer"}},
            }
        )
    )
    result = run(
        "exec",
        "CELL8 wrong shape",
        "--output-format",
        "json",
        "--output-schema",
        str(schema),
        stdin="",
    )
    assert result.returncode == 1
    assert "payload does not match the schema:" in result.stderr
    assert len(_cells(requests, "CELL8")) == 3
    assert result.stdout == ""


# --- cells 9-11: the SDK, --background, --resume ----------------------------------


@pytest.mark.asyncio
async def test_sdk_pydantic_schema_and_retry_round_trip(exec_server, tmp_path):
    """Cell 9. The SDK path: a pydantic schema, one retry, and the typed object
    reconstructed from the validated payload span on the event stream."""
    from pydantic import BaseModel

    from local_operator.output_contract import (  # noqa: F401 — import weight
        OUTPUT_FORMATS,
    )
    from local_operator.sdk import SessionRoots, SessionSpec, events, open_session

    run, requests, root = exec_server

    class Invoice(BaseModel):
        cell: int
        ok: bool

    roots = SessionRoots(
        config_dir=str(root),
        agent_home=str(tmp_path / "home"),
        cwd=str(tmp_path),
        allow_volatile=True,
    )
    spec = SessionSpec(
        hosting="openai-compatible",
        model="exec-fixture",
        output_format="json",
        output_schema=Invoice,
        output_retries=1,
    )
    async with open_session(spec, roots=roots) as session:
        stream = events(session)
        await session.prompt("CELL9 sdk retry")
        collected = []
        while not stream._queue.empty():
            collected.append(stream._queue.get_nowait())
        await stream.aclose()

    validations = [e for e in collected if e.type == "output_validation"]
    assert [(v.attempt, v.max_attempts, v.ok) for v in validations] == [
        (1, 2, False),
        (2, 2, True),
    ]
    ends = [e for e in collected if e.type == "agent_end"]
    assert ends and ends[-1].error is None, "the turn ends clean once the retry validates"
    invoice = Invoice.model_validate_json(validations[-1].payload_text)
    assert invoice.cell == 9 and invoice.ok is True
    assert len(_cells(requests, "CELL9")) == 2


def test_background_reports_failed_status_and_log(exec_server, tmp_path):
    """Cell 10. `--background` on an always-invalid cell: the durable record
    reaches ``failed`` and its log carries the contract error. The schema file
    is named RELATIVELY from the child's cwd: the CLI absolutises it for the
    detached worker, which re-reads it on its own side — a "cannot read
    --output-schema" in the log would mean the path stopped crossing the
    boundary (review R-2)."""
    run, requests, root = exec_server
    (tmp_path / "report.schema.json").write_text('{"type": "object", "required": ["cell"]}')
    launched = run(
        "exec",
        "CELL10 background",
        "--output-format",
        "json",
        "--output-schema",
        "report.schema.json",
        "--background",
        stdin="",
    )
    assert "Background job " in launched.stderr
    job_id = launched.stderr.split("Background job ", 1)[1].split(":", 1)[0]

    state = _await_terminal(job_id)
    assert state["status"] == "failed"
    log_text = Path(state["log"]).read_text(errors="replace")
    assert "cannot read --output-schema" not in log_text
    assert "final response did not satisfy the output contract (json) after 3 attempts" in log_text
    assert "exec failed:" in log_text

    report = run("exec", "--status", job_id)
    assert report.returncode == 0
    assert json.loads(report.stdout)["status"] == "failed"


def test_resume_enforces_only_where_the_flag_is_given(exec_server):
    """Cell 11. The contract is this invocation's, never persisted: run free,
    resume with the flag (enforced), resume again without (unenforced), and the
    config store gains no key."""
    run, requests, root = exec_server
    free = run("exec", "CELL11A free run", stdin="")
    assert free.returncode == 0
    assert "```json" in free.stdout, "unenforced stdout is the raw message, fence and all"
    session_id = free.stderr.split("session_id=", 1)[1].split()[0]

    enforced = run(
        "exec",
        "CELL11B resumed with the flag",
        "--resume",
        session_id,
        "--output-format",
        "json",
        stdin="",
    )
    assert enforced.returncode == 0
    assert json.loads(enforced.stdout) == {"cell": 11}

    again = run(
        "exec",
        "CELL11C resumed without the flag",
        "--resume",
        session_id,
        stdin="",
    )
    assert again.returncode == 0
    assert "```json" in again.stdout, "the flag did not persist into this resume"

    config_text = (root / "config.yml").read_text()
    assert "output_format" not in config_text
    assert "output_retries" not in config_text
