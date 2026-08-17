"""Tests for headless exec mode, the print renderer, and the background worker.

Engine-free by construction: a scripted ``FakeSession`` implements the
``SessionProtocol`` surface and is injected through the documented seams
(``exec_mode.default_session_factory`` and ``exec_worker.run``'s factory
parameter). Background spawning is verified by monkeypatching
``subprocess.Popen`` — no real detached processes are created.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_operator import exec_mode, exec_worker
from local_operator.exec_mode import ExecArgs, build_worker_argv, slugify
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelSpec,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.headless_print import PrintRenderer, printable_event, run_print_mode
from local_operator.session.protocol import CompactionOutcome

# --- Fakes ---------------------------------------------------------------------


class FakeSession:
    """Scripted SessionProtocol: emits a fixed event list per prompt call."""

    def __init__(self, scripts: list[list[AgentEvent]]) -> None:
        self.scripts = scripts
        self.prompts: list[str] = []
        self.handlers: list[Any] = []
        self.disposed = False
        self._script_index = 0

    # identity / state
    @property
    def session_id(self) -> str:
        return "fake-session"

    @property
    def agent_id(self) -> str:
        return "fake-agent"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def model_label(self) -> str:
        return "fake/model"

    @property
    def model(self) -> ModelSpec:
        return ModelSpec(provider="fake", model_id="fake-model")

    def set_model(self, model: ModelSpec) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    @property
    def conversation_name(self) -> str:
        return getattr(self, "_conversation_name", "")

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        self._conversation_name = (text or "").strip()
        return self._conversation_name

    async def complete_once(self, system: str, prompt: str) -> str:
        return ""

    def history(self) -> list[AgentMessage]:
        return []

    async def seed_history(self, messages: list[Message]) -> None:
        pass

    # driving turns
    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)
        events = (
            self.scripts[self._script_index]
            if self._script_index < len(self.scripts)
            else self.scripts[-1]
        )
        self._script_index += 1
        for event in events:
            for handler in list(self.handlers):
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def set_ask_handler(self, handler: object | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot, and that
        # install is what makes the tool exist; fakes only need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        pass

    # events
    def subscribe(self, handler: Any) -> Any:
        self.handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self.handlers:
                self.handlers.remove(handler)

        return unsubscribe

    # lifecycle
    async def dispose(self) -> None:
        self.disposed = True

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # exec mode never opens an aside; present only so the fake still
        # satisfies SessionProtocol, which is what these tests type against.
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        return None

    async def compact_now(self) -> CompactionOutcome:
        # No history to compact: this fake never carries a conversation, which
        # is the state a real session answers with the same refusal.
        return CompactionOutcome(
            ran=False, reason="nothing_to_compact", detail="nothing to compact"
        )


def _success_script(reply: str = "Hello from the agent") -> list[AgentEvent]:
    """One turn: assistant streams text, runs one tool, ends cleanly."""
    message = Message.assistant(reply)
    return [
        AgentStartEvent(),
        MessageStartEvent(message=message),
        MessageUpdateEvent(message=message, delta=reply),
        MessageEndEvent(message=message),
        ToolExecutionStartEvent(tool_call_id="t1", tool_name="bash", args={"command": "ls"}),
        ToolExecutionEndEvent(
            tool_call_id="t1",
            tool_name="bash",
            result=ToolResult(
                tool_call_id="t1", tool_name="bash", content=[TextContent(text="ok")]
            ),
        ),
        AgentEndEvent(messages=[message]),
    ]


def _error_script() -> list[AgentEvent]:
    message = Message.assistant("")
    return [
        AgentStartEvent(),
        MessageStartEvent(message=message),
        MessageEndEvent(message=message),
        AgentEndEvent(messages=[message], error="provider exploded"),
    ]


@pytest.fixture
def fake_factory(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted FakeSession as the exec session factory."""

    def _install(session: FakeSession) -> None:
        monkeypatch.setattr(exec_mode, "default_session_factory", lambda: session)

    return _install


# --- exec_mode: slug + argv serialization --------------------------------------


def test_slugify_rules() -> None:
    assert slugify("Make a file called test.txt!") == "Make-a-file-called-test-txt-"
    # Unicode letters are alnum in Python — kept, not dashed.
    assert slugify("café ☕ work") == "café---work"
    assert slugify("") == "task"
    assert len(slugify("x" * 100)) <= 40


def test_build_worker_argv_roundtrip() -> None:
    args = ExecArgs(
        json_mode=True,
        yolo=True,
        train=True,
        agent_name="A",
        agent_id="id1",
        hosting="openai",
        model="gpt-4o",
    )
    argv = build_worker_argv("do it", args)
    assert argv[:3] == [sys.executable, "-m", "local_operator.exec_worker"]
    assert argv[3:5] == ["--prompt", "do it"]
    # Every set flag serializes; parse it back through the worker parser.
    parsed = exec_worker.build_parser().parse_args(argv[3:])
    assert parsed.prompt == "do it"
    assert parsed.json_mode is True
    assert parsed.yolo is True
    assert parsed.train is True  # CL-05
    assert parsed.agent == "A"
    assert parsed.agent_id == "id1"
    assert parsed.hosting == "openai"
    assert parsed.model == "gpt-4o"


def test_build_worker_argv_train_threaded_to_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """CL-05: ExecArgs.train reaches the session-factory namespace via argv."""
    argv = build_worker_argv("t", ExecArgs(train=True))
    parsed = exec_worker.build_parser().parse_args(argv[3:])
    assert parsed.train is True
    # And the worker's factory passes it into the session args namespace.
    seen: dict[str, Any] = {}

    def fake_create_session(session_args, *managers, **kwargs):
        seen["train"] = session_args.train
        return None

    monkeypatch.setattr("local_operator.config.ConfigManager", lambda *a: object())
    monkeypatch.setattr("local_operator.credentials.CredentialManager", lambda *a: object())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", lambda *a: object())
    monkeypatch.setattr("local_operator.session_factory.create_session", fake_create_session)
    exec_worker._default_session_factory(parsed)
    assert seen["train"] is True


def test_build_worker_argv_omits_unset_flags() -> None:
    argv = build_worker_argv("bare", ExecArgs())
    parsed = exec_worker.build_parser().parse_args(argv[3:])
    assert parsed.prompt == "bare"
    assert parsed.json_mode is False
    assert parsed.agent is None
    assert parsed.hosting is None


# --- exec_mode foreground -------------------------------------------------------


def test_run_exec_foreground_success(fake_factory, capsys) -> None:
    session = FakeSession([_success_script()])
    fake_factory(session)
    code = exec_mode.run_exec("say hello", ExecArgs())
    captured = capsys.readouterr()
    assert code == 0
    # Text mode prints the last assistant text on stdout...
    assert "Hello from the agent" in captured.out
    # ...and the tool row renders as a dim one-liner on stderr.
    assert "bash" in captured.err
    assert session.prompts == ["say hello"]
    assert session.disposed is True


def test_run_exec_foreground_error_exits_nonzero(fake_factory, capsys) -> None:
    session = FakeSession([_error_script()])
    fake_factory(session)
    code = exec_mode.run_exec("doomed task", ExecArgs())
    captured = capsys.readouterr()
    assert code == 1
    assert "provider exploded" in captured.err
    assert session.disposed is True


def test_run_exec_foreground_json_mode(fake_factory, capsys) -> None:
    reply = Message.assistant("streamed")
    script: list[AgentEvent] = [
        AgentStartEvent(),
        MessageStartEvent(message=reply),
        MessageUpdateEvent(message=reply, delta="streamed"),
        MessageEndEvent(message=reply),
        AgentEndEvent(messages=[reply]),
    ]
    fake_factory(FakeSession([script]))
    code = exec_mode.run_exec("json please", ExecArgs(json_mode=True))
    captured = capsys.readouterr()
    assert code == 0
    lines = [json.loads(line) for line in captured.out.strip().splitlines()]
    assert [line["type"] for line in lines] == [
        "agent_start",
        "message_start",
        "message_update",
        "message_end",
        "agent_end",
    ]
    # Quadratic-growth fix: message_update keeps ONLY the delta — plus the
    # message_id so JSON consumers can attribute deltas (CL-15).
    update = lines[2]
    assert update == {
        "type": "message_update",
        "message_id": reply.id,
        "delta": "streamed",
    }


def test_printable_event_strips_provider_payload() -> None:
    message = Message.assistant("x", provider_payload={"encrypted": "blob"})
    event = MessageEndEvent(message=message)
    out = printable_event(event)
    # The payload key is gone entirely, and nothing leaks the secret value.
    assert "provider_payload" not in out["message"]
    assert "encrypted" not in json.dumps(out)


def test_printable_event_message_update_carries_message_id() -> None:
    """CL-15: message_update JSON lines carry message_id."""
    message = Message.assistant("abc")
    out = printable_event(MessageUpdateEvent(message=message, delta="abc"))
    assert out["message_id"] == message.id
    assert out["delta"] == "abc"
    assert out["type"] == "message_update"


def test_run_exec_prompt_raising_exits_one(fake_factory, capsys) -> None:
    """CL-19: a prompt() that RAISES maps to exit 1 with the error on
    stderr — never the interactive red banner."""

    class RaisingSession(FakeSession):
        async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
            raise RuntimeError("turn blew up")

    fake_factory(RaisingSession([]))
    code = exec_mode.run_exec("explode", ExecArgs())
    assert code == 1
    assert "turn blew up" in capsys.readouterr().err


def test_renderer_tracks_failure() -> None:
    renderer = PrintRenderer(json_mode=False)
    renderer.handle(AgentEndEvent(error="boom"))
    assert renderer.failed is True
    renderer2 = PrintRenderer(json_mode=False)
    renderer2.handle(AgentEndEvent(aborted=True))
    assert renderer2.failed is True
    renderer3 = PrintRenderer(json_mode=False)
    renderer3.handle(AgentEndEvent())
    assert renderer3.failed is False


@pytest.mark.asyncio
async def test_run_print_mode_prompts_sequentially(capsys) -> None:
    session = FakeSession([_success_script("one"), _success_script("two")])

    code = await run_print_mode(session, ["first", "second"])
    assert code == 0
    assert session.prompts == ["first", "second"]


# --- exec_mode background --------------------------------------------------------


def test_run_exec_background_spawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(exec_mode, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", lambda args: ("test", "m"))

    popen_mock = MagicMock()
    popen_mock.return_value.pid = 4321
    monkeypatch.setattr("local_operator.exec_mode.subprocess.Popen", popen_mock)

    code = exec_mode.run_exec(
        "write a long report about penguins",
        ExecArgs(background=True, json_mode=True, yolo=True, hosting="openai"),
    )
    assert code == 0

    # Detached argv, one new session on POSIX.
    popen_mock.assert_called_once()
    argv = popen_mock.call_args[0][0]
    assert argv[:5] == [
        sys.executable,
        "-m",
        "local_operator.exec_worker",
        "--prompt",
        "write a long report about penguins",
    ]
    assert "--json" in argv and "--yolo" in argv
    assert "--job-id" in argv  # CL-09 terminal-record wiring
    kwargs = popen_mock.call_args[1]
    if sys.platform != "win32":
        assert kwargs.get("start_new_session") is True

    # Log path printed and registered; JSONL ledger appended (CL-11).
    # STDERR: --json and --background are independent flags, so these two
    # notices must not precede the event stream on stdout.
    out = capsys.readouterr().err
    assert "Started background job" in out
    log_line = next(line for line in out.splitlines() if line.startswith("Log: "))
    log_path = Path(log_line.removeprefix("Log: "))
    assert log_path == logs_dir / log_path.name
    assert log_path.name.startswith("exec-")
    assert "write-a-long-report-about-penguins" in log_path.name
    assert log_path.exists()
    assert "write a long report about penguins" in log_path.read_text()

    records = exec_mode.read_job_records()
    assert len(records) == 1
    record = records[0]
    assert record["pid"] == 4321
    assert record["prompt"] == "write a long report about penguins"
    assert record["log"] == str(log_path)
    assert record["finished_at"] is None and record["exit_code"] is None

    # A second spawn APPENDS a second JSONL line (never rewrites the file).
    exec_mode.run_exec("second task", ExecArgs(background=True))
    records = exec_mode.read_job_records()
    assert len(records) == 2
    assert records[1]["prompt"] == "second task"
    lines = (logs_dir / exec_mode.JOBS_FILE).read_text().splitlines()
    assert len(lines) == 2


def test_ledger_reader_tolerates_partial_line(tmp_path: Path, monkeypatch) -> None:
    """CL-11: a truncated trailing line (crash mid-write) never breaks reads."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(exec_mode, "LOGS_DIR", logs_dir)
    good = json.dumps({"id": "abc", "prompt": "ok"})
    (logs_dir / exec_mode.JOBS_FILE).write_text(
        good + "\n" + '{"id": "de", "prom', encoding="utf-8"
    )
    records = exec_mode.read_job_records()
    assert len(records) == 1
    assert records[0]["id"] == "abc"


def test_logs_dir_and_log_file_permissions(monkeypatch, tmp_path: Path) -> None:
    """CL-10: LOGS_DIR is 0700 and job logs are created 0600."""
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(exec_mode, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", lambda args: ("test", "m"))
    popen_mock = MagicMock()
    popen_mock.return_value.pid = 1
    monkeypatch.setattr("local_operator.exec_mode.subprocess.Popen", popen_mock)

    assert exec_mode.run_exec("perm task", ExecArgs(background=True)) == 0
    assert (logs_dir.stat().st_mode & 0o777) == 0o700
    log_files = list(logs_dir.glob("exec-*.log"))
    assert len(log_files) == 1
    assert (log_files[0].stat().st_mode & 0o777) == 0o600


def test_background_preflight_blocks_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    """CL-09: a failed hosting/model resolution returns non-zero WITHOUT
    spawning the worker or writing a log."""
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(exec_mode, "LOGS_DIR", logs_dir)

    def broken(args):
        raise ValueError("Model name is not configured.")

    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", broken)
    popen_mock = MagicMock()
    monkeypatch.setattr("local_operator.exec_mode.subprocess.Popen", popen_mock)

    code = exec_mode.run_exec("doomed", ExecArgs(background=True))
    assert code != 0
    popen_mock.assert_not_called()
    # stderr: a preflight failure on the --json path must stay off the data
    # channel, like every other diagnostic.
    out = capsys.readouterr().err
    assert "Model name is not configured." in out
    assert not logs_dir.exists() or not list(logs_dir.glob("exec-*.log"))


def test_worker_records_exit_in_ledger(monkeypatch, tmp_path: Path, capsys) -> None:
    """CL-09: main() with --job-id appends finished_at + exit_code."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(exec_mode, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(sys, "argv", ["exec_worker", "--prompt", "x", "--job-id", "job1"])
    monkeypatch.setattr(exec_worker, "run", lambda _p, session_factory=None: 0)

    assert exec_worker.main() == 0
    records = exec_mode.read_job_records()
    assert any(r["id"] == "job1" and r["exit_code"] == 0 and r["finished_at"] for r in records)


def test_headless_approval_denial_notice(fake_factory, monkeypatch, capsys) -> None:
    """CL-04: a non-tty approval denial prints the --yolo notice to stderr."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    import local_operator.session_factory as sf

    gate = sf._make_request_approval(yolo=False)

    async def _gate() -> bool:
        return await gate("exec", "rm -rf /")

    approved = asyncio.run(_gate())
    assert approved is False
    err = capsys.readouterr().err
    assert "approval required but no tty; run with --yolo to auto-approve" in err


def test_yolo_gate_approves_without_tty(monkeypatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    import local_operator.session_factory as sf

    gate = sf._make_request_approval(yolo=True)

    async def _gate() -> bool:
        return await gate("exec", "anything")

    assert asyncio.run(_gate()) is True


# --- exec_worker -----------------------------------------------------------------


def test_exec_worker_success(fake_factory, capsys) -> None:
    session = FakeSession([_success_script("worker says hi")])
    parsed = exec_worker.build_parser().parse_args(["--prompt", "greet me"])
    code = exec_worker.run(parsed, session_factory=lambda: session)
    captured = capsys.readouterr()
    assert code == 0
    assert "worker says hi" in captured.out
    assert session.disposed is True


def test_exec_worker_error_exit_code(fake_factory) -> None:
    session = FakeSession([_error_script()])
    parsed = exec_worker.build_parser().parse_args(["--prompt", "doomed"])
    assert exec_worker.run(parsed, session_factory=lambda: session) == 1


def test_exec_worker_main_wraps_errors(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """main() maps unexpected failures to exit 1 with the log-line on stderr."""
    monkeypatch.setattr(sys, "argv", ["exec_worker", "--prompt", "x"])

    def boom(_parsed: argparse.Namespace, session_factory=None) -> int:
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(exec_worker, "run", boom)
    assert exec_worker.main() == 1
    assert "exec_worker error: worker exploded" in capsys.readouterr().err


def test_exec_worker_parser_requires_prompt() -> None:
    with pytest.raises(SystemExit):
        exec_worker.build_parser().parse_args([])


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM semantics are POSIX")
def test_exec_worker_sigterm_yields_130(tmp_path: Path) -> None:
    """CL-03/CL-19: a real SIGTERM to a running worker exits 130 with a
    clean log (no traceback). Drives the worker through a subprocess with a
    stubbed session whose prompt parks until abort — exactly the shape a
    background turn has when SIGTERM arrives."""
    import os as _os
    import subprocess as sp
    import time

    repo_root = Path(exec_worker.__file__).resolve().parent.parent
    script = (
        "import asyncio\n"
        "import local_operator.exec_worker as ew\n"
        "from local_operator.exec_worker import EXIT_INTERRUPTED\n"
        "class Slow:\n"
        "    def __init__(self):\n"
        "        self.disposed = False\n"
        "        self._abort = asyncio.Event()\n"
        "    def subscribe(self, handler):\n"
        "        return lambda: None\n"
        "    def abort(self, reason):\n"
        "        self._abort.set()\n"
        "    async def prompt(self, text, images=None):\n"
        # READY is printed from inside the turn, which is the only point where
        # the signal handler is provably installed AND the turn has started.
        # A fixed sleep here raced under full-suite load: the child took SIGTERM
        # before installing the handler, died with rc=-15, and both streams came
        # back empty.
        "        print('READY', flush=True)\n"
        "        await self._abort.wait()\n"
        "    async def dispose(self):\n"
        "        self.disposed = True\n"
        "parsed = ew.build_parser().parse_args(['--prompt', 'sleepy'])\n"
        "code = ew.run(parsed, session_factory=Slow)\n"
        "print('EXIT', code)\n"
        "import sys\n"
        "sys.exit(code)\n"
    )
    env = dict(_os.environ)
    env["PYTHONPATH"] = str(repo_root) + _os.pathsep + env.get("PYTHONPATH", "")
    proc = sp.Popen(
        [sys.executable, "-c", script],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    # Block until the child says the handler is installed and the turn is live.
    assert proc.stdout is not None
    deadline = time.monotonic() + 30.0
    ready = False
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break  # child exited early
        if line.strip() == "READY":
            ready = True
            break
    if not ready:
        proc.kill()
        remainder, stderr = proc.communicate(timeout=15)
        raise AssertionError(f"worker never signalled READY: {remainder!r} {stderr!r}")
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=15)
    assert proc.returncode == 130, f"stdout={stdout!r} stderr={stderr!r}"
    assert "EXIT 130" in stdout
    assert "Traceback" not in stderr
