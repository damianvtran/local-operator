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
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from local_operator import exec_mode, exec_worker
from local_operator.exec_mode import ExecArgs, build_worker_argv, slugify
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from local_operator.harness.types import Message
from local_operator.harness.types import ToolResult
from local_operator.headless_print import PrintRenderer, printable_event, run_print_mode


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

    # driving turns
    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
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

    def steer(self, text: str) -> None:
        pass

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
        json_mode=True, yolo=True, agent_name="A", agent_id="id1", hosting="openai", model="gpt-4o"
    )
    argv = build_worker_argv("do it", args)
    assert argv[:3] == [sys.executable, "-m", "local_operator.exec_worker"]
    assert argv[3:5] == ["--prompt", "do it"]
    # Every set flag serializes; parse it back through the worker parser.
    parsed = exec_worker.build_parser().parse_args(argv[3:])
    assert parsed.prompt == "do it"
    assert parsed.json_mode is True
    assert parsed.yolo is True
    assert parsed.agent == "A"
    assert parsed.agent_id == "id1"
    assert parsed.hosting == "openai"
    assert parsed.model == "gpt-4o"


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
    script = [
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
    # Quadratic-growth fix: message_update keeps ONLY the delta.
    update = lines[2]
    assert update == {"type": "message_update", "delta": "streamed"}


def test_printable_event_strips_provider_payload() -> None:
    message = Message.assistant("x", provider_payload={"encrypted": "blob"})
    event = MessageEndEvent(message=message)
    out = printable_event(event)
    # The payload key is gone entirely, and nothing leaks the secret value.
    assert "provider_payload" not in out["message"]
    assert "encrypted" not in json.dumps(out)


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
    renderer_seen: list[str] = []

    code = await run_print_mode(session, ["first", "second"])
    assert code == 0
    assert session.prompts == ["first", "second"]
    # Last assistant text wins in text mode.
    captured = capsys.readouterr()
    assert "two" in captured.out


# --- exec_mode background --------------------------------------------------------


def test_run_exec_background_spawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(exec_mode, "LOGS_DIR", logs_dir)

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
    assert argv[:5] == [sys.executable, "-m", "local_operator.exec_worker", "--prompt",
                        "write a long report about penguins"]
    assert "--json" in argv and "--yolo" in argv
    kwargs = popen_mock.call_args[1]
    if sys.platform != "win32":
        assert kwargs.get("start_new_session") is True

    # Log path printed and registered; jobs.json ledger appended.
    out = capsys.readouterr().out
    assert "Started background job" in out
    log_line = next(line for line in out.splitlines() if line.startswith("Log: "))
    log_path = Path(log_line.removeprefix("Log: "))
    assert log_path == logs_dir / log_path.name
    assert log_path.name.startswith("exec-")
    assert "write-a-long-report-about-penguins" in log_path.name
    assert log_path.exists()
    assert "write a long report about penguins" in log_path.read_text()

    jobs_path = logs_dir / "exec-jobs.json"
    records = json.loads(jobs_path.read_text())
    assert len(records) == 1
    record = records[0]
    assert record["pid"] == 4321
    assert record["prompt"] == "write a long report about penguins"
    assert record["log"] == str(log_path)
    assert set(record) == {"id", "started_at", "prompt", "log", "pid"}

    # A second spawn APPENDS rather than overwrites.
    exec_mode.run_exec("second task", ExecArgs(background=True))
    records = json.loads(jobs_path.read_text())
    assert len(records) == 2
    assert records[1]["prompt"] == "second task"


# --- exec_worker -----------------------------------------------------------------


def test_exec_worker_success(fake_factory, capsys) -> None:
    session = FakeSession([_success_script("worker says hi")])
    monkeypatch_default_factory = lambda: None  # noqa: E731 — placeholder for clarity
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
    parsed = exec_worker.build_parser().parse_args(["--prompt", "x"])
    monkeypatch.setattr(sys, "argv", ["exec_worker", "--prompt", "x"])

    def boom(_parsed: argparse.Namespace, session_factory=None) -> int:
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(exec_worker, "run", boom)
    assert exec_worker.main() == 1
    assert "exec_worker error: worker exploded" in capsys.readouterr().err


def test_exec_worker_parser_requires_prompt() -> None:
    with pytest.raises(SystemExit):
        exec_worker.build_parser().parse_args([])
