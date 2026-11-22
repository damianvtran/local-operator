"""Tests for opt-in verbose trace logging on ``local_operator.sdk``.

Covers:
* ``format_event_trace`` across all five traced ``AgentEvent`` types
  (``tool_execution_start``, ``tool_execution_end``, ``output_validation``,
  ``turn_end``, ``agent_end``) plus non-traced events returning ``None``.
* Bounded, single-line output so large file reads/writes never flood logs.
* Secret redaction: both repo credential shapes (``redaction_shapes``) and
  environment variables whose names carry ``KEY``, ``TOKEN``, ``SECRET``,
  ``PASSWORD``, or ``AUTHORIZATION`` (non-empty, >= 8 chars), including
  secrets that span the truncation boundary.
* ``trace_events`` helper (default logger, custom logger, stream sink,
  idempotent unsubscribe, defensive error isolation).
* ``open_session(..., verbose=True/False)`` lifecycle: silent by default,
  subscribes and logs real turns when enabled, and unsubscribes cleanly on
  normal or exceptional exit before ``dispose()``.
"""

from __future__ import annotations

import io
import json
import logging
import random
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import pytest

from local_operator import sdk
from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    ChatRequest,
    Message,
    MessageUpdateEvent,
    OutputValidationEvent,
    StreamEndEvent,
    StreamEvent,
    StreamStartEvent,
    StreamToolCallDelta,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
)
from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionRoots:
    """Isolated scratch roots matching ``tests/unit/test_sdk.py``."""
    from local_operator import agent_shell

    home = tmp_path / "home"
    root = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "decoy-config"))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "decoy-home"))
    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
    return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)


class _StubSession:
    """Minimal subscribe/emit stub for unit-testing ``trace_events``."""

    def __init__(self) -> None:
        self.handlers: list[Callable[[Any], None]] = []
        self.unsub_calls = 0

    def subscribe(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self.handlers.append(handler)

        def _unsub() -> None:
            self.unsub_calls += 1
            if handler in self.handlers:
                self.handlers.remove(handler)

        return _unsub

    def emit(self, event: Any) -> None:
        for handler in list(self.handlers):
            handler(event)


# --- format_event_trace -----------------------------------------------------------


def test_format_tool_execution_start_includes_short_id_and_bounded_json_args() -> None:
    long_content = "line1\nline2\t" + ("x" * 300)
    event = ToolExecutionStartEvent(
        tool_call_id="call_1234567890abcdef",
        tool_name="write",
        args={"path": "notes.txt", "content": long_content},
    )

    line = sdk.format_event_trace(event)
    assert line is not None
    assert "\n" not in line
    assert "\r" not in line
    assert "write" in line
    assert "call_1234567" in line
    assert "call_1234567890abcdef" not in line
    assert '"path": "notes.txt"' in line
    assert "x" * 200 not in line
    assert "..." in line


def test_format_tool_execution_start_bounds_wide_arg_payloads() -> None:
    wide_args = {f"k{i}": f"value_{i:03d}" for i in range(100)}
    event = ToolExecutionStartEvent(
        tool_call_id="call_wide",
        tool_name="search",
        args=wide_args,
    )

    line = sdk.format_event_trace(event)
    assert line is not None
    assert "\n" not in line
    assert len(line) < 650
    assert line.endswith("...")


def test_format_tool_execution_end_returned_and_error_with_duration() -> None:
    ok_event = ToolExecutionEndEvent(
        tool_call_id="call_1234567890abcdef",
        tool_name="read",
        result=ToolResult(
            tool_call_id="call_1234567890abcdef",
            tool_name="read",
            content=[TextContent(text="first line\nsecond line\n" + ("y" * 250))],
        ),
        duration_s=0.0124,
        is_error=False,
    )
    ok_line = sdk.format_event_trace(ok_event)
    assert ok_line is not None
    assert "\n" not in ok_line
    assert "read" in ok_line
    assert " returned " in ok_line
    assert "0.012s" in ok_line
    assert "first line second line" in ok_line
    assert "y" * 200 not in ok_line

    err_event = ToolExecutionEndEvent(
        tool_call_id="call_err987654321",
        tool_name="bash",
        result=ToolResult(
            tool_call_id="call_err987654321",
            tool_name="bash",
            content=[TextContent(text="Permission denied: /etc/shadow\nstack detail")],
            is_error=True,
        ),
        duration_s=0.25,
        is_error=True,
    )
    err_line = sdk.format_event_trace(err_event)
    assert err_line is not None
    assert "\n" not in err_line
    assert "bash" in err_line
    assert "error" in err_line
    assert "0.250s" in err_line
    assert "Permission denied: /etc/shadow" in err_line

    # bash reports a process exit code in the text, not as a tool error.
    ok_event.result.content = [TextContent(text="exit code: 7")]
    returned = sdk.format_event_trace(ok_event)
    assert returned is not None
    assert " returned " in returned and "exit code: 7" in returned


def test_format_output_validation_ok_retry_and_exhausted() -> None:
    ok = OutputValidationEvent(format="json", attempt=1, max_attempts=3, ok=True)
    retry = OutputValidationEvent(
        format="json",
        attempt=1,
        max_attempts=3,
        ok=False,
        exhausted=False,
        error="missing required field 'summary'",
    )
    exhausted = OutputValidationEvent(
        format="markdown",
        attempt=3,
        max_attempts=3,
        ok=False,
        exhausted=True,
        error="schema validation failed\non line 2",
    )

    ok_line = sdk.format_event_trace(ok)
    retry_line = sdk.format_event_trace(retry)
    exh_line = sdk.format_event_trace(exhausted)

    assert ok_line is not None
    assert "1/3" in ok_line and "ok" in ok_line
    assert retry_line is not None
    assert "1/3" in retry_line and "retry" in retry_line
    assert "missing required field 'summary'" in retry_line
    assert exh_line is not None
    assert "\n" not in exh_line
    assert "3/3" in exh_line and "exhausted" in exh_line
    assert "schema validation failed on line 2" in exh_line


def test_format_turn_end_model_tokens_and_cost() -> None:
    msg = Message.assistant(
        "Done",
        usage=Usage(
            provider="anthropic",
            model_id="claude-sonnet-4-5",
            input_tokens=1200,
            cache_read_tokens=800,
            output_tokens=150,
            usd_cost=0.0042,
        ),
    )
    line = sdk.format_event_trace(TurnEndEvent(message=msg))
    assert line is not None
    assert "claude-sonnet-4-5" in line
    assert "in=1200" in line
    assert "cache_read=800" in line
    assert "out=150" in line
    assert "$0.0042" in line

    # Estimated cost fallback when provider usd_cost is absent.
    est_msg = Message.assistant(
        "Done",
        usage=Usage(
            model_id="gpt-5",
            input_tokens=500,
            cache_read_tokens=0,
            output_tokens=40,
            estimated_usd_cost=0.0015,
        ),
    )
    est_line = sdk.format_event_trace(TurnEndEvent(message=est_msg))
    assert est_line is not None
    assert "gpt-5" in est_line
    assert "in=500" in est_line
    assert "cache_read=0" in est_line
    assert "out=40" in est_line
    assert "$0.0015" in est_line

    # TurnEndEvent without usage still formats cleanly.
    bare_line = sdk.format_event_trace(TurnEndEvent())
    assert bare_line is not None
    assert "turn" in bare_line.lower()


def test_format_agent_end_completed_aborted_and_error() -> None:
    completed = sdk.format_event_trace(AgentEndEvent())
    aborted = sdk.format_event_trace(AgentEndEvent(aborted=True))
    errored = sdk.format_event_trace(
        AgentEndEvent(error="rate limit exceeded\nretry later", failing_model="openai/gpt-5")
    )

    assert completed is not None and "completed" in completed
    assert aborted is not None and "aborted" in aborted
    assert errored is not None
    assert "\n" not in errored
    assert "error" in errored
    assert "rate limit exceeded retry later" in errored


def test_format_event_trace_ignores_non_traced_events() -> None:
    assert sdk.format_event_trace(AgentStartEvent()) is None
    assert sdk.format_event_trace(TurnStartEvent()) is None
    assert (
        sdk.format_event_trace(MessageUpdateEvent(message=Message.assistant("hi"), delta="hi"))
        is None
    )


# --- redaction --------------------------------------------------------------------


def test_format_event_trace_redacts_env_secrets_and_credential_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_key = "test-live-api-key-987654321"
    secret_tok = "tok-super-secret-value-abcd"
    secret_pwd = "MyP@ssw0rdSecret!"
    secret_auth = "Basic dXNlcjpzZWNyZXRwYXNz"
    short_val = "short12"
    monkeypatch.setenv("CUSTOM_API_KEY", secret_key)
    monkeypatch.setenv("SERVICE_TOKEN", secret_tok)
    monkeypatch.setenv("DB_PASSWORD", secret_pwd)
    monkeypatch.setenv("HTTP_AUTHORIZATION", secret_auth)
    monkeypatch.setenv("SHORT_PASSWORD", short_val)
    monkeypatch.setenv("PUBLIC_APP_MODE", "ordinary_public_value_12345")

    start = ToolExecutionStartEvent(
        tool_call_id="call_sec1",
        tool_name="bash",
        args={
            "command": (
                f"curl -H '{secret_auth}' -H 'X-Key: {secret_key}' https://api.test/{short_val}"
            ),
            "note": "ordinary_public_value_12345",
            "bearer": "Authorization: Bearer raw_unregistered_bearer_token_999999",
        },
    )
    start_line = sdk.format_event_trace(start)
    assert start_line is not None
    assert secret_key not in start_line
    assert secret_auth not in start_line
    assert "raw_unregistered_bearer_token_999999" not in start_line
    assert REDACTION_MARKER in start_line
    assert short_val not in start_line
    assert "ordinary_public_value_12345" in start_line

    # Error strings in tool_execution_end and agent_end must also be redacted.
    end = ToolExecutionEndEvent(
        tool_call_id="call_sec1",
        tool_name="bash",
        result=ToolResult(
            tool_call_id="call_sec1",
            tool_name="bash",
            content=[TextContent(text=f"failed with {secret_tok} and {secret_pwd}")],
            is_error=True,
        ),
        duration_s=0.05,
        is_error=True,
    )
    end_line = sdk.format_event_trace(end)
    assert end_line is not None
    assert secret_tok not in end_line
    assert secret_pwd not in end_line
    assert REDACTION_MARKER in end_line

    agent_err = sdk.format_event_trace(AgentEndEvent(error=f"auth rejected for {secret_key}"))
    assert agent_err is not None
    assert secret_key not in agent_err
    assert REDACTION_MARKER in agent_err


def test_format_event_trace_redacts_secret_crossing_truncation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secret starting before the 160-char arg cutoff and extending past it
    must be redacted BEFORE truncation so its prefix never leaks."""
    boundary_secret = "SECRET_BOUNDARY_" + ("Z" * 180)
    monkeypatch.setenv("BOUNDARY_SECRET", boundary_secret)

    payload = ("a" * 140) + " " + boundary_secret + " tail"
    event = ToolExecutionStartEvent(
        tool_call_id="call_bound",
        tool_name="write",
        args={"content": payload},
    )
    line = sdk.format_event_trace(event)
    assert line is not None
    assert "SECRET_BOUNDARY_" not in line
    assert REDACTION_MARKER in line


# --- trace_events helper ----------------------------------------------------------


def test_format_event_trace_withholds_credential_subtrees_before_clipping() -> None:
    value = "".join(random.Random(4427).choices(string.ascii_letters + string.digits, k=420))
    event = ToolExecutionStartEvent(
        tool_call_id="call_fixture",
        tool_name="fixture",
        args={"api_key": value, "nested": [{"db_password": {"unknown": [value]}}], "public": "ok"},
    )

    line = sdk.format_event_trace(event)
    assert line is not None
    assert value[:12] not in line
    assert '"api_key": "' + REDACTION_MARKER + '"' in line
    assert '"db_password": "' + REDACTION_MARKER + '"' in line
    assert '"public": "ok"' in line


def test_format_event_trace_scrubs_complete_name_value_pair() -> None:
    value = "".join(random.Random(4428).choices(string.ascii_letters + string.digits, k=420))
    event = ToolExecutionStartEvent(
        tool_call_id="call_fixture",
        tool_name="fixture",
        args={"env": [{"value": value, "name": "DB_PASSWORD"}]},
    )

    line = sdk.format_event_trace(event)
    assert line is not None
    assert value[:12] not in line
    assert f'"value": "{REDACTION_MARKER}"' in line
    assert '"name": "DB_PASSWORD"' in line


def test_format_event_trace_withholds_over_window_payload() -> None:
    pem = "-----BEGIN PRIVATE KEY-----\nsynthetic-pem-body\n" + ("A" * 5000)
    pem += "\n-----END PRIVATE KEY-----"
    event = AgentEndEvent(error=pem)

    line = sdk.format_event_trace(event)
    assert line is not None
    assert "synthetic-pem-body" not in line
    assert "BEGIN PRIVATE KEY" not in line
    assert REDACTION_MARKER in line


def test_format_event_trace_redacts_call_id_before_shortening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "syntheticId943AbCd21"
    monkeypatch.setenv("FIXTURE_API_KEY", value)
    event = ToolExecutionStartEvent(tool_call_id=value, tool_name="fixture", args={})

    line = sdk.format_event_trace(event)
    assert line is not None
    assert value[:12] not in line
    assert REDACTION_MARKER in line


def test_format_event_trace_does_not_emit_raw_control_characters() -> None:
    controls = "\x1b[31m\x00\x07\x7f\x9b\u2028\u202e"
    event = AgentEndEvent(error=f"left{controls}right")

    line = sdk.format_event_trace(event)
    assert line is not None
    assert "left" in line and "right" in line
    assert not any(char in line for char in controls if not char.isprintable())


def test_trace_events_logs_to_default_logger_and_unsubscribes_idempotently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stub = _StubSession()
    with caplog.at_level(logging.INFO, logger="local_operator.sdk"):
        unsub = sdk.trace_events(stub)
        stub.emit(AgentStartEvent())  # ignored
        stub.emit(
            ToolExecutionStartEvent(
                tool_call_id="call_001", tool_name="glob", args={"pattern": "*.py"}
            )
        )
        stub.emit(AgentEndEvent())
        unsub()
        unsub()  # idempotent
        stub.emit(AgentEndEvent(aborted=True))  # must not be logged after unsubscribe

    sdk_records = [r for r in caplog.records if r.name == "local_operator.sdk"]
    assert len(sdk_records) == 2
    assert sdk_records[0].levelno == logging.INFO
    assert "glob" in sdk_records[0].getMessage()
    assert "completed" in sdk_records[1].getMessage()
    assert stub.unsub_calls == 1
    assert not stub.handlers


def test_trace_events_supports_stream_and_custom_logger() -> None:
    stub = _StubSession()
    buf = io.StringIO()
    unsub = sdk.trace_events(stub, stream=buf)
    stub.emit(OutputValidationEvent(format="json", attempt=2, max_attempts=3, ok=True))
    unsub()

    output = buf.getvalue().splitlines()
    assert len(output) == 1
    assert "2/3" in output[0] and "ok" in output[0]

    # Explicit custom logger receives records instead of the default logger.
    custom_logger = logging.getLogger("test.custom.sdk.trace")
    captured: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _ListHandler()
    custom_logger.addHandler(handler)
    custom_logger.setLevel(logging.INFO)
    custom_logger.propagate = False
    try:
        unsub2 = sdk.trace_events(stub, logger=custom_logger)
        stub.emit(AgentEndEvent())
        unsub2()
    finally:
        custom_logger.removeHandler(handler)

    assert len(captured) == 1
    assert "completed" in captured[0]


def test_trace_events_survives_broken_stream_without_breaking_dispatch() -> None:
    class _BrokenStream:
        def write(self, _data: str) -> int:
            raise OSError("broken pipe")

    stub = _StubSession()
    unsub = sdk.trace_events(stub, stream=_BrokenStream())  # type: ignore[arg-type]
    # Must never propagate into the session emitter.
    stub.emit(AgentEndEvent())
    unsub()


# --- open_session(verbose=...) integration ----------------------------------------


@pytest.mark.asyncio
async def test_open_session_verbose_default_silent_and_opt_in_traces_real_turn(
    scratch: SessionRoots, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.providers.clients import MockClient

    credentials = {
        "SDK_QA_PASSWORD": "qA!7-zX",
        "DB_PASSWORD_FALLBACK": "pR!9-fallback-4Z",
        "EXAMPLE_TOKEN_PRIMARY": "tQ!2-primary-7X",
    }
    control = "\x1b[31m"
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    path = Path(scratch.cwd) / "fixture.txt"
    path.write_text(f"visible {' '.join(credentials.values())} {control}text\n", encoding="utf-8")
    spec = SessionSpec(
        hosting="test",
        model="mock",
        tools=("read", "bash"),
        approvals=ApprovalPolicy.declared(("read", "bash")),
    )
    real_stream = MockClient.stream

    async def tool_stream(
        self: MockClient, request: ChatRequest, *args: Any, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        if request.wire_messages()[-1].role != "user":
            async for event in real_stream(self, request, *args, **kwargs):
                yield event
            return
        yield StreamStartEvent(response_id="fixture-response")
        for index, (name, arguments) in enumerate(
            [("read", {"path": str(path)}), ("bash", {"command": "exit 7"})]
        ):
            yield StreamToolCallDelta(index=index, id=f"call_{name}", name=name)
            yield StreamToolCallDelta(index=index, argument_delta=json.dumps(arguments))
        yield StreamEndEvent(stop_reason="toolUse", usage=Usage(input_tokens=1, output_tokens=1))

    monkeypatch.setattr(MockClient, "stream", tool_stream)
    with caplog.at_level(logging.INFO, logger="local_operator.sdk"):
        async with sdk.open_session(spec, roots=scratch) as session:
            await session.prompt("read the fixture and exercise an exit status")
    assert not [rec for rec in caplog.records if rec.name == "local_operator.sdk"]

    output = io.StringIO()
    with caplog.at_level(logging.INFO, logger="local_operator.sdk"):
        async with sdk.open_session(spec, roots=scratch, verbose=True) as session:
            unsubscribe = sdk.trace_events(session, stream=output)
            try:
                await session.prompt("read the fixture and exercise an exit status")
            finally:
                unsubscribe()
        after_count = len([r for r in caplog.records if r.name == "local_operator.sdk"])
        await session._emit(AgentEndEvent())  # type: ignore[attr-defined]
        assert len([r for r in caplog.records if r.name == "local_operator.sdk"]) == after_count

    sdk_lines = [r.getMessage() for r in caplog.records if r.name == "local_operator.sdk"]
    assert output.getvalue().splitlines() == sdk_lines
    assert any(line.startswith("[turn:end]") for line in sdk_lines)
    assert any(line.startswith("[agent:end] completed") for line in sdk_lines)
    read_line = next(line for line in sdk_lines if line.startswith("[tool:end] read "))
    assert "visible" in read_line and read_line.count(REDACTION_MARKER) == len(credentials)
    assert all(value not in read_line for value in credentials.values())
    assert control not in read_line
    assert all(char.isprintable() for char in read_line)
    bash_line = next(line for line in sdk_lines if line.startswith("[tool:end] bash "))
    assert " returned " in bash_line and "exit code: 7" in bash_line


@pytest.mark.asyncio
@pytest.mark.parametrize("verbose", [False, True])
async def test_open_session_attached_trace_preserves_backlog_and_logs_each_frame_once(
    scratch: SessionRoots,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    verbose: bool,
) -> None:
    from local_operator.session.attached import AttachedSession

    viewer = AttachedSession(
        config_dir=Path(scratch.config_dir), session_id="viewer", takeover_factory=lambda: None
    )
    viewer._ready_for_events = True
    start = ToolExecutionStartEvent(tool_call_id="call_backlog", tool_name="read", args={})
    before_subscribe = ToolExecutionStartEvent(
        tool_call_id="call_pending", tool_name="read", args={}
    )
    live = ToolExecutionStartEvent(tool_call_id="call_live", tool_name="read", args={})
    viewer._emit_or_buffer(start)

    async def build(*args: Any, **kwargs: Any) -> AttachedSession:
        return viewer

    monkeypatch.setattr(sdk, "_build_session", build)
    with caplog.at_level(logging.INFO, logger="local_operator.sdk"):
        async with sdk.open_session(
            SessionSpec(resume="viewer"), roots=scratch, mode="attach", verbose=verbose
        ) as session:
            viewer._emit_or_buffer(before_subscribe)
            async with sdk.events(session) as stream:
                viewer._emit_or_buffer(live)
                assert await anext(stream) == start
                assert [await anext(stream) for _ in range(2)] == [before_subscribe, live]
        viewer._emit_or_buffer(live)

    lines = [rec.getMessage() for rec in caplog.records if rec.name == "local_operator.sdk"]
    assert lines == (
        [
            f"[tool:start] read ({event.tool_call_id}) {{}}"
            for event in (start, before_subscribe, live)
        ]
        if verbose
        else []
    )


@pytest.mark.parametrize("caller_subscribed", [False, True])
def test_attached_trace_follows_sync_seed_order_without_consuming_replay(
    scratch: SessionRoots, caller_subscribed: bool
) -> None:
    from local_operator.session.attached import AttachedSession
    from local_operator.session.frontend_state import FrontendSessionState

    viewer = AttachedSession(
        config_dir=Path(scratch.config_dir), session_id="viewer", takeover_factory=lambda: None
    )
    start = ToolExecutionStartEvent(tool_call_id="call_fixture", tool_name="read", args={})
    end = ToolExecutionEndEvent(
        tool_call_id="call_fixture",
        tool_name="read",
        result=ToolResult(tool_call_id="call_fixture", content=[]),
    )
    buffer = io.StringIO()
    received: list[Any] = []
    expected_lines = []
    expected_call_ids = ["call_fixture", "call_fixture"]
    if caller_subscribed:
        viewer.subscribe(received.append)
        sdk.trace_events(viewer, stream=buffer)
    else:
        # An older ready frame can still be waiting for the first caller.
        viewer._ready_for_events = True
        viewer._emit_or_buffer(
            ToolExecutionStartEvent(tool_call_id="call_older", tool_name="read", args={})
        )
        viewer._ready_for_events = False
        expected_lines.append("[tool:start] read (call_older) {}")
        expected_call_ids.insert(1, "call_older")
    viewer._install_frontend(
        FrontendSessionState(
            session_id="viewer",
            epoch="e1",
            streaming=True,
            live_events=[start.model_dump(mode="json")],
        )
    )
    viewer._on_wire_event(end.model_dump(mode="json"))
    if not caller_subscribed:
        sdk.trace_events(viewer, stream=buffer)
    viewer._finish_sync()
    if not caller_subscribed:
        viewer.subscribe(received.append)

    assert received[0].type == "agent_start"
    assert [event.tool_call_id for event in received[1:]] == expected_call_ids
    assert isinstance(received[1], ToolExecutionStartEvent)
    assert isinstance(received[-1], ToolExecutionEndEvent)
    lines = buffer.getvalue().splitlines()
    assert lines[:-1] == [*expected_lines, "[tool:start] read (call_fixture) {}"]
    assert lines[-1].startswith("[tool:end] read (call_fixture) returned")


@pytest.mark.asyncio
async def test_open_session_verbose_unsubscribes_before_dispose_on_exception(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = SessionSpec(hosting="test", model="mock")
    order: list[str] = []

    real_trace = sdk.trace_events

    def _spy_trace(session: Any, **kwargs: Any) -> Callable[[], None]:
        unsub = real_trace(session, **kwargs)
        orig_dispose = session.dispose

        async def _spy_dispose() -> None:
            order.append("dispose")
            await orig_dispose()

        session.dispose = _spy_dispose

        def _spy_unsub() -> None:
            order.append("unsub")
            unsub()

        return _spy_unsub

    monkeypatch.setattr(sdk, "trace_events", _spy_trace)

    with pytest.raises(RuntimeError, match="boom"):
        async with sdk.open_session(spec, roots=scratch, verbose=True):
            raise RuntimeError("boom")

    assert order == ["unsub", "dispose"]


def test_importing_sdk_still_leaves_redaction_and_engine_off_import_graph() -> None:
    probe = (
        "import json, importlib, sys; "
        "importlib.import_module('local_operator.sdk'); "
        "print(json.dumps(sorted(sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    modules = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    assert "local_operator.redaction_shapes" not in modules
    assert "local_operator.harness.types" not in modules
