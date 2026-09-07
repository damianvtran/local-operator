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
import io
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.console import Console

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
    NoticeEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.headless_print import PrintRenderer, printable_event, run_print_mode
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.session.naming import ConversationName
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

    @property
    def effective_model(self) -> ModelSpec:
        # The fake never falls back, so selection and effective agree.
        return self.model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: ModelSpec, *, explicit: bool = False) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    @property
    def conversation_name(self) -> str:
        return self.conversation_name_state.text

    @property
    def conversation_name_state(self) -> ConversationName:
        # The real holder, created on first read: `user_set` precedence (a
        # human rename outranks every generated title, forever) is behaviour
        # the TUI reads before it spends a re-title call, so a fake that
        # reimplemented it as a bare string would hide a regression in it.
        state = getattr(self, "_name_state", None)
        if state is None:
            state = self._name_state = ConversationName()
        return state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return self.conversation_name_state.set(text, user_set=user_set)

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

    def queued_steering(self) -> list[Any]:
        return []

    def steer_message(self, message: Any) -> None:
        pass

    def recall_steering(self, message: Any) -> bool:
        return False

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

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def running_subagents(self) -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

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
    # Opt-in: a detached run publishes no record unless asked.
    assert "--control" not in argv
    assert parsed.control is False


def test_build_worker_argv_threads_control_to_the_worker() -> None:
    """`--background --control` is the same request run elsewhere.

    A detached run is the one that most needs steering — nobody is watching its
    log — so a flag dropped at this process boundary would be accepted by the
    front end and silently lost, the failure the ``resume`` field records.
    """
    argv = build_worker_argv("do it", ExecArgs(control=True))
    assert "--control" in argv
    parsed = exec_worker.build_parser().parse_args(argv[3:])
    assert parsed.control is True


# --- exec_mode foreground -------------------------------------------------------


def test_run_exec_starts_no_control_surface_by_default(fake_factory, monkeypatch, capsys) -> None:
    """The default exec run stays invisible: no record, no socket, no line.

    Opt-in is the whole design (`lop sessions` filters on nothing, so every
    scripted run would otherwise become a `lop send` target), and this is the
    pin on it.
    """
    started: list[bool] = []
    monkeypatch.setattr(
        "local_operator.session.runtime.exec_control.start_exec_control",
        lambda *a, **k: started.append(True),
    )
    session = FakeSession([_success_script()])
    fake_factory(session)
    assert exec_mode.run_exec("say hello", ExecArgs()) == 0
    assert started == []
    assert "lop exec control" not in capsys.readouterr().err


def test_run_exec_control_prints_the_endpoint_on_stderr(fake_factory, monkeypatch, capsys) -> None:
    """stdout is the payload stream; chrome that lands in it corrupts the only
    output the run has."""

    class _Control:
        endpoint_line = "lop exec control: session_id=x pid=1 port=2 record=/r"

        async def aclose(self) -> None:
            closed.append(True)

    closed: list[bool] = []

    async def fake_start(session, *, enabled, cwd, yolo=False):  # noqa: ANN001
        assert enabled is True
        return _Control()

    monkeypatch.setattr(
        "local_operator.session.runtime.exec_control.maybe_start_exec_control", fake_start
    )
    session = FakeSession([_success_script()])
    fake_factory(session)
    assert exec_mode.run_exec("say hello", ExecArgs(control=True)) == 0
    captured = capsys.readouterr()
    assert "lop exec control:" in captured.err
    assert "lop exec control:" not in captured.out
    assert closed == [True]


def test_run_exec_control_closes_before_the_session_is_disposed(fake_factory, monkeypatch) -> None:
    """The ordering that lets an attached supervisor see a deliberate end.

    Closing after the dispose would leave the runtime reading through its handle
    into a session being torn down, and the supervisor would see a bare EOF it
    cannot tell from a crash.
    """
    order: list[str] = []

    class _Control:
        endpoint_line = "endpoint"

        async def aclose(self) -> None:
            order.append("control-closed")

    async def fake_start(session, *, enabled, cwd, yolo=False):  # noqa: ANN001
        return _Control()

    monkeypatch.setattr(
        "local_operator.session.runtime.exec_control.maybe_start_exec_control", fake_start
    )
    session = FakeSession([_success_script()])
    original_dispose = session.dispose

    async def tracking_dispose() -> None:
        order.append("session-disposed")
        await original_dispose()

    session.dispose = tracking_dispose  # type: ignore[method-assign]
    fake_factory(session)
    assert exec_mode.run_exec("say hello", ExecArgs(control=True)) == 0
    assert order == ["control-closed", "session-disposed"]


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
    # message_id so JSON consumers can attribute deltas (CL-15), and the
    # session_id every json-mode line carries for stateless per-line parsers.
    # Asserted as an exact shape (minus the session stamp, which the fake
    # session supplies) because the property under test is what is ABSENT:
    # no full-message snapshot may creep back in.
    update = lines[2]
    assert {key: value for key, value in update.items() if key != "session_id"} == {
        "type": "message_update",
        "message_id": reply.id,
        "delta": "streamed",
    }
    assert "message" not in update, "the full message snapshot must not return"


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


def test_an_error_notice_is_marked_by_a_glyph_not_only_by_colour() -> None:
    """Design round 3, D11. Piped logs and NO_COLOR strip the only signal.

    This renderer writes to a real terminal, but its output is also redirected
    into logs and read with colour disabled — and there an error notice was
    byte-identical to an informational one. It matters most for the line that
    prompted it: an unrunnable tool call used to print `✗ <name> failed`, which
    DID carry a marker, so moving that diagnostic onto a notice dropped one.

    `info` stays bare on purpose: a marker on every routine line is noise, and
    it is the one kind with nothing to warn about.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, width=100)
    renderer = PrintRenderer(json_mode=False, console=console)

    renderer.handle(NoticeEvent(text="Tool not found: reed_file", kind="error"))
    renderer.handle(NoticeEvent(text="running low on context", kind="warning"))
    renderer.handle(NoticeEvent(text="compacted", kind="info"))

    lines = buffer.getvalue().splitlines()
    assert lines[0] == "✗ Tool not found: reed_file"
    assert lines[1] == "! running low on context"
    assert lines[2] == "compacted"


def test_a_notice_cannot_smuggle_control_sequences_to_the_terminal() -> None:
    """Round 7, R7-1. Notice text is no longer only ours.

    The unrunnable-call diagnostic carries a MODEL-CHOSEN tool name, so an
    erase-display escape inside it reaches a real terminal and clears the
    operator's screen. The `✗ <name> failed` line this diagnostic replaced was
    stripped for exactly that reason; moving the message onto a notice moved it
    off the guard.

    Asserted on the RENDERER rather than the producer, because that is where
    the guard now lives: the next notice to carry untrusted text should not
    have to remember to sanitize itself.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, no_color=True, highlight=False, width=100)
    renderer = PrintRenderer(json_mode=False, console=console)

    renderer.handle(NoticeEvent(text="Tool not found: ru\x1b[2Jn", kind="error"))

    out = buffer.getvalue()
    assert "\x1b" not in out, f"a control sequence reached the terminal: {out!r}"
    assert "2J" not in out, f"an erase-display escape survived stripping: {out!r}"
    # The message itself still arrives, with its severity marker.
    assert out.startswith("✗ Tool not found: ")


def test_a_refusal_with_markup_shaped_prose_still_prints_and_fails(fake_factory, capsys) -> None:
    """Review R1-1. ``agent_end.error`` now carries MODEL-AUTHORED prose (a
    provider's refusal message), and rendering it as rich markup meant prose
    like ``[/see our policy]`` raised MarkupError inside the subscriber —
    BEFORE the outcome tracker ran, so exec printed a traceback instead of the
    error line and exited 0: the silent-refusal bug resurfacing on adversarial
    text. The text is data, never markup.
    """
    refusal = "model refused: I can't comply [/see our policy] with that. (finish_reason=stop)"
    reply = Message(role="assistant", stop_reason="refusal")
    script: list[AgentEvent] = [
        AgentStartEvent(),
        MessageStartEvent(message=reply),
        MessageEndEvent(message=reply),
        AgentEndEvent(messages=[reply], error=refusal),
    ]
    fake_factory(FakeSession([script]))
    code = exec_mode.run_exec("refused task", ExecArgs())
    captured = capsys.readouterr()
    assert code == 1
    assert "[/see our policy]" in captured.err


def test_a_notice_cannot_forge_a_row_or_crash_the_renderer() -> None:
    """Design round 4, D14/D15. The tool name in a notice is model-chosen.

    Three hazards beyond control sequences, all reachable from a hallucinated
    tool name and none covered by stripping alone:

    * square brackets are Rich MARKUP — `[bold]x` renders the wrong name, and
      `[/red]oops` raises `MarkupError` inside the renderer, which the session's
      emit path swallows, so the notice disappears entirely. That is the exact
      silence this diagnostic exists to prevent;
    * newlines survive stripping by design, so a name containing one forges a
      second, unmarked row that can read as a clean success;
    * both were pre-existing on the `✗ <name> failed` line this replaced.
    """

    def render(name: str) -> str:
        buffer = io.StringIO()
        console = Console(file=buffer, no_color=True, highlight=False, width=100)
        PrintRenderer(json_mode=False, console=console).handle(
            NoticeEvent(text=f"Tool not found: {name}", kind="error")
        )
        return buffer.getvalue()

    # Unbalanced markup must not raise, and must not be interpreted.
    assert render("[/red]oops") == "✗ Tool not found: [/red]oops\n"
    # Balanced markup must not be interpreted either — the name is shown as-is.
    assert render("[bold]x") == "✗ Tool not found: [bold]x\n"
    # A newline must not forge a second row that carries its own claim.
    forged = render("a\n✓ 3 subagents finished cleanly")
    assert forged.count("\n") == 1, f"the name forged an extra row: {forged!r}"
    assert forged.startswith("✗ ")


def test_no_renderer_branch_interprets_a_tool_name_as_markup() -> None:
    """Round 14, R14-2. The rule has to hold for every branch, not the newest.

    The tool NAME is model-chosen, and a `[` in it is Rich markup: `[/red]x`
    raises `MarkupError`, which the session's emit path swallows, so the row
    vanishes and the operator watches a tool run with no line at all. The
    notice branch was fixed for D14; the two tool-event branches above it had
    the same defect (pre-existing on main) and the fix's own comment
    generalised the rule further than the code did.
    """
    name = "[/red]oops"

    def render(event: object) -> str:
        buffer = io.StringIO()
        console = Console(file=buffer, no_color=True, highlight=False, width=100)
        PrintRenderer(json_mode=False, console=console).handle(event)  # type: ignore[arg-type]
        return buffer.getvalue()

    started = render(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name=name, args={}, intent="do it")
    )
    assert started == "● [/red]oops do it\n"

    result = ToolResult(
        tool_call_id="c1", tool_name=name, is_error=True, content=[TextContent(text="x")]
    )
    ended = render(
        ToolExecutionEndEvent(tool_call_id="c1", tool_name=name, result=result, is_error=True)
    )
    assert ended == "✗ [/red]oops failed\n"

    assert render(NoticeEvent(text=f"Tool not found: {name}", kind="error")) == (
        "✗ Tool not found: [/red]oops\n"
    )


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


@pytest.mark.asyncio
async def test_run_print_mode_runs_before_dispose_ahead_of_the_dispose() -> None:
    """The hook exists because the dispose is this function's own contract, so
    a caller cannot sequence anything ahead of it from the outside."""
    order: list[str] = []
    session = FakeSession([_success_script()])
    original = session.dispose

    async def tracking() -> None:
        order.append("disposed")
        await original()

    session.dispose = tracking  # type: ignore[method-assign]

    async def hook() -> None:
        order.append("hook")

    assert await run_print_mode(session, ["go"], before_dispose=hook) == 0
    assert order == ["hook", "disposed"]


@pytest.mark.asyncio
async def test_run_print_mode_runs_before_dispose_when_the_prompt_raises() -> None:
    """A control surface must be closed on the failure path too, or a crashed
    run leaves a published record for a scanner to reap."""
    order: list[str] = []

    class Exploding(FakeSession):
        async def prompt(self, text, images=None):  # noqa: ANN001, ANN201
            raise RuntimeError("boom")

        async def dispose(self) -> None:
            order.append("disposed")
            await super().dispose()

    async def hook() -> None:
        order.append("hook")

    with pytest.raises(RuntimeError):
        await run_print_mode(Exploding([_success_script()]), ["go"], before_dispose=hook)
    assert order == ["hook", "disposed"]


# --- exec_mode background --------------------------------------------------------


def _redirect_logs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point exec's logs/ledger root at ``tmp_path`` and return it.

    Sets ``LOCAL_OPERATOR_CONFIG_DIR``, which is the ONLY seam now that
    ``exec_mode.logs_dir()`` resolves through ``paths.config_dir()`` on every
    call rather than freezing a module constant at import. That is deliberately
    stricter than the ``monkeypatch.setattr(exec_mode, "LOGS_DIR", ...)`` these
    tests used before: patching the constant redirected the ledger while every
    OTHER root exec touches stayed on the real home, so a test could still reach
    outside its sandbox through a path it was not thinking about.
    """
    root = tmp_path / "config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
    return root / "logs"


def test_run_exec_background_spawn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    logs_dir = _redirect_logs_dir(monkeypatch, tmp_path)
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
    kwargs = popen_mock.call_args[1]

    # argv[0] is the process LABEL when a branded interpreter image exists
    # (`executable=` then carries the real image), and the bare interpreter
    # when it does not. Both are correct; what must never drift is the module
    # and the request that follow it. See `local_operator.procname`.
    if kwargs.get("executable"):
        assert os.path.basename(kwargs["executable"]) == "Local Operator"
        assert argv[0].startswith("Local Operator [exec] job=")
    else:
        assert argv[0] == sys.executable
    assert argv[1:5] == [
        "-m",
        "local_operator.exec_worker",
        "--prompt",
        "write a long report about penguins",
    ]
    assert "--json" in argv and "--yolo" in argv
    assert "--job-id" in argv  # CL-09 terminal-record wiring
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


def test_spawn_background_unconfigured_hosting_returns_one(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Item 18 contract: the ``exec --background`` preflight failure exits 1.

    Pins the exact regression that round-1 review MAJOR 1 caught — the sites
    returned ``-1``, which ``exit(main())`` maps to process exit 255, exactly
    the wrapped-negative outcome item 18 exists to eliminate. A dry-run that
    raises ``ValueError`` (nothing configured) must surface as a clean ``1`` and
    must NOT reach ``subprocess.Popen``: a preflight failure never spawns.
    """
    monkeypatch.setattr(
        exec_mode,
        "resolve_hosting_model_dry",
        lambda args: (_ for _ in ()).throw(ValueError("Hosting platform is not configured.")),
    )
    popen_mock = MagicMock()
    monkeypatch.setattr("local_operator.exec_mode.subprocess.Popen", popen_mock)

    code = exec_mode.run_exec("do a thing", ExecArgs(background=True))

    assert code == 1  # not -1 / 255
    popen_mock.assert_not_called()
    assert "Hosting platform is not configured." in capsys.readouterr().err


def test_ledger_reader_tolerates_partial_line(tmp_path: Path, monkeypatch) -> None:
    """CL-11: a truncated trailing line (crash mid-write) never breaks reads."""
    logs_dir = _redirect_logs_dir(monkeypatch, tmp_path)
    logs_dir.mkdir(parents=True)
    good = json.dumps({"id": "abc", "prompt": "ok"})
    (logs_dir / exec_mode.JOBS_FILE).write_text(
        good + "\n" + '{"id": "de", "prom', encoding="utf-8"
    )
    records = exec_mode.read_job_records()
    assert len(records) == 1
    assert records[0]["id"] == "abc"


def test_logs_dir_and_log_file_permissions(monkeypatch, tmp_path: Path) -> None:
    """CL-10: the logs dir is 0700 and job logs are created 0600."""
    logs_dir = _redirect_logs_dir(monkeypatch, tmp_path)
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
    logs_dir = _redirect_logs_dir(monkeypatch, tmp_path)

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
    logs_dir = _redirect_logs_dir(monkeypatch, tmp_path)
    logs_dir.mkdir(parents=True)
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


# --- exec config-root resolution (#737 regression guards) -------------------------
#
# Both sites below hardcoded ``Path.home() / ".local-operator"`` and were fixed in
# #737 to resolve through ``paths.config_dir()``. That fix shipped UNGUARDED: the
# whole exec suite passed with either site reverted, because the only test that
# reached ``_default_session_factory`` stubbed the managers with
# ``lambda *a: object()`` and threw the directory away. These tests assert on the
# ROOT THAT WAS ACTUALLY RESOLVED, so a revert to the hardcoded root fails them.
#
# ``LOCAL_OPERATOR_CONFIG_DIR`` is set with ``monkeypatch.setenv`` (the convention
# everywhere else in the suite), which unsets it at teardown; the autouse
# ``isolate_environment`` fixture in ``tests/conftest.py`` scrubs it and redirects
# ``HOME`` to a scratch dir, so the "home root" a mutant would resolve is that
# scratch dir and never the operator's real ``~/.local-operator``.


def test_worker_session_factory_resolves_the_config_dir_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``exec --background``'s worker builds its managers under the override.

    The BACKGROUND worker inherits the spawner's environment, so a hardcoded root
    here made ``exec --background`` ignore ``LOCAL_OPERATOR_CONFIG_DIR`` while the
    foreground run honoured it — one entry point resolving two different roots
    depending on a flag. It also feeds the analytics session-name backfill through
    ``create_session``'s store-maintenance pass, which writes to whatever root it
    is handed, so a wrong root here strands names in a ledger nothing reads.

    Asserted on the DIRECTORY EACH MANAGER RECEIVED, not on the fact that a call
    happened: the pre-existing ``test_build_worker_argv_train_threaded_to_worker``
    stubs the same three managers with ``lambda *a: object()`` and discards the
    argument, which is exactly why the revert of this site passed 36/36.
    """
    override = tmp_path / "override-config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(override))
    home_root = Path.home() / ".local-operator"

    seen: dict[str, Path] = {}

    def capture(name: str) -> Callable[..., object]:
        def factory(directory: Path, *_rest: object, **_kwargs: object) -> object:
            seen[name] = Path(directory)
            return object()

        return factory

    monkeypatch.setattr("local_operator.config.ConfigManager", capture("config"))
    monkeypatch.setattr("local_operator.credentials.CredentialManager", capture("credentials"))
    monkeypatch.setattr("local_operator.agents.AgentRegistry", capture("agents"))
    monkeypatch.setattr(
        "local_operator.session_factory.create_session", lambda *a, **k: None  # noqa: ARG005
    )

    parsed = exec_worker.build_parser().parse_args(["--prompt", "x"])
    exec_worker._default_session_factory(parsed)

    assert seen == {
        "config": override,
        "credentials": override,
        "agents": override,
    }, (
        f"the worker built its managers under {sorted(set(map(str, seen.values())))} "
        f"instead of the override {override}; the home root is {home_root}"
    )


def test_worker_session_factory_writes_nothing_outside_the_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same site, asserted on the SIDE EFFECTS of the real managers.

    The stub test above pins the argument; this one pins where the bytes land,
    with the genuine ``ConfigManager``/``CredentialManager``/``AgentRegistry``
    constructed. That distinction is the whole lesson of #737's analytics half:
    a redirected read path is not automatically a redirected WRITE path, and only
    looking at the filesystem afterwards tells them apart. Constructing these
    managers creates ``agents/`` and ``credentials.env``, so a hardcoded root
    leaves that litter under ``HOME`` (the scratch ``HOME`` here — never the
    operator's real one, which is what makes this safe to assert on).
    """
    override = tmp_path / "override-config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(override))
    home_root = Path.home() / ".local-operator"
    assert not home_root.exists(), "the scratch HOME must start clean for this assertion"

    monkeypatch.setattr(
        "local_operator.session_factory.create_session", lambda *a, **k: None  # noqa: ARG005
    )

    parsed = exec_worker.build_parser().parse_args(["--prompt", "x"])
    exec_worker._default_session_factory(parsed)

    # Asserted FIRST because it is the actual harm: an isolated exec run laying
    # down credentials and an agent registry in the operator's real home.
    assert not home_root.exists(), (
        f"the worker created {sorted(p.name for p in home_root.rglob('*'))} under the "
        f"home root {home_root} while LOCAL_OPERATOR_CONFIG_DIR pointed at {override}"
    )
    assert override.is_dir(), f"nothing was created under the override {override}"


def test_preflight_resolves_agents_and_config_from_the_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``resolve_hosting_model_dry`` reads the override's agents and config.

    Preflight exists to resolve hosting/model through the EXACT path the worker
    will use, so reading a different root than the worker defeats its purpose:
    with the override set, the hardcoded root validated against the developer's
    real agents and config and then spawned a worker that used the override's.

    Both roots are populated with DIFFERENT, individually valid answers, so the
    returned pair names which root was read — a test that only seeded the
    override would pass on a mutant by failing to find anything at either root
    for the wrong reason. The agent is seeded only in the override, so
    ``--agent-id`` resolving at all is itself evidence of the root used.
    """
    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.config import ConfigManager

    home_root = Path.home() / ".local-operator"
    ConfigManager(home_root).update_config({"hosting": "openai", "model_name": "home-model"})

    override = tmp_path / "override-config"
    ConfigManager(override).update_config({"hosting": "anthropic", "model_name": "override-model"})
    agent = AgentRegistry(override).create_agent(
        AgentEditFields(
            name="guarded",
            security_prompt=None,
            hosting="anthropic",
            model="agent-model",
            description=None,
            last_message=None,
            temperature=None,
            tags=[],
            categories=[],
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
    )
    monkeypatch.setenv(CONFIG_DIR_ENV, str(override))

    # No selector: the answer comes from the config file, so it names the root.
    assert exec_mode.resolve_hosting_model_dry(ExecArgs()) == (
        "anthropic",
        "override-model",
    ), "preflight read the config file at the home root instead of the override"

    # And the registry lookup: this id exists ONLY under the override, so a
    # preflight reading the home root raises "No agent found with ID".
    assert exec_mode.resolve_hosting_model_dry(ExecArgs(agent_id=agent.id)) == (
        "anthropic",
        "agent-model",
    ), "preflight read the agent registry at the home root instead of the override"


def test_background_logs_and_ledger_follow_the_config_dir_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The jobs ledger and the job log live under the override, not the home root.

    ``exec_mode.logs_dir()`` was a module constant hardcoding the home root — the
    fourth copy of the expression #737 fixed at the other three sites in this file.
    It split the jobs ledger away from the root the rest of exec mode honours: an
    isolated ``exec --background`` built its config and agents under the override
    and then wrote its log and its ledger into the operator's real home, which is
    both the isolation defect #737 exists to prevent and litter nothing cleans up.

    Asserted on the FILES THAT APPEAR, not on the resolver's return value: a
    resolver can be correct while a caller keeps a stale copy of the old root, and
    only looking at the filesystem afterwards tells those apart. The whole spawn
    runs for real (``Popen`` alone is faked), so this covers the log path, the
    0700 directory and the ledger write in one pass.
    """
    override = tmp_path / "override-config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(override))
    home_logs = Path.home() / ".local-operator" / "logs"
    monkeypatch.setattr(exec_mode, "resolve_hosting_model_dry", lambda args: ("test", "m"))

    popen_mock = MagicMock()
    popen_mock.return_value.pid = 5150
    monkeypatch.setattr("local_operator.exec_mode.subprocess.Popen", popen_mock)

    assert exec_mode.run_exec("isolated background task", ExecArgs(background=True)) == 0

    # Asserted FIRST because it is the actual harm: a sandboxed run leaving its
    # log and its job ledger in the operator's real home.
    assert not home_logs.exists(), (
        f"the spawn wrote {sorted(p.name for p in home_logs.rglob('*'))} into the home "
        f"root {home_logs} while LOCAL_OPERATOR_CONFIG_DIR pointed at {override}"
    )

    logs_root = override / "logs"
    assert [
        p.name for p in logs_root.glob("exec-*.log")
    ], f"no job log under the override's logs root {logs_root}"
    assert (
        logs_root / exec_mode.JOBS_FILE
    ).exists(), f"the jobs ledger did not land under the override's logs root {logs_root}"
    # The ledger is READ back through the same resolver, so a reader left on the
    # old root would find nothing even though the write succeeded.
    assert [r["pid"] for r in exec_mode.read_job_records()] == [5150]


def test_worker_exit_record_follows_the_config_dir_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``update_job_exit`` writes the terminal record under the override too.

    A separate entry point from the spawn: the detached worker calls this in its
    own process at exit, inheriting the spawner's environment. With the root
    hardcoded, the spawn record and the terminal record could land in two
    different ledgers — the job would look permanently unfinished to any reader.
    """
    override = tmp_path / "override-config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(override))
    home_logs = Path.home() / ".local-operator" / "logs"

    monkeypatch.setattr(sys, "argv", ["exec_worker", "--prompt", "x", "--job-id", "jobx"])
    monkeypatch.setattr(exec_worker, "run", lambda _p, session_factory=None: 0)

    assert exec_worker.main() == 0

    assert not home_logs.exists(), (
        f"the worker wrote its terminal record into the home root {home_logs} "
        f"while LOCAL_OPERATOR_CONFIG_DIR pointed at {override}"
    )
    assert (override / "logs" / exec_mode.JOBS_FILE).exists()
    records = exec_mode.read_job_records()
    assert any(r["id"] == "jobx" and r["exit_code"] == 0 for r in records)
