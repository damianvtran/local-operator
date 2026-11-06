"""Divergence guard tests: what a turn ASSERTS, at project-root granularity.

The guard is observation only — the last test in this file drives the real
``AgentLoop`` and asserts the flagged call still RAN, because a detector that
quietly gates would be the failure it exists to prevent
(``~/workspace/incidents/agent-context-contamination-20261106/INCIDENT.md``).

Every test holds a fixture home rather than the operator's, so a run can never
read or write the real ``~/.local-operator`` (the audit path is overridden the
same way).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.divergence import (
    DivergenceJudge,
    _skip_roots,
    asserted_paths,
    origin_parent_session_id,
    project_root,
    root_key,
    shell_asserted_paths,
)
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.rows import divergence_notice
from local_operator.harness.types import (
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)

MODEL = ModelSpec(provider="test", model_id="m")


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    """An isolated ``$HOME``; the judge takes it as a parameter for this reason."""
    root = tmp_path / "home"
    (root / "known-project").mkdir(parents=True)
    return root


def _judge(home: Path, tmp_path: Path, **kwargs: Any) -> DivergenceJudge:
    """A judge over a FIXTURE root set.

    ``skip_roots=()`` by default because a pytest home lives under the OS temp
    root, which is itself a shipped skip root — with the real list every
    assertion here would be skipped for a reason the test is not about. The
    skip tests pass the real list explicitly (``_skip_roots(None, home)``).
    """
    kwargs.setdefault("skip_roots", ())
    return DivergenceJudge(
        session_id="sess0001",
        parent_session_id="parent01",
        home=home,
        audit_path=tmp_path / "audit" / "divergence.jsonl",
        **kwargs,
    )


def _rows(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "audit" / "divergence.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Shell extraction: every creating form the operator named
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("mkdir -p ~/squeeze/src", "squeeze/src"),
        ("mkdir ~/squeeze", "squeeze"),
        ("touch ~/squeeze/index.html", "squeeze/index.html"),
        ("cp report.md ~/squeeze/report.md", "squeeze/report.md"),
        ("mv ~/scratch/x.py ~/squeeze/x.py", "squeeze/x.py"),
        ("echo hello > ~/squeeze/README.md", "squeeze/README.md"),
        ("echo hello >> ~/squeeze/log.txt", "squeeze/log.txt"),
        ("ls | tee ~/squeeze/listing.txt", "squeeze/listing.txt"),
        ("git clone https://example.invalid/x.git ~/squeeze", "squeeze"),
        ("uv venv ~/squeeze/.venv", "squeeze/.venv"),
    ],
)
def test_every_creating_form_yields_its_target(command: str, expected: str, home: Path) -> None:
    assert str(home / expected) in shell_asserted_paths(command, cwd=home, home=home)


def test_copy_creates_only_its_destination(home: Path) -> None:
    """``cp src.txt dst.txt`` reads the source; only the destination is asserted."""
    created = shell_asserted_paths("cp ~/squeeze/src.txt ~/squeeze/dst.txt", cwd=home, home=home)
    assert created == [str(home / "squeeze" / "dst.txt")]


def test_cd_base_resolves_relative_creations(home: Path) -> None:
    """``cd ~/squeeze && mkdir src`` names ``src`` under the directory it entered."""
    created = shell_asserted_paths("cd ~/squeeze && mkdir src", cwd=home, home=home)
    assert created == [str(home / "squeeze" / "src")]


# ---------------------------------------------------------------------------
# Probes are not assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "cd ~/squeeze 2>/dev/null",
        "ls ~/squeeze 2>/dev/null",
        "ls ~/squeeze || true",
        "ls ~/squeeze || :",
        "test -d ~/squeeze 2>/dev/null",
    ],
)
def test_failure_swallowing_segments_are_dropped(command: str, home: Path) -> None:
    assert shell_asserted_paths(command, cwd=home, home=home) == []


def test_a_probe_does_not_silence_the_creation_beside_it(home: Path) -> None:
    """Per SEGMENT, never per command string: the probe and the write differ."""
    created = shell_asserted_paths(
        "cd ~/squeeze 2>/dev/null && mkdir -p ~/squeeze/src", cwd=home, home=home
    )
    assert created == [str(home / "squeeze" / "src")]


def test_non_creating_reads_are_not_shell_targets(home: Path) -> None:
    assert shell_asserted_paths("cat ~/squeeze/x.py | head -3", cwd=home, home=home) == []
    assert shell_asserted_paths("ls -la ~/squeeze", cwd=home, home=home) == []


# ---------------------------------------------------------------------------
# File-tool arguments
# ---------------------------------------------------------------------------


def test_write_and_edit_assert_their_path(home: Path) -> None:
    write = asserted_paths(
        "write", {"path": "~/squeeze/index.html", "content": "x"}, cwd=home, home=home
    )
    edit = asserted_paths(
        "edit", {"path": "~/squeeze/index.html", "edits": []}, cwd=home, home=home
    )
    assert write == [(str(home / "squeeze" / "index.html"), True)]
    assert edit == [(str(home / "squeeze" / "index.html"), True)]


def test_reads_assert_without_writing(home: Path) -> None:
    """A read is a weaker assertion than a write, and the flag says so."""
    read = asserted_paths("read", {"path": "~/squeeze"}, cwd=home, home=home)
    assert read == [(str(home / "squeeze"), False)]


def test_relative_targets_resolve_against_cwd(home: Path) -> None:
    resolved = asserted_paths("write", {"path": "out.txt", "content": ""}, cwd=home, home=home)
    assert resolved == [(str(home / "out.txt"), True)]


def test_browser_only_asserts_for_actions_that_write_a_file(home: Path) -> None:
    shot = asserted_paths(
        "browser", {"action": "screenshot", "path": "~/squeeze/shot.png"}, cwd=home, home=home
    )
    navigation = asserted_paths(
        "browser", {"action": "open", "url": "https://x"}, cwd=home, home=home
    )
    assert shot == [(str(home / "squeeze" / "shot.png"), True)]
    assert navigation == []


def test_virtual_schemes_are_not_filesystem_targets(home: Path) -> None:
    for raw in ("scratchpad://notes.md", "skill://minerva-admin-apis", "guide://delivery"):
        assert asserted_paths("write", {"path": raw}, cwd=home, home=home) == []


def test_shell_virtual_schemes_are_not_targets(home: Path) -> None:
    assert shell_asserted_paths("mkdir scratchpad://x", cwd=home, home=home) == []


# ---------------------------------------------------------------------------
# Root granularity
# ---------------------------------------------------------------------------


def test_project_root_granularity(home: Path) -> None:
    assert project_root(home / "squeeze" / "src" / "app.py", home) == "~/squeeze"
    assert project_root(Path("/Users/damian/squeeze/x.html"), home) == "/Users/damian/squeeze"
    assert project_root(Path("/opt/acme/app/x.py"), home) == "/opt/acme"
    assert root_key("~/squeeze") == "squeeze"
    assert root_key("/Users/damian/squeeze") == "squeeze"


def test_a_file_under_a_named_root_is_explained(home: Path, tmp_path: Path) -> None:
    """The T3→T4 step: naming ``~/known-project`` explains every file inside it."""
    judge = _judge(home, tmp_path)
    system = ["You are working in ~/known-project on the report service."]
    events = judge.observe_turn(
        turn_index=1,
        tool="write",
        arguments={"path": "~/known-project/deep/nested/file.py", "content": ""},
        system_blocks=system,
    )
    assert events == []
    assert _rows(tmp_path) == []


def test_the_root_basename_in_a_user_message_explains_it(home: Path, tmp_path: Path) -> None:
    """Baseline flag #1: the operator wrote prose ("in my Downloads"), not a path."""
    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=1,
        tool="write",
        arguments={"path": "~/Downloads/order.pdf"},
        messages=[Message.user("the agreement I signed is in my Downloads folder")],
    )
    assert events == []


def test_earlier_tool_output_explains_a_root(home: Path, tmp_path: Path) -> None:
    """T5's whole allowance: the model is allowed to use what it just read."""
    judge = _judge(home, tmp_path)
    history = [
        Message.user("start"),
        Message(role="tool", content=[TextContent(text="squeeze\nsqueeze.tar.gz")]),
    ]
    events = judge.observe_turn(
        turn_index=2,
        tool="read",
        arguments={"path": "~/squeeze/squeeze.tar.gz"},
        messages=history,
    )
    assert events == []


def test_the_models_own_assistant_text_does_not_explain_a_root(home: Path, tmp_path: Path) -> None:
    """The laundering case, stated as a test: reasoning may not justify itself."""
    judge = _judge(home, tmp_path)
    history = [
        Message.user("start"),
        Message.assistant("I will build ~/squeeze now."),
    ]
    events = judge.observe_turn(
        turn_index=2,
        tool="read",
        arguments={"path": "~/squeeze"},
        messages=history,
    )
    assert len(events) == 1
    assert events[0].root == "~/squeeze"


# ---------------------------------------------------------------------------
# Once per root, then escalate
# ---------------------------------------------------------------------------


def test_one_row_per_novel_root_within_a_turn(home: Path, tmp_path: Path) -> None:
    judge = _judge(home, tmp_path)
    first = judge.observe_turn(turn_index=1, tool="bash", arguments={"command": "mkdir ~/squeeze"})
    second = judge.observe_turn(
        turn_index=1, tool="bash", arguments={"command": "mkdir ~/squeeze/src"}
    )
    assert len(first) == 1
    assert second == []
    assert len(_rows(tmp_path)) == 1


def test_escalates_on_a_second_turn(home: Path, tmp_path: Path) -> None:
    judge = _judge(home, tmp_path)
    turn1 = judge.observe_turn(turn_index=1, tool="read", arguments={"path": "~/squeeze"})
    turn2 = judge.observe_turn(turn_index=2, tool="read", arguments={"path": "~/squeeze/src"})
    turn3 = judge.observe_turn(
        turn_index=3, tool="read", arguments={"path": "~/squeeze/src/app.py"}
    )
    assert [event.escalated for event in turn1] == [False]
    assert [(event.escalated, event.reason) for event in turn2] == [(True, "repeat")]
    assert turn3 == [], "at most two rows per root: 117 mentions must not flood"
    rows = _rows(tmp_path)
    assert [row["escalated"] for row in rows] == [False, True]


def test_escalates_on_a_write_after_a_read(home: Path, tmp_path: Path) -> None:
    judge = _judge(home, tmp_path)
    judge.observe_turn(turn_index=1, tool="read", arguments={"path": "~/squeeze"})
    wrote = judge.observe_turn(
        turn_index=1, tool="write", arguments={"path": "~/squeeze/index.html", "content": ""}
    )
    assert [(event.escalated, event.reason) for event in wrote] == [(True, "write")]


def test_a_write_on_first_sighting_is_already_escalated(home: Path, tmp_path: Path) -> None:
    """The incident's shape: the first assertion was a creation, not a look."""
    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=1, tool="write", arguments={"path": "~/squeeze/index.html", "content": ""}
    )
    assert [(event.escalated, event.reason) for event in events] == [(True, "write")]
    assert len(_rows(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Skip roots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "~/Library/Caches/com.example/x",
        "~/.cache/uv/wheels",
        "~/.npm/_logs/x.log",
        "~/.local/share/uv/python",
        "/dev/null",
        "~/Library/Application Support/App/x.json",
    ],
)
def test_skipped_roots_never_fire(path: str, home: Path, tmp_path: Path) -> None:
    judge = _judge(home, tmp_path, skip_roots=_skip_roots(None, home))
    assert judge.observe_turn(turn_index=1, tool="write", arguments={"path": path}) == []


def test_the_sessions_own_cwd_root_is_skipped(home: Path, tmp_path: Path) -> None:
    project = home / "known-project"
    judge = _judge(home, tmp_path, skip_roots=_skip_roots(project, home))
    events = judge.observe_turn(
        turn_index=1,
        tool="bash",
        arguments={"command": "mkdir -p build/out"},
        cwd=project,
    )
    assert events == [], "a relative creation under the session's own cwd is ordinary work"


def test_a_session_running_at_home_still_polices_home(home: Path, tmp_path: Path) -> None:
    """THE exception the incident needs: cwd == $HOME must explain NOTHING.

    169 of the 170 desktop sessions that persist a cwd on this machine run at
    ``$HOME``; treating that as "the session's own root" would skip every
    project on the machine, ``~/squeeze`` included.
    """
    # The shipped list itself must not carry home as a skip root here...
    assert home not in {root for root, _why in _skip_roots(home, home)}
    # ...and the predicate must therefore see the creation at cwd == home.
    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=1,
        tool="bash",
        arguments={"command": "mkdir -p ~/squeeze/src"},
        cwd=home,
    )
    assert [event.root for event in events] == ["~/squeeze"]


def test_the_scratchpad_is_skipped(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = tmp_path / "pad"
    pad.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_SCRATCHPAD", str(pad))
    judge = _judge(home, tmp_path, skip_roots=_skip_roots(None, home))
    wrote = judge.observe_turn(turn_index=1, tool="write", arguments={"path": str(pad / "x.md")})
    assert wrote == []


# ---------------------------------------------------------------------------
# The audit row
# ---------------------------------------------------------------------------


def test_the_audit_row_carries_the_operators_fields(home: Path, tmp_path: Path) -> None:
    judge = _judge(home, tmp_path)
    judge.observe_turn(
        turn_index=7,
        tool="bash",
        arguments={"command": "mkdir -p ~/squeeze/src"},
        call_id="call_abc",
    )
    rows = _rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "sess0001"
    assert row["parent"] == "parent01"
    assert row["turn"] == 7
    assert row["root"] == "~/squeeze"
    assert row["tool"] == "bash"
    assert row["call_id"] == "call_abc"
    assert row["escalated"] is True
    assert row["sources"] == [
        "system_prompt",
        "custom_instructions",
        "user_messages",
        "earlier_tool_output",
    ]
    assert row["recorded"] is True


def test_a_failed_write_is_not_reported_as_success(home: Path, tmp_path: Path) -> None:
    """The audit row is the control; losing it silently would be the bypass."""
    auditor = tmp_path / "audit" / "divergence.jsonl"
    auditor.parent.mkdir(parents=True)
    auditor.mkdir()  # a DIRECTORY at the record's path: the open must fail
    judge = DivergenceJudge(
        session_id="sess0001",
        home=home,
        audit_path=auditor,
        on_notice=lambda payload: None,
        skip_roots=(),
    )
    events = judge.observe_turn(turn_index=1, tool="write", arguments={"path": "~/squeeze/x"})
    assert len(events) == 1, "the observation still surfaces; only the record failed"
    assert events[0].recorded is False
    assert "audit row NOT written" in divergence_notice(events[0].as_notice_payload())


def test_origin_parent_session_id_reads_origin_json(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "origin.json").write_text(json.dumps({"origin": "subagent", "parent": "abc123"}))
    assert origin_parent_session_id(child) == "abc123"
    assert origin_parent_session_id(tmp_path / "missing") == ""
    (child / "origin.json").write_text("{not json")
    assert origin_parent_session_id(child) == ""


# ---------------------------------------------------------------------------
# Observation only: the real loop still runs the flagged call
# ---------------------------------------------------------------------------


class _OneCallStream:
    """Stream one tool call, then stop — the smallest real turn."""

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments
        self.calls = 0

    def __call__(self, request: Any, signal: Any):
        self.calls += 1
        first = self.calls == 1
        events: list[StreamEvent] = (
            [
                StreamToolCallDelta(
                    index=0, id="call_div", name=self.name, argument_delta=self.arguments
                ),
                StreamEndEvent(stop_reason="tool_use"),
            ]
            if first
            else [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
        )

        async def gen():
            for event in events:
                yield event

        return gen()


@pytest.mark.asyncio
async def test_a_flagged_call_still_executes_and_the_event_is_written(
    home: Path, tmp_path: Path
) -> None:
    executed: list[str] = []
    notices: list[dict[str, Any]] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append(str(args.get("path")))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="write",
            content=[TextContent(text="wrote it")],
        )

    judge = DivergenceJudge(
        session_id="sess0001",
        home=home,
        audit_path=tmp_path / "audit" / "divergence.jsonl",
        on_notice=notices.append,
        skip_roots=(),
    )
    tool = AgentTool(
        name="write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
        execute=execute,
    )
    context = LoopContext(
        messages=[Message.user("build the software")],
        system_blocks=["You are a personal assistant."],
        tools=[tool],
        tool_context=ToolContext(cwd=str(home)),
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_OneCallStream("write", '{"path":"~/squeeze/index.html","content":"<html>"}'),
        divergence_judge=judge,
    )

    results = []
    async for event in AgentLoop().run([], context, config, None):
        if getattr(event, "result", None) is not None:
            results.append(event.result)

    # 1. OBSERVATION ONLY: the call ran exactly as it would have without a judge.
    assert executed == ["~/squeeze/index.html"]
    assert results and not results[0].is_error
    # 2. And the event was written where the operator can find it.
    rows = _rows(tmp_path)
    assert len(rows) == 1 and rows[0]["root"] == "~/squeeze"
    # 3. The visible half is the host callback's payload, and it says it ran.
    assert notices and "observation only" in divergence_notice(notices[0])


@pytest.mark.asyncio
async def test_a_judge_that_raises_cannot_break_the_turn(home: Path, tmp_path: Path) -> None:
    """An observer that can kill a turn is a gate with a disguise on."""

    class Exploding:
        def observe_turn(self, **kwargs: Any) -> None:
            raise RuntimeError("observer bug")

    executed: list[str] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append("ran")
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="bash", content=[TextContent(text="ok")]
        )

    tool = AgentTool(
        name="bash",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        execute=execute,
    )
    context = LoopContext(
        messages=[Message.user("go")],
        tools=[tool],
        tool_context=ToolContext(cwd=str(home)),
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_OneCallStream("bash", '{"command":"mkdir -p ~/squeeze"}'),
        divergence_judge=Exploding(),
    )

    async for _event in AgentLoop().run([], context, config, None):
        pass
    assert executed == ["ran"]


# ---------------------------------------------------------------------------
# The row: wire value, model-visibility, and what the operator reads
# ---------------------------------------------------------------------------


def test_the_notice_marker_is_a_stable_wire_value() -> None:
    """Persisted rows are matched on this literal by every replaying surface."""
    from local_operator.harness.message_types import DIVERGENCE_NOTICE_MESSAGE_TYPE

    assert DIVERGENCE_NOTICE_MESSAGE_TYPE == "divergence_notice"


def test_the_notice_is_not_injected_into_the_models_context() -> None:
    """The row is for the OPERATOR; the model gets no notice about its own detector.

    Unlisted custom types are dropped by the shared renderer as bookkeeping —
    the same treatment the credential-redaction marker gets, for the same
    reason. A notice injected as a user turn would tell the model something it
    cannot act on and would cost context on every following turn.
    """
    from local_operator.harness.message_types import DIVERGENCE_NOTICE_MESSAGE_TYPE
    from local_operator.harness.render import _default_convert_to_llm
    from local_operator.harness.types import CustomMessage

    history = [
        Message.user("hello"),
        CustomMessage(
            custom_type=DIVERGENCE_NOTICE_MESSAGE_TYPE,
            attribution="system",
            details={"root": "~/squeeze", "tool": "bash", "path": "/Users/x/squeeze"},
        ),
    ]
    rendered = _default_convert_to_llm(history)
    assert [message.role for message in rendered] == ["user"]
    assert all("squeeze" not in json.dumps(message.model_dump()) for message in rendered)


def test_the_notice_names_the_root_the_check_and_that_the_call_ran() -> None:
    text = divergence_notice(
        {
            "root": "~/squeeze",
            "path": "/Users/damian/squeeze/src/app.py",
            "tool": "bash",
            "turn": 12,
            "escalated": True,
            "reason": "write",
            "recorded": True,
        }
    )
    assert "~/squeeze" in text
    assert "WRITE" in text
    assert "no system prompt, custom instruction, user message or earlier tool output" in text
    assert "the call ran (observation only)" in text
    assert "NOT written" not in text


def test_an_escalation_row_says_it_came_back() -> None:
    text = divergence_notice(
        {"root": "~/squeeze", "tool": "read", "turn": 41, "escalated": True, "reason": "repeat"}
    )
    assert "named AGAIN in turn 41" in text


def test_a_lost_audit_row_is_disclosed_in_the_notice() -> None:
    text = divergence_notice({"root": "~/x", "recorded": False, "escalated": False})
    assert "audit row NOT written" in text


def test_heredoc_bodies_contribute_no_targets() -> None:
    """Measured artefact: prose after a ``>`` inside a ``cat > f <<'EOF'`` body."""
    command = (
        "cat > /tmp/note.md <<'EOF'\n"
        "totals were ~/6,371,817,635) and ~/30_000_000: and /=8:\n"
        "EOF\n"
    )
    targets = shell_asserted_paths(command, cwd=None, home=Path("/home/x"))
    # The OPENER's redirect target is a real creation and must survive; the
    # body's prose must contribute nothing.
    assert [Path(target).name for target in targets] == ["note.md"]


def test_the_notice_paints_as_a_warning_block_in_the_tui_fold() -> None:
    """An unbranched custom type paints NOWHERE; this is the branch's proof.

    Driven through the real projection entry point (``project_settled_rows``)
    with the replay state the sidebar uses, so the assertion is about the fold
    that paints a resumed transcript, not about a helper's return value.
    """
    from local_operator.harness.message_types import DIVERGENCE_NOTICE_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage
    from local_operator.tui.session_presentation import (
        NoticeBlock,
        PreparedReplay,
        project_settled_rows,
    )

    replay = PreparedReplay()
    history = [
        Message.user("build the thing"),
        CustomMessage(
            custom_type=DIVERGENCE_NOTICE_MESSAGE_TYPE,
            attribution="system",
            details={
                "root": "~/squeeze",
                "path": "/Users/damian/squeeze/index.html",
                "tool": "write",
                "turn": 12,
                "escalated": True,
                "reason": "write",
                "recorded": True,
            },
        ),
    ]
    assert project_settled_rows(replay, history, fold_width=80) is True
    painted = [block for block in replay.blocks if isinstance(block, NoticeBlock)]
    assert painted, "the divergence row painted nothing"
    assert any("~/squeeze" in block.text() for block in painted)
    # ``warning`` ink, not ``info``: nothing failed, but a task substitution is
    # a state the operator has to look at (``harness/rows.py`` decides the tier
    # so both surfaces agree).
    assert any(getattr(block, "notice_kind", "") == "warning" for block in painted)


@pytest.mark.asyncio
async def test_the_session_writes_the_notice_row_to_the_transcript() -> None:
    """The wiring half: the judge's callback becomes a persisted transcript row.

    Called as an unbound method with a stand-in holding only the two
    collaborators it uses, because the point is the METHOD's behaviour — which
    message type and which details reach the transcript — and building a whole
    ``Session`` here would measure the session, not this row.
    """
    from local_operator.harness.message_types import DIVERGENCE_NOTICE_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage
    from local_operator.session.session import Session

    written: list[Any] = []

    class _Transcript:
        async def append_message(self, message: Any) -> None:
            written.append(message)

    class _StandIn:
        _transcript = _Transcript()

        def _spawn_background(self, coro: Any) -> None:
            import asyncio

            asyncio.ensure_future(coro)

    payload = {"root": "~/squeeze", "tool": "bash", "escalated": True, "recorded": True}
    Session._record_divergence_notice(_StandIn(), payload)  # type: ignore[arg-type]
    for _ in range(10):
        if written:
            break
        await __import__("asyncio").sleep(0)

    assert len(written) == 1
    message = written[0]
    assert isinstance(message, CustomMessage)
    assert message.custom_type == DIVERGENCE_NOTICE_MESSAGE_TYPE
    assert message.details == payload
