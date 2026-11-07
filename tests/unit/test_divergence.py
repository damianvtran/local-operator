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
    MAX_NOTICED_ROOTS,
    MAX_ROWS_PER_ROOT,
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
    AgentMessage,
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
    ToolExecutionEndEvent,
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
        messages=[
            Message.user("build the software"),
            Message(role="tool", content=[TextContent(text="exit code: 0")]),
        ],
        system_blocks=["You are a personal assistant."],
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


def test_a_row_claims_only_the_sources_the_judge_was_given(home: Path, tmp_path: Path) -> None:
    """A fixed source list claims a check that may never have happened.

    The shipped version wrote all four on every row, including rows where the
    caller had passed no history and no system prompt — the same class of false
    statement as asserting absence over a carrier the judge never read.
    """
    judge = _judge(home, tmp_path)
    judge.observe_turn(turn_index=1, tool="write", arguments={"path": "~/squeeze/x"})
    rows = _rows(tmp_path)
    assert rows and rows[0]["sources"] == []
    assert rows[0]["unseen"] == []


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
    assert "No audit row was written" in divergence_notice(events[0].as_notice_payload())[0]


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
        if isinstance(event, ToolExecutionEndEvent):
            results.append(event.result)

    # 1. OBSERVATION ONLY: the call ran exactly as it would have without a judge.
    assert executed == ["~/squeeze/index.html"]
    assert results and not results[0].is_error
    # 2. And the event was written where the operator can find it.
    rows = _rows(tmp_path)
    assert len(rows) == 1 and rows[0]["root"] == "~/squeeze"
    # 3. The visible half is the host callback's payload, and it says it ran.
    assert notices and "observation only" in divergence_notice(notices[0])[0]


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
    text, kind = divergence_notice(
        {
            "root": "~/squeeze",
            "path": "/Users/damian/squeeze/src/app.py",
            "tool": "bash",
            "turn": 12,
            "escalated": True,
            "reason": "write",
            "first": False,
            "sources": [
                "system_prompt",
                "custom_instructions",
                "user_messages",
                "earlier_tool_output",
            ],
            "recorded": True,
        }
    )
    assert "~/squeeze" in text
    # D11: the reassurance precedes the path, so a long path cannot push it to
    # the last cells of the row (D2's position). D12: no em dash before it to
    # wrap orphaned at the start of a line.
    assert "(ran; observation only)" in text
    assert text.index("(ran; observation only)") < 90, "the reassurance must not be the tail"
    assert text.index("(ran; observation only)") < text.index("/Users/damian/squeeze")
    assert "—" not in text
    assert "WRITE" not in text and "AGAIN" not in text
    assert "`" not in text
    # An escalation row omits the provenance clause: the first row said it (D6).
    assert "Not in " not in text
    assert kind == "warning"


def test_a_first_sighting_is_a_note_and_says_what_was_checked() -> None:
    """D1: a first sighting and an escalation must not paint the same row."""
    text, kind = divergence_notice(
        {
            "root": "~/squeeze",
            "path": "/Users/damian/squeeze/src/app.py",
            "tool": "read",
            "first": True,
            "escalated": False,
            "sources": ["system_prompt", "user_messages"],
        }
    )
    assert kind == "note"
    assert text.startswith("~/squeeze is new to this session")
    assert "Not in the system prompt or user messages" in text
    assert "earlier tool output" not in text, "the clause names only what was read"


def test_an_escalation_row_says_it_came_back_without_a_turn_number() -> None:
    """D5: "turn 41" pointed at nothing the transcript paints."""
    text, kind = divergence_notice(
        {"root": "~/squeeze", "tool": "read", "turn": 41, "escalated": True, "reason": "repeat"}
    )
    assert "named again, in a later turn" in text.lower() or "again, in a later turn" in text
    assert "turn 41" not in text
    assert kind == "warning"


def test_an_unreadable_carrier_is_named_and_absence_is_not_claimed_over_it() -> None:
    text, _kind = divergence_notice(
        {
            "root": "~/squeeze",
            "tool": "write",
            "first": True,
            "escalated": False,
            "sources": ["user_messages"],
            "unseen": ["image_attachments", "masked_text"],
        }
    )
    # D9: one sentence, and it names both halves — what was read, then what
    # could not be. The hedge is not a trailing clause of its own any more.
    assert (
        "Not in user messages; image attachments and text masked before storage "
        "could not be checked"
    ) in text
    assert "is new to this session" not in text, "the headline must weaken too"


def test_the_headline_weakens_when_nothing_could_be_checked() -> None:
    """D9: an absolute headline backed by zero checked sources is a false claim."""
    text, _kind = divergence_notice(
        {"root": "~/squeeze", "tool": "write", "first": True, "escalated": False, "sources": []}
    )
    assert "is new to this session" not in text
    assert "is not named in what this session could check" in text


def test_an_escalation_row_does_not_repeat_the_unseen_hedge() -> None:
    """D10: the hedge is said once per root, in ``note`` ink, not on every row."""
    text, kind = divergence_notice(
        {
            "root": "~/squeeze",
            "path": "/Users/damian/squeeze/app.py",
            "tool": "write",
            "first": False,
            "escalated": True,
            "reason": "write",
            "sources": ["user_messages"],
            "unseen": ["compacted_history"],
        }
    )
    assert "could not be checked" not in text
    assert "Could not check" not in text, "the standalone hedge form is gone too"
    assert kind == "warning"


def test_a_lost_audit_row_is_disclosed_first_and_keeps_the_warning_tier() -> None:
    """D4: the only variant reporting a real failure must not look the same."""
    text, kind = divergence_notice({"root": "~/x", "recorded": False, "escalated": False})
    assert text.startswith("No audit row was written")
    assert kind == "warning", "a lost record is a warning even on a first sighting"


def test_the_collapse_row_says_the_rest_is_in_the_audit_log() -> None:
    text, kind = divergence_notice({"collapsed": True, "roots": 5, "root": "~/z"})
    assert "audit log only" in text and "~/z" in text
    assert kind == "note"


def test_the_collapse_row_never_claims_the_audit_log_when_the_write_failed() -> None:
    """D8: a failed write must not be reported as success — on ANY row.

    The collapse row sits directly beneath the per-root rows that say no audit
    row was written. Saying "further ones are in the audit log only" there
    claims durability on the one screen where there is none, and every root past
    the cap is silent by design, so this row is the only thing that can say so.
    """
    text, kind = divergence_notice(
        {
            "collapsed": True,
            "roots": 5,
            "root": "~/billing-api",
            "recorded": False,
            "unseen": ["compacted_history"],
        }
    )
    assert text.startswith("No audit row was written")
    assert "audit log only" not in text
    assert "not being recorded anywhere" in text
    assert "turns compacted away could not be checked" in text
    assert kind == "warning"


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


# ---------------------------------------------------------------------------
# The live boundary: a `str` cwd, no fixture skip list (QA round 1, case 2)
# ---------------------------------------------------------------------------


def test_the_skip_list_accepts_the_string_a_tool_context_carries(home: Path) -> None:
    """``ToolContext.cwd`` is declared ``str``; ``_skip_roots`` must take it.

    The first shipped version called ``cwd.resolve()`` and raised
    ``AttributeError`` out of the observer, which the loop swallowed at DEBUG:
    the guard wrote nothing in any live session while the ``Path``-passing
    replay evidence stayed green.
    """
    assert _skip_roots(str(home), home)


@pytest.mark.asyncio
async def test_a_live_loop_with_a_string_cwd_writes_the_row(home: Path, tmp_path: Path) -> None:
    """The end-to-end regression: production's judge, production's cwd type.

    No ``skip_roots`` override (production derives the shipped list), a real
    ``ToolContext(cwd=str(...))``, and the real ``AgentLoop``/tool-result path —
    the exact combination the round-1 live test missed by passing
    ``skip_roots=()``. The asserted path is outside every skip root and has to
    fire, or the guard is inert again.
    """
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
        stream_fn=_OneCallStream("write", '{"path":"/opt/acme/index.html","content":"<html>"}'),
        divergence_judge=judge,
    )

    results = []
    async for event in AgentLoop().run([], context, config, None):
        if isinstance(event, ToolExecutionEndEvent):
            results.append(event.result)

    assert executed == ["/opt/acme/index.html"], "the flagged call still ran"
    assert results and not results[0].is_error
    rows = _rows(tmp_path)
    assert len(rows) == 1, "a live session must write the audit row"
    assert rows[0]["root"] == "/opt/acme"
    assert notices and notices[0]["root"] == "/opt/acme"


# ---------------------------------------------------------------------------
# MAJOR-1: instructions arrive as CustomMessage/user turns, not only as messages
# ---------------------------------------------------------------------------


def test_a_wake_prompt_naming_the_root_explains_it(home: Path, tmp_path: Path) -> None:
    """The round-1 reproducer: the sender named the root, so it is not novel.

    ``CustomMessage`` carries no ``role``, so folding only ``role == "user"``
    missed every delivered instruction and then asserted the root "appears in
    no … user message" — false absence, the class this guard must never commit.
    """
    from local_operator.harness.types import CustomMessage

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=2,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("review the tunnels MR"),
            CustomMessage(
                custom_type="wake_prompt",
                attribution="system",
                details={"text": "pick this up and keep the work inside ~/squeeze"},
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert events == []


def test_a_session_state_delta_naming_the_root_explains_it(home: Path, tmp_path: Path) -> None:
    """Second carrier: ``append_only_state`` providers ship later state as a delta.

    For every subagent the system blocks are the frozen prefix, so a goal/team
    section added after turn 1 reaches the model as a ``session_state`` user turn
    and nowhere else.
    """
    from local_operator.harness.types import CustomMessage

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=3,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("review the tunnels MR"),
            CustomMessage(
                custom_type="session_state",
                attribution="system",
                details={"text": "[session-state]\n## Goal\nship ~/squeeze this week"},
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert events == []


#: Details every arm of the renderer can read. Deliberately a superset of the
#: shapes it composes from: the point of the sweep below is that NO arm has to
#: be known to the test for its carrier to be found.
_PROBE_DETAILS = {
    "text": "delivered by probe",
    "summary": "delivered by probe",
    "tool": "bash",
    "description": "write to ~/probe/app.py",
    "waited_s": 60,
    "kind": "approval",
    "status": "answered",
    "answers": {},
}


def _declared_custom_types() -> dict[str, str]:
    """Every custom type the PACKAGE declares, as ``{value: declaring file}``.

    Derived from source, never from the folded set: a module-level
    ``*_MESSAGE_TYPE`` / ``*_CUSTOM_TYPE`` assignment with a literal value. That
    is how ``GATE_TIMEOUT_CUSTOM_TYPE`` and every other producer constant enters
    the universe without this test keeping a list of modules — the enumeration
    the folded set provably could not do for itself.
    """
    import ast

    import local_operator

    root = Path(local_operator.__file__).parent
    found: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                continue
            for target in node.targets:
                name = getattr(target, "id", "")
                if name.endswith(("_MESSAGE_TYPE", "_CUSTOM_TYPE")):
                    found.setdefault(value.value, str(path.relative_to(root)))
    return found


def _renderer_compared_types() -> set[str]:
    """Every custom-type value ``harness/render.py`` compares ``custom_type`` to.

    The renderer's OWN source is the authority for what it can inject: each
    ``message.custom_type == X`` / ``in (…)`` comparator is read out of the AST
    and resolved through the module namespace, so an imported constant, an
    inline literal and a family frozenset all arrive as the strings it matches.
    """
    import ast

    from local_operator.harness import render

    tree = ast.parse(Path(render.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()

    def resolve(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
        elif isinstance(node, ast.Name):
            value = getattr(render, node.id, None)
            if isinstance(value, str):
                found.add(value)
            elif isinstance(value, (frozenset, set, tuple, list)):
                found.update(item for item in value if isinstance(item, str))
        elif isinstance(node, (ast.Tuple, ast.Set, ast.List)):
            for element in node.elts:
                resolve(element)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "custom_type"):
            continue
        for comparator in node.comparators:
            resolve(comparator)
    return found


def _renders_to_the_model(custom_type: str) -> bool:
    """Does the REAL renderer emit this carried type as a user turn?"""
    from local_operator.harness.render import _default_convert_to_llm
    from local_operator.harness.types import CustomMessage

    rendered = _default_convert_to_llm(
        [CustomMessage(custom_type=custom_type, attribution="system", details=dict(_PROBE_DETAILS))]
    )
    return any(message.role == "user" for message in rendered)


def test_the_folded_set_is_exactly_what_the_renderer_delivers() -> None:
    """The carrier set is DERIVED from the renderer, in both directions.

    The round-2 review's systemic half: the previous version of this test
    iterated ``MODEL_VISIBLE_INJECTED_TYPES`` and asserted each member renders
    (set → renderer), with two hardcoded names for the negatives. It was
    therefore blind to a renderer arm the set never learned about — which is
    exactly how ``gate_timed_out_unattended`` reached the model unfolded while
    the notice claimed the path was in no user message.

    Here the set is never its own witness. The universe is every custom type
    the package DECLARES plus every value ``render.py`` compares ``custom_type``
    against; each is driven through the real ``_default_convert_to_llm``; and
    the types it emits as a user turn must equal the folded set exactly. Add a
    carrier to the renderer without folding it and this fails on the name.
    """
    from local_operator.harness.message_types import MODEL_VISIBLE_INJECTED_TYPES

    universe = set(_declared_custom_types()) | _renderer_compared_types()
    carried = {name for name in universe if _renders_to_the_model(name)}

    unfolded = sorted(carried - MODEL_VISIBLE_INJECTED_TYPES)
    assert not unfolded, f"the renderer delivers these to the model unfolded: {unfolded}"
    # The other direction: a member nothing declares or compares is a claim the
    # renderer never backed, and it would make the corpus over-inclusive.
    unbacked = sorted(MODEL_VISIBLE_INJECTED_TYPES - universe)
    assert not unbacked, f"folded types no constant or renderer comparison declares: {unbacked}"
    assert carried == set(MODEL_VISIBLE_INJECTED_TYPES)


def test_a_gate_expiry_that_names_the_root_explains_it(home: Path, tmp_path: Path) -> None:
    """Round-2 MAJOR: this carrier's path is COMPOSED, and it still explains.

    The reviewer's reproducer, end to end: a parked gate expired unattended while
    the history was being built, its row is in the history the judge is handed,
    and the row's ``description`` carries the resolved path the model was told.
    Before the fix this fired an event with ``unseen: ()`` and told the operator
    the root appeared in no user message — false absence on a carrier sitting in
    the very list passed in.
    """
    from local_operator.harness.types import CustomMessage

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=2,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("build the thing at ~/desk"),
            CustomMessage(
                custom_type="gate_timed_out_unattended",
                attribution="system",
                details={
                    "tool": "write",
                    "description": "write to ~/squeeze/app.py",
                    "kind": "approval",
                    "waited_s": 86400,
                },
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert events == [], "a gate row the model was told about must explain the root"


def test_a_gate_row_about_another_root_leaves_the_claim_true(
    home: Path, tmp_path: Path
) -> None:
    """The same carrier, NOT naming the root: the notice must still be honest.

    Folding the type must not become a blanket excuse. When the gate row is
    about somewhere else there is nothing to explain, the row fires, and it may
    keep claiming the user messages were read — because they were.
    """
    from local_operator.harness.types import CustomMessage

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=2,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("build the thing at ~/desk"),
            CustomMessage(
                custom_type="gate_timed_out_unattended",
                attribution="system",
                details={"tool": "write", "description": "write to ~/desk/notes.md"},
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1
    assert events[0].unseen == ()
    assert "user_messages" in events[0].sources
    text, _kind = divergence_notice(events[0].as_notice_payload())
    assert "Not in the system prompt" in text
    assert "could not be checked" not in text


def test_unreadable_carriers_are_named_on_the_row(home: Path, tmp_path: Path) -> None:
    """Masked storage, images and a compaction cut are named, never assumed clean.

    Each is a carrier the model received that this judge cannot read; the record
    says which were present so the absence claim is scoped (QA round 1, case 6
    measured the masked-storage one live).
    """
    from local_operator.harness.types import CustomMessage, ImageContent

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=4,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("build the thing"),
            CustomMessage(
                custom_type="compaction_summary",
                attribution="system",
                details={"text": "earlier turns summarised"},
            ),
            Message(
                role="tool",
                content=[TextContent(text=f"the value was {'[re' + 'dacted]'}")],
            ),
            Message(
                role="user",
                content=[TextContent(text="see the screenshot"), ImageContent(data="aGk=")],
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1
    assert events[0].unseen == ("masked_text", "image_attachments", "compacted_history")
    text, _kind = divergence_notice(events[0].as_notice_payload())
    # D9/D10: one sentence on the first sighting — the sources this judge read,
    # then the carriers it could not — instead of a trailing "Could not check:"
    # clause in the row's last cells.
    assert (
        "earlier tool output; text masked before storage, image attachments and turns "
        "compacted away could not be checked"
    ) in text


def test_a_compaction_summary_is_a_carrier_the_model_wrote(home: Path, tmp_path: Path) -> None:
    """MINOR-1: model-authored prose must not authorise the model's own path.

    ``compaction_summary`` is a model's prose (``compaction/marker.py`` says so),
    and it reaches the model as a user turn, so it is in the carrier set. Folding
    it as an INSTRUCTION source would let a model that invented a root explain
    its own next assertion with its own explanation — the laundering this module
    forbids. It is disclosed as ``compacted_history`` instead, so the operator
    learns the turns could not be checked rather than hearing nothing.
    """
    from local_operator.harness.types import CustomMessage

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=4,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("review the tunnels MR"),
            CustomMessage(
                custom_type="compaction_summary",
                attribution="system",
                details={"summary": "the operator asked for ~/squeeze"},
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1, "the model's own summary must not silence the notice"
    assert "compacted_history" in events[0].unseen
    text, _kind = divergence_notice(events[0].as_notice_payload())
    assert "turns compacted away could not be checked" in text


def test_a_replaced_history_does_not_claim_sources_it_no_longer_holds(
    home: Path, tmp_path: Path
) -> None:
    """MINOR-2: ``sources`` describes the CURRENT corpus, not the session's past.

    Compaction hands the judge a different list object. ``_reset_corpus`` used to
    keep the accumulated ``sources``, so a row claimed ``earlier_tool_output``
    was checked while the replacement held no tool text at all — a claim the
    record cannot support.
    """
    judge = _judge(home, tmp_path)
    first_history: list[AgentMessage] = [
        Message.user("hello"),
        Message(role="tool", content=[TextContent(text="a listing from ~/other")]),
    ]
    judge.observe_turn(
        turn_index=1,
        tool="read",
        arguments={"path": "~/novel/a"},
        messages=first_history,
        system_blocks=["You are a personal assistant."],
    )
    # A NEW list object is a replaced history (a compaction, a resume).
    compacted: list[AgentMessage] = [Message.user("hello, compacted")]
    events = judge.observe_turn(
        turn_index=9,
        tool="write",
        arguments={"path": "~/novel/b"},
        messages=compacted,
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1
    assert "earlier_tool_output" not in events[0].sources
    assert "user_messages" in events[0].sources
    text, _kind = divergence_notice(events[0].as_notice_payload())
    assert "earlier tool output" not in text


# ---------------------------------------------------------------------------
# MAJOR-2: the per-call cost of a repeated mention
# ---------------------------------------------------------------------------


def test_a_repeated_miss_is_bounded_by_the_announced_root_short_circuit(
    home: Path, tmp_path: Path
) -> None:
    """The burst is bounded by the CALLER, not by a per-root cursor.

    Round 2 measured one ``_is_explained`` call across five turns: once a root
    is registered, ``_observe_turn`` skips the search entirely, and once it is
    explained it is memoised. A per-root cursor of "how far I searched" was
    therefore unreachable state that read as load-bearing, so it is gone; this
    pins what actually bounds the cost.
    """
    judge = _judge(home, tmp_path)
    history: list[AgentMessage] = [Message.user("hello")]
    judge.observe_turn(
        turn_index=1, tool="read", arguments={"path": "~/novel/a"}, messages=history
    )
    calls = 0
    original = judge._is_explained

    def counted(root: str) -> bool:
        nonlocal calls
        calls += 1
        return original(root)

    judge._is_explained = counted  # type: ignore[method-assign]
    for turn in range(2, 6):
        history = [*history, Message(role="tool", content=[TextContent(text=f"out {turn}")])]
        judge.observe_turn(
            turn_index=turn,
            tool="read",
            arguments={"path": f"~/novel/f{turn}"},
            messages=history,
        )
    assert calls == 0, "a root this session announced is never re-searched"


def test_an_announced_root_is_not_rescanned_on_every_mention(home: Path, tmp_path: Path) -> None:
    """117 mentions in one session must cost the corpus scan once, not 117 times."""
    judge = _judge(home, tmp_path)
    history = [Message.user("hello")]
    judge.observe_turn(
        turn_index=1, tool="read", arguments={"path": "~/squeeze/a"}, messages=history
    )
    seen = len(judge._events)
    assert seen == 1
    for turn in range(2, 8):
        history = [*history, Message(role="tool", content=[TextContent(text=f"out {turn}")])]
        judge.observe_turn(
            turn_index=turn,
            tool="write",
            arguments={"path": f"~/squeeze/f{turn}"},
            messages=history,
        )
    # Announced once, escalated once (the first write), then silent: the
    # once-per-root decision holds even though every call re-derived candidates
    # from a growing history — and none of those calls re-scanned the corpus,
    # because a root already in ``_roots`` never reaches ``_is_explained``.
    assert len(judge._events) - seen == 1
    assert judge._roots["~/squeeze"].rows == MAX_ROWS_PER_ROOT

# ---------------------------------------------------------------------------
# The turn index under compaction (code review NIT-1)
# ---------------------------------------------------------------------------


def test_the_turn_index_stays_monotonic_when_compaction_shortens_the_history(
    home: Path, tmp_path: Path
) -> None:
    judge = _judge(home, tmp_path)
    judge.observe_turn(
        turn_index=9,
        tool="read",
        arguments={"path": "~/rootx/a"},
        messages=[Message.user("first")],
    )
    # Compaction replaced the history the loop counts assistant messages in, so
    # the loop's next index is LOWER. The escalation must still be an escalation.
    events = judge.observe_turn(
        turn_index=2,
        tool="read",
        arguments={"path": "~/rootx/b"},
        messages=[Message.user("summarised")],
    )
    assert len(events) == 1
    assert events[0].escalated is True
    assert events[0].reason == "repeat"
    assert events[0].turn_index > 9


# ---------------------------------------------------------------------------
# The transcript's cap on distinct roots (design round 1, D6)
# ---------------------------------------------------------------------------


def test_the_notice_stream_is_capped_and_the_audit_is_not(home: Path, tmp_path: Path) -> None:
    notices: list[dict[str, Any]] = []
    judge = _judge(home, tmp_path, on_notice=notices.append)
    for root in ("r1", "r2", "r3", "r4", "r5"):
        for write in (False, True):
            judge.observe_turn(
                turn_index=1 + write,
                tool="write" if write else "read",
                arguments={"path": f"~/{root}/f{int(write)}"},
                messages=[Message.user("ordinary brief")],
            )
    assert len(_rows(tmp_path)) == 10, "every root still gets its audit rows"
    assert notices[-1].get("collapsed") is True
    painted_roots = {notice.get("root") for notice in notices if not notice.get("collapsed")}
    assert len(painted_roots) == MAX_NOTICED_ROOTS, "the transcript stops at the cap"
    assert painted_roots == {"~/r1", "~/r2", "~/r3"}
    assert "~/r5" not in json.dumps(notices[:-1])


# ---------------------------------------------------------------------------
# The two acceptance numbers, now asserted (code review MINOR-3)
# ---------------------------------------------------------------------------

#: The ceiling the module's own docstring states (~4.8 flags per 100 turns on the
#: 20-session ordinary sample). The rate below is measured against committed
#: fixtures, so a precision regression fails CI instead of shipping green.
MAX_ROWS_PER_100_TURNS = 4.8

#: Ordinary work: the operator names the project, tools read inside it, the model
#: acts inside it. This is the shape 76% of the baseline's literal flags were
#: made of — the model doing the task, not diverging.
_ORDINARY_SESSIONS: tuple[tuple[str, list[tuple[str, dict[str, Any]]], list[str]], ...] = (
    (
        "Review PR 3533 in ~/local-operator and report what changed.",
        [("read", {"path": "~/local-operator/local_operator/divergence.py"})],
        ["diff --git a/local_operator/divergence.py b/local_operator/divergence.py"],
    ),
    (
        "The build is red in ~/oss/oh-my-pi — fix it.",
        [("bash", {"command": "cd ~/oss/oh-my-pi && make build"})],
        ["src/mobile/app.ts written"],
    ),
    (
        "Check the deployment record in ~/tools/omp-mobile.",
        [("read", {"path": "~/tools/omp-mobile/docs/DEPLOYMENT.md"})],
        ["docs/DEPLOYMENT.md: 12 lines"],
    ),
    (
        "Run the core suite in ~/minervaai.",
        [("bash", {"command": "cd ~/minervaai/core && pytest -x -q"})],
        ["412 passed"],
    ),
)


def test_ordinary_work_stays_under_the_false_positive_budget(home: Path, tmp_path: Path) -> None:
    judge = _judge(home, tmp_path)
    turns = 0
    for prompt, calls, outputs in _ORDINARY_SESSIONS:
        history: list[Any] = [Message.user(prompt)]
        for index, (tool, arguments) in enumerate(calls):
            turns += 1
            judge.observe_turn(
                turn_index=turns,
                tool=tool,
                arguments=arguments,
                messages=list(history),
                system_blocks=["You are a personal assistant."],
            )
            history = [
                *history,
                Message(role="tool", content=[TextContent(text=outputs[index % len(outputs)])]),
            ]
    rows = _rows(tmp_path)
    rate = len(rows) * 100 / max(turns, 1)
    assert rate <= MAX_ROWS_PER_100_TURNS, f"{rate:.2f} rows/100 turns over {turns} turns: {rows}"


def test_the_incident_control_still_fires(home: Path, tmp_path: Path) -> None:
    """The positive control: a foreign project root appears from nowhere.

    ``73084269786c``'s shape — one QA brief, tool output about the tunnel work,
    then a turn that builds ``~/squeeze`` — replayed through the shipped judge.
    The real transcript is gone from disk, so the control is a committed
    fixture of that shape rather than the session itself.
    """
    judge = _judge(home, tmp_path)
    history: list[Any] = [
        Message.user("Run an ordinary QA review of the tunnels MR and report findings."),
        Message(role="tool", content=[TextContent(text="tests/unit/test_tunnels.py 29 passed")]),
    ]
    judge.observe_turn(
        turn_index=1,
        tool="read",
        arguments={"path": "~/local-operator/tests/unit/test_tunnels.py"},
        messages=history,
        system_blocks=["You are a personal assistant."],
    )
    history = [*history, Message(role="tool", content=[TextContent(text="read 29 lines")])]
    events = judge.observe_turn(
        turn_index=2,
        tool="bash",
        arguments={"command": "mkdir -p ~/squeeze/src/squeeze/{engine,service}"},
        messages=history,
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1, "the control stopped firing"
    assert events[0].root == "~/squeeze"
    assert events[0].escalated is True
    assert events[0].reason == "write"
    assert events[0].turn_index == 2
    # The claim it makes is about the sources it read, and the notice says it ran.
    assert "user_messages" in events[0].sources
    text, kind = divergence_notice(events[0].as_notice_payload())
    assert "(ran; observation only)" in text and kind == "warning"


@pytest.mark.asyncio
async def test_a_real_session_writes_the_row_and_the_call_still_runs(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1 case 2, on the SHIPPED construction path.

    The unit-level live test above proves the loop boundary; this proves the
    seam QA actually failed on: a real ``Session`` builds the judge itself
    (``session/session.py``, no ``skip_roots``), derives the skip list from the
    ``str`` cwd its own ``ToolContext`` carries, and writes the row to the
    configured audit path. Before the boundary fix this produced ZERO rows and
    returned the tool result unchanged — green evidence, dead feature.

    Env isolation is load-bearing: ``HOME`` and the config dir are redirected
    into the test's own tree so a run can never read or write the operator's
    ``~/.local-operator``.
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    scratch = home.parent
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(scratch / "config"))
    # The analytics mirror is a process-wide singleton with its own writer
    # thread. Point it at this test's tree (never the operator's database) and
    # close it afterwards, so the thread cannot outlive the test.
    recorder = reset_recorder_for_test(AnalyticsStore(scratch / "analytics.db"))

    executed: list[str] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append(str(args.get("path")))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="write",
            content=[TextContent(text="wrote it")],
        )

    tool = AgentTool(
        name="write",
        description="write a file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
        execute=execute,
    )
    session = Session(
        model=MODEL,
        stream_fn=_OneCallStream("write", '{"path":"/opt/acme/index.html","content":"<html>"}'),
        tools=[tool],
        transcript=Transcript(scratch / "sess"),
        system_blocks_provider=lambda: ["You are a personal assistant."],
    )
    try:
        await session.prompt("scaffold a tiny project from scratch")
    finally:
        await session.dispose()
        recorder.close()
        reset_recorder_for_test(AnalyticsStore(scratch / "analytics-2.db")).close()

    # 1. OBSERVATION ONLY: the call ran, with its side effect, unchanged.
    assert executed == ["/opt/acme/index.html"]
    # 2. And the session's own judge wrote the row where the operator finds it.
    audit = scratch / "config" / "audit" / "divergence.jsonl"
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, "a real Session must write the audit row"
    assert rows[0]["root"] == "/opt/acme"
    assert rows[0]["tool"] == "write"
    assert rows[0]["recorded"] is True
