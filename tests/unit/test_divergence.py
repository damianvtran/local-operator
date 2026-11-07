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
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.divergence import (
    MAX_NOTICED_ROOTS,
    MAX_ROWS_PER_ROOT,
    MODEL_AUTHORED_CARRIERS,
    SELF_ECHO_RECEIPT_TOOLS,
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
from local_operator.harness.wake_types import WakeOrigin
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


def test_a_wake_prompt_is_the_models_own_text_and_does_not_explain_the_root(
    home: Path,
    tmp_path: Path,
) -> None:
    """Round-3 MAJOR: a wake message the MODEL scheduled cannot explain a root.

    ``details["text"]`` ends with the scheduling model's own ``WakeParams.message``,
    verbatim (``harness/wake.py`` -> ``tools/builtin.py``). Round 1 read this
    carrier as the operator's words and asserted it silenced the notice; under
    the rule the module now states — text THIS session's model can author is
    never an instruction source — the notice must FIRE here. It is a missed
    notice if it does not: the reviewer reproduced ``events: 0`` on this exact
    shape, and a session can wake itself into a project it invented.

    ROUND 4: the carrier is split by AUTHOR, so this is the ``model`` arm —
    stamped, not inferred, exactly as the delivery stamps it. The
    operator-authored arm is the test below, and it must fold.
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
                attribution="user",
                details={
                    "text": "pick this up and keep the work inside ~/squeeze",
                    "wake_author": "model",
                },
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1, "the model's own wake message must not silence the notice"
    assert "user_messages" in events[0].sources
    text, _kind = divergence_notice(events[0].as_notice_payload())
    assert "Not in the system prompt, custom instructions or user messages" in text


def test_the_operators_own_scheduled_wake_explains_the_root(
    home: Path,
    tmp_path: Path,
) -> None:
    """Round-4 MAJOR: the ONE carrier with two authors is split by author.

    ``format_wake_delivery_text`` renders the schedule's message verbatim and no
    user turn is ever synthesised for a wake (``desktop_wakes.py``: a fake turn
    would be replayed as something the user said), so for a wake the OPERATOR
    scheduled the delivery is the only copy of their instruction. Excluding the
    carrier by name told them that instruction was absent from the session —
    QA round 4 reproduced it twice on the CLI and desktop writers. Stamped
    ``operator``, it folds.
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
                attribution="user",
                details={
                    "text": "(alarm) Scheduled wake w1 (1/1).\n\nwork on the app under ~/squeeze",
                    "wake_author": "operator",
                },
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert events == [], "the operator's own scheduled instruction named the root"


def test_an_unattributed_wake_is_not_folded_and_not_claimed_as_checked(
    home: Path,
    tmp_path: Path,
) -> None:
    """The honesty arm: attribution genuinely unavailable must say so.

    A stored delivery from before ``WakeSchedule.origin`` existed cannot be
    attributed from here. It is not folded (a model must never explain a root
    with its own wake text) but the row must not claim the carrier was read
    either — ``unseen`` names it, and ``harness/rows.py`` renders the hedged
    sentence for exactly that. A hedged notice is acceptable; a false one is not.
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
                attribution="user",
                details={"text": "legacy delivery\n\nwork in ~/squeeze"},
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1
    assert "unattributed_wake" in events[0].unseen
    text, _kind = divergence_notice(events[0].as_notice_payload())
    assert "is not named in what this session could check" in text
    assert "wake deliveries with no recorded author" in text


@pytest.mark.parametrize("tool_name", sorted(SELF_ECHO_RECEIPT_TOOLS))
def test_a_tool_receipt_that_echoes_the_models_own_arguments_is_not_a_source(
    tool_name: str, home: Path, tmp_path: Path
) -> None:
    """QA4-1: the creating tool's receipt is not a second author.

    Measured on the real path in QA round 4: the exclusion of the carrier
    changed nothing for a model-created wake or todo because the receipt had
    already put the root in the corpus as ``earlier_tool_output``. These four
    tools render the model's own arguments back to it, so folding them would let
    a session wake itself into a project it invented. The control below pins the
    other half: a tool result that carries the ENVIRONMENT's answer still folds.
    """
    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=2,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("review the tunnels MR"),
            Message(
                role="tool",
                tool_name=tool_name,
                content=[TextContent(text=f"{tool_name} receipt: write ~/squeeze/app.py")],
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1, f"the {tool_name} receipt only echoes the model's own arguments"


def test_a_tool_result_that_carries_the_environments_answer_still_explains(
    home: Path, tmp_path: Path
) -> None:
    """The control for QA4-1: only the SELF-ECHOING tools lose their voice.

    A ``bash`` result is the environment's answer — the 48% of the baseline this
    arm exists for — and still explains a root. Excluding tool results wholesale
    would be false absence in the other direction.
    """
    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=2,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("review the tunnels MR"),
            Message(
                role="tool",
                tool_name="bash",
                content=[TextContent(text="total 8\ndrwxr-xr-x  squeeze/")],
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert events == []


def test_a_listed_tools_stored_record_is_excluded_and_an_unlisted_tools_is_not(
    home: Path, tmp_path: Path
) -> None:
    """R5-MINOR-1, the DECISION PIN: the exclusion is by NAME, on purpose.

    ``project op=show`` / ``op=list`` returns records the operator or another
    session wrote, so excluding every ``project`` result costs a real source —
    the note on :data:`SELF_ECHO_RECEIPT_TOOLS` states the trade and why
    narrowing by op was rejected (a read op's result is an unattributable
    aggregate that also carries the model's own earlier description, so folding
    it would reopen the two-call laundering path). This test pins BOTH halves in
    the same test so the trade cannot drift into an implicit one: a listed
    tool's stored-record result does not explain the root, and an unlisted
    tool's result of the same shape does.
    """
    stored_record = "project show: description 'minerva build' names ~/squeeze/app.py"
    listed = _judge(home, tmp_path)
    assert (
        len(
            listed.observe_turn(
                turn_index=2,
                tool="write",
                arguments={"path": "~/squeeze/app.py"},
                messages=[
                    Message.user("review the tunnels MR"),
                    Message(
                        role="tool",
                        tool_name="project",
                        content=[TextContent(text=stored_record)],
                    ),
                ],
                system_blocks=["You are a personal assistant."],
            )
        )
        == 1
    ), "a project read is excluded by name — documented, not accidental"
    unlisted = _judge(home, tmp_path)
    assert (
        unlisted.observe_turn(
            turn_index=2,
            tool="write",
            arguments={"path": "~/squeeze/app.py"},
            messages=[
                Message.user("review the tunnels MR"),
                Message(role="tool", tool_name="bash", content=[TextContent(text=stored_record)]),
            ],
            system_blocks=["You are a personal assistant."],
        )
        == []
    ), "the SAME text from an unlisted tool still counts"


def test_a_todo_reminder_is_the_models_own_plan_and_does_not_explain_the_root(
    home: Path,
    tmp_path: Path,
) -> None:
    """Round-3 MAJOR, second carrier: the todo list is the model's own plan.

    ``_todo_reminder_text`` lists ``TodoParams.items`` verbatim, so folding it
    would disable the guard for every root the session planned in todos — turn N
    writes the item, the reminder lands at the yield boundary, and turn N+1's
    write reads as already explained. Reviewer's reproducer: ``events: 0``.
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
                custom_type="todo_reminder",
                attribution="system",
                details={
                    "text": (
                        "<system-reminder>\nInjected by the harness at the turn "
                        "boundary.\n- write the app at ~/squeeze\n</system-reminder>"
                    )
                },
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1, "the model's own plan must not silence the notice"
    text, _kind = divergence_notice(events[0].as_notice_payload())
    assert "Not in the system prompt, custom instructions or user messages" in text


@pytest.mark.parametrize("custom_type", sorted(MODEL_AUTHORED_CARRIERS))
def test_no_model_authored_carrier_explains_a_root(
    custom_type: str, home: Path, tmp_path: Path
) -> None:
    """The set's contract, member by member — including ones no other test names.

    The two reproducers above cover the carriers the reviewer measured; this one
    holds the RULE for the whole set, so a future member added to
    :data:`MODEL_AUTHORED_CARRIERS` without its own reproducer cannot be a name
    that is excluded on paper and folded in practice. Every member names the root
    in the fields its renderer arm reads, and the notice must still fire: text
    THIS session's model can author is never an instruction source.

    The rule is scoped to the asserting session (round-4 MINOR): ``job_result``,
    ``peer_message``, ``hub_message`` and ``ask_response`` stay folded, because
    they are OTHER models' words arriving as external context. This set is only
    the self-citation channel.
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
                custom_type=custom_type,
                attribution="system",
                details={
                    "text": "carry on inside ~/squeeze",
                    "summary": "the operator asked for ~/squeeze",
                    "tool": "write",
                    "description": "write to ~/squeeze/app.py",
                },
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert len(events) == 1, f"{custom_type} named ~/squeeze and silenced the notice"


def test_the_operator_naming_the_root_still_explains_it_beside_a_wake_prompt(
    home: Path,
    tmp_path: Path,
) -> None:
    """The bound on the exclusion: an operator turn naming the root still folds.

    Excluding model-authored carriers costs at most a redundant notice, and this
    is the row that proves the cost stays bounded — the wake prompt beside the
    operator's own words changes nothing, because the operator's turn is the
    source, and the wake echo is simply not one.
    """
    from local_operator.harness.types import CustomMessage

    judge = _judge(home, tmp_path)
    events = judge.observe_turn(
        turn_index=2,
        tool="write",
        arguments={"path": "~/squeeze/app.py"},
        messages=[
            Message.user("build it under ~/squeeze for me"),
            CustomMessage(
                custom_type="wake_prompt",
                attribution="system",
                details={"text": "wake delivery\n\nkeep going in ~/squeeze"},
            ),
        ],
        system_blocks=["You are a personal assistant."],
    )
    assert events == [], "the operator's own turn named the root, so it is not novel"


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


def _renderer_compared_types(source: str | None = None) -> set[str]:
    """Every custom-type value ``harness/render.py`` compares ``custom_type`` to.

    The renderer's OWN source is the authority for what it can inject: each
    ``message.custom_type == X`` / ``in (…)`` comparator is read out of the AST
    and resolved through the module namespace, so an imported constant, an
    inline literal and a family frozenset all arrive as the strings it matches.

    FAILS CLOSED (round-3 MINOR). A sweep that understands only the arm shape
    someone happened to write certifies rather than checks: hoisting
    ``custom_type`` into a local, interpolating it into an f-string, keying a
    dispatch table with it, or spelling a comparator as ``PREFIX + "_x"`` each
    leave this function returning a SMALLER set without failing — and the twin
    test below then reports a shorter universe as agreement while a carrier
    really is delivered unfolded. So nothing is skipped silently:

    * every ``.custom_type`` attribute use in the module must be the direct left
      of a ``Compare``, else this raises (that covers the hoist, the f-string,
      the dispatch key and any other use the value could escape through);
    * every comparator must resolve statically to strings — a literal, a module
      name bound to a ``str``/collection, or a tuple/set/list of those — else
      this raises (that covers ``PREFIX + "_x"`` and ``f"x_{suffix}"``);
    * an empty result raises, because a resolver that yields nothing is the
      renderer arm being invisible rather than absent.

    Residual, documented rather than silent: an arm that never spells the
    attribute at all is outside both halves — ``getattr(message,
    "custom_type")``, a table keyed by the instance, ``match message: case
    CustomMessage(custom_type="…")``, and a helper that lives in ANOTHER module
    and compares the value there. The ``match`` case is the one that misleads:
    the SUBJECT form ``match message.custom_type:`` hits the direct-left rule
    above and raises, which invites the conclusion that ``match`` is covered —
    the capture-pattern form spells no attribute here and is not seen at all.
    That shape is not a refactor of an existing arm — it is a new arm — and the
    convention this sweep relies on is that a carrier arm compares
    ``message.custom_type`` directly, in this module.

    ``source`` exists so ``test_the_sweep_fails_closed_on_every_evasion`` can
    drive adversarial variants through this same code; production calls pass
    nothing and read the real module.
    """
    import ast

    from local_operator.harness import render

    if source is None:
        source = Path(render.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    found: set[str] = set()

    def resolve(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.Name):
            value = getattr(render, node.id, None)
            if isinstance(value, str):
                return [value]
            if isinstance(value, (frozenset, set, tuple, list)):
                members = [item for item in value if isinstance(item, str)]
                if members:
                    return members
        elif isinstance(node, (ast.Tuple, ast.Set, ast.List)):
            resolved: list[str] = []
            for element in node.elts:
                resolved.extend(resolve(element))
            return resolved
        raise AssertionError(
            f"render.py compares custom_type against a value this sweep cannot resolve "
            f"statically ({ast.dump(node)[:120]}, line {getattr(node, 'lineno', '?')}). "
            "Spell the arm as `message.custom_type == <module-level constant>` or teach "
            "this sweep the new shape — a pass here would certify a carrier it never saw."
        )

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "custom_type"):
            continue
        parent = parents.get(node)
        if not (isinstance(parent, ast.Compare) and parent.left is node):
            raise AssertionError(
                f"render.py uses `.custom_type` at line {node.lineno} in a shape this "
                "sweep cannot follow (only the direct left of a Compare is understood). "
                "Keep each carrier arm as `message.custom_type == <constant>`: a hoisted "
                "local, an f-string or a dispatch-table key would empty the renderer half "
                "of this sweep while it still reports success."
            )
        for comparator in parent.comparators:
            found.update(resolve(comparator))

    if not found:
        raise AssertionError(
            "no custom_type comparisons resolved in render.py — the renderer's arms are "
            "invisible to this sweep, so an empty unfolded set would mean nothing"
        )
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


#: Sources standing in for the four ordinary refactors of a carrier arm that round
#: 3 showed the sweep silently missing, plus the literal arm it must still read.
#: Each is a whole module: the sweep parses it exactly as it parses ``render.py``.
_SWEEP_VARIANTS: dict[str, str] = {
    "literal arm (control)": 'def f(message):\n    return message.custom_type == "sweep_probe"\n',
    "concatenated comparator": (
        "def f(message):\n" '    return message.custom_type == "sweep" + "_probe"\n'
    ),
    "f-string comparator": (
        'def f(message):\n    return message.custom_type == f"sweep_{suffix}"\n'
    ),
    "dispatch table": "def f(message):\n    return _ROUTES[message.custom_type](message)\n",
    "hoisted custom_type local": (
        "def f(message):\n"
        "    custom_type = message.custom_type\n"
        '    return custom_type == "sweep_probe"\n'
    ),
}


def test_the_sweep_fails_closed_on_every_evasion_shape() -> None:
    """Round-3 MINOR: a sweep that only reads the shapes it knows CERTIFIES.

    The reviewer inserted each shape into a copy of ``render.py`` and ran the
    sweep: the literal arm failed by name (the demonstration was real), and a
    concatenated comparator, an f-string comparator, a dispatch table and a
    hoisted ``custom_type`` local were all MISSED while the variant really did
    deliver the missed type to the model as a user turn. The failure mode is the
    worst one for a gate: silence that reads as agreement.

    So the control must still resolve, and each evasion must RAISE rather than
    return a smaller set. Fail-closed is the only behaviour that cannot be
    satisfied by writing the arm differently.
    """
    assert _renderer_compared_types(_SWEEP_VARIANTS["literal arm (control)"]) == {
        "sweep_probe"
    }, "the control must resolve, or the rest of this test proves nothing"

    for name in (
        "concatenated comparator",
        "f-string comparator",
        "dispatch table",
        "hoisted custom_type local",
    ):
        with pytest.raises(AssertionError):
            _renderer_compared_types(_SWEEP_VARIANTS[name])

    # The real module goes through the same fail-closed path, so a shape it
    # cannot follow fails the folded-set test above instead of shrinking it.
    assert _renderer_compared_types()


def _rendered_user_text(message: AgentMessage) -> str:
    """The user-turn text the REAL renderer delivers for one message."""
    from local_operator.harness.render import _default_convert_to_llm

    chunks: list[str] = []
    for rendered in _default_convert_to_llm([message]):
        if rendered.role != "user":
            continue
        content = rendered.content
        if isinstance(content, str):
            chunks.append(content)
            continue
        for block in content or []:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def test_every_folded_carrier_contributes_text_to_the_corpus() -> None:
    """Round-3 MINOR: folding a type obliges extraction to know its shape.

    The sibling test above pins MEMBERSHIP — that the folded set is exactly what
    the renderer delivers — and nothing pinned EXTRACTION, so the reviewer's next
    structured carrier repeated round-2 MAJOR-1: folded, arm composing its text
    from ``tool`` + ``description`` (the gate arm's shape), ``_custom_text``
    returned ``''``, and the row again claimed the root was "new to this session …
    Not in … user messages" while a green sweep stood beside it.

    A probe that simply carries ``text`` would pass vacuously for exactly that
    carrier, so each folded type is driven twice: with the full probe (a carrier
    that ships ``details["text"]``, which the judge must read back) and with the
    probe minus ``text`` (a carrier that COMPOSES from its payload — if the
    renderer still delivers words with no ``text`` key, the judge must find text
    there too, and the words it folds must be the ones the model read).
    """
    from local_operator.divergence import MODEL_AUTHORED_CARRIERS, _custom_text
    from local_operator.harness.message_types import MODEL_VISIBLE_INJECTED_TYPES
    from local_operator.harness.types import CustomMessage

    folded = sorted(MODEL_VISIBLE_INJECTED_TYPES - MODEL_AUTHORED_CARRIERS)
    assert folded, "the folded set must have members, or this probe checks nothing"
    assert MODEL_AUTHORED_CARRIERS <= MODEL_VISIBLE_INJECTED_TYPES, (
        "an excluded name that is not a carrier excludes nothing — a typo here folds "
        "the type it meant to leave out"
    )
    structured = {key: value for key, value in _PROBE_DETAILS.items() if key != "text"}

    for name in folded:
        for details, shape in (
            (dict(_PROBE_DETAILS), "ships a text field"),
            (dict(structured), "composes from its payload"),
        ):
            message = CustomMessage(custom_type=name, attribution="system", details=details)
            delivered = _rendered_user_text(message)
            if not delivered:
                # Not this type's shape: the renderer read nothing from these
                # details, so there is no delivery to compare against. The other
                # probe is the one that covers this type.
                continue
            extracted = _custom_text(message)
            assert extracted, (
                f"{name} ({shape}): the renderer delivers {delivered[:60]!r} to the "
                "model but _custom_text reads nothing — the corpus holds an empty "
                "source while the notice claims the carrier was checked"
            )
            assert extracted in delivered, (
                f"{name} ({shape}): _custom_text reads {extracted!r} but the renderer "
                f"delivers {delivered[:60]!r} — the judge is folding words the model "
                "never read"
            )


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


def test_a_gate_row_about_another_root_leaves_the_claim_true(home: Path, tmp_path: Path) -> None:
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
    judge.observe_turn(turn_index=1, tool="read", arguments={"path": "~/novel/a"}, messages=history)
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


# ---------------------------------------------------------------------------
# The live paths: a real wake, a real delivery, a real judge
# ---------------------------------------------------------------------------
#
# QA round 4's finding about the round before it: the unit pin that bounded the
# wake exclusion built a history the LIVE path cannot produce (it put the root in
# a typed user turn beside the carrier), so it stayed green while the operator
# path regressed. These two tests therefore drive the PRODUCERS — ``arm_wake``,
# the session's own scheduler, the real ``wake`` tool, ``format_wake_delivery_text``
# and ``Session._deliver_wake`` — and assert on the audit rows and the transcript,
# never on a hand-built message list.


class _WakePathStream:
    """A request-driven model: each decision is about what the request carries.

    A positional script drifts the moment the runtime issues a call of its own,
    and for these tests the whole point is WHICH request the woken turn is
    handed — so the decision is read off the real request.
    """

    def __init__(
        self,
        *,
        root: str,
        create_wake: dict[str, Any] | None,
        unnamed: str | None = None,
        trigger: str = "(alarm) Scheduled wake",
    ) -> None:
        self.root = root
        self.create_wake = create_wake
        #: Which delivered text makes this the WOKEN turn. A single fire says
        #: "(alarm) Scheduled wake"; the resume catch-up says "(alarm) The
        #: session resumed" and folds several rows into one prompt, so the test
        #: that drives the fold picks its own marker rather than matching one
        #: hard-coded string and silently never waking.
        self.trigger = trigger
        #: A SECOND root in the same command that NOTHING named — the operator
        #: case's non-vacuity control: one turn, one tool, one judge, and the
        #: row that fires must be this one while the operator's root stays quiet.
        self.unnamed = unnamed
        self.calls = 0

    @staticmethod
    def _text(request: Any) -> str:
        parts: list[str] = []
        for message in getattr(request, "messages", []) or []:
            for block in getattr(message, "content", []) or []:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _tool_names(request: Any) -> list[str]:
        return [
            str(getattr(message, "tool_name", "") or "")
            for message in getattr(request, "messages", []) or []
            if getattr(message, "role", None) == "tool"
        ]

    def __call__(self, request: Any, signal: Any):
        from local_operator.harness.types import StreamEndEvent, StreamTextDelta

        self.calls += 1
        blob = self._text(request)
        tools = self._tool_names(request)
        events: list[StreamEvent] = []
        if self.trigger in blob:
            # THE WOKEN TURN: the carrier is in this request. Write to the root
            # the test is watching, then finish.
            if "bash" in tools:
                events = [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
            else:
                targets = [self.root, self.unnamed] if self.unnamed else [self.root]
                command = " && ".join(
                    [f"mkdir -p {target}/src" for target in targets]
                    + [f"printf 'WRITTEN' > {target}/src/app.py" for target in targets]
                )
                events = [
                    StreamToolCallDelta(
                        index=0,
                        id=f"call_wake_{self.calls}",
                        name="bash",
                        argument_delta=json.dumps({"command": command}),
                    ),
                    StreamEndEvent(stop_reason="tool_use"),
                ]
        elif self.create_wake is not None and "wake" not in tools:
            events = [
                StreamToolCallDelta(
                    index=0,
                    id=f"call_wake_{self.calls}",
                    name="wake",
                    argument_delta=json.dumps(self.create_wake),
                ),
                StreamEndEvent(stop_reason="tool_use"),
            ]
        else:
            events = [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]

        async def gen():
            for event in events:
                yield event

        return gen()


async def _wait_until(predicate: Any, timeout: float = 30.0) -> bool:
    """Poll a condition the scheduler fires asynchronously (bounded, never a sleep)."""
    import asyncio
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


def _wake_carriers(session: Any) -> list[Any]:
    return [
        message
        for message in list(session._divergence_judge._messages_ref or [])
        if getattr(message, "custom_type", "") == "wake_prompt"
    ]


def _divergence_rows(config_dir: Path) -> list[dict[str, Any]]:
    path = config_dir / "audit" / "divergence.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_an_operator_scheduled_wake_explains_its_root_end_to_end(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-4 MAJOR, driven through the CLI's own writer.

    ``arm_wake(..., origin="cli")`` is what ``lop wake create`` calls; the fired
    delivery is the ONLY copy of the operator's words (no user turn is
    synthesised for a wake), so the woken turn's write to the root they named
    must not produce a row. QA round 4 failed this twice on the real path while
    the unit pin stayed green — this is that regression, driven by the producer.
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools
    from local_operator.wakes.arm import arm_wake

    scratch = home.parent
    config_dir = scratch / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    recorder = reset_recorder_for_test(AnalyticsStore(scratch / "analytics.db"))
    # THE SKIP LIST WOULD MAKE THESE TESTS VACUOUS: a pytest home lives under
    # ``$TMPDIR``, which the shipped skip list suppresses WHOLESALE, so every
    # root under the fixture home would be skipped before the carrier rule was
    # ever consulted — a green "no row" that proves nothing (and a "row" test
    # that can never fire). Skipping is not what these tests measure; it has its
    # own tests. Neutralised through the one seam the real derivation reads, the
    # same way ``_judge(skip_roots=())`` neutralises it for the rest of this file.
    monkeypatch.setattr("local_operator.session.spec.volatile_roots", lambda: ())
    root = "~/wake-operator-root"
    message = f"when you wake, check {root}/src and report"

    def build(stream_fn: Any) -> Any:
        return Session(
            model=MODEL,
            stream_fn=stream_fn,
            tools=create_tools(ToolContext(), enabled=["bash"]),
            # THE SHIPPED LOCATION: ``arm_wake`` resolves a session id under
            # ``<config>/sessions/`` and refuses an id with no transcript there.
            transcript=Transcript(config_dir / "sessions" / "sess"),
            session_id="sess",
            system_blocks_provider=lambda: ["You are a personal assistant."],
            yolo=True,
            compaction_settings=CompactionSettings(enabled=False),
        )

    # The conversation exists before the operator schedules anything (the CLI
    # refuses a session id with no transcript).
    first = build(_WakePathStream(root=root, create_wake=None))
    try:
        await first.prompt("hello there")
    finally:
        await first.dispose()

    outcome = await arm_wake(config_dir, "sess", {"message": message, "in": "2s"}, origin="cli")
    assert outcome.schedules[0].origin == "cli", "the CLI's writer must stamp its own origin"

    unnamed = f"{root}-unnamed"
    session = build(_WakePathStream(root=root, create_wake=None, unnamed=unnamed))
    try:
        # WAIT ON THE JUDGE, not on the file: the carrier enters the judged
        # history before the woken turn's call runs, while the tool's side effect
        # lands first — polling the file alone races the post-execution hook.
        assert await _wait_until(
            lambda: _wake_carriers(session)
        ), "the wake never fired through the real scheduler"
        carriers = _wake_carriers(session)
        target = home / "wake-operator-root" / "src" / "app.py"
        assert await _wait_until(
            target.exists
        ), "the woken turn never wrote to the root the operator named"
    finally:
        await session.dispose()
        recorder.close()
        reset_recorder_for_test(AnalyticsStore(scratch / "analytics-2.db")).close()

    # The delivery carried the attribution, from the row the writer stamped.
    assert carriers, "the woken turn must have received the wake_prompt carrier"
    assert carriers[0].details["wake_author"] == "operator"
    assert (home / "wake-operator-root" / "src" / "app.py").read_text() == "WRITTEN"
    # NON-VACUITY, on the same turn and the same judge: the root NOTHING named
    # fires, so an empty result for the operator's root is the carrier being
    # folded and not the observation being blind.
    rows = _divergence_rows(config_dir)
    assert [row["root"] for row in rows] == [unnamed], rows
    # THE REGRESSION: the operator's own instruction named the root.
    assert root not in {
        row["root"] for row in rows
    }, "the operator named ~/wake-operator-root in the wake they scheduled"


@pytest.mark.asyncio
async def test_a_model_scheduled_wake_does_not_explain_its_root_end_to_end(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror image, through the real ``wake`` tool and scheduler.

    The message is the MODEL's own ``WakeParams.message``. The notice must fire
    once the woken turn writes to that root (round-3 MAJOR: a session can wake
    itself into a project it invented), and the row must not hedge — the author
    was recorded, so nothing here was unreadable. The creating tool's receipt
    echoes the same text and is not a second author (QA4-1).
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools

    scratch = home.parent
    config_dir = scratch / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    recorder = reset_recorder_for_test(AnalyticsStore(scratch / "analytics.db"))
    # THE SKIP LIST WOULD MAKE THESE TESTS VACUOUS: a pytest home lives under
    # ``$TMPDIR``, which the shipped skip list suppresses WHOLESALE, so every
    # root under the fixture home would be skipped before the carrier rule was
    # ever consulted — a green "no row" that proves nothing (and a "row" test
    # that can never fire). Skipping is not what these tests measure; it has its
    # own tests. Neutralised through the one seam the real derivation reads, the
    # same way ``_judge(skip_roots=())`` neutralises it for the rest of this file.
    monkeypatch.setattr("local_operator.session.spec.volatile_roots", lambda: ())
    root = "~/wake-model-root"
    create = {
        "op": "create",
        "message": f"continue the build in {root}/src/app.py",
        "in": "2s",
    }

    session = Session(
        model=MODEL,
        stream_fn=_WakePathStream(root=root, create_wake=create),
        tools=create_tools(ToolContext(), enabled=["bash"]),
        transcript=Transcript(scratch / "sess"),
        session_id="sess",
        system_blocks_provider=lambda: ["You are a personal assistant."],
        yolo=True,
        compaction_settings=CompactionSettings(enabled=False),
    )
    # The wake tool's builder is a createIf: it returns nothing unless the
    # context already carries a scheduler, so it is built from the session's own
    # per-turn context — the same list the loop reads at dispatch.
    session._tools = [
        *session._tools,
        *create_tools(session._build_tool_context(), enabled=["wake"]),
    ]
    session._context.tools = session._tools
    try:
        await session.prompt("start the job")
        assert await _wait_until(
            lambda: (home / "wake-model-root" / "src" / "app.py").exists()
        ), "the wake the model scheduled never fired"
        # The ROW is written by the judge at the write call, so waiting on it (not
        # on the file) is what makes the carrier read below non-racy.
        assert await _wait_until(
            lambda: _divergence_rows(config_dir)
        ), "the woken turn's write produced no audit row"
        carriers = _wake_carriers(session)
        events = list(session._divergence_judge._events)
    finally:
        await session.dispose()
        recorder.close()
        reset_recorder_for_test(AnalyticsStore(scratch / "analytics-2.db")).close()
    rows = _divergence_rows(config_dir)

    assert carriers and carriers[0].details["wake_author"] == "model"
    assert len(rows) == 1, "the model's own wake text must not silence the notice"
    assert rows[0]["root"] == root
    assert rows[0]["unseen"] == [], "the author was recorded, so nothing was unreadable"
    assert events, "the row must surface as a notice on the real path"
    text, _kind = divergence_notice(events[0].as_notice_payload())
    # No hedge: the delivery recorded its author, so nothing about it was
    # unreadable. The claim it does make is true — the operator's only turn is
    # "start the job", and the model's own wake text is not an instruction.
    assert "could not be checked" not in text
    assert "wake-model-root" in text


async def _rewrite_schedules(session_dir: Path, mutate: Any) -> list[dict[str, Any]]:
    """Rewrite the transcript's NEWEST ``wake_schedules`` snapshot in place.

    The transcript entry is the truth ``load()`` reads; the index under
    ``<config>/wakes/`` is derived (``wakes/arm.py``'s own note). A test that
    wants a row this build would never write — one produced by a build that
    predates a field, or one whose due instant is already past — therefore has
    to rewrite THIS, exactly as a pre-upgrade runtime left it. Returns the rows
    it wrote, so the test can assert what reached disk.
    """
    from local_operator.harness.wake import WAKE_SCHEDULES_CUSTOM_TYPE
    from local_operator.session.transcript import Transcript, read_latest_custom_entry

    entry = read_latest_custom_entry(session_dir, WAKE_SCHEDULES_CUSTOM_TYPE)
    assert entry is not None, "no wake_schedules snapshot to rewrite"
    rows = list(entry.payload["details"]["schedules"])
    mutate(rows)
    await Transcript(session_dir).append_custom(WAKE_SCHEDULES_CUSTOM_TYPE, {"schedules": rows})
    return rows


@pytest.mark.asyncio
async def test_a_legacy_wake_row_hedges_its_live_delivery_end_to_end(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5-MAJOR-1: a row written before ``origin`` existed must HEDGE, not assert.

    The hedge was built for the stored-history case and was unreachable for a
    LIVE fire: the field defaulted to ``"tool"``, so a row with no ``origin``
    read back as model-authored, the delivery stamped ``wake_author="model"``,
    the judge saw a known author and never recorded ``unattributed_wake`` — and
    the operator read "… Not in the system prompt, custom instructions or user
    messages" about their OWN instruction. Not a corner: every persisted row on
    a live machine carries no ``origin`` (167 of 167 sampled), and an edit
    carries ``tool`` forward so editing did not repair one.

    Driven through the real writer (``arm_wake`` stamping ``cli``), a store
    rewrite that removes the key exactly as a pre-field build would, then the
    real scheduler, the real delivery and the real judge.
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools
    from local_operator.wakes.arm import arm_wake

    scratch = home.parent
    config_dir = scratch / "config"
    session_dir = config_dir / "sessions" / "sess"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    recorder = reset_recorder_for_test(AnalyticsStore(scratch / "analytics.db"))
    # The fixture home lives under $TMPDIR, which the shipped skip list
    # suppresses wholesale — neutralised through the one seam the real
    # derivation reads, so an empty result here is the fold and not blindness.
    monkeypatch.setattr("local_operator.session.spec.volatile_roots", lambda: ())
    root = "~/wake-legacy-root"
    message = f"when you wake, check {root}/src and report"

    def build(stream_fn: Any) -> Any:
        return Session(
            model=MODEL,
            stream_fn=stream_fn,
            tools=create_tools(ToolContext(), enabled=["bash"]),
            transcript=Transcript(config_dir / "sessions" / "sess"),
            session_id="sess",
            system_blocks_provider=lambda: ["You are a personal assistant."],
            yolo=True,
            compaction_settings=CompactionSettings(enabled=False),
        )

    first = build(_WakePathStream(root=root, create_wake=None))
    try:
        await first.prompt("hello there")
    finally:
        await first.dispose()

    outcome = await arm_wake(config_dir, "sess", {"message": message, "in": "2s"}, origin="cli")
    assert outcome.schedules[0].origin == "cli"
    # What a build that predates the field wrote: the KEY IS ABSENT (not
    # ``"tool"``), which is the distinction the default has to preserve.
    written = await _rewrite_schedules(
        session_dir, lambda rows: [row.pop("origin", None) for row in rows]
    )
    assert all("origin" not in row for row in written), written

    session = build(_WakePathStream(root=root, create_wake=None))
    try:
        assert await _wait_until(
            lambda: _wake_carriers(session)
        ), "the legacy row never fired through the real scheduler"
        carriers = _wake_carriers(session)
        target = home / "wake-legacy-root" / "src" / "app.py"
        assert await _wait_until(
            lambda: target.exists()
        ), "the woken turn never wrote to the root the legacy row named"
        events = list(session._divergence_judge._events)
    finally:
        await session.dispose()
        recorder.close()
        reset_recorder_for_test(AnalyticsStore(scratch / "analytics-2.db")).close()

    assert carriers, "the woken turn must have received the wake_prompt carrier"
    assert carriers[0].details["wake_author"] == "", (
        "a row with no recorded author must deliver with NO author, not 'model' — "
        f"got {carriers[0].details.get('wake_author')!r}"
    )
    rows = _divergence_rows(config_dir)
    assert [row["root"] for row in rows] == [root], rows
    assert "unattributed_wake" in rows[0]["unseen"], rows[0]["unseen"]
    assert events, "the row must surface as a notice on the real path"
    text, _kind = divergence_notice(events[0].as_notice_payload())
    # THE REGRESSION, in the wording the operator reads: the author was never
    # recorded, so the notice may not claim the absence was checked.
    assert "is not named in what this session could check" in text
    assert "wake deliveries with no recorded author" in text
    assert "is new to this session" not in text


@pytest.mark.parametrize(
    ("authors", "expected_author", "expect_row", "expect_hedge"),
    [
        (("cli",), "operator", False, False),
        (("cli", "tool"), "", True, True),
        (("tool",), "model", True, False),
        # R6-NIT-1: the R5-MAJOR shape itself, a row written before ``origin``
        # existed. Alone and folded with an operator row it must hedge too, and
        # neither member may be able to borrow the other's author.
        ((None,), "", True, True),
        (("cli", None), "", True, True),
        (("tool", None), "", True, True),
    ],
    ids=[
        "operator-only",
        "mixed",
        "tool-only",
        "legacy-only",
        "operator-legacy",
        "tool-legacy",
    ],
)
@pytest.mark.asyncio
async def test_a_resume_catchup_attributes_its_fold_only_when_every_row_agrees(
    authors: tuple[WakeOrigin | None, ...],
    expected_author: str,
    expect_row: bool,
    expect_hedge: bool,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QA round 5: the fold is three-valued, so a MIXED catch-up hedges.

    An aggregate of several rows is attributable only when every member agrees.
    Collapsing a mixed fold to ``"model"`` was the defect: one model row folded
    in with the operator's stamped a known author, so the judge folded the
    whole blob, never recorded ``unattributed_wake``, and the operator's own
    root got the unhedged false sentence. All-operator must stay silent (the
    round-4 pass), all-model must still fire a true notice, and mixed must fire
    the HEDGED one.
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools
    from local_operator.wakes.arm import arm_wake

    scratch = home.parent
    config_dir = scratch / "config"
    session_dir = config_dir / "sessions" / "sess"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    recorder = reset_recorder_for_test(AnalyticsStore(scratch / "analytics.db"))
    monkeypatch.setattr("local_operator.session.spec.volatile_roots", lambda: ())
    root = "~/wake-catchup-root"

    def build(stream_fn: Any) -> Any:
        return Session(
            model=MODEL,
            stream_fn=stream_fn,
            tools=create_tools(ToolContext(), enabled=["bash"]),
            transcript=Transcript(config_dir / "sessions" / "sess"),
            session_id="sess",
            system_blocks_provider=lambda: ["You are a personal assistant."],
            yolo=True,
            compaction_settings=CompactionSettings(enabled=False),
        )

    first = build(_WakePathStream(root=root, create_wake=None))
    try:
        await first.prompt("hello there")
    finally:
        await first.dispose()

    for origin in authors:
        # ``None`` is the LEGACY member: a row written by a build that predates
        # the field. Armed as the model's, then stripped in the snapshot below,
        # which is exactly the bytes such a build left behind (``arm_wake``
        # refuses an unknown literal, and only the stored row may carry no
        # ``origin`` at all).
        await arm_wake(
            config_dir,
            "sess",
            {"message": f"check {root}/src and report", "in": "2s"},
            origin="tool" if origin is None else origin,
        )
    overdue = int(time.time() * 1000) - 60_000
    legacy = [index for index, origin in enumerate(authors) if origin is None]

    def make_overdue(rows: list[dict[str, Any]]) -> None:
        for index, row in enumerate(rows):
            # Overdue at load is what routes a row into the catch-up fold.
            row["next_due_at"] = overdue
            if index in legacy:
                row.pop("origin", None)

    rewritten = await _rewrite_schedules(session_dir, make_overdue)
    assert len(rewritten) == len(authors)

    session = build(
        _WakePathStream(root=root, create_wake=None, trigger="(alarm) The session resumed")
    )
    try:
        # ``_prepare_missed_wake_catchup`` ran in __init__; the take is held
        # until load's grace deadline (LOAD_GRACE_MS) so a resumed session does
        # not paint a catch-up over a prompt it is about to run. The wait itself
        # is not what this test measures and has its own tests, so the deadline
        # is dropped rather than slept through — the same neutralisation the
        # skip list gets above.
        session._resume_grace_ends_ms = 0.0
        session._handle_missed_wakes()
        assert await _wait_until(lambda: _wake_carriers(session)), "the catch-up never delivered"
        carriers = _wake_carriers(session)
        target = home / "wake-catchup-root" / "src" / "app.py"
        assert await _wait_until(
            lambda: target.exists()
        ), "the woken turn never wrote to the root the folded row named"
        events = list(session._divergence_judge._events)
    finally:
        await session.dispose()
        recorder.close()
        reset_recorder_for_test(AnalyticsStore(scratch / "analytics-2.db")).close()

    assert carriers, "the catch-up must reach the model as a carrier"
    assert carriers[0].details["wake_author"] == expected_author
    rows = _divergence_rows(config_dir)
    if not expect_row:
        # NON-VACUITY: the turn ran and wrote (asserted above), so an empty
        # result is the operator's root being folded, not the judge being blind.
        assert rows == [], "an all-operator fold explains the operator's own root"
        assert events == []
        return
    assert [row["root"] for row in rows] == [root], rows
    assert events, "the notice must surface on the real path"
    text, _kind = divergence_notice(events[0].as_notice_payload())
    if expect_hedge:
        assert "unattributed_wake" in rows[0]["unseen"], rows[0]["unseen"]
        assert "is not named in what this session could check" in text
        assert "wake deliveries with no recorded author" in text
        assert "is new to this session" not in text, "a mixed fold is unattributable"
    else:
        assert rows[0]["unseen"] == [], "every folded row was model-written"
        assert "could not be checked" not in text
        assert "wake-catchup-root" in text


async def _patch_wake_over_http(
    config_dir: Path, session_id: str, wake_id: str, body: dict[str, Any]
) -> Any:
    """Drive the desktop's PATCH route against the real app, as the client does.

    The route is a surface in its own right: it is where the operator's body is
    accepted, and the only place that decides which writer the request reaches.
    One app, one request, torn down with the ``with`` — the rig rule for a
    scripted browser applies in spirit to any server a test starts.
    """
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_wakes

    app = FastAPI()
    app.include_router(desktop_wakes.router)
    app.state.config_manager = ConfigManager(config_dir)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer divergence-edit-token"},
    ) as client:
        response = await client.patch(f"/v1/desktop/wakes/{session_id}/{wake_id}", json=body)
    return response


@pytest.mark.parametrize("surface", ["tui", "desktop-route", "desktop-owner"])
@pytest.mark.asyncio
async def test_an_operator_reword_of_a_model_armed_row_is_the_operators_end_to_end(
    surface: str, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6-MAJOR-1, through each of the three surfaces that can reach it.

    ``origin`` is documented as who WROTE ``message``, and every edit surface
    replaces ``message``. Until this round the row kept ``tool``, the delivery
    stamped ``wake_author="model"``, and the guard told the operator their own
    sentence was "new to this session" — the false notice the whole module
    exists to prevent. Each leg drives the surface's OWN writer (the page's save
    through the session's one writer; the desktop PATCH over HTTP; the live
    runtime's ``wake`` branch), then the real scheduler fires the edited row and
    the real judge reads the delivery.

    The assertion is the strongest one available: the delivery names the
    operator, the woken turn still runs and writes (non-vacuity), and there is
    NO row at all — their own text is a source the guard can find.
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tools.registry import create_tools
    from local_operator.wakes.arm import arm_wake

    scratch = home.parent
    config_dir = scratch / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "divergence-edit-token")
    recorder = reset_recorder_for_test(AnalyticsStore(scratch / "analytics.db"))
    monkeypatch.setattr("local_operator.session.spec.volatile_roots", lambda: ())
    model_root = "~/wake-edit-model-root"
    operator_root = "~/wake-edit-operator-root"
    model_message = f"check {model_root}/src and report"
    operator_message = f"check {operator_root}/src and report"
    # The two live surfaces (page, runtime branch) need the row to exist at
    # boot, then take a margin the edit fits inside; the cold route edits before
    # anything loads the row, so it fires on load like the legacy case.
    live = surface in ("tui", "desktop-owner")
    now_ms = int(time.time() * 1000)
    # The desktop's own id shape (``^[a-f0-9]{12}$``, the sessions store's
    # admission rule): a shorter id is refused by the route before any wake is
    # read, which would make the cold-path leg assert the wrong refusal.
    session_id = "a1b2c3d4e5f6"

    def build(stream_fn: Any) -> Any:
        return Session(
            model=MODEL,
            stream_fn=stream_fn,
            tools=create_tools(ToolContext(), enabled=["bash"]),
            transcript=Transcript(config_dir / "sessions" / session_id),
            session_id=session_id,
            system_blocks_provider=lambda: ["You are a personal assistant."],
            yolo=True,
            compaction_settings=CompactionSettings(enabled=False),
        )

    first = build(_WakePathStream(root=operator_root, create_wake=None))
    try:
        await first.prompt("hello there")
    finally:
        await first.dispose()

    # Armed by the MODEL: ``arm_wake``'s default is what the agent's own
    # ``wake`` tool stamps, and it is the author the edit has to replace.
    armed = await arm_wake(
        config_dir,
        session_id,
        {"message": model_message, "in": "10s" if live else "2s"},
        origin="tool",
    )
    assert armed.schedules[0].origin == "tool"

    if surface == "desktop-route":
        status = await _patch_wake_over_http(
            config_dir, session_id, "w1", {"message": operator_message}
        )
        assert status.status_code == 200, status.text

    session = build(_WakePathStream(root=operator_root, create_wake=None))
    try:
        if surface == "tui":
            # The page's own ``_save``/``_commit`` pair, minus the widgets: the
            # pure model on the rows the page owns, then the session's ONE
            # writer — no second persistence path is invented here.
            from local_operator.tui import schedule as schedule_model

            rows = list(session.wake_scheduler.schedules)
            target = next(row for row in rows if row.id == "w1")
            values = schedule_model.prefilled(target)
            result = schedule_model.edit(
                schedule_model.FormValues(
                    message=operator_message,
                    when=values.when,
                    repeat=values.repeat,
                    ends=values.ends,
                    notify=values.notify,
                ),
                rows,
                "w1",
                now_ms,
            )
            assert result.ok and result.schedule is not None, result.error
            await session.set_wake_schedules(schedule_model.with_schedule(rows, result.schedule))
        elif surface == "desktop-owner":
            # The ladder word the route sends when a live runtime owns the
            # session (``route_shared_slash("wake", …)``); the dispatch to it is
            # pinned in ``tests/unit/session/runtime/test_serving_wake_command``.
            from local_operator.session.frontend_state import SlashResult
            from local_operator.session.runtime.serving import ServingSessionHandle

            handle = ServingSessionHandle.__new__(ServingSessionHandle)
            outcome = await handle._wake_slash(
                session,
                json.dumps(
                    {"op": "edit", "wake_id": "w1", "request": {"message": operator_message}}
                ),
                SlashResult,
            )
            assert outcome.kind == "notice", outcome.text

        assert await _wait_until(
            lambda: _wake_carriers(session)
        ), f"the wake edited through the {surface} surface never fired"
        carriers = _wake_carriers(session)
        target_file = home / "wake-edit-operator-root" / "src" / "app.py"
        assert await _wait_until(
            lambda: target_file.exists()
        ), "the woken turn never wrote the root the operator named"
        events = list(session._divergence_judge._events)
    finally:
        await session.dispose()
        recorder.close()
        reset_recorder_for_test(AnalyticsStore(scratch / "analytics-2.db")).close()

    assert carriers, "the woken turn must have received the wake_prompt carrier"
    delivered = carriers[0].details["text"]
    assert operator_message in delivered, "the row that fired is not the edited one"
    assert model_message not in delivered
    assert carriers[0].details["wake_author"] == "operator", (
        "the operator's replacement text delivered as model authorship — the "
        f"false notice this round exists to close (surface {surface})"
    )
    # NON-VACUITY: the turn ran and wrote (above), so an empty audit means the
    # operator's own words explain the root — not that the judge was blind.
    assert _divergence_rows(config_dir) == []
    assert events == []
