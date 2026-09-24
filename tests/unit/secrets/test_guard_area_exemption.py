"""Arms for the escalation exemption of the guard's own source and corpus.

The module under test decides ONE thing — whether a call is a read of the guard's
own area — and these arms pin both halves of that decision. The positive half is
the operator's request; the negative half is the security requirement, and it is
the half that decides whether the exemption ships: an agent must not be able to
confer the exemption on itself by naming a path in a command, a pattern, or any
other text. Only a file-reading tool's own ``path`` argument may.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.guard_area import EXEMPT_SOURCES, READING_TOOLS
from local_operator.harness.guard_area import exempt_from_escalation as exempt
from local_operator.harness.guard_area import reads_exempt_source, source_is_exempt
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    ChatRequest,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.redaction_shapes import REDACTION_MARKER
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from local_operator.variables import VariableStore
from tests.unit.secrets.credential_shape_corpus import POSITIVE_CASES

#: The two files the operator named, spelled from this checkout's root. Derived
#: from the module's own package root rather than from the CWD so the arms hold
#: whatever directory pytest was started in.
_PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "local_operator"
SHAPE_TABLE = _PACKAGE_ROOT / "redaction_shapes.py"
CORPUS = _PACKAGE_ROOT.parent / "tests" / "unit" / "secrets" / "credential_shape_corpus.py"
#: The corpus's own test module: exempt because it was MEASURED to escalate when
#: read (21 hits, 18 of them escalating) — see ``guard_area._EXEMPT_RELATIVE_PATHS``.
CORPUS_TESTS = _PACKAGE_ROOT.parent / "tests" / "unit" / "secrets" / "test_credential_shapes.py"


def test_the_exempt_set_is_the_guards_source_and_corpus_and_nothing_else() -> None:
    assert EXEMPT_SOURCES == {
        SHAPE_TABLE.resolve(),
        CORPUS.resolve(),
        CORPUS_TESTS.resolve(),
    }
    # No tree and no class rule: a sibling test module and the rest of the
    # package are NOT exempt, which is the broader version the operator declined.
    assert reads_exempt_source("read", {"path": "tests/unit/secrets/conftest.py"}) is False
    assert reads_exempt_source("read", {"path": "tests/unit/secrets/"}) is False
    assert reads_exempt_source("read", {"path": "local_operator/"}) is False
    assert reads_exempt_source("read", {"path": "local_operator/incidents.py"}) is False


@pytest.mark.parametrize("tool", sorted(READING_TOOLS))
def test_each_reading_tool_is_exempt_for_either_file(tool: str) -> None:
    assert reads_exempt_source(tool, {"path": str(SHAPE_TABLE)}) is True
    assert reads_exempt_source(tool, {"path": str(CORPUS)}) is True


def test_a_relative_spelling_resolves_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(SHAPE_TABLE.parent.parent)
    assert reads_exempt_source("read", {"path": "local_operator/redaction_shapes.py"}) is True
    assert reads_exempt_source("read", {"path": "./local_operator/redaction_shapes.py"}) is True


def test_a_bash_command_that_merely_names_the_path_is_not_exempt() -> None:
    """THE SECURITY ARM. Naming the path in a command confers nothing.

    ``cat``, a python one-liner and a shell pipeline all read the file in fact,
    and all of them are refused: the exemption may only be conferred by a
    file-reading tool's STRUCTURED path argument, because a command string is
    text an agent (or a file the agent read) can steer.
    """
    commands = (
        f"cat {SHAPE_TABLE}",
        f"sed -n '1,20p' {CORPUS}",
        f"python -c \"print(open('{SHAPE_TABLE}').read())\"",
        f"grep -h . {CORPUS} | head",
        f"cat {SHAPE_TABLE} > /dev/null # credential-shape corpus",
    )
    for command in commands:
        assert reads_exempt_source("bash", {"command": command}) is False, command


def test_text_that_names_the_path_elsewhere_is_not_exempt() -> None:
    """A pattern, a URL, an edit target and an excerpt are all just text."""
    assert reads_exempt_source("grep", {"pattern": str(CORPUS)}) is False
    assert reads_exempt_source("grep", {"pattern": "redaction_shapes", "path": "tests/"}) is False
    assert reads_exempt_source("read", {"path": f"skill://guard-area/{CORPUS.name}"}) is False
    assert reads_exempt_source("read", {"path": f"spill://{CORPUS.name}"}) is False
    assert reads_exempt_source("web_read", {"url": f"https://example.invalid/{CORPUS}"}) is False


def test_a_path_that_only_resembles_the_exempt_one_is_refused() -> None:
    """Exact resolved equality, not a suffix or substring match."""
    decoys = (
        SHAPE_TABLE.parent / "notes" / SHAPE_TABLE.name,
        SHAPE_TABLE.parent.parent / "vendor" / SHAPE_TABLE.name,
        Path(str(SHAPE_TABLE) + ".bak"),
        SHAPE_TABLE.parent / "redaction_shapes.pyc",
    )
    for decoy in decoys:
        assert reads_exempt_source("read", {"path": str(decoy)}) is False, decoy


def test_a_call_without_a_usable_path_is_not_exempt() -> None:
    assert reads_exempt_source("read", None) is False
    assert reads_exempt_source("read", {}) is False
    assert reads_exempt_source("read", {"path": ""}) is False
    assert reads_exempt_source("read", {"path": 42}) is False
    assert reads_exempt_source("", {"path": str(CORPUS)}) is False


# --- the escalation gate, driven through the REAL loop ------------------------
#
# Everything below runs the actual decision path: the REAL ``read`` tool (so the
# arguments under test are the arguments the tool was handed), the REAL loop
# (``AgentLoop._append_results`` → ``_redact_content``, which is what publishes
# the call's identity), and the REAL host hook (``Session.
# _redact_tool_result_text``, a bound method exactly as the loop receives it).

#: The corpus's own ESCALATING case, DERIVED rather than spelled. This file must
#: not carry a credential-shaped literal of its own — the guard would have to
#: mask it while it was being written — and an arm that pinned one would go on
#: disagreeing with the corpus the day the corpus changed its mind.
ESCALATING_TEXT = next(case.text for case in POSITIVE_CASES if case.reason == "amqp DSN")

#: A registered session credential for the loud-direction arm. Deliberately NOT
#: credential-shaped: what is under test is that registering a value changes
#: nothing about whether the guard still escalates around it.
REGISTERED_VALUE = "REGISTERED-VALUE-2f1c9d"


class _OneCallStream:
    """One tool call, then a plain stop: the shortest road into ``_append_results``."""

    def __init__(self, name: str, arguments: str) -> None:
        self.tool_name = name
        self.arguments = arguments
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        turn: list[Any] = (
            [
                StreamToolCallDelta(
                    index=0, id="c1", name=self.tool_name, argument_delta=self.arguments
                )
            ]
            if len(self.requests) == 1
            else [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
        )

        async def gen() -> AsyncIterator[Any]:
            for event in turn:
                yield event

        return gen()


async def _never_streams(_request: Any, _signal: Any = None) -> AsyncIterator[Any]:
    """The Session's own provider stub: these arms never reach a stream."""

    if False:  # pragma: no cover - makes this an async generator
        yield None
    raise AssertionError("the guard-area arms never reach the provider stream")


def _session(tmp: Path) -> Session:
    """A real Session, built the way the session factory builds one."""
    return Session(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        stream_fn=_never_streams,
        tools=[],
        transcript=Transcript(tmp / "session"),
        system_blocks_provider=lambda *_a: [],
        yolo=True,
        cwd=str(tmp),
        variables=VariableStore(cwd=str(tmp)),
    )


def _read_tool() -> AgentTool:
    """The REAL ``read`` tool, not a stand-in for it."""
    return AgentTool(
        name="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=builtin.execute_read,
    )


def _echo_tool(name: str, text: str) -> AgentTool:
    """A tool that returns ``text``: the stand-in for a command's own output."""

    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any,
        on_update: Any,
        context: Any,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text=text)]
        )

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        execute=execute,
    )


async def _drive(session: Session, tool: AgentTool, stream: Any) -> list[str]:
    """Run one scripted turn and return the redacted tool rows' text."""
    config = LoopConfig(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        redact_tool_result=session._redact_tool_result_text,
    )
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("go")], LoopContext(tools=[tool]), config, None
        )
    ]
    end = events[-1]
    assert isinstance(end, AgentEndEvent), "the scripted turn did not end"
    rows = [m for m in end.messages if isinstance(m, Message) and m.role == "tool"]
    assert rows, "no tool row landed: the redaction path never ran"
    return [part.text for row in rows for part in row.content if isinstance(part, TextContent)]


def _escalation_flags(session: Session) -> list[bool]:
    return [flag for _tool, _labels, _summary, flag in session._pending_shape_incidents]


@pytest.mark.asyncio
async def test_a_read_of_the_guard_area_files_no_escalation(tmp_path: Path) -> None:
    """THE OPERATOR'S CASE. Reading the corpus masks, and asks for no rotation.

    Both halves are asserted on the row the model would actually see: the mask
    marker IS there (masking is not exempt), and the corpus's escalating case is
    NOT (the report that would have demanded a rotation is the thing that stops).

    One property is asserted rather than assumed — the read really carries the
    material that escalates — so this arm cannot pass by reading a summary that
    happened to contain nothing shaped.
    """
    corpus = CORPUS.read_text(encoding="utf-8")
    assert ESCALATING_TEXT in corpus, "the corpus no longer carries its escalating case"
    session = _session(tmp_path)
    # ``raw=True`` on purpose: a plain ``read`` of a ``.py`` file returns the
    # harness's structural summary with the bodies elided, and the guard's own
    # literals live in those bodies — so the raw read is the one that carries
    # them, and therefore the one this arm has to drive.
    stream = _OneCallStream("read", json.dumps({"path": str(CORPUS), "raw": True}))

    rows = await _drive(session, _read_tool(), stream)

    assert len(rows[0]) > 1000, "the read returned almost nothing; the arm proved nothing"
    assert REDACTION_MARKER in rows[0], "the corpus read was NOT masked"
    assert ESCALATING_TEXT not in rows[0], "the corpus's escalating case reached the model"
    assert _escalation_flags(session) == [], "a read of the guard's own area still escalated"


@pytest.mark.asyncio
async def test_a_bash_command_that_merely_names_the_path_still_escalates(tmp_path: Path) -> None:
    """THE SECURITY ARM. ``cat <the corpus>`` is not a read of it, for this rule.

    A command string is text: an agent (or a file the agent read) can write a
    filename into it, so a rule that honoured it would hand out the exemption for
    an echo. The command really does produce the corpus's bytes here, and the
    rotated demand still fires.
    """
    session = _session(tmp_path)
    command = f"cat {CORPUS}"
    stream = _OneCallStream("bash", json.dumps({"command": command}))

    rows = await _drive(session, _echo_tool("bash", CORPUS.read_text(encoding="utf-8")), stream)

    assert REDACTION_MARKER in rows[0], "the bash result was not masked"
    assert _escalation_flags(session) == [
        True
    ], "a bash command whose ARGUMENT names the guard's file conferred the exemption"
    assert session._pending_shape_incidents[0][0] == "bash"


@pytest.mark.asyncio
async def test_a_result_that_prints_the_path_inside_credential_text_still_escalates(
    tmp_path: Path,
) -> None:
    """THE OTHER SECURITY ARM: an OUTPUT naming the path confers nothing.

    The read is of an ordinary file; the file's text happens to print the corpus
    path on the line above escalating material. Nothing about that text is a
    decision input here — the decision was settled from the CALL's arguments.
    """
    session = _session(tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text(f"# the guard's own area: {CORPUS}\n{ESCALATING_TEXT}\n", encoding="utf-8")
    stream = _OneCallStream("read", json.dumps({"path": str(notes)}))

    rows = await _drive(session, _read_tool(), stream)

    assert REDACTION_MARKER in rows[0], "the result was not masked"
    assert _escalation_flags(session) == [
        True
    ], "a tool OUTPUT that printed the guard's path silenced an escalation"


@pytest.mark.asyncio
async def test_a_registered_credential_still_escalates_loudly_outside_the_guard_area(
    tmp_path: Path,
) -> None:
    """Registration changes nothing about escalation, and the demand is audible.

    Driven outside the exempt area with a registered session value in the same
    result: the value is masked (registration still works), and the rotated
    demand still reaches the JOURNAL — the operator-facing surface — rather than
    stopping at the queue.
    """
    session = _session(tmp_path)
    store = session.variables
    assert store.register_redaction(REGISTERED_VALUE) is True
    notes = tmp_path / "notes.txt"
    notes.write_text(f"REGISTER={REGISTERED_VALUE}\n{ESCALATING_TEXT}\n", encoding="utf-8")
    stream = _OneCallStream("read", json.dumps({"path": str(notes)}))

    rows = await _drive(session, _read_tool(), stream)
    await session._flush_shape_incidents()

    body = (tmp_path / "session" / "transcript.jsonl").read_text(encoding="utf-8")
    assert REGISTERED_VALUE not in rows[0], "a registered credential was not masked"
    assert ESCALATING_TEXT not in rows[0], "the escalating case reached the model"
    assert "rotate" in body, "the escalated notice no longer reaches the transcript"


def test_the_publication_is_per_call_and_resets() -> None:
    """Unset means ESCALATE: the honest default for a surface nobody published."""
    assert source_is_exempt() is False
    with exempt("read", {"path": str(CORPUS)}) as published:
        assert published is True
        assert source_is_exempt() is True
        # A nested call replaces the answer and restores it on exit, so an
        # unrelated tool reading nothing exempt cannot inherit the outer verdict.
        with exempt("bash", {"command": f"cat {CORPUS}"}) as inner:
            assert inner is False
            assert source_is_exempt() is False
        assert source_is_exempt() is True
    assert source_is_exempt() is False
