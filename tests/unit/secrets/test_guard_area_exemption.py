"""Arms for the escalation exemption of the guard's own source and corpus.

The module under test decides ONE thing — whether a call is a read of the guard's
own area — and these arms pin both halves of that decision. The positive half is
the operator's request; the negative half is the security requirement, and it is
the half that decides whether the exemption ships: an agent must not be able to
confer the exemption on itself by naming a path in a command, a pattern, or any
other text. Only a file-reading tool's own ``path`` argument may.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness import guard_area
from local_operator.harness.guard_area import EXEMPT_SOURCES, READING_TOOLS
from local_operator.harness.guard_area import exempt_from_escalation as exempt
from local_operator.harness.guard_area import reads_exempt_source, source_is_exempt
from local_operator.harness.loop import AgentLoop, LoopContext, _scrub_history_arguments
from local_operator.harness.redaction import current_tool_source, tool_source
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
    ToolCall,
    ToolContext,
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


def test_a_relative_spelling_follows_the_session_root_not_the_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ROUND-1 BLOCKER, at the matcher.

    The process CWD stands at the repo root, so the exempt spelling *would*
    resolve to a real exempt file if the process CWD were consulted. The
    SESSION root is somewhere else entirely, and that is the root the reader
    uses -- so the only correct answer is False, the escalating one. A matcher
    that answers True here exempts a read of a file the reader never opens; an
    agent-authored decoy at that spelling has its escalation suppressed.
    """
    monkeypatch.chdir(SHAPE_TABLE.parent.parent)
    session_root = tmp_path / "elsewhere"
    (session_root / "local_operator").mkdir(parents=True)
    # Sanity: the spelling resolves to a genuinely exempt file from THIS cwd,
    # so the arm cannot pass merely because the spelling is wrong.
    assert reads_exempt_source("read", {"path": "local_operator/redaction_shapes.py"}) is True
    assert (
        reads_exempt_source(
            "read", {"path": "local_operator/redaction_shapes.py"}, str(session_root)
        )
        is False
    ), "a relative spelling was resolved against the process CWD, not the reader's root"
    # The same spelling is still exempt -- and only -- when the session root IS
    # the tree the exempt files live in.
    assert (
        reads_exempt_source(
            "read", {"path": "local_operator/redaction_shapes.py"}, str(SHAPE_TABLE.parent.parent)
        )
        is True
    )


@pytest.mark.parametrize(
    "raw",
    (
        "~nosuchuser000/redaction_shapes.py",
        "notes\x00.txt",
        "notes\ud800.txt",
    ),
    ids=("unresolvable-tilde-user", "embedded-nul", "lone-surrogate"),
)
def test_a_malformed_path_fails_safe_instead_of_raising(raw: str, tmp_path: Path) -> None:
    """FAIL SAFE, never raise. This runs for EVERY read/grep result.

    The reader tolerates all three of these on purpose (``_resolve_workspace_path``
    catches ``RuntimeError`` from ``expanduser`` and ``(OSError, ValueError)``
    from ``resolve``), and the pre-PR path returned an ordinary result for them.
    A matcher that raises takes the whole turn down before ``AgentEndEvent``;
    the honest answer for a path that cannot be resolved is the ESCALATING one.
    """
    assert reads_exempt_source("read", {"path": raw}, str(tmp_path)) is False
    assert reads_exempt_source("grep", {"path": raw}, str(tmp_path)) is False


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


@pytest.mark.parametrize("tool", sorted(READING_TOOLS))
@pytest.mark.parametrize(
    "spelling",
    (" {path}", "{path} ", "  {path}  ", "\t{path}\n"),
    ids=("leading", "trailing", "both", "tab-and-newline"),
)
def test_a_whitespace_padded_exempt_spelling_is_exempt_for_both_readers(
    tool: str, spelling: str
) -> None:
    """R2-1. The guard and the READER must normalise a ``path`` identically.

    Both readers strip before resolving, so a padded spelling names the SAME
    file as the bare one and both must answer True. This is the arm that reds if
    either reader stops stripping: ``grep`` used to hand the resolver the raw
    argument, so a padded spelling matched the guard's stripped form while the
    reader opened the whitespace-bearing name — a different file, not in
    ``EXEMPT_SOURCES``, with the rotation demand suppressed.

    Parametrized over the WHOLE whitespace class, not one trailing space, because
    which whitespace Python's ``str.strip`` removes (leading, trailing, tabs,
    newlines) is the dimension a later "normalise with ``rstrip()``" would break.
    """
    for path in (SHAPE_TABLE, CORPUS, CORPUS_TESTS):
        assert (
            reads_exempt_source(tool, {"path": spelling.format(path=path)}) is True
        ), f"{tool} did not exempt the padded spelling of {path.name!r}"
    # The bare spelling is exempt too, so the arm cannot pass by the padded
    # form resolving somewhere else entirely.
    assert reads_exempt_source(tool, {"path": str(CORPUS)}) is True


@pytest.mark.parametrize("tool", sorted(READING_TOOLS))
def test_the_guard_and_the_readers_share_one_normalisation(tool: str) -> None:
    """THE STRUCTURAL PIN. One function, three callers, no second spelling.

    ``normalise_path_argument`` is the single normalisation a ``path`` argument
    gets, and it is what the guard and BOTH readers must call before resolving.
    Asserted from the parsed AST rather than from ``inspect.getsource``: both
    readers are wrapped by ``_guard`` (a bare closure, no ``functools.wraps``), so
    ``getsource`` hands back the wrapper and would pass whatever the inner
    function did -- the one shape where a source-text pin is worse than nothing.

    A reader that reverted to its own inline ``.strip()`` keeps the R2-1 behaviour
    arms green for as long as the two spellings happen to agree, and reintroduces
    the class the first time they do not. This reds on that revert directly.
    """
    assert builtin.normalise_path_argument("  a/b.py  ") == "a/b.py"
    assert builtin.normalise_path_argument("a/b.py") == "a/b.py"
    assert builtin.normalise_path_argument("\t\n") == ""
    assert builtin.normalise_path_argument("") == ""

    def calls_it(path: Path, func_name: str) -> bool:
        """True when ``func_name``'s body contains a call to the normaliser."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                return any(
                    isinstance(inner, ast.Name) and inner.id == "normalise_path_argument"
                    for inner in ast.walk(node)
                )
        raise AssertionError(f"{func_name} is gone from {path.name}")

    readers = Path(builtin.__file__)
    for name in ("execute_read", "execute_grep"):
        assert calls_it(readers, name), (
            f"{name} no longer routes its path through the shared normaliser "
            f"(the call under test was {tool})"
        )
    guard = Path(inspect.getfile(reads_exempt_source))
    assert calls_it(
        guard, "reads_exempt_source"
    ), "the guard no longer shares the readers' normalisation"


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


def _grep_tool() -> AgentTool:
    """The REAL ``grep`` tool, not a stand-in for it."""
    return AgentTool(
        name="grep",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        },
        execute=builtin.execute_grep,
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


async def _drive(
    session: Session, tool: AgentTool, stream: Any, session_cwd: str | None = None
) -> list[str]:
    """Run one scripted turn and return the redacted tool rows' text.

    ``session_cwd`` is the SESSION root the reader resolves a relative ``path``
    against (``ToolContext.cwd``). It is a parameter rather than always
    ``session._cwd`` because the dimension the round-1 blocker lived in is
    exactly when this root differs from the process CWD: an arm that never
    varies the two agrees with itself and proves nothing about that class.
    """
    return await _drive_tools(session, [tool], stream, session_cwd)


async def _drive_tools(
    session: Session, tools: list[AgentTool], stream: Any, session_cwd: str | None = None
) -> list[str]:
    """``_drive`` for a turn that offers MORE THAN ONE tool.

    The R3-2 arm needs a real two-call batch (``read`` plus ``bash``) and the
    R4-1 arm needs an ``eval`` beside the ``read`` it dispatches into; both are
    the same drive, so the body lives here and ``_drive`` is the one-tool call.
    """
    config = LoopConfig(
        model=ModelSpec(provider="test", model_id="unit-model", context_window=1000),
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        redact_tool_result=session._redact_tool_result_text,
    )
    tool_context = ToolContext(cwd=session_cwd) if session_cwd is not None else None
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("go")],
            LoopContext(tools=tools, tool_context=tool_context),
            config,
            None,
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
async def test_a_grep_scoped_to_the_guard_area_files_no_escalation(tmp_path: Path) -> None:
    """The second reading tool, over its own result path.

    ``grep`` is in the matcher for the same reason ``read`` is: its ``path``
    argument is the file whose lines it returns. The result here IS the corpus's
    matching lines, so this is the case that decides whether the exemption is
    about reading the file or merely about the ``read`` tool.
    """
    session = _session(tmp_path)
    grep = AgentTool(
        name="grep",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        },
        execute=builtin.execute_grep,
    )
    stream = _OneCallStream("grep", json.dumps({"pattern": "a", "path": str(CORPUS)}))

    rows = await _drive(session, grep, stream)

    # The lines really came back, and really carried shaped material, or this arm
    # would be asserting silence about nothing.
    assert f"{CORPUS.name}:" in rows[0] or REDACTION_MARKER in rows[0], "grep returned nothing"
    assert _escalation_flags(session) == [], "a grep scoped to the guard's area still escalated"


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


def test_publishing_a_call_identity_alone_confers_nothing() -> None:
    """The carrier that names the tool is NOT the carrier that exempts it.

    ``tool_source`` also wraps the scrub of a call's own ARGUMENTS, so if the
    exemption rode there, a credential typed into a second argument of a reading
    tool — a ``grep`` pattern, with the call scoped to the corpus to buy the
    exemption — would be laundered past the guard.
    """
    with tool_source("read", {"path": str(CORPUS)}):
        assert current_tool_source() == ("read", str(CORPUS))
        assert source_is_exempt() is False
    with exempt("read", {"path": str(CORPUS)}):
        assert source_is_exempt() is True


@pytest.mark.asyncio
async def test_a_credential_in_a_reading_tools_other_argument_still_escalates(
    tmp_path: Path,
) -> None:
    """The exemption covers the RESULT's bytes, never the call's argument text.

    A ``grep`` scoped to the corpus whose PATTERN is the credential it went to
    look for is a real call an agent can make, and the call's own arguments are
    what would reach the model there. It is reported: the argument scrub runs
    under ``tool_source`` and deliberately not under the exemption.
    """
    session = _session(tmp_path)
    message = Message(
        role="assistant",
        content=[TextContent(text="")],
        tool_calls=[
            ToolCall(
                id="c1",
                name="grep",
                arguments={"path": str(CORPUS), "pattern": ESCALATING_TEXT},
            )
        ],
    )

    _scrub_history_arguments(message, session._redact_tool_result_text)

    assert _escalation_flags(session) == [
        True
    ], "a credential in a reading tool's own argument was laundered past the guard"
    assert session._pending_shape_incidents[0][0] == "grep"


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


# --- the DECISIVE arm: process CWD and session root DISAGREE -------------------
#
# Every arm above leaves the two roots equal, which is exactly why the two-root
# union shipped: nothing here ever asked what happens when the root the reader
# uses is not the root the process stands in. These three arms are that question.
# The process CWD is chdir'd to the repo root (where the exempt spellings DO
# resolve to real exempt files); the session root is a plain temp dir, which is
# the default shape of an installed `lop` run launched outside the repo.


@pytest.mark.asyncio
async def test_a_decoy_at_the_exempt_spelling_still_escalates_from_another_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKER REGRESSION ARM. An agent-authored decoy must NOT be exempt.

    The session root holds ``local_operator/redaction_shapes.py`` -- a file the
    agent wrote, carrying the corpus's escalating case -- while the process CWD
    stands at the repo root, so the same relative spelling ALSO resolves to the
    real exempt file there. Before the fix the matcher took the process-CWD arm
    and returned True, so the read was masked with no rotate-it demand; the file
    read is the decoy, and the operator was never told. The absolute spelling of
    the SAME decoy escalated, which is what isolates the exemption (not the
    bytes) as the difference.
    """
    session_root = tmp_path / "session"
    (session_root / "local_operator").mkdir(parents=True)
    decoy = session_root / "local_operator" / "redaction_shapes.py"
    decoy.write_text(f"{ESCALATING_TEXT}\n", encoding="utf-8")
    monkeypatch.chdir(SHAPE_TABLE.parent.parent)  # the process CWD that used to win

    relative = _OneCallStream(
        "read", json.dumps({"path": "local_operator/redaction_shapes.py", "raw": True})
    )
    rows = await _drive(_session(tmp_path), _read_tool(), relative, str(session_root))

    # The decoy really was read, and really carried escalating material: without
    # this the arm could pass by reading something that never had a shape in it.
    assert ESCALATING_TEXT in decoy.read_text(encoding="utf-8")
    assert len(rows[0]) > 40, "the read returned almost nothing; the arm proved nothing"
    assert REDACTION_MARKER in rows[0], "masking is not exempt and still must apply"
    assert ESCALATING_TEXT not in rows[0], "the decoy's escalating case reached the model"

    session = _session(tmp_path)
    rows = await _drive(
        session,
        _read_tool(),
        _OneCallStream(
            "read", json.dumps({"path": "local_operator/redaction_shapes.py", "raw": True})
        ),
        str(session_root),
    )
    assert _escalation_flags(session) == [
        True
    ], "a relative spelling resolved against the process CWD exempted a decoy"
    assert len(rows[0]) > 40


@pytest.mark.asyncio
async def test_a_laundered_value_in_that_decoy_reaches_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE LAUNDERING CASE. ``write`` a real value into the decoy, ``read`` it relatively.

    The reviewer's sharpest vector: an agent writes an escalating value into a
    file it names with the exempt relative spelling and reads it back. Masking
    still applies (the value never reaches the model), but the rotation demand
    is the thing that tells the operator a real secret was handled -- and the
    union of roots is what silenced it. Asserted on all three surfaces the
    requirement names: the model's text, the pending queue, and the JOURNAL.
    """
    session_dir = tmp_path / "session"
    (session_dir / "local_operator").mkdir(parents=True)
    decoy = session_dir / "local_operator" / "redaction_shapes.py"
    decoy.write_text(f"PAYLOAD = {ESCALATING_TEXT!r}\n", encoding="utf-8")
    monkeypatch.chdir(SHAPE_TABLE.parent.parent)

    session = _session(tmp_path)
    stream = _OneCallStream(
        "read", json.dumps({"path": "local_operator/redaction_shapes.py", "raw": True})
    )
    rows = await _drive(session, _read_tool(), stream, str(session_dir))
    # Read the queue BEFORE flushing: ``_flush_shape_incidents`` drains it into
    # the journal, so an assertion on the flags afterwards would read an empty
    # list and pass for the wrong reason.
    flags = _escalation_flags(session)
    await session._flush_shape_incidents()

    body = (tmp_path / "session" / "transcript.jsonl").read_text(encoding="utf-8")
    assert ESCALATING_TEXT not in rows[0], "the laundered value reached the model"
    assert flags == [True], "the laundering vector was not escalated"
    assert "rotate" in body, "the demand never reached the journal"


# --- R2-1: the whitespace spelling ------------------------------------------
#
# Round 1 was a different ROOT. This is the same class one transform along: the
# guard and the reader disagreeing about what the ``path`` ARGUMENT means. Unlike
# round 1 it needs no unusual layout -- the session root resolving to the real
# exempt file is the whole precondition, which is what an ordinary checkout does
# -- so these arms set that up with a symlink rather than by writing an attacker
# file into the checkout they are reading.
#
# The defect: the guard resolved the STRIPPED argument while ``grep`` resolved the
# raw one. For ``path = <exempt spelling> + " "`` the guard decided about the real
# exempt file and exempted the call, while the reader opened the PADDED name --
# the agent's file, not in ``EXEMPT_SOURCES``. Masking still hid the value from
# the model; the rotation demand, the only thing the operator is told, was
# dropped, and nothing else reports it. The FIXED behaviour is that both readers
# strip, so a padded spelling opens the SAME file the guard's verdict is about,
# and the agent's padded name is not reachable through ``read``/``grep`` at all.

#: A token the agent writes INSIDE its file, so "the reader opened the agent's
#: file" is decidable from the result text. Deliberately not the grep PATTERN:
#: grep echoes the pattern back in its no-match row, so a marker used as the
#: pattern would read as "the file was opened" in exactly the case where it was
#: not. The pattern is a match-all ``.`` instead, which puts the whole (two-line)
#: file -- marker and escalating case together -- into the result.
ATTACKER_BODY = "ATTACKER-BODY-9f3a17"
MATCH_ALL = "."
DECOY_BODY = f"{ATTACKER_BODY} = 1\n{ESCALATING_TEXT!r}\n"


def _grep_args(spelling: str) -> dict[str, str]:
    return {"pattern": MATCH_ALL, "path": spelling}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("read", "grep"))
async def test_a_padded_spelling_opens_the_exempt_file_not_a_file_beside_it(
    tmp_path: Path, tool: str
) -> None:
    """R2-1 BLOCKER ARM: the padded spelling must open the file the guard judged.

    The session root holds ``local_operator/redaction_shapes.py`` -- a SYMLINK to
    the real exempt file, so the stripped spelling resolves to the guard's own
    source exactly as it does in an ordinary checkout -- beside an agent-authored
    file at that same spelling plus a trailing space, carrying ``ATTACKER_BODY``.

    FIXED behaviour, both readers: the padded argument strips, opens the real
    exempt file, is exempt, and the agent's file is never read -- so ``ATTACKER_BODY``
    cannot appear in the result. The regression this reds on: ``grep`` resolves the
    raw argument, opens the padded name, and returns the AGENT's bytes while the
    guard's verdict (from the stripped name) still says "the guard's own source" --
    the marker lands in the model's text from a file the guard never exempted.
    """
    session_dir = tmp_path / "session"
    (session_dir / "local_operator").mkdir(parents=True)
    # The exempt file, reached the way the guard's stripped spelling reaches it.
    (session_dir / "local_operator" / "redaction_shapes.py").symlink_to(SHAPE_TABLE)
    attacker = session_dir / "local_operator" / "redaction_shapes.py "
    attacker.write_text(DECOY_BODY, encoding="utf-8")
    assert attacker.is_file(), "the padded attacker file was not created"

    session = _session(session_dir)
    padded = "local_operator/redaction_shapes.py "
    args = {"path": padded, "raw": True} if tool == "read" else _grep_args(padded)
    rows = await _drive(
        session,
        _read_tool() if tool == "read" else _grep_tool(),
        _OneCallStream(tool, json.dumps(args)),
        str(session_dir),
    )

    assert rows, f"no {tool} row: the turn did not complete"
    assert ATTACKER_BODY not in rows[0], (
        f"{tool} opened the PADDED name (the agent's file) while the guard's verdict "
        "came from the stripped one -- the whitespace spelling is the whole bypass"
    )
    assert _escalation_flags(session) == [], (
        f"{tool} did not exempt a read of the real exempt file reached through a " "padded spelling"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ("read", "grep"))
async def test_the_padded_and_clean_spellings_both_escalate_for_a_decoy(
    tmp_path: Path, tool: str
) -> None:
    """The isolating control, BOTH ways: identical bytes, differing only in the name.

    A plain session root whose ``local_operator/redaction_shapes.py`` is an
    agent-authored decoy carrying the corpus's escalating case. Two spellings of
    that SAME decoy -- the clean name and the name plus a trailing space -- must
    both open it and both escalate: it is not in ``EXEMPT_SOURCES``, and stripping
    the padding is exactly what makes the reader agree with the guard about which
    file that is. Under the defect ``grep`` resolved the padded spelling raw, found
    no such file, and answered an ordinary missing-path row with no demand at all,
    while the clean spelling escalated -- the same bytes decided two ways.
    """
    session_dir = tmp_path / "session"
    (session_dir / "local_operator").mkdir(parents=True)
    (session_dir / "local_operator" / "redaction_shapes.py").write_text(
        DECOY_BODY, encoding="utf-8"
    )

    seen: dict[str, list[bool]] = {}
    for label, spelling in (
        ("clean", "local_operator/redaction_shapes.py"),
        ("padded", "local_operator/redaction_shapes.py "),
    ):
        session = _session(session_dir)
        args = {"path": spelling, "raw": True} if tool == "read" else _grep_args(spelling)
        rows = await _drive(
            session,
            _read_tool() if tool == "read" else _grep_tool(),
            _OneCallStream(tool, json.dumps(args)),
            str(session_dir),
        )
        seen[label] = _escalation_flags(session)
        # The decoy really was the file opened, in both spellings: otherwise the
        # escalation below could be silence about a file nobody read.
        assert ATTACKER_BODY in rows[0], f"{tool} did not open the decoy for the {label!r} spelling"

    assert seen["clean"] == [True], "the clean control did not escalate; it isolates nothing"
    assert seen["padded"] == [True], (
        f"{tool} answered the same decoy differently on a trailing space "
        f"(padded={seen['padded']}, clean={seen['clean']})"
    )


@pytest.mark.parametrize("tool", sorted(READING_TOOLS))
def test_the_padded_and_bare_spellings_agree_for_the_real_exempt_file(tool: str) -> None:
    """The positive half: padding must not LOSE the operator's own exemption either.

    A fix that over-corrected -- refusing every padded spelling -- would silence
    the request the exemption exists for. The whole whitespace class, on purpose:
    which characters ``str.strip`` removes (leading, trailing, tab, newline) is the
    dimension a later partial normalisation would break.
    """
    bare = str(CORPUS)
    assert reads_exempt_source(tool, {"path": bare}) is True
    for padded in (f"{bare} ", f" {bare}", f"\t{bare}\n", f"  {bare}  "):
        assert (
            reads_exempt_source(tool, {"path": padded}) is True
        ), f"{tool} exempted the bare spelling but not {padded!r}"


@pytest.mark.asyncio
async def test_a_malformed_path_returns_an_ordinary_result_and_the_turn_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE SCRIPTED TURN THAT CRASHED, re-run. Not the unit call -- the turn.

    ``reads_exempt_source`` runs for EVERY read/grep result, so a raise there
    aborted the turn before ``AgentEndEvent``. Each of the three inputs the two
    streams named is driven through the real ``read`` tool here, and the turn is
    required to END with an ordinary error row. The lone surrogate additionally
    exercises ``_error_batch_fingerprint``, which digests model-visible error
    text and raised on the same input once the guard no longer did.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    monkeypatch.chdir(session_dir)
    for raw in ("~nosuchuser000/redaction_shapes.py", "notes\x00.txt", "notes\ud800.txt"):
        session = _session(tmp_path)
        stream = _OneCallStream("read", json.dumps({"path": raw}))
        rows = await _drive(session, _read_tool(), stream, str(session_dir))
        assert rows, f"no tool row for {raw!r}: the turn did not complete"
        assert (
            "Does not exist" in rows[0] or "does not exist" in rows[0]
        ), f"expected the ordinary missing-path result for {raw!r}, got {rows[0][:120]!r}"


# --- R3-1 / R3-2 / R4-1: the remediation's own regression arms -----------------
#
# Round 3 blocked this PR on two defects and the remediation that closed them
# shipped WITHOUT an arm for either: rebinding both fixes off in memory left this
# whole module green (review round 4, R4-2), which is the shape AGENTS.md warns
# about -- a guard that cannot observe the defect it names. These arms are the
# pins, one per class, plus the second site the same class survived on:
#
#   * the redaction hook must be handed the ORIGINAL arguments; the stored copy in
#     ``context.messages`` is the scrubbed one (R3-1);
#   * the guard must CONSUME the verdict recorded at dispatch and never re-resolve
#     it at redaction time -- on the native path (R3-2) and on the eval bridge
#     (R4-1).
#
# Every arm asserts its own premise (the reader really opened the agent's file,
# the retarget really would have flipped a fresh resolution), so none can pass by
# reading something that never carried a shape, or by a mutation that changed
# nothing. None of them writes anything into this checkout: ``EXEMPT_SOURCES`` is
# rebound to a temp file, which is also what keeps each arm's "exempt" side a real
# resolved path rather than a spelling.

#: A bare credential-shaped token the shape pass replaces WHOLESALE, DERIVED from
#: the corpus rather than spelled here. It is the path SEGMENT in the R3-1 arm,
#: and that arm's premise is that scrubbing it rewrites the spelling the guard
#: would otherwise resolve.
SHAPED_SEGMENT = next(
    case.text for case in POSITIVE_CASES if case.reason == "a GitHub personal access token"
)


class _BatchStream:
    """Several tool calls in ONE assistant turn -- i.e. ONE batch.

    ``_OneCallStream`` cannot express the R3-2 attack: what makes it work is that
    a batch's calls are gathered together before any result is appended, so a
    sibling's mutation always lands before the redaction.
    """

    def __init__(self, calls: list[tuple[str, str, str]]) -> None:
        self.calls = calls
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        turn: list[Any] = (
            [
                StreamToolCallDelta(index=index, id=call_id, name=name, argument_delta=arguments)
                for index, (call_id, name, arguments) in enumerate(self.calls)
            ]
            if len(self.requests) == 1
            else [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]
        )

        async def gen() -> AsyncIterator[Any]:
            for event in turn:
                yield event

        return gen()


def _bash_tool() -> AgentTool:
    """The REAL ``bash`` tool -- the only thing that can move a link mid-batch."""
    return AgentTool(
        name="bash",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
        execute=builtin.execute_bash,
    )


def _shaped_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """The R3-1 layout, and the spelling that makes one argument name two files.

    ``<shaped>/../local_operator/redaction_shapes.py`` resolves THROUGH the
    symlink to the agent's decoy, while the scrubbed spelling
    ``[redacted]/../local_operator/redaction_shapes.py`` cancels lexically to the
    exempt file. One argument, two files: the whole defect.
    """
    root = tmp_path / "root"
    exempt = root / "local_operator" / "redaction_shapes.py"
    exempt.parent.mkdir(parents=True)
    exempt.write_text("# the exempt file\n", encoding="utf-8")
    (root / "a" / "b" / "c").mkdir(parents=True)
    decoy = root / "a" / "b" / "local_operator" / "redaction_shapes.py"
    decoy.parent.mkdir(parents=True)
    decoy.write_text(f"{ATTACKER_BODY}\nPAYLOAD = {ESCALATING_TEXT!r}\n", encoding="utf-8")
    (root / SHAPED_SEGMENT).symlink_to(root / "a" / "b" / "c")
    monkeypatch.setattr(guard_area, "EXEMPT_SOURCES", frozenset({exempt.resolve()}))
    return root, f"{SHAPED_SEGMENT}/../local_operator/redaction_shapes.py"


def _link_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    """The R3-2 layout: a link the caller can retarget, and an exempt tree beside it."""
    root = tmp_path / "root"
    exempt_dir = root / "exemptdir"
    exempt = exempt_dir / "local_operator" / "redaction_shapes.py"
    exempt.parent.mkdir(parents=True)
    exempt.write_text("# the exempt file\n", encoding="utf-8")
    decoy_dir = root / "decoydir" / "local_operator"
    decoy_dir.mkdir(parents=True)
    (decoy_dir / "redaction_shapes.py").write_text(
        f"{ATTACKER_BODY}\nPAYLOAD = {ESCALATING_TEXT!r}\n", encoding="utf-8"
    )
    (root / "lnk").symlink_to(root / "decoydir")
    monkeypatch.setattr(guard_area, "EXEMPT_SOURCES", frozenset({exempt.resolve()}))
    return root, exempt_dir, "lnk/local_operator/redaction_shapes.py"


@pytest.mark.parametrize("tool", sorted(READING_TOOLS))
@pytest.mark.asyncio
async def test_the_hook_is_handed_the_original_arguments_not_the_scrubbed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """R3-1 REGRESSION ARM, both readers. The hook's input is the CALLER's string.

    ``_call_arguments`` used to recover a call's arguments from
    ``context.messages``, which holds the SCRUBBED assistant turn. A
    credential-SHAPED path segment was therefore rewritten before the hook saw
    it, and every consumer of that string -- the guard's fallback resolution, and
    the ``tool_source`` identity a notice is built from -- described a different
    call than the one that ran. The arm asserts the property the fix restores,
    on the arguments the hook is actually handed.

    Reds on the pre-remediation code: the hook then receives the masked spelling,
    with ``SHAPED_SEGMENT`` already gone (measured: rebinding
    ``_call_arguments`` back to the ``messages`` search turns this arm red).
    """
    root, relative = _shaped_layout(tmp_path, monkeypatch)
    # The premise, stated as two readings: the shaped spelling resolves to the
    # DECOY, and the scrubbed spelling -- the same string after the shape pass --
    # resolves to the exempt file. Without both, the arm would be about nothing.
    assert reads_exempt_source(tool, {"path": relative}, str(root)) is False
    assert (
        reads_exempt_source(
            tool, {"path": relative.replace(SHAPED_SEGMENT, REDACTION_MARKER)}, str(root)
        )
        is True
    )

    session = _session(tmp_path)
    seen: list[dict[str, Any]] = []
    original = AgentLoop._redact_content

    async def spy(
        content: list[Any],
        redact: Any,
        tool_name: str,
        arguments: Any,
        session_cwd: str | None = None,
        exempt_verdict: bool | None = None,
    ) -> list[Any]:
        seen.append(dict(arguments or {}))
        return await original(content, redact, tool_name, arguments, session_cwd, exempt_verdict)

    monkeypatch.setattr(AgentLoop, "_redact_content", staticmethod(spy))
    args = {"path": relative, "raw": True} if tool == "read" else {"pattern": ".", "path": relative}
    rows = await _drive_tools(
        session,
        [_read_tool() if tool == "read" else _grep_tool()],
        _BatchStream([("c1", tool, json.dumps(args))]),
        str(root),
    )

    assert seen, "the redaction hook never ran"
    assert (
        seen[0].get("path") == relative
    ), f"{tool}: the hook was handed a scrubbed spelling, not the one the reader resolved"
    assert ATTACKER_BODY in rows[0], f"{tool} did not open the agent's file"
    assert REDACTION_MARKER in rows[0], "masking is not exempt and must still apply"
    assert _escalation_flags(session) == [True], f"{tool}: the decoy read did not escalate"


@pytest.mark.asyncio
async def test_a_batch_that_retargets_the_read_path_still_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-2 REGRESSION ARM. A same-batch mutation must not decide the verdict.

    A batch GATHERS its results before any are appended, so a ``read`` plus a
    ``bash ln -sfn`` in ONE batch always lands the mutation before the redaction.
    A guard that resolved the exemption at redaction time therefore decided about
    a file the reader never opened -- clearing the rotation demand for the
    agent's own bytes, 3/3 deterministic on the pre-remediation head (QA round 3).

    The read is FIRST in the batch, so its verdict is recorded at dispatch --
    before the mutation -- and the reader opens the decoy. Reds when the recorded
    verdict is not consumed: the redaction-time resolution then sees the
    retargeted link (the exempt file) and answers exempt.
    """
    root, exempt_dir, relative = _link_layout(tmp_path, monkeypatch)
    link = root / "lnk"
    session = _session(tmp_path)

    rows = await _drive_tools(
        session,
        [_read_tool(), _bash_tool()],
        _BatchStream(
            [
                ("c1", "read", json.dumps({"path": relative, "raw": True})),
                ("c2", "bash", json.dumps({"command": f"ln -sfn {exempt_dir} {link}"})),
            ]
        ),
        str(root),
    )

    assert ATTACKER_BODY in rows[0], "the reader did not open the agent's file"
    assert link.resolve() == exempt_dir.resolve(), "the batch never retargeted the link"
    assert _escalation_flags(session) == [
        True
    ], "a same-batch mutation cleared the escalation for the bytes the reader read"


@pytest.mark.asyncio
async def test_the_eval_bridge_consumes_the_verdict_recorded_at_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4-1 REGRESSION ARM: the same class, on the eval bridge.

    The bridge redacts a nested call's result with ``planned.args`` and used to
    leave the guard to re-resolve the exemption at THAT moment -- the R3-2 defect
    one surface along, and reachable without a batch (measured 7 of 30 attempts
    clearing the escalation for a NON-exempt file, review round 4).

    The mutation WINDOW is injected rather than raced: a wrapper around
    ``_redact_content`` retargets the link exactly between the nested READER's
    resolution and that redaction, and the arm asserts the retarget really would
    have flipped a fresh resolution (the isolating control) while the escalation
    for the agent's file still fires. Deterministic on purpose -- a raced flipper
    would make this arm the flaky thing it exists to prevent.

    Reds while the bridge passes no verdict: the guard then re-resolves, finds
    the exempt tree the retarget installed, and files nothing.
    """
    root, exempt_dir, relative = _link_layout(tmp_path, monkeypatch)
    link = root / "lnk"
    session = _session(tmp_path)
    nested_text: list[str] = []
    control: list[bool] = []
    original = AgentLoop._redact_content

    async def retarget_at_the_window(
        content: list[Any],
        redact: Any,
        tool_name: str,
        arguments: Any,
        session_cwd: str | None = None,
        exempt_verdict: bool | None = None,
    ) -> list[Any]:
        if tool_name == "read" and not control:
            link.unlink()
            link.symlink_to(exempt_dir)
            control.append(reads_exempt_source("read", {"path": relative}, str(root)))
        return await original(content, redact, tool_name, arguments, session_cwd, exempt_verdict)

    monkeypatch.setattr(AgentLoop, "_redact_content", staticmethod(retarget_at_the_window))

    async def execute(
        tool_call_id: str, args: dict[str, Any], signal: Any, on_update: Any, context: Any
    ) -> ToolResult:
        nested = await context.dispatch_tool("read", {"path": relative, "raw": True})
        nested_text.append("".join(part.get("text", "") for part in nested.get("content", [])))
        # A fixed, shape-free row: what is under test is the NESTED call's
        # escalation, so the outer row must contribute none of its own.
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="eval",
            content=[TextContent(text="nested dispatch done")],
        )

    rows = await _drive_tools(
        session,
        [
            AgentTool(
                name="eval", parameters={"type": "object", "properties": {}}, execute=execute
            ),
            _read_tool(),
        ],
        _BatchStream([("c1", "eval", json.dumps({}))]),
        str(root),
    )

    assert (
        nested_text and ATTACKER_BODY in nested_text[0]
    ), "the nested reader did not open the agent's file"
    assert control == [True], "the retarget did not move the spelling onto the exempt file"
    assert rows == ["nested dispatch done"], "the outer row is not the one under test"
    assert _escalation_flags(session) == [
        True
    ], "the bridge let a mutation between the reader and the redaction decide the verdict"


def test_a_handed_in_verdict_is_consumed_and_never_re_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE SEAM both fixes rest on: an answer in hand means no resolution at all.

    The arms above assert OUTCOMES; this one asserts the property those outcomes
    depend on, and it reds if ``exempt_from_escalation`` starts resolving when it
    was handed a verdict -- the shape a later "optimisation" would take. The
    second half is the guard against over-correcting: with NO verdict the
    fallback must still resolve, or a never-dispatched call would answer False
    for a genuine read of the exempt file.
    """
    calls: list[str] = []
    original = guard_area._resolve_workspace_path

    def counting(raw: str, cwd: str) -> tuple[Path, bool, bool]:
        calls.append(raw)
        return original(raw, cwd)

    monkeypatch.setattr(guard_area, "_resolve_workspace_path", counting)
    with guard_area.exempt_from_escalation("read", {"path": "a/b.py"}, ".", True):
        assert source_is_exempt() is True
    assert calls == [], "a handed-in verdict was re-resolved anyway"
    with guard_area.exempt_from_escalation("read", {"path": "a/b.py"}, "."):
        assert source_is_exempt() is False
    assert calls == ["a/b.py"], "the no-verdict fallback stopped resolving"
