"""Tool-call fault classification, measured through the REAL AgentLoop.

The rate arithmetic is trivial; the CLASSIFICATION is the claim, so every case
here drives an actual turn and asserts what the loop reported through
``LoopConfig.record_tool_call`` rather than unit-testing a helper.

The split that matters is whether a fault is the MODEL's: only those three
classes feed the benchmarking figure, and a misclassification would either
inflate the number (crediting a denied call) or defame the model (blaming it
for an HTTP 500). Each class is therefore pinned at its own source.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)

MODEL = ModelSpec(provider="test", model_id="m")


class _Scripted:
    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.n = 0

    def __call__(self, request, signal: AbortSignal | None):
        turn = self.turns[self.n]
        self.n += 1

        async def gen():
            for event in turn:
                yield event

        return gen()


def _ok_tool(
    name: str = "read", concurrency: Literal["shared", "exclusive"] = "shared"
) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(
        name=name,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        concurrency=concurrency,
        execute=execute,
    )


def _raising_tool(name: str = "web_fetch") -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        raise RuntimeError("HTTP 500 from upstream")

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"url": {"type": "string"}}},
        execute=execute,
    )


def _calls(*specs: tuple[int, str, str, str]) -> list[StreamEvent]:
    out: list[StreamEvent] = []
    for index, cid, name, args in specs:
        out.append(StreamToolCallDelta(index=index, id=cid, name=name, argument_delta=""))
        out.append(StreamToolCallDelta(index=index, id=cid, name=None, argument_delta=args))
    return out


async def _run(
    turn: list[StreamEvent], tools: list[AgentTool], **context_kwargs: Any
) -> list[tuple[str, str, str, float]]:
    recorded: list[tuple[str, str, str, float]] = []
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted([turn, [StreamEndEvent(stop_reason="stop")]]),
        record_tool_call=lambda name, origin, fault, ms: recorded.append((name, origin, fault, ms)),
    )
    context = LoopContext(
        system_blocks=["sys"],
        tools=tools,
        tool_context=ToolContext(session_id="s1", **context_kwargs),
    )
    async for _ in AgentLoop().run([], context, config):
        pass
    return recorded


def _faults(recorded) -> dict[str, str]:
    return {name: fault for name, _origin, fault, _ms in recorded}


@pytest.mark.asyncio
async def test_a_clean_call_records_no_fault():
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    assert recorded == [("read", "model", "", recorded[0][3])]
    assert recorded[0][3] >= 0, "a dispatched call reports its duration"


@pytest.mark.asyncio
async def test_a_hallucinated_tool_name_is_a_model_fault():
    """The model named a tool that does not exist. Its NAME is retained."""
    recorded = await _run(
        _calls((0, "c1", "reed_file", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    assert _faults(recorded) == {"reed_file": "unknown_tool"}


@pytest.mark.asyncio
async def test_a_schema_violation_is_a_model_fault():
    """``path`` is declared a string; the model sent an int."""
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":123}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    assert _faults(recorded) == {"read": "invalid_arguments"}


@pytest.mark.asyncio
async def test_a_duplicate_call_id_is_a_model_fault():
    """One id emitted twice: the first wins, the twin is a malformed emission.

    Classified by its ``details`` MARKER and not by its text \u2014 a duplicate is
    parked with ``tool is None`` exactly like an unknown tool, so the two are
    indistinguishable structurally at ``park``.
    """
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}'), (1, "c1", "read", '{"path":"b"}'))
        + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    faults = sorted(fault for _n, _o, fault, _ms in recorded)
    assert faults == ["", "duplicate_id"]


@pytest.mark.asyncio
async def test_a_tool_that_raises_is_an_execution_error_not_a_model_fault():
    """HTTP 500 is the world failing, not the model emitting a bad call."""
    recorded = await _run(
        _calls((0, "c1", "web_fetch", '{"url":"x"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_raising_tool()],
    )
    assert _faults(recorded) == {"web_fetch": "execution"}


@pytest.mark.asyncio
async def test_a_denied_call_is_recorded_but_is_not_the_model_s_fault():
    """The user's decision. Recorded so counts reconcile, excluded from both rates."""

    async def deny(tool_name, summary, job_id):
        return False

    tool = _ok_tool()
    tool.approval_tier = "write"
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [tool],
        request_approval=deny,
    )
    assert _faults(recorded) == {"read": "denied"}


@pytest.mark.asyncio
async def test_an_approval_gate_crash_is_our_fault_not_the_model_s():
    async def explode(tool_name, summary, job_id):
        raise RuntimeError("gate is broken")

    tool = _ok_tool()
    tool.approval_tier = "write"
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [tool],
        request_approval=explode,
    )
    assert _faults(recorded) == {"read": "gate_failed"}


@pytest.mark.asyncio
async def test_the_nested_eval_bridge_records_with_its_own_origin():
    """``dispatch_tool`` never reaches ``park`` and would otherwise be uncounted.

    Missing it would silently under-count every eval-driven tool call \u2014 exactly
    the composition-heavy runs the accuracy figure is most wanted for. ``origin``
    separates them because a nested call is the model's CODE calling a tool, not
    the model emitting one, and conflating the two would let a scripted retry
    loop dominate the figure.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        await context.dispatch_tool("read", {"path": "a"})
        await context.dispatch_tool("nope_tool", {"path": "a"})
        await context.dispatch_tool("read", {"path": 123})
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="eval", content=[TextContent(text="done")]
        )

    eval_tool = AgentTool(
        name="eval",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        execute=execute,
    )
    recorded = await _run(
        _calls((0, "c1", "eval", '{"code":"x"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool(), eval_tool],
    )
    nested = [(n, f) for n, o, f, _ms in recorded if o == "nested"]
    assert nested == [("read", ""), ("nope_tool", "unknown_tool"), ("read", "invalid_arguments")]
    # The outer eval call itself is a MODEL call and recorded as one.
    assert ("eval", "model", "") in [(n, o, f) for n, o, f, _ms in recorded]


@pytest.mark.asyncio
async def test_a_raising_callback_cannot_break_a_turn():
    """Analytics must never raise into a turn \u2014 the package's stated contract.

    A host hook that throws is a host bug, and the loop still has to finish the
    turn and pair every tool result. Guarded in the loop rather than trusted to
    the host, because a measurement that can kill the thing it measures is not
    a measurement.
    """
    calls: list[str] = []

    def explode(name, origin, fault, ms):
        calls.append(name)
        raise RuntimeError("analytics is broken")

    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted(
            [
                _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
                [StreamEndEvent(stop_reason="stop")],
            ]
        ),
        record_tool_call=explode,
    )
    context = LoopContext(
        system_blocks=["sys"], tools=[_ok_tool()], tool_context=ToolContext(session_id="s1")
    )
    results = [event async for event in AgentLoop().run([], context, config)]
    assert calls == ["read"], "the hook was reached"
    assert results, "the turn completed despite the hook raising"


@pytest.mark.asyncio
async def test_no_callback_configured_is_a_silent_no_op():
    """A host that does not want analytics pays nothing and sees no error."""
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted(
            [
                _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
                [StreamEndEvent(stop_reason="stop")],
            ]
        ),
    )
    context = LoopContext(
        system_blocks=["sys"], tools=[_ok_tool()], tool_context=ToolContext(session_id="s1")
    )
    assert [event async for event in AgentLoop().run([], context, config)]


@pytest.mark.asyncio
async def test_a_session_less_context_records_nothing():
    """No session id means no row to attribute: recording it would be a lie."""
    recorded: list[Any] = []
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted(
            [
                _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
                [StreamEndEvent(stop_reason="stop")],
            ]
        ),
        record_tool_call=lambda *a: recorded.append(a),
    )
    context = LoopContext(
        system_blocks=["sys"], tools=[_ok_tool()], tool_context=ToolContext(session_id="")
    )
    async for _ in AgentLoop().run([], context, config):
        pass
    assert recorded == []


@pytest.mark.asyncio
async def test_the_origins_the_loop_emits_are_exactly_the_ones_the_reader_partitions_on():
    """Pins the writer to the reader across a deliberate layering gap.

    ``harness`` carries no analytics dependency by design (see
    ``LoopConfig.record_tool_call``), so the loop passes these strings as
    literals while ``analytics.model`` names them as constants. Nothing but this
    test stops the two sides drifting on a spelling — and a drift would not
    fail, it would silently move every model-emitted call into the nested
    bucket, emptying the benchmarking figure's denominator instead of raising.
    """
    from local_operator.analytics.model import ORIGIN_MODEL, ORIGIN_NESTED

    async def execute(tool_call_id, args, signal, on_update, context):
        await context.dispatch_tool("read", {"path": "a"})
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="eval", content=[TextContent(text="done")]
        )

    eval_tool = AgentTool(
        name="eval",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        execute=execute,
    )
    recorded = await _run(
        _calls((0, "c1", "eval", '{"code":"x"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool(), eval_tool],
    )
    emitted_origins = {origin for _n, origin, _f, _ms in recorded}
    assert emitted_origins == {ORIGIN_MODEL, ORIGIN_NESTED}


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_fault", ["denied", "gate_failed"])
async def test_shipped_eval_nested_exclusion_is_not_a_failed_tool_call(
    tmp_path, monkeypatch, nested_fault
):
    """Exercise the shipped exec-tier eval subprocess, bridge, store and screen.

    Approve eval itself, then deny (or break approval for) its nested write.
    The successful read afterwards proves the kernel/bridge continued rather
    than a denied outer eval accidentally making this a vacuous assertion.
    """
    import json
    import os

    from local_operator.analytics.store import AnalyticsStore
    from local_operator.tools import eval as eval_tool
    from local_operator.tui.widgets.session_panel import build_session_report
    from tests.unit.analytics.test_store import _snap
    from tests.unit.tui.test_session_panel import _section, runtime

    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    approvals = []

    async def approve(tool_name, summary, job_id):
        approvals.append(tool_name)
        if tool_name == "eval":
            return True
        if nested_fault == "gate_failed":
            raise RuntimeError("synthetic approval failure")
        return False

    write = _ok_tool("write_file")
    write.approval_tier = "write"
    read = _ok_tool()
    read.approval_tier = "read"
    ev = eval_tool.build_eval_tool()
    assert ev.approval_tier == "exec"
    args = json.dumps({"code": 'tool("write_file", path="a")\ntool("read", path="b")'})
    try:
        recorded = await _run(
            _calls((0, "c1", "eval", args)) + [StreamEndEvent(stop_reason="toolUse")],
            [ev, write, read],
            cwd=str(tmp_path),
            request_approval=approve,
        )
    finally:
        await eval_tool.close_session_kernel("s1")
    assert approvals == ["eval", "write_file"], recorded
    assert [(n, o, f) for n, o, f, _ in recorded] == [
        ("write_file", "nested", nested_fault),
        ("read", "nested", ""),
        ("eval", "model", ""),
    ]
    store = AnalyticsStore(tmp_path / "nested.db")
    try:
        store.record_batch([_snap(session_id="s1")])
        store.record_tool_calls(
            [(i, "s1", n, o, f, ms) for i, (n, o, f, ms) in enumerate(recorded)]
        )
        report = store.session_report("s1")
        stats = report.tool_calls
        assert stats is not None
        assert (stats.recorded, stats.ok + stats.nested_ok, stats.all_excluded) == (3, 2, 1)
        assert stats.excluded == 0 and stats.nested_excluded == 1
        assert stats.emitted == 1 and stats.tool_call_error_rate == 0
        for width in (40, 58, 100):
            rows = _section(build_session_report(report, runtime(), width).plain, "Tool surface")
            headline = next(row for row in rows if "Tool calls" in row)
            assert "failed" not in headline
            if width == 100:
                assert "2 ok" in headline
            nested = next(row for row in rows if "└ excluded" in row)
            assert nested.split("excluded", 1)[1].strip().startswith("1")
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The classification BOUNDARY: malformed argument vs unsatisfiable request.
#
# Both arrive at the same generic handler as an is_error result from a tool
# body, and before the marker existed both recorded as `execution` — which is
# how a session with four model-caused invalid calls rendered a 0.0% error
# rate. These pin the boundary in BOTH directions, because the permissive
# direction (calling an execution error a model fault) inflates a published
# benchmark and is the worse failure.
# ---------------------------------------------------------------------------


def _shipped_read_tool(tmp_path) -> AgentTool:
    """The REAL shipped ``read``, not a stand-in: the claim under test is that
    its own argument parsing is classified correctly."""
    from local_operator.tools.registry import create_tools

    context = ToolContext(cwd=str(tmp_path), session_id="s1")
    return next(t for t in create_tools(context) if t.name == "read")


@pytest.mark.asyncio
async def test_a_malformed_range_is_a_model_fault_not_an_execution_error(tmp_path):
    """The operator's exact repro: ``range='"270-330"'`` — a line range with
    literal quote characters embedded.

    ``range`` is typed ``str | None``, so this passes the loop's schema check
    and fails inside the tool body. It is unambiguously the model emitting an
    argument the tool cannot use, and recording it as ``execution`` is what
    laundered it out of the accuracy figure.
    """
    (tmp_path / "a.txt").write_text("l1\nl2\nl3\n")
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a.txt","range":"\\"270-330\\""}'))
        + [StreamEndEvent(stop_reason="toolUse")],
        [_shipped_read_tool(tmp_path)],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"read": "invalid_arguments"}


@pytest.mark.asyncio
async def test_a_wellformed_range_on_a_missing_file_stays_an_execution_error(tmp_path):
    """The regression that matters most. A path that does not exist is
    UNSATISFIABLE, not malformed — the file may have vanished after the model
    planned the call — so it must NOT be credited to the model."""
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"ghost.txt","range":"1-5"}'))
        + [StreamEndEvent(stop_reason="toolUse")],
        [_shipped_read_tool(tmp_path)],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"read": "execution"}


@pytest.mark.asyncio
async def test_a_malformed_regex_is_a_model_fault_but_a_missing_path_is_not(tmp_path):
    """``grep``'s two rejections sit three lines apart in the same function and
    fall on opposite sides of the boundary."""
    from local_operator.tools.registry import create_tools

    context = ToolContext(cwd=str(tmp_path), session_id="s1")
    grep = next(t for t in create_tools(context) if t.name == "grep")

    bad_regex = await _run(
        _calls((0, "c1", "grep", '{"pattern":"("}')) + [StreamEndEvent(stop_reason="toolUse")],
        [grep],
        cwd=str(tmp_path),
    )
    assert _faults(bad_regex) == {"grep": "invalid_arguments"}

    missing_path = await _run(
        _calls((0, "c1", "grep", '{"pattern":"x","path":"no/such/dir"}'))
        + [StreamEndEvent(stop_reason="toolUse")],
        [grep],
        cwd=str(tmp_path),
    )
    assert _faults(missing_path) == {"grep": "execution"}


@pytest.mark.asyncio
async def test_a_tool_body_raising_the_typed_error_is_translated_by_the_executor():
    """A tool may RAISE rather than hand-build a marked result; the executor
    performs the translation so every tool gets it for free."""
    from local_operator.harness.types import InvalidToolArgumentsError

    async def execute(tool_call_id, args, signal, on_update, context):
        raise InvalidToolArgumentsError("not a duration: 'soonish'")

    tool = AgentTool(
        name="wake",
        parameters={"type": "object", "properties": {"in": {"type": "string"}}},
        execute=execute,
    )
    recorded = await _run(
        _calls((0, "c1", "wake", '{"in":"soonish"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [tool],
    )
    assert _faults(recorded) == {"wake": "invalid_arguments"}


def test_the_tool_bodys_fault_class_is_one_the_reader_counts():
    """The value a tool body writes must be one ``MODEL_FAULTS`` recognises.

    ``harness`` carries no analytics dependency (see ``LoopConfig``), so the
    writer's spelling and the reader's frozenset are pinned only here. A drift
    would not raise — it would write a fault name no rate counts, silently
    returning the figure to the under-reporting this PR fixes.

    Deliberately NOT asserting ``loop.FAULT_KEY == types.FAULT_KEY``: the loop
    now IMPORTS both names, so that compares an object with itself and can
    never fail. What is worth pinning is the cross-package agreement below.
    """
    from local_operator.analytics.model import EXCLUDED_FAULTS, MODEL_FAULTS
    from local_operator.harness import types as types_module

    assert types_module.FAULT_INVALID_ARGUMENTS in MODEL_FAULTS
    assert types_module.FAULT_INVALID_ARGUMENTS not in EXCLUDED_FAULTS
    # The key is what the store reads out of ``details``; the store's own
    # schema comment documents this exact string.
    assert types_module.FAULT_KEY == "__fault"


@pytest.mark.asyncio
async def test_the_builtin_guard_does_not_swallow_the_typed_error():
    """``_guard`` wraps every builtin and catches ``Exception`` INSIDE the tool,
    so the loop's translation branch never sees a raise from one.

    Without its own branch there, the guard's catch-all converted a deliberate
    argument-shape rejection into an unmarked "failed unexpectedly" traceback —
    the marker lost, the call recorded as ``execution``, and the raise mechanism
    silently useless for exactly the tools it was built for. A genuine internal
    error must still take the traceback path.
    """
    from local_operator.harness.types import InvalidToolArgumentsError
    from local_operator.tools.builtin import _guard

    @_guard("demo")
    async def malformed(tool_call_id, args, signal=None, on_update=None, context=None):
        raise InvalidToolArgumentsError("not a duration: 'soonish'")

    @_guard("demo")
    async def internal(tool_call_id, args, signal=None, on_update=None, context=None):
        raise RuntimeError("genuine internal failure")

    rejected = await malformed("c", {}, None, None, None)
    assert rejected.is_error and rejected.details == {"__fault": "invalid_arguments"}
    assert "Traceback" not in rejected.text

    crashed = await internal("c", {}, None, None, None)
    assert crashed.is_error
    assert not (crashed.details or {}).get("__fault"), "an internal error is not a model fault"


# ---------------------------------------------------------------------------
# `_validation_error`: a pydantic rejection is a model fault ONLY when it is a
# pure function of the arguments.
#
# This is the change with the widest blast radius in this area — it reclassifies
# every params-model rejection across every tool — and it shipped without
# behavioural coverage, which is exactly how the `effort` case below survived to
# review. Both directions are pinned here.
# ---------------------------------------------------------------------------


def _shipped_task_tool(tmp_path) -> AgentTool:
    """The REAL ``task``, whose schema is rendered from live config at BUILD
    time while its validator re-reads config at CALL time — the gap under test.

    ``task`` is ``createIf``-gated on a launcher, so one is supplied; nothing in
    these tests launches anything, the call is refused during validation.
    """
    from local_operator.tools.registry import create_tools

    def _launcher(
        label: str, prompt: str, *, agent: str = "task", effort: str | None = None
    ) -> str:
        raise AssertionError("validation must refuse the call before any launch")

    context = ToolContext(cwd=str(tmp_path), session_id="s1", subagent_launcher=_launcher)
    return next(t for t in create_tools(context) if t.name == "task")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,why",
    [
        ({"path": "a.txt", "bogus": 1}, "extra=forbid"),
        ({"path": "a.txt", "range": 5}, "declared type"),
    ],
)
async def test_a_static_shape_violation_is_a_model_fault(tmp_path, args, why):
    """`extra="forbid"` and a wrong declared type are decidable from the
    arguments alone, so they are the model violating the tool's contract."""
    (tmp_path / "a.txt").write_text("l1\nl2\n")
    recorded = await _run(
        _calls((0, "c1", "read", json.dumps(args))) + [StreamEndEvent(stop_reason="toolUse")],
        [_shipped_read_tool(tmp_path)],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"read": "invalid_arguments"}, why


@pytest.mark.asyncio
async def test_a_cross_field_validator_is_still_a_model_fault(tmp_path):
    """A cross-field rule is a pure function of the arguments — the model could
    have satisfied it — so the environmental escape hatch must not catch it."""
    from local_operator.tools.registry import create_tools

    context = ToolContext(cwd=str(tmp_path), session_id="s1")
    edit = next(t for t in create_tools(context) if t.name == "edit")
    recorded = await _run(
        _calls((0, "c1", "edit", json.dumps({"path": "a.txt", "old_text": "a"})))
        + [StreamEndEvent(stop_reason="toolUse")],
        [edit],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"edit": "invalid_arguments"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage,why",
    [
        ("values:\n  subagents:\n    models: {}\n", "operator removed the tier"),
        ("values: [ this is not: valid yaml\n", "config.yml became unreadable"),
    ],
)
async def test_an_effort_tier_that_vanished_after_build_is_not_the_models_fault(
    tmp_path, monkeypatch, damage, why
):
    """The blocker this suite missed: `effort`'s enum is rendered into the
    schema at BUILD time, but `_validate_effort_tier` re-reads config at CALL
    time. The model emits the one value its own schema offered and is refused
    because the world moved.

    The corrupt-config variant is the sharper one: `configured_effort_tiers`
    swallows a read failure and reports "no tiers" ON PURPOSE, so that a broken
    config costs the operator a tier picker rather than a session. Billing that
    to the model would put an operator config error into a published accuracy
    figure — the over-claiming direction `InvalidToolArgumentsError` names as
    the worse one.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    (config_dir / "config.yml").write_text(
        "values:\n  subagents:\n    models:\n      med: anthropic/claude-sonnet-4-5\n"
    )

    # Build the tool while the tier exists, so the advertised enum contains it.
    task = _shipped_task_tool(tmp_path)
    effort = task.parameters["properties"]["effort"]
    assert "med" in json.dumps(effort), "the schema must have offered the tier"

    (config_dir / "config.yml").write_text(damage)

    recorded = await _run(
        _calls((0, "c1", "task", json.dumps({"label": "x", "prompt": "y", "effort": "med"})))
        + [StreamEndEvent(stop_reason="toolUse")],
        [task],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"task": "execution"}, why


@pytest.mark.asyncio
async def test_a_malformed_effort_value_is_still_a_model_fault(tmp_path, monkeypatch):
    """The escape hatch must not blanket-exempt the field: a non-string
    `effort` fails on declared type, before the config-reading validator, and
    is decidable from the arguments alone."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    recorded = await _run(
        _calls((0, "c1", "task", json.dumps({"label": "x", "prompt": "y", "effort": 123})))
        + [StreamEndEvent(stop_reason="toolUse")],
        [_shipped_task_tool(tmp_path)],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"task": "invalid_arguments"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args,expected,why",
    [
        ({"op": "create", "message": "m", "in": "soonish"}, "invalid_arguments", "duration"),
        ({"op": "create", "message": "m", "at": "whenever"}, "invalid_arguments", "time"),
        ({"op": "create", "message": "m", "at": "2001-01-01T00:00:00Z"}, "execution", "past"),
    ],
)
async def test_the_real_wake_tool_separates_malformed_from_unsatisfiable(
    tmp_path, args, expected, why
):
    """Driven through the REAL ``wake``, not a synthetic stand-in.

    Two fixture tests in this module raise the typed error from a tool NAMED
    ``wake``, which made the suite read as though the shipped tool were covered
    when it was not. ``'soonish'`` could never be a duration; a timestamp in
    2001 is perfectly well-formed and merely unsatisfiable.
    """
    from local_operator.tools.registry import create_tools

    class _Scheduler:
        schedules: list[Any] = []

        async def update(self, schedules: list[Any]) -> None:
            return None

    context = ToolContext(cwd=str(tmp_path), session_id="s1", wake_scheduler=_Scheduler())
    wake = next(t for t in create_tools(context) if t.name == "wake")
    recorded = await _run(
        _calls((0, "c1", "wake", json.dumps(args))) + [StreamEndEvent(stop_reason="toolUse")],
        [wake],
        cwd=str(tmp_path),
        wake_scheduler=_Scheduler(),
    )
    assert _faults(recorded) == {"wake": expected}, why


@pytest.mark.asyncio
async def test_a_malformed_spill_handle_is_a_model_fault(tmp_path):
    """The handle rides inside ``path``, a plain schema string, so a value that
    is not a handle at all can only be rejected in the tool body — the same
    argument its sibling regex branch already carries."""
    recorded = await _run(
        _calls((0, "c1", "read", json.dumps({"path": "spill://not-a-valid-handle!!"})))
        + [StreamEndEvent(stop_reason="toolUse")],
        [_shipped_read_tool(tmp_path)],
        cwd=str(tmp_path),
    )
    assert _faults(recorded) == {"read": "invalid_arguments"}


@pytest.mark.asyncio
async def test_validity_does_not_depend_on_which_tool_the_model_picked(tmp_path):
    """``web_fetch``/``web_search`` built their own validation results and so
    classified an identical modelling mistake differently from ``read``.

    A benchmark whose denominator moves with tool identity is the hidden
    dependence the analytics origin-partition prose exists to prevent.
    """
    from local_operator.tools.registry import create_tools

    context = ToolContext(cwd=str(tmp_path), session_id="s1")
    built = {t.name: t for t in create_tools(context)}
    faults = {}
    for name, args in (
        ("read", {"path": "a.txt", "bogus": 1}),
        ("web_fetch", {"url": "https://example.com", "bogus": 1}),
        ("web_search", {"query": "x", "bogus": 1}),
    ):
        if name not in built:  # web tools are createIf-gated on configuration
            continue
        recorded = await _run(
            _calls((0, "c1", name, json.dumps(args))) + [StreamEndEvent(stop_reason="toolUse")],
            [built[name]],
            cwd=str(tmp_path),
        )
        faults[name] = _faults(recorded)[name]
    assert set(faults.values()) == {"invalid_arguments"}, faults
    assert "read" in faults and len(faults) > 1, f"web tools were not exercised: {faults}"
