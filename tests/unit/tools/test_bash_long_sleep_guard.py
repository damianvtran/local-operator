"""The bash long-sleep guard: refuse a foreground call that is mostly a sleep.

Two properties, as for the search guard. It must fire on the MEASURED shapes —
the commands a child used on 2026-09-24 to poll a background pytest run while
its parent's hub notes waited out every sleep (session ``f7318cc06bdd``) — and
it must never fire on the sleeps a model writes legitimately: a short settle
after a kill, a readiness loop, a background job, a sleep the parser cannot
read as a literal. The allow list is the contract, so each shape is a named case.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.tools import builtin, sleep_guard

_MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

C = sleep_guard.check_long_sleep

#: Verbatim from the child's transcript (``~/.local-operator/sessions/
#: f7318cc06bdd/transcript.jsonl``), each a FOREGROUND bash call. The last two
#: are the calls the parent's notes arrived during and were only read after.
MEASURED = [
    'sleep 1500; tail -c 250 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"; echo; '
    'echo "--- pyright ---"; tail -3 "$LOCAL_OPERATOR_SCRATCHPAD/pyright-final.log"; uptime',
    'sleep 1500; tail -c 250 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"; echo; uptime',
    'sleep 1800; tail -c 200 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; echo; '
    "sysctl vm.swapusage | tail -1; uptime",
    'sleep 900; tail -c 400 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final.log"; echo; echo "---"; uptime',
    'sleep 600; head -3 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; '
    'tail -c 200 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; echo; sysctl vm.swapusage; uptime',
    'sleep 300; echo "waited 5 min"; uptime',
    'sleep 180; cat "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"',
]


@pytest.mark.parametrize("command", MEASURED)
def test_the_measured_commands_from_the_live_child_are_refused(command: str) -> None:
    message = C(command)
    assert message is not None, command
    # The refusal names the replacement, not just the problem.
    assert "background: true" in message
    assert "`wait`" in message
    assert "jobs op='peek'" in message
    assert sleep_guard.ALLOW_ENV in message
    # The diagnosis must be TRUE about the operator's own reach: a steer DOES
    # interrupt a foreground bash and detaches it to a background job; only a
    # note is deaf. The old sentence claimed neither could reach it, in the
    # direction that costs the operator the lever they lacked in the incident
    # (design review D4).
    assert "deaf to a hub note" in message
    assert "a steer does reach it" in message
    assert "cannot be interrupted by a hub message" not in message
    # The promise about a message is the true one: a note queued BEFORE the wait
    # parks is delivered at the next boundary rather than shortening that park,
    # so the clause has to say WHILE (review round 2, A4).
    assert "while it is parked" in message
    assert "a message arrives, or " not in message
    # The hatch is PER SEGMENT, which the copy has to say: `VAR=1 sleep 900;
    # sleep 900` is still refused, and an exported value is not read at all
    # (design review D3).
    assert "prefixed on each long sleep" in message
    assert "per segment" in message
    assert "an exported value is not read" in message
    # ...and the reader's own vocabulary, not a third one for the same event
    # (design review D5): `wait` answers in "the wait was cancelled" and takes
    # `wait_ms` in ms.
    assert "the wait is cancelled" in message
    assert "`wait_ms` in ms" in message
    # D1: every bullet LEADS with its actionable token, because a failed-call
    # card paints each advice line as ONE cropped row (91 cells at a 100-column
    # frame, 72 at 80, 52 at 60, 41 at 50) — the unhedged copy put `wait` at cells
    # 86-92 and the hatch variable at 57-90, so both were cut exactly on the
    # frames operators run. This pins the copy the MODEL reads and the operator
    # reads once the card is rebuilt; the plan-time refusal's LIVE row is the
    # not-run ending, which is clipped at 200 chars before any widget sees it and
    # is not what this assertion describes (design review D6).
    bullets = [line for line in message.splitlines() if line.startswith("  - ")]
    assert bullets, message
    for bullet in bullets:
        spans = _token_spans(bullet)
        assert spans, bullet
        assert spans[0][0] <= 8, bullet
        assert spans[0][1] <= _ADVICE_TOKEN_LEAD_CELLS, bullet
        # A SECOND token on the same line is what an operator on a narrow pane
        # loses first: with `background: true` leading bullet 1, `wait` ended at
        # cell 52 against the 52-cell lane at 60 columns and painted as an
        # unclosed `` then `wait… `` (design review D7).
        if len(spans) > 1:
            assert spans[1][0] <= _ADVICE_TOKEN_LEAD_CELLS, bullet
    assert _token_spans(bullets[0])[1][1] <= 52, bullets[0]


#: The same child's SHORT sleeps, which must keep working.
MEASURED_ALLOWED = [
    "kill -TERM 23076 23031 2>/dev/null; sleep 3; ps -p 23076,23031 -o pid,command 2>&1 | tail -2",
    'cd ~/wt && pkill -f "wt" 2>/dev/null; sleep 2; pgrep -fl "pyright|pytest" | grep -c wt',
    "for i in $(seq 1 60); do curl -fsS http://127.0.0.1:13380/v1/capabilities >/dev/null "
    "&& break; sleep 1; done",
]


@pytest.mark.parametrize("command", MEASURED_ALLOWED)
def test_the_short_sleeps_from_the_same_child_still_run(command: str) -> None:
    assert C(command) is None, command


BLOCKED = [
    "sleep 121",
    "sleep 25m",
    "sleep 1h",
    "sleep infinity",
    "/bin/sleep 600",
    "sleep 60; sleep 70",  # the foreground total is what holds the call
    "cd x && sleep 600 && tail f",
    "sleep 1 && sleep 1000 && x",
    "while true; do tail f; done; sleep 900",  # after the loop closes: top level again
    "sleep 1500\ntail -c 250 log",
    "sleep 900 2>/dev/null; tail log",
    "sleep 900 | cat",  # the pipeline waits for its first stage
    # M2: a trailing shell comment must not hide the sleep (it changes nothing
    # about how long the call holds the session).
    "sleep 121  # poll the suite",
    "sleep 1500  # wait",
    "cd x && sleep 900  # watch",
    # (`sleep 900;# poll` is NOT listed: the `;` splits the segment before the
    # comment is reached, so it refuses with or without the fix and reading it
    # as coverage would be misleading — the two cells above are the ones that
    # discriminate, and the round-2 differential confirms it.)
]


@pytest.mark.parametrize("command", BLOCKED)
def test_long_foreground_sleeps_are_refused(command: str) -> None:
    assert C(command) is not None, command


NEVER_BLOCKED = [
    # at or under the threshold
    "sleep 120",
    "sleep 2; pgrep x",
    "sleep 0.5",
    "sleep 1m",
    # backgrounded by the shell: holds nothing
    "sleep 900 &",
    "x & sleep 5",
    "(sleep 600; echo) &",
    "{ sleep 600; echo; } &",
    # inside a compound: may itself be backgrounded, and a loop sleep is a poll
    # interval the guard cannot weigh against the loop's own exit
    "while true; do sleep 300; done",
    "until test -f x; do sleep 300; done",
    "for i in 1 2; do\n sleep 900\ndone",
    "if true; then sleep 900; fi",
    # not a literal duration
    'sleep "$N"',
    "sleep $((60*30))",
    "sleep 1m 30x",
    # a LATER pipeline stage, and a `2>&1` the shared splitter reads as `&`
    "yes | sleep 900",
    "sleep 900 2>&1; tail log",
    # not a sleep command at all
    "timeout 900 sleep 900",
    'echo "sleep 900"',
    'git commit -m "sleep 900; tail"',
    "cat <<EOF\nsleep 900\nEOF",
    # a `#` that is not the start of a word is data, not a comment
    'sleep "1#2"',
    "echo hi # sleep 900",
    "printf '%s' '# sleep 900'",
    # the escape hatch, read per segment off the command itself
    f"{sleep_guard.ALLOW_ENV}=1 sleep 900; tail f",
]


@pytest.mark.parametrize("command", NEVER_BLOCKED)
def test_legitimate_sleeps_are_not_blocked(command: str) -> None:
    assert C(command) is None, command


def test_a_falsy_grant_does_not_open_the_hatch() -> None:
    assert C(f"{sleep_guard.ALLOW_ENV}=0 sleep 900") is not None


def test_an_inherited_grant_is_not_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grant is per call, read off the command. An exported value would make
    it silently global and invisible in the transcript."""
    monkeypatch.setenv(sleep_guard.ALLOW_ENV, "1")
    assert C("sleep 900") is not None


def test_the_threshold_is_the_one_constant() -> None:
    limit = sleep_guard.LONG_SLEEP_THRESHOLD_SECONDS
    assert C(f"sleep {int(limit)}") is None
    assert C(f"sleep {int(limit) + 1}") is not None


# --- through the real bash tool --------------------------------------------


class _ToolsThenText:
    """Scripted provider: request one ``bash`` call, then answer in prose."""

    def __init__(self, args: dict[str, object]) -> None:
        self.args = args
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        first = len(self.requests) == 1

        async def gen():
            if first:
                yield StreamToolCallDelta(index=0, id="c", name="bash", argument_delta="")
                yield StreamToolCallDelta(index=0, argument_delta=json.dumps(self.args))
            else:
                yield StreamTextDelta(delta="understood")
            yield StreamEndEvent(stop_reason="toolUse" if first else "stop")

        return gen()


async def _run_loop(
    tools: list[AgentTool], context: ToolContext, stream: _ToolsThenText
) -> list[object]:
    """Drive the real :class:`AgentLoop` for one turn over ``tools``."""
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import LoopConfig, Message

    config = LoopConfig(
        model=_MODEL,
        # The loop hands the provider a rendered history; the filter is what the
        # harness's own tests use to keep only real messages.
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
    )
    loop_context = LoopContext(
        system_blocks=["stable"],
        messages=[Message.user("go")],
        tools=tools,
        tool_context=context,
    )
    events: list[object] = []
    async for event in AgentLoop().run(loop_context.messages, loop_context, config, None):
        events.append(event)
    return events


def _call(name: str, args: dict[str, object]):
    from local_operator.harness.types import ToolCall

    return ToolCall(id="c", name=name, arguments=dict(args))


async def _plan(tools: list[AgentTool], context: ToolContext, args: dict[str, object]):
    """Plan one ``bash`` call through the real seam, with ``tools`` as the
    session's live inventory (that list is what the refusal copy is built from)."""
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import LoopConfig, Message

    loop_context = LoopContext(system_blocks=[], tools=tools, tool_context=context)
    return await AgentLoop()._plan_call(  # noqa: SLF001 — the planning seam is the unit
        _call("bash", args),
        loop_context,
        # A stream function is required by the config but never reached: these
        # calls are refused during planning, before any model request.
        LoopConfig(
            model=_MODEL,
            convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
            stream_fn=lambda request, signal: None,
        ),
    )


async def _bash(args: dict[str, object], context: ToolContext):
    tool = builtin.build_bash_tool()
    return await tool.execute("c", args, None, None, context)  # type: ignore[operator]


@pytest.mark.asyncio
async def test_the_bash_tool_refuses_before_spawning(tmp_path) -> None:
    """Refused at the top of ``execute_bash``: nothing runs, nothing waits."""
    marker = tmp_path / "ran"
    context = ToolContext(cwd=str(tmp_path), session_id="sg")
    started = time.perf_counter()
    result = await _bash({"command": f"touch {marker}; sleep 1500; tail -c 250 log"}, context)
    assert time.perf_counter() - started < 5.0
    assert result.is_error is True
    assert "sleeps 1500 s (25 min) in the foreground" in result.text
    assert not marker.exists()


#: (offered inventory, must-name, must-not-name) for every reader class the
#: design round measured. ``must_not_name`` is the whole point of D2: a refusal
#: that names a replacement the reader does not hold routes it into
#: ``Tool not found``.
READER_CLASSES = [
    # an interactive root: all three
    (
        frozenset({"bash", "read", "wait", "jobs", "wake", "hub"}),
        ["`wait`", "`jobs op='peek'`", "`wake`"],
        [],
    ),
    # a coder child: non-delegating, can background, keeps jobs and wait, has no
    # wake (its one-prompt session could not keep one anyway)
    (
        frozenset({"bash", "read", "wait", "jobs", "hub"}),
        ["`wait`", "`jobs op='peek'`"],
        ["`wake`"],
    ),
    # a declared-inventory session (`lop exec --tools bash,read`): none of them,
    # and it must be told so rather than handed options that error
    (frozenset({"bash", "read"}), [], ["`wait`", "`jobs op='peek'`", "`wake`"]),
]


@pytest.mark.parametrize(("offered", "named", "absent"), READER_CLASSES)
def test_the_advice_names_only_tools_the_reader_holds(
    offered: frozenset[str], named: list[str], absent: list[str]
) -> None:
    message = sleep_guard.check_long_sleep("sleep 1500; tail log", offered=offered)
    assert message is not None
    for token in named:
        assert token in message, (token, message)
    # ``absent`` is checked against the ADVICE (the bullet lines), not the whole
    # message: the no-tool reader is told in prose that the session offers none
    # of the three, which necessarily names them.
    bullets = "\n".join(line for line in message.splitlines() if line.startswith("  - "))
    for token in absent:
        assert token not in bullets, (token, message)


def test_a_reader_with_none_of_the_three_is_told_so_and_led_to_the_hatch() -> None:
    """The class that made the hedged middle unacceptable: with no `wait`,
    no `jobs` and no `wake`, the escape hatch is the only bullet that works, so
    it leads and the reader is told plainly (design review D2)."""
    message = sleep_guard.check_long_sleep("sleep 1500; tail log", offered=frozenset({"bash"}))
    assert message is not None
    lines = message.splitlines()
    assert "No `wait`, `jobs` or `wake` here." in message
    first_bullet = next(line for line in lines if line.startswith("  - "))
    assert sleep_guard.ALLOW_ENV in first_bullet


def test_the_none_class_statement_fits_a_50_column_lane_and_keeps_the_lead_in() -> None:
    """D8: the statement used to be one 108-cell line ending `…nothing here to
    hand the waiting to. Do one of:` — painted as `…so there is nothing he…`
    with the bullets' lead-in swallowed at every width. Every other class puts
    `Do one of:` on its own 11-cell line, so this one must too."""
    message = sleep_guard.check_long_sleep("sleep 1500; tail log", offered=frozenset({"bash"}))
    assert message is not None
    lines = message.splitlines()
    statement = next(line for line in lines if line.startswith("No `wait`"))
    assert len(statement) <= 41, statement
    assert "Do one of:" not in statement
    assert lines[lines.index(statement) + 1] == "Do one of:"


@pytest.mark.asyncio
async def test_the_loop_hands_the_hook_the_readers_own_inventory(tmp_path) -> None:
    """The inventory must reach the refusal through the REAL planning seam, not
    just the constructor: a session whose inventory is narrowed refuses with
    copy that names nothing it cannot run (design review D2)."""
    from local_operator.tools.registry import create_tools

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="sg-inv", jobs=manager)
    try:
        tools = [tool for tool in create_tools(context) if tool.name in ("bash", "read")]
        planned = await _plan(tools, context, {"command": "sleep 1500; tail log"})
        assert planned.failure is not None
        text = planned.failure.text
        assert "No `wait`, `jobs` or `wake` here." in text
        assert "`wait` on the job id" not in text
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_the_loop_never_prompts_approval_for_a_refused_long_sleep(tmp_path) -> None:
    """The finding, driven through the REAL loop.

    The approval gate runs inside the runner, BEFORE the tool body, so a refusal
    that lived only in ``execute`` made an interactive operator approve a call
    that was then refused (review Q-2). The plan-time hook is what closes it:
    this asserts the gate's callback is never invoked for the long-sleep call,
    and that the model still gets the refusal text telling it what to do.
    """
    from local_operator.tools.registry import create_tools

    approvals: list[tuple[str, str]] = []

    async def request_approval(tool_name: str, summary: str, **kwargs: object) -> bool:
        approvals.append((tool_name, summary))
        return True

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="sg-gate", jobs=manager)
    context.request_approval = request_approval  # type: ignore[assignment]
    tools = list(create_tools(context))
    stream = _ToolsThenText(
        {"command": 'sleep 1500; tail -c 250 "$LOCAL_OPERATOR_SCRATCHPAD/suite.log"'}
    )
    events = await _run_loop(tools, context, stream)
    try:
        assert approvals == [], f"the gate was asked about a refused call: {approvals}"
        text = json.dumps([str(event) for event in events])
        assert "sleeps 1500 s (25 min) in the foreground" in text
        assert "background: true" in text
        # One model turn, no second call: the model was told why and stopped.
        assert len(stream.requests) == 2
    finally:
        await manager.dispose()


#: The tools an interactive root session holds, i.e. every replacement the copy
#: may name. Passed explicitly so a test that is not ABOUT the inventory still
#: exercises the fully-offered copy rather than the speculative ``None`` path.
_ALL_TOOLS = frozenset({"bash", "read", "grep", "wait", "jobs", "wake", "hub"})


def _token_spans(line: str) -> list[tuple[int, int]]:
    """1-based ``(start, end)`` cells of each backticked token on ``line``.

    Backticks are paired left to right, which is how the copy writes them: a
    token is one ``...`` pair, never a stray backtick.
    """
    starts = [index + 1 for index, char in enumerate(line) if char == "`"]
    return [(starts[i], starts[i + 1]) for i in range(0, len(starts) - 1, 2)]


#: How far into a bullet its actionable token may run. The operator's failed-call
#: card crops each advice line at 74 cells in the canonical 80-column frame (54 at
#: 60, 44 at 50), so a token starting past ~40 cells is a token the operator
#: cannot read (design review D1).
_ADVICE_TOKEN_LEAD_CELLS = 40

#: Every spelling pydantic coerces to a BOOL, split by what it means. The first
#: seven passed ``validate_tool_arguments`` looking like a background call to the
#: raw-argument hook — which read the string and found it truthy — so the call
#: reached the approval gate and was refused only by the body (review A1).
FALSY_BACKGROUND_SPELLINGS = ["false", "no", "0", "off", "n", "f", "FALSE", "False"]
TRUTHY_BACKGROUND_SPELLINGS = ["true", "yes", "1", "on", "y", "t", "TRUE", "True"]


@pytest.mark.parametrize("spelling", FALSY_BACKGROUND_SPELLINGS)
def test_the_hook_refuses_when_background_is_a_falsy_spelling(spelling: str) -> None:
    """``background: "false"`` IS a foreground call: the hook must refuse it, so
    the refusal happens before the operator is asked to approve anything."""
    refusal = builtin._long_foreground_sleep_refusal(
        {"command": "sleep 1500; tail -c 250 log", "background": spelling}, _ALL_TOOLS
    )
    assert refusal is not None, spelling
    assert "sleeps 1500 s (25 min) in the foreground" in refusal


@pytest.mark.parametrize("spelling", TRUTHY_BACKGROUND_SPELLINGS)
def test_the_hook_allows_when_background_is_a_truthy_spelling(spelling: str) -> None:
    """...and the other half: a real background call is never refused, whichever
    spelling the model used."""
    assert (
        builtin._long_foreground_sleep_refusal(
            {"command": "sleep 1500; tail -c 250 log", "background": spelling}, _ALL_TOOLS
        )
        is None
    ), spelling


@pytest.mark.parametrize("spelling", FALSY_BACKGROUND_SPELLINGS)
@pytest.mark.asyncio
async def test_a_falsy_background_spelling_never_reaches_the_approval_gate(
    tmp_path, spelling: str
) -> None:
    """The finding, through the REAL loop: the gate's callback is not invoked.

    This is the arm that discriminated in review — with the raw-argument hook the
    gate was asked (`approvals: [('bash', 'run: sleep 1500')]`) and only the body
    refused the call.
    """
    from local_operator.tools.registry import create_tools

    approvals: list[str] = []

    async def request_approval(tool_name: str, summary: str, **kwargs: object) -> bool:
        approvals.append(summary)
        return True

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="sg-spell", jobs=manager)
    context.request_approval = request_approval  # type: ignore[assignment]
    tools = list(create_tools(context))
    stream = _ToolsThenText({"command": "sleep 1500; tail -c 250 log", "background": spelling})
    try:
        events = await _run_loop(tools, context, stream)
        assert approvals == [], f"{spelling!r} reached the gate: {approvals}"
        assert "in the foreground" in json.dumps([str(event) for event in events])
    finally:
        await manager.dispose()


def test_the_hook_leaves_a_shape_it_cannot_coerce_to_the_body(tmp_path) -> None:
    """A hook that raises is SILENTLY SKIPPED by the loop, which would let the
    call reach the gate — so an uncoercible shape returns None here and is the
    body's ``ValidationError`` to report."""
    refusal = builtin._long_foreground_sleep_refusal(
        {"command": "sleep 1500", "timeout": "soon"}, _ALL_TOOLS
    )
    assert refusal is None


@pytest.mark.asyncio
async def test_a_background_call_is_never_refused(tmp_path) -> None:
    """The replacement the refusal names must itself be allowed."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="sg-bg", jobs=manager)
    result = await _bash({"command": "sleep 1500; echo done", "background": True}, context)
    try:
        assert result.is_error is False, result.text
        assert result.details is not None and result.details.get("backgrounded") is True
    finally:
        await manager.cancel(result.details["job_id"]) if result.details else None
        await manager.dispose()


@pytest.mark.asyncio
async def test_both_paths_of_the_guard_report_the_same_fault_class(tmp_path) -> None:
    """One predicate, one bucket. The plan-time refusal is a MODEL fault
    (``invalid_arguments``); the execute-time path used to be an unmarked
    ``execution`` one, so an identical call was counted differently depending on
    which layer caught it (review A2)."""
    from local_operator.harness.loop import FAULT_KEY
    from local_operator.tools.registry import create_tools

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="sg-fault", jobs=manager)
    try:
        tools = list(create_tools(context))
        planned = await _plan(tools, context, {"command": "sleep 1500; tail log"})
        direct = await _bash({"command": "sleep 1500; tail log"}, context)
        failure = planned.failure
        assert failure is not None and failure.details is not None
        assert direct.details is not None
        assert failure.details[FAULT_KEY] == direct.details[FAULT_KEY] == "invalid_arguments", (
            failure.details,
            direct.details,
        )
    finally:
        await manager.dispose()


@pytest.mark.asyncio
async def test_the_escape_hatch_runs_the_command(tmp_path) -> None:
    context = ToolContext(cwd=str(tmp_path), session_id="sg-hatch")
    # A long sleep would stall the test, so prove the grant on a command whose
    # sleep is long only on paper: the timeout ends it, and the result is a
    # timeout rather than the refusal.
    result = await asyncio.wait_for(
        _bash(
            {"command": f"{sleep_guard.ALLOW_ENV}=1 sleep 900", "timeout": 0.5},
            context,
        ),
        timeout=30,
    )
    assert "in the foreground" not in result.text
    assert result.text.startswith("TIMEOUT after 0.5s")
