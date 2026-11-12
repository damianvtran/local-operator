"""``Session.verify_completion``: the harness completion verifier primitive.

What is pinned is the BOUNDARY, not the wording: the verifier child's tools
are a capability it physically has (read-only, no web floor, the host's tool
and nothing else), the parent never gains the host tool, a cap really cancels
the child, its result never reaches the parent's conversation, its prompt is
only what the launcher gave it, and its request rows are attributed.

The stream double is the smallest thing with the real seams this needs:
``fork`` (each child gets its own identity-bound stream, as
``SessionStreamFn.fork`` does), ``bind_request_actor`` and a root-tree
``subscribe_requests`` that sees every request-final record.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from local_operator.agent_profiles import AgentProfile
from local_operator.harness.subagent import ChildSurface, run_subagent
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.harness.verifier import (
    VERIFIER_ROLE,
    VerifierLimits,
    build_verifier_prompt,
    parse_verdict,
    verifier_profile,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.session.test_launch_subagent import MODEL, wait_for

Reply = Callable[[ChatRequest], list[Any]]


def text_turn(text: str) -> list[Any]:
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]


_call_ids = itertools.count()


def tool_turn(name: str, args: dict[str, Any]) -> list[Any]:
    call_id = f"call-{next(_call_ids)}"
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name),
        StreamToolCallDelta(index=0, argument_delta=json.dumps(args)),
        StreamEndEvent(stop_reason="toolUse"),
    ]


class TreeStream:
    """A root stream and its forks, sharing one script and one observer list."""

    def __init__(self, reply: Reply, *, root: "TreeStream | None" = None) -> None:
        self.reply = reply
        self.root = root or self
        self.job_id: str | None = None
        self.actor_role: str | None = None
        if root is None:
            self.requests: list[ChatRequest] = []
            self.by_job: dict[str | None, list[ChatRequest]] = {}
            self.observers: list[Callable[[Any], None]] = []
            self.rows: list[Any] = []

    # -- the seams the session and subagent machinery look for ---------------
    def fork(self, _name: str) -> "TreeStream":
        return TreeStream(self.reply, root=self.root)

    def bind_request_actor(self, *, job_id: str | None, actor_role: str | None) -> None:
        self.job_id, self.actor_role = job_id, actor_role

    def subscribe_requests(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self.root.observers.append(handler)

        def unsubscribe() -> None:
            if handler in self.root.observers:
                self.root.observers.remove(handler)

        return unsubscribe

    async def close(self) -> None:
        return None

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        root = self.root
        root.requests.append(request)
        root.by_job.setdefault(self.job_id, []).append(request)
        events = self.reply(request)
        stream = self

        async def gen():
            try:
                for event in events:
                    if isinstance(event, float):
                        # A slow provider: the wall cap's case.
                        await asyncio.sleep(event)
                        continue
                    yield event
            finally:
                row = SimpleNamespace(
                    request_id=f"r{len(root.rows)}",
                    job_id=stream.job_id,
                    actor_role=stream.actor_role,
                )
                root.rows.append(row)
                for observer in list(root.observers):
                    observer(row)

        return gen()


def first_user_text(request: ChatRequest) -> str:
    return next(m for m in request.messages if m.role == "user").text or ""


def is_verifier(request: ChatRequest) -> bool:
    return first_user_text(request).startswith(f"[role: {VERIFIER_ROLE}]")


def make_parent(tmp_path, stream: TreeStream) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )


def inspect_tool(calls: list[dict[str, Any]], output: str = "B2 = 42") -> AgentTool:
    async def execute(tool_call_id, args, signal=None, on_update=None, context=None):
        calls.append(dict(args))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="inspect_command",
            content=[TextContent(text=output)],
        )

    return AgentTool(
        name="inspect_command",
        description="Run a read-only command on the target machine.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        approval_tier="read",
        execute=execute,
    )


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    # The child's WIP checkpointer snapshots its cwd with git after each tool
    # call; a scratch cwd keeps that off the repository checkout.
    monkeypatch.chdir(tmp_path)
    return tmp_path / "config"


# -- 1. the role's tool set ---------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_child_is_read_only_and_holds_the_host_tool(tmp_path, config_dir):
    stream = TreeStream(lambda request: text_turn("all met\nVERDICT: PASS"))
    parent = make_parent(tmp_path, stream)

    verdict = await parent.verify_completion(
        request="Put 42 in B2.", claim="Done.", host_tools=[inspect_tool([])]
    )

    assert verdict.status == "PASS" and verdict.ended_by == "verdict"
    [first] = [r for r in stream.requests if is_verifier(r)][:1]
    names = {tool.name for tool in first.tools}
    assert "inspect_command" in names
    # Never a writer, a delegator, a messenger or the network.
    forbidden = {"write", "edit", "task", "hub", "web_search", "web_fetch", "agent", "team"}
    assert not names & forbidden, names & forbidden
    # Everything it does hold is the read-only allowlist plus the host tool
    # (``bash`` only where the host can sandbox it).
    assert names <= {"read", "glob", "grep", "bash", "inspect_command"}
    assert {"read", "glob", "grep"} <= names
    await parent.dispose()


# -- 2. the ceiling drops the web floor ---------------------------------------


async def _first_child_tools(tmp_path, surface: ChildSurface | None) -> set[str]:
    stream = TreeStream(lambda request: text_turn("ok"))
    parent = make_parent(tmp_path, stream)
    profile = AgentProfile(name="probe", instructions="probe", tools=("read",))
    job_id = run_subagent(
        "probe",
        "go",
        parent_session=parent,
        jobs_manager=parent.jobs,
        agent="probe",
        profile_override=profile,
        surface=surface,
    )
    await wait_for(lambda: (job := parent.jobs.get(job_id)) is not None and job.status != "running")
    tools = {tool.name for tool in stream.by_job[job_id][0].tools}
    await parent.dispose()
    return tools


@pytest.mark.asyncio
async def test_surface_ceiling_drops_the_network_floor(tmp_path, config_dir):
    floored = await _first_child_tools(tmp_path / "a", None)
    # Control: a role-limited child is handed the web floor today. Without
    # this the assertion below could pass on a build that has no web tools.
    assert {"web_search", "web_fetch"} & floored, floored
    capped = await _first_child_tools(tmp_path / "b", ChildSurface(ceiling=frozenset({"read"})))
    assert capped == {"read"}, capped


# -- 3. host tools reach the child only ---------------------------------------


@pytest.mark.asyncio
async def test_host_tools_exist_in_the_child_not_the_parent(tmp_path, config_dir):
    calls: list[dict[str, Any]] = []

    def reply(request: ChatRequest) -> list[Any]:
        if is_verifier(request):
            if not any(m.role == "tool" or m.role == "toolResult" for m in request.messages):
                return tool_turn("inspect_command", {"command": "cat /tmp/x"})
            return text_turn("B2 holds 42\nVERDICT: PASS")
        return text_turn("parent turn")

    stream = TreeStream(reply)
    parent = make_parent(tmp_path, stream)
    verdict = await parent.verify_completion(
        request="Put 42 in B2.", claim="Done.", host_tools=[inspect_tool(calls)]
    )
    assert calls == [{"command": "cat /tmp/x"}]
    assert verdict.commands == 1 and verdict.status == "PASS"
    assert "inspect_command" not in {tool.name for tool in parent._tools}
    await parent.dispose()


# -- 4. the request cap cancels ------------------------------------------------


@pytest.mark.asyncio
async def test_request_cap_cancels_the_child(tmp_path, config_dir):
    # A verifier that never answers: every turn is another read.
    stream = TreeStream(lambda request: tool_turn("glob", {"pattern": "*.nothing"}))
    parent = make_parent(tmp_path, stream)
    limits = VerifierLimits(max_requests=3, max_commands=10, wall_s=30)

    verdict = await parent.verify_completion(request="r", claim="c", limits=limits)

    assert verdict.ended_by == "requests"
    assert verdict.status == "INCONCLUSIVE"
    made = len(stream.by_job[verdict.job_id])
    assert limits.max_requests <= made <= limits.max_requests + 1, made
    assert verdict.job_id is not None
    job = parent.jobs.get(verdict.job_id)
    assert job is not None and job.status != "running"
    await parent.dispose()


@pytest.mark.asyncio
async def test_command_cap_refuses_and_cancels(tmp_path, config_dir):
    calls: list[dict[str, Any]] = []
    stream = TreeStream(lambda request: tool_turn("inspect_command", {"command": "ls"}))
    parent = make_parent(tmp_path, stream)
    limits = VerifierLimits(max_requests=50, max_commands=2, wall_s=30)

    verdict = await parent.verify_completion(
        request="r", claim="c", host_tools=[inspect_tool(calls)], limits=limits
    )

    assert verdict.ended_by == "commands"
    assert len(calls) == 2  # the third never reached the host
    await parent.dispose()


@pytest.mark.asyncio
async def test_wall_cap_cancels_a_slow_child(tmp_path, config_dir):
    def reply(request: ChatRequest) -> list[Any]:
        return [30.0, *text_turn("VERDICT: PASS")]

    stream = TreeStream(reply)
    parent = make_parent(tmp_path, stream)
    verdict = await parent.verify_completion(
        request="r", claim="c", limits=VerifierLimits(max_requests=10_000, wall_s=0.3)
    )
    assert verdict.ended_by == "wall" and verdict.status == "INCONCLUSIVE"
    await parent.dispose()


# -- 7. never announced to the parent ------------------------------------------


@pytest.mark.asyncio
async def test_verifier_result_is_not_delivered_to_the_parent(tmp_path, config_dir):
    stream = TreeStream(lambda request: text_turn("ANSWER-MARKER\nVERDICT: FAIL"))
    parent = make_parent(tmp_path, stream)

    verdict = await parent.verify_completion(request="r", claim="c")
    assert verdict.job_id is not None
    job = parent.jobs.get(verdict.job_id)
    assert job is not None and job.consumed is True
    # Give any (wrongly) scheduled delivery a chance to run.
    for _ in range(20):
        await asyncio.sleep(0)
    parent_requests = [r for r in stream.requests if not is_verifier(r)]
    assert parent_requests == []
    # Neither the model's history nor the pending-delivery queue holds it.
    history = json.dumps([m.model_dump(mode="json") for m in parent.history()])
    assert "ANSWER-MARKER" not in history, "the verifier's answer reached the parent"
    assert verdict.job_id not in parent._deferred_job_results
    await parent.dispose()


# -- 8. the prompt is only request / claim / observation -----------------------


@pytest.mark.asyncio
async def test_verifier_prompt_is_exactly_role_plus_request_claim_observation(tmp_path, config_dir):
    stream = TreeStream(lambda request: text_turn("VERDICT: PASS"))
    parent = make_parent(tmp_path, stream)
    await parent.verify_completion(
        request="Make the background transparent.",
        claim="Set opacity to 50%.",
        observation="Slide 1 selected",
    )
    [first] = [r for r in stream.requests if is_verifier(r)][:1]
    expected = verifier_profile().preamble + build_verifier_prompt(
        request="Make the background transparent.",
        claim="Set opacity to 50%.",
        observation="Slide 1 selected",
    )
    assert first_user_text(first) == expected
    await parent.dispose()


# -- 9. request rows carry the verifier's identity ------------------------------


@pytest.mark.asyncio
async def test_verifier_rows_carry_job_id_and_role(tmp_path, config_dir):
    stream = TreeStream(lambda request: text_turn("VERDICT: PASS"))
    parent = make_parent(tmp_path, stream)
    verdict = await parent.verify_completion(request="r", claim="c")
    rows = [row for row in stream.rows if row.job_id == verdict.job_id]
    assert rows and all(row.actor_role == VERIFIER_ROLE for row in rows)
    assert verdict.requests == len(rows)
    await parent.dispose()


# -- refusals are skips, never a launch -----------------------------------------


@pytest.mark.asyncio
async def test_no_request_observation_refuses_rather_than_running_uncapped(tmp_path, config_dir):
    plain_calls: list[ChatRequest] = []

    def plain(request: ChatRequest, signal: AbortSignal | None):
        plain_calls.append(request)

        async def gen():
            yield StreamTextDelta(delta="VERDICT: PASS")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    parent = make_parent(tmp_path, plain)  # type: ignore[arg-type]
    verdict = await parent.verify_completion(request="r", claim="c")
    assert verdict.ended_by == "refused" and verdict.status == "INCONCLUSIVE"
    assert plain_calls == [] and parent.jobs.list() == []
    await parent.dispose()


def test_parse_verdict_is_strict() -> None:
    assert parse_verdict("x\nVERDICT: PASS") == ("PASS", "x")
    assert parse_verdict("**VERDICT: fail**")[0] == "FAIL"
    # The last line wins over a quoted format.
    assert parse_verdict("VERDICT: PASS means ok\nB2 is 41\nVERDICT: FAIL")[0] == "FAIL"
    assert parse_verdict("looks good to me")[0] == "INCONCLUSIVE"
    assert parse_verdict(None)[0] == "INCONCLUSIVE"


def test_limits_refuse_nonsense() -> None:
    bad_values: list[dict[str, Any]] = [
        {"max_requests": 0},
        {"max_commands": True},
        {"wall_s": float("inf")},
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            VerifierLimits(**bad)


# -- review round 1 regressions (PR #3837) -------------------------------------


@pytest.mark.asyncio
async def test_a_task_verifier_role_keeps_its_shell_and_the_harness_verifier_is_confined(
    tmp_path, monkeypatch
):
    """R2 / Q1: confinement rides the harness LAUNCH, never the ``verifier``
    name. An operator registry row called ``verifier`` (this machine has one:
    it needs ``gh``/``curl``) launched by ``task`` keeps an ordinary shell and
    its network floor; the harness-built verifier, same name, is confined."""
    from local_operator.harness import subagent as subagent_mod
    from local_operator.harness.subagent import is_readonly_shell_role
    from local_operator.harness.verifier import VERIFIER_TOOL_ALLOWLIST
    from tests.unit.session.test_launch_subagent import (
        OneShotStream,
        _bash_of,
        _sandbox,
        make_session,
    )

    _sandbox(monkeypatch, available=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.chdir(tmp_path)
    # The attached-profile path (``lop exec --profile verifier``) reads the
    # same one predicate.
    assert not is_readonly_shell_role("verifier")
    parent = make_session(tmp_path, OneShotStream())

    ordinary = await subagent_mod._build_child_session(
        label="verifier",
        prompt="p",
        parent_session=parent,
        model_spec=None,
        job_id="job-operator-verifier",
        agent="verifier",
        profile=AgentProfile(name="verifier", tools=("read", "bash", "web_fetch")),
    )
    bash = _bash_of(ordinary)
    assert bash is not None and bash.approval_tier == "exec"
    assert "SANDBOXED" not in bash.description
    assert ordinary._shell_write_root is None and not ordinary._readonly_shell
    assert "web_fetch" in {tool.name for tool in ordinary._tools}

    harness = await subagent_mod._build_child_session(
        label="Verify completion",
        prompt="p",
        parent_session=parent,
        model_spec=None,
        job_id="job-harness-verifier",
        agent=VERIFIER_ROLE,
        profile=verifier_profile(),
        surface=ChildSurface(ceiling=frozenset(VERIFIER_TOOL_ALLOWLIST), readonly_shell=True),
    )
    confined = _bash_of(harness)
    assert confined is not None and confined.approval_tier == "read"
    assert "SANDBOXED" in confined.description
    assert harness._shell_write_root and harness._readonly_shell
    assert not {"web_search", "web_fetch"} & {tool.name for tool in harness._tools}
    await harness.dispose()
    await ordinary.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_cancelling_the_caller_cancels_the_verifier_job(tmp_path, config_dir):
    """R3: a cancelled caller must not leave the child running uncapped."""

    stream = TreeStream(lambda request: [30.0, *text_turn("VERDICT: PASS")])
    parent = make_parent(tmp_path, stream)
    started = asyncio.Event()
    info: dict[str, Any] = {}

    def on_started(payload: dict[str, Any]) -> None:
        info.update(payload)
        started.set()

    caller = asyncio.create_task(
        parent.verify_completion(
            request="r",
            claim="c",
            limits=VerifierLimits(max_requests=100, wall_s=60),
            on_started=on_started,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    await wait_for(lambda: bool(stream.by_job.get(info["job_id"])))
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    job = parent.jobs.get(info["job_id"])
    assert job is not None and job.status == "cancelled", getattr(job, "status", None)
    # The request observer is gone too: nothing of this call stays subscribed.
    assert stream.observers == []
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_verdict_on_the_capped_request_is_kept(tmp_path, config_dir):
    """Q2: the answer given ON request ``max_requests`` wins over the cap."""

    limits = VerifierLimits(max_requests=3, max_commands=10, wall_s=30)

    def reply(request: ChatRequest) -> list[Any]:
        done = sum(1 for m in request.messages if m.role in ("tool", "toolResult"))
        if done < limits.max_requests - 1:
            return tool_turn("glob", {"pattern": "*.nothing"})
        return text_turn("all met\nVERDICT: PASS")

    stream = TreeStream(reply)
    parent = make_parent(tmp_path, stream)
    verdict = await parent.verify_completion(request="r", claim="c", limits=limits)

    assert len(stream.by_job[verdict.job_id]) == limits.max_requests
    assert (verdict.status, verdict.ended_by) == ("PASS", "verdict"), verdict
    await parent.dispose()


def test_parse_verdict_accepts_common_decorations() -> None:
    """M3: decorations a model adds to its last line are not a failure."""

    for line in (
        "`VERDICT: PASS`",
        "## VERDICT: PASS",
        "> VERDICT: PASS",
        "Final VERDICT: PASS",
        "**Final verdict:** PASS",
        "- **VERDICT**: `PASS`",
        "verdict: pass",
    ):
        assert parse_verdict(f"B2 holds 42\n{line}") == ("PASS", "B2 holds 42"), line
    # Still strict where it matters: prose is not a verdict.
    assert parse_verdict("the verdict is clearly a pass")[0] == "INCONCLUSIVE"
    assert parse_verdict("VERDICT: PASSABLE")[0] == "INCONCLUSIVE"


@pytest.mark.asyncio
async def test_recorded_tools_are_what_the_child_was_given(tmp_path, config_dir, monkeypatch):
    """M5: on a host that cannot sandbox, the confined verifier gets NO
    ``bash``, and neither ``verifier_started`` nor the verdict may list one."""
    from tests.unit.session.test_launch_subagent import _sandbox

    _sandbox(monkeypatch, available=False)
    stream = TreeStream(lambda request: text_turn("VERDICT: PASS"))
    parent = make_parent(tmp_path, stream)
    started: list[dict[str, Any]] = []

    verdict = await parent.verify_completion(request="r", claim="c", on_started=started.append)

    given = {tool.name for tool in stream.by_job[verdict.job_id][0].tools}
    assert "bash" not in given
    assert "bash" not in verdict.tools and "bash" not in started[0]["tools"]
    assert set(verdict.tools) == given, (verdict.tools, given)
    await parent.dispose()
