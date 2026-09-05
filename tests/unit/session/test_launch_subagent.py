"""Session._launch_subagent: the production caller of the subagent engine.

``_launch_subagent`` registers one one-shot child run on the parent session's
job manager and emits ``subagent_start`` / ``subagent_end`` on the parent
event stream. These tests drive it through a real ``run_subagent`` against a
ScriptedStream, so the child actually runs its own loop and the parent sees
the full lifecycle.
"""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import weakref

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import (
    AbortSignal,
    AgentEvent,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    SubagentEndEvent,
    SubagentStartEvent,
    Usage,
)
from local_operator.mobile.projection import (
    ProjectionFold,
    SessionProjection,
    fold_messages_to_entries,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


class FailingStream:
    """Fails the child's provider turn after the runner has started."""

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            if False:  # pragma: no cover - makes this an async generator
                yield
            raise RuntimeError("provider failed")

        return gen()


class OneShotStream:
    """Serves exactly one text-only turn (the child's run)."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="child did the work")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, **kwargs) -> Session:
    transcript = Transcript(tmp_path / "sess")
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=kwargs.pop("system_blocks_provider", lambda: ["stable", "env"]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_launch_subagent_runs_child_and_emits_lifecycle(tmp_path, monkeypatch):
    """_launch_subagent registers a task job, the child runs via the parent's
    stream_fn, and subagent_start/end land on the parent stream."""
    # The child writes its transcript under config_dir(); keep it hermetic.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    stream = OneShotStream()
    parent = make_session(tmp_path, stream)

    events: list[AgentEvent] = []
    parent.subscribe(events.append)

    job_id = parent._launch_subagent(label="sub", prompt="go do a thing")
    assert isinstance(job_id, str) and job_id  # a non-empty job id

    # The child is registered on the parent's job manager.
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.type == "task"
    assert job.label == "sub"

    # The parent row links to the child's separately owned ledger only while
    # live; settlement replaces the edge with detached accounting.
    await wait_for(lambda: job.child_jobs is not None)
    assert job.child_jobs is not parent.jobs

    # Wait for the child run to settle and the parent stream to see the end.
    await wait_for(lambda: any(e.type == "subagent_end" for e in events))

    starts = [e for e in events if isinstance(e, SubagentStartEvent)]
    ends = [e for e in events if isinstance(e, SubagentEndEvent)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].job_id == job_id
    assert starts[0].label == "sub"
    assert ends[0].job_id == job_id
    assert ends[0].status == "completed"
    assert "child did the work" in (ends[0].result_text or "")
    await wait_for(lambda: job.status == "completed")
    assert job.child_jobs is None
    assert job.descendant_usage == []

    # The child actually ran its own provider turn through the shared stream.
    assert stream.requests
    assert stream.requests[0].messages
    assert isinstance(stream.requests[0].messages[0], Message)
    assert stream.requests[0].messages[0].text == "go do a thing"

    # ``set_subagent_details`` is metadata-only (the freeze fix keeps child
    # transcript I/O off the Textual loop), so it fills lineage/prompt but NOT
    # the transcript. The full-screen subagent conversation (#298) is hydrated
    # off-loop and published through ``set_subagent_hydrated_details``.
    fold = ProjectionFold(SessionProjection(session_id="root", pid=1))
    fold.set_subagent_details(parent.subagent_comms)
    [row] = fold.projection.subagents
    assert row.prompt == "go do a thing"
    assert row.launch_message_id == f"subagent-launch:{job_id}"
    assert row.result_text == "child did the work"
    assert row.transcript == []  # not hydrated synchronously on the event path

    # The worker path publishes off-loop metadata (todos), but a child
    # transcript is NEVER placed on the projection wire: the list projection is
    # a full repaint pushed ~30x/s and the daemon's control-socket reader caps a
    # single frame at 1 MB, so embedding a per-subagent transcript across a deep
    # roster overran that cap and wedged real-time updates. The transcript is
    # served lazily instead, via ``/api/sessions/{sid}/agents/{job_id}/history``
    # (see ``ProjectionFold.set_subagent_hydrated_details``'s own comment). The
    # call still returns True — the row exists — and must leave the transcript
    # empty even when handed a fully rendered conversation.
    node = parent.subagent_comms.node(job_id)
    assert node is not None and node.session_dir is not None
    hydrated = fold_messages_to_entries(Transcript(node.session_dir).build_llm_history())
    assert fold.set_subagent_hydrated_details(job_id, hydrated, [])
    assert row.transcript == []

    # Item 12: once the task settles, the idle TOP-LEVEL parent re-wakes with
    # the result instead of polling `jobs`; the shared provider sees a second
    # request carrying that job-result custom message.
    await wait_for(lambda: len(stream.requests) >= 2)
    assert any("background job 'sub' completed" in m.text for m in stream.requests[1].messages)

    await parent.dispose()


@pytest.mark.asyncio
async def test_completed_subagent_disposes_its_owned_browser_surface(tmp_path, monkeypatch):
    """Runner completion reaches Session.dispose, whose browser close is the
    fallback when a child misses the before-handoff instruction."""
    from local_operator.harness import subagent as subagent_mod
    from local_operator.tools import builtin

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    calls: list[tuple[str, dict[str, str], str]] = []

    async def fake_bridge_call(tool_call_id, action, params, *, surface=""):
        calls.append((action, params, surface))
        return {}, None

    original_build = subagent_mod._build_child_session

    async def build_with_owned_tab(**kwargs):
        child = await original_build(**kwargs)
        child._browser.surface_id = "bridge:44:childnonce"
        return child

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge_call)
    monkeypatch.setattr(subagent_mod, "_build_child_session", build_with_owned_tab)
    parent = make_session(tmp_path, OneShotStream())

    job_id = parent._launch_subagent(label="sub", prompt="finish and hand off")
    await wait_for(
        lambda: (job := parent.jobs.get(job_id)) is not None and job.status == "completed"
    )

    assert calls == [("close", {"tab": "bridge:44:childnonce"}, "bridge:44:childnonce")]
    await parent.dispose()


@pytest.mark.asyncio
async def test_completion_survives_cancellation_during_end_event_fanout(tmp_path, monkeypatch):
    """A result is final before its end event fan-out begins.

    Cancellation can still interrupt that awaited fan-out, but it must not
    rewrite the durable result or emit a contradictory terminal event. The
    restored projection models the reconnect after the manager row was swept.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    end_emit_started = asyncio.Event()
    completed_events: list[SubagentEndEvent] = []

    async def suspend_first_completed_end(event: AgentEvent) -> None:
        if not isinstance(event, SubagentEndEvent) or event.status != "completed":
            return
        completed_events.append(event)
        if len(completed_events) == 1:
            end_emit_started.set()
            await asyncio.Event().wait()

    parent.subscribe(suspend_first_completed_end)
    job_id = parent._launch_subagent(label="sub", prompt="go do a thing")
    await asyncio.wait_for(end_emit_started.wait(), timeout=5)

    assert await parent.jobs.cancel(job_id) is True
    snapshot = parent.subagent_comms.snapshot()
    restored_parent = make_session(tmp_path / "restored", OneShotStream())
    restored = SubagentComms(restored_parent)
    restored.restore(snapshot)
    fold = ProjectionFold(SessionProjection(session_id="restored", pid=1))
    fold.set_subagent_details(restored)
    [row] = fold.projection.subagents

    assert [(event.status, event.result_text) for event in completed_events] == [
        ("completed", "child did the work"),
        ("completed", "child did the work"),
    ]
    assert row.status == "completed"
    assert row.result_text == "child did the work"
    assert row.error_text == ""
    assert row.progress == ""
    assert row.activity == ""
    job = parent.jobs.get(job_id)
    assert job is not None
    assert (job.status, job.result_text, job.error_text) == (
        "completed",
        "child did the work",
        None,
    )
    await parent.dispose()
    await restored_parent.dispose()


@pytest.mark.asyncio
async def test_failure_survives_cancellation_during_end_event_fanout(tmp_path, monkeypatch):
    """Interrupted delivery retries the authoritative failure for later handlers."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, FailingStream())
    end_emit_started = asyncio.Event()
    first_handler_events: list[SubagentEndEvent] = []
    later_handler_events: list[SubagentEndEvent] = []

    async def suspend_first_failed_end(event: AgentEvent) -> None:
        if not isinstance(event, SubagentEndEvent) or event.status != "failed":
            return
        first_handler_events.append(event)
        if len(first_handler_events) == 1:
            end_emit_started.set()
            await asyncio.Event().wait()

    def later_handler(event: AgentEvent) -> None:
        if isinstance(event, SubagentEndEvent):
            later_handler_events.append(event)

    parent.subscribe(suspend_first_failed_end)
    parent.subscribe(later_handler)
    job_id = parent._launch_subagent(label="sub", prompt="fail this child")
    await asyncio.wait_for(end_emit_started.wait(), timeout=5)

    assert await parent.jobs.cancel(job_id) is True
    await wait_for(lambda: len(later_handler_events) == 1)
    [row] = parent.subagent_comms.roster()
    job = parent.jobs.get(job_id)

    assert [(event.status, event.error_text) for event in first_handler_events] == [
        ("failed", "provider failed"),
        ("failed", "provider failed"),
    ]
    assert [(event.status, event.error_text) for event in later_handler_events] == [
        ("failed", "provider failed")
    ]
    assert (row.status, row.error_text, row.result_text) == (
        "failed",
        "provider failed",
        None,
    )
    assert job is not None
    assert (job.status, job.error_text, job.result_text) == (
        "failed",
        "provider failed",
        None,
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_completed_event_delivery_interruption_reaches_later_subscriber(
    tmp_path, monkeypatch
):
    """A cancelled handler cannot strand subscribers later in the fan-out."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    end_emit_started = asyncio.Event()
    interrupted = False
    later_events: list[SubagentEndEvent] = []

    async def interrupt_first_delivery(event: AgentEvent) -> None:
        nonlocal interrupted
        if isinstance(event, SubagentEndEvent) and event.status == "completed" and not interrupted:
            interrupted = True
            end_emit_started.set()
            raise asyncio.CancelledError

    def later_subscriber(event: AgentEvent) -> None:
        if isinstance(event, SubagentEndEvent):
            later_events.append(event)

    parent.subscribe(interrupt_first_delivery)
    parent.subscribe(later_subscriber)
    job_id = parent._launch_subagent(label="sub", prompt="go do a thing")
    await asyncio.wait_for(end_emit_started.wait(), timeout=5)
    await wait_for(lambda: len(later_events) == 1)

    [event] = later_events
    [row] = parent.subagent_comms.roster()
    job = parent.jobs.get(job_id)
    assert (event.status, event.result_text, event.error_text) == (
        "completed",
        "child did the work",
        None,
    )
    assert (row.status, row.result_text, row.error_text) == (
        "completed",
        "child did the work",
        None,
    )
    assert job is not None
    assert (job.status, job.result_text, job.error_text) == (
        "completed",
        "child did the work",
        None,
    )
    await parent.dispose()


def _agent_fields(**overrides):
    from typing import Any

    from local_operator.agents import AgentEditFields

    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


def test_subagent_only_loads_explicit_specialist_instructions(tmp_path):
    from local_operator.agents import AgentRegistry
    from local_operator.harness.subagent import _specialist_instructions

    registry = AgentRegistry(tmp_path / "config")
    private = registry.create_agent(
        _agent_fields(name="private-chat", description="Personal notes")
    )
    specialist = registry.create_agent(
        _agent_fields(
            name="dashboard-release",
            description="Release the dashboard",
            categories=["specialist"],
        )
    )
    registry.set_agent_system_prompt(private.id, "PRIVATE USER CONTEXT")
    registry.set_agent_system_prompt(specialist.id, "Follow the release checklist.")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    assert _specialist_instructions("private-chat", parent) == ""
    assert _specialist_instructions("dashboard-release", parent) == "Follow the release checklist."


@pytest.mark.asyncio
async def test_specialist_launch_records_the_exact_effective_prompt(tmp_path, monkeypatch):
    from local_operator.agents import AgentRegistry

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    registry = AgentRegistry(tmp_path / "agents")
    specialist = registry.create_agent(
        _agent_fields(
            name="dashboard-release",
            description="Release the dashboard",
            categories=["specialist"],
        )
    )
    registry.set_agent_system_prompt(specialist.id, "Follow the release checklist.")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    job_id = parent._launch_subagent(
        label="release", prompt="Deploy dashboard.", agent="dashboard-release"
    )
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.prompt == "Deploy dashboard."
    assert job.effective_prompt == "Follow the release checklist.\n\nDeploy dashboard."
    assert job.launch_message_id
    node = parent.subagent_comms.node(job_id)
    assert node is not None
    assert node.effective_prompt == job.effective_prompt
    assert node.launch_message_id == job.launch_message_id

    await wait_for(lambda: job.status == "completed")
    session_dir = parent.subagent_comms.session_dir_of(job_id)
    assert session_dir is not None
    transcript_entries = Transcript(session_dir).entries()
    assert any(entry.id == job.launch_message_id for entry in transcript_entries)
    [snapshot] = parent.subagent_comms.snapshot()
    assert snapshot["effective_prompt"] == job.effective_prompt
    assert snapshot["launch_message_id"] == job.launch_message_id
    await parent.dispose()


def test_attach_team_layers_specialist_manager_instructions(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.teams import Team, TeamMember

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    registry = AgentRegistry(tmp_path / "config")
    manager = registry.create_agent(
        AgentEditFields(
            name="dashboard-release",
            security_prompt=None,
            hosting=None,
            model=None,
            description="Release the dashboard",
            tags=None,
            categories=["specialist"],
            last_message=None,
            temperature=None,
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
    registry.set_agent_system_prompt(manager.id, "Follow the dashboard release checklist.")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    parent.attach_team(
        Team(
            id="t1",
            name="feature-release",
            created_date=datetime.now(timezone.utc),
            manager="dashboard-release",
            members=[TeamMember(role="coder")],
            instructions="Review before merge.",
        )
    )

    brief = parent._goal_state.team_brief
    assert "Follow the dashboard release checklist." in brief
    assert "Review before merge." in brief


def test_attach_agent_profile_resolves_roles_and_specialists(tmp_path, monkeypatch):
    """`/agent` scope: registry role wins, explicit specialist works, an
    ordinary conversational row is refused even on an exact name match."""
    from local_operator.agents import AgentRegistry

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    registry = AgentRegistry(tmp_path / "config")
    role = registry.create_agent(
        _agent_fields(name="auditor", description="Audit changes", tags=["role"])
    )
    registry.set_agent_system_prompt(role.id, "You audit changes.")
    specialist = registry.create_agent(
        _agent_fields(
            name="dashboard-sme",
            description="Dashboard release practices",
            categories=["specialist"],
        )
    )
    registry.set_agent_system_prompt(specialist.id, "Follow the dashboard checklist.")
    private = registry.create_agent(_agent_fields(name="private-chat", description="Notes"))
    registry.set_agent_system_prompt(private.id, "PRIVATE USER CONTEXT")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    # A registered role rides the tail with its role preamble.
    assert parent.attach_agent_profile("auditor") == "auditor"
    assert "You audit changes." in parent._goal_state.agent_brief
    assert parent._goal_state.agent_brief.startswith("[role: auditor]")

    # A later /agent REPLACES the earlier agent brief (switching hats, not
    # stacking them) — and a specialist resolves through its explicit marker.
    assert parent.attach_agent_profile("dashboard-sme") == "dashboard-sme"
    assert "Follow the dashboard checklist." in parent._goal_state.agent_brief
    assert "You audit changes." not in parent._goal_state.agent_brief
    assert parent._goal_state.agent_brief.startswith("[agent: dashboard-sme]")

    # An ordinary conversational agent must NOT be attachable by name.
    assert parent.attach_agent_profile("private-chat") is None
    assert "PRIVATE USER CONTEXT" not in parent._goal_state.agent_brief

    # A packaged starter resolves without being installed, so /agent works on
    # a fresh registry too.
    assert parent.attach_agent_profile("reviewer") == "reviewer"
    assert parent._goal_state.agent_brief.startswith("[role: reviewer]")

    assert parent.attach_agent_profile("no-such-name") is None


def test_a_specialist_named_after_a_seed_attaches_its_own_prompt(tmp_path, monkeypatch):
    """A1 regression: a user's specialist named after a packaged seed word must
    ATTACH the specialist's prompt, not the seed's persona.

    `resolve_profile` honours only role rows and otherwise returns the SEED, so
    resolving a specialist AFTER that fallthrough silently shadowed it: bare
    /agent listed the specialist while attach applied the seed. The fix
    resolves the operator's own specialist before the seed, so listing and
    attach agree.
    """
    from local_operator.agents import AgentRegistry

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    registry = AgentRegistry(tmp_path / "config")
    # A specialist deliberately named "reviewer" — a packaged seed word.
    specialist = registry.create_agent(
        _agent_fields(
            name="reviewer",
            description="Our house reviewer with release-specific rules",
            categories=["specialist"],
        )
    )
    registry.set_agent_system_prompt(specialist.id, "HOUSE REVIEW RULES: block on missing tests.")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    assert parent.attach_agent_profile("reviewer") == "reviewer"
    brief = parent._goal_state.agent_brief
    # The SPECIALIST's prompt and marker, never the packaged seed's persona.
    assert "HOUSE REVIEW RULES" in brief, brief
    assert brief.startswith("[agent: reviewer]"), brief
    assert "[role: reviewer]" not in brief, brief


def test_agent_brief_coexists_with_team_brief(tmp_path, monkeypatch):
    """The two briefs live in separate fields: attaching an agent must not
    drop the roster a /team manager is coordinating, and vice versa."""
    from datetime import datetime, timezone

    from local_operator.teams import Team, TeamMember

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    parent.attach_team(
        Team(
            id="t1",
            name="feature-release",
            created_date=datetime.now(timezone.utc),
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="Review before merge.",
        )
    )
    assert parent.attach_agent_profile("reviewer") == "reviewer"
    assert "Review before merge." in parent._goal_state.team_brief
    assert parent._goal_state.agent_brief.startswith("[role: reviewer]")


def test_a_team_manager_specialist_named_after_a_seed_wins_over_the_seed(tmp_path, monkeypatch):
    """The team twin of A1: a manager that is the operator's own specialist
    named after a packaged seed word must layer the SPECIALIST's prompt, not
    the seed's. attach_team shares the same resolver as /agent attach, so the
    two paths cannot disagree about which persona wins."""
    from datetime import datetime, timezone

    from local_operator.agents import AgentRegistry
    from local_operator.teams import Team, TeamMember

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    registry = AgentRegistry(tmp_path / "config")
    # A specialist deliberately named "reviewer" (a packaged seed word), used
    # here as a team's MANAGER.
    manager = registry.create_agent(
        _agent_fields(
            name="reviewer",
            description="Our house reviewer",
            categories=["specialist"],
        )
    )
    registry.set_agent_system_prompt(manager.id, "HOUSE MANAGER RULES: gate on evidence.")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    parent.attach_team(
        Team(
            id="t1",
            name="feature-release",
            created_date=datetime.now(timezone.utc),
            manager="reviewer",
            members=[TeamMember(role="coder")],
            instructions="Review before merge.",
        )
    )
    brief = parent._goal_state.team_brief
    # The manager specialist's prompt, wrapped, never the packaged reviewer seed.
    assert "HOUSE MANAGER RULES" in brief, brief
    assert "<manager-profile>" in brief, brief
    assert "[role: reviewer]" not in brief, brief
    # The team's collaboration brief still rides after the manager preamble.
    assert "Review before merge." in brief, brief


def test_clear_agent_profile_returns_to_base_instructions(tmp_path, monkeypatch):
    """U1: attaching then clearing leaves the agent brief empty (session back on
    its base instructions) without disturbing a separately-held team brief."""
    from local_operator.agents import AgentRegistry

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    registry = AgentRegistry(tmp_path / "config")
    parent = make_session(tmp_path, OneShotStream(), agent_registry=registry)

    assert parent.attach_agent_profile("reviewer") == "reviewer"
    assert parent._goal_state.agent_brief.startswith("[role: reviewer]")
    # The NAME is stamped alongside the brief so the band (U2) can name it.
    assert parent.active_agent == "reviewer"

    parent.clear_agent_profile()
    assert parent._goal_state.agent_brief == ""
    # M1: the NAME must clear too, not just the brief. The band reads
    # ``active_agent``; if the detach dropped only the brief, the segment would
    # keep painting ``◉ reviewer`` while the notice says "no agent active". This
    # assertion is on the REAL Session on purpose — the pilot's FakeSession
    # double already blanked both, so only a real-Session check catches the
    # source drifting from the double.
    assert parent.active_agent == ""
    # Idempotent: clearing again is a harmless no-op.
    parent.clear_agent_profile()
    assert parent._goal_state.agent_brief == ""
    assert parent.active_agent == ""


@pytest.mark.asyncio
async def test_a_team_parent_stamps_the_member_brief_on_the_child(tmp_path, monkeypatch):
    """A manager session's children inherit collaboration and project context
    without the manager restating them in the task prompt."""
    from datetime import datetime, timezone

    from local_operator.teams import Team, TeamMember

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = OneShotStream()
    parent = make_session(tmp_path, stream)
    parent.attach_team(
        Team(
            id="t1",
            name="feature-release",
            created_date=datetime.now(timezone.utc),
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="Review before merge.",
            project="user-dashboard",
        )
    )
    events: list[AgentEvent] = []
    parent.subscribe(events.append)
    parent._launch_subagent(label="code", prompt="implement the button", agent="coder")
    await wait_for(lambda: any(e.type == "subagent_end" for e in events))
    child_prompt = stream.requests[0].messages[0].text
    assert "[team: feature-release]" in child_prompt
    assert "You are coder on this team" in child_prompt
    assert "Review before merge." in child_prompt
    assert "user-dashboard" in child_prompt
    assert "implement the button" in child_prompt
    await parent.dispose()


@pytest.mark.asyncio
async def test_settled_row_does_not_retain_disposed_child_session(tmp_path, monkeypatch):
    """The accounting snapshot must not keep an observability-to-owner edge."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    captured: list[weakref.ReferenceType[Session]] = []

    def remember_child(event):
        if isinstance(event, SubagentStartEvent):
            record = parent.subagent_comms._records[event.job_id]
            assert record.child is not None
            captured.append(weakref.ref(record.child))

    parent.subscribe(remember_child)
    job_id = parent._launch_subagent(label="collectible", prompt="finish")
    await wait_for(
        lambda: (row := parent.jobs.get(job_id)) is not None and row.status == "completed"
    )
    await wait_for(lambda: bool(captured))
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)
    assert captured[0]() is None
    assert parent.jobs.get(job_id) is not None
    await parent.dispose()


@pytest.mark.asyncio
async def test_launch_subagent_is_wired_as_subagent_launcher(tmp_path, monkeypatch):
    """The ToolContext built for a turn carries _launch_subagent as the
    subagent_launcher, so the task tool can call it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = OneShotStream()
    parent = make_session(tmp_path, stream)
    ctx = parent._build_tool_context()
    assert ctx.subagent_launcher is not None
    # The launcher registers on the SAME manager the task/wait/jobs tools see.
    assert ctx.jobs is parent.jobs
    await parent.dispose()


@pytest.mark.asyncio
async def test_launch_subagent_cancels_on_parent_dispose(tmp_path, monkeypatch):
    """A still-running child is cancelled when the parent session disposes,
    because it lives on the parent's job manager."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    class _HangStream:
        """Turns never finish, so the child keeps running until dispose."""

        def __call__(self, request, signal):
            async def gen():
                await asyncio.sleep(30)
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    parent = make_session(tmp_path, _HangStream())
    job_id = parent._launch_subagent(label="slow", prompt="never finish")

    # Give the child a moment to register AND start its turn, so the runner's
    # inner coroutine is genuinely awaited before disposal (cancelling a task
    # that never reached `await coro` leaks an un-awaited coroutine warning).
    def _running():
        job = parent.jobs.get(job_id)
        return job is not None and job.status == "running"

    await wait_for(_running)
    await asyncio.sleep(0.05)
    assert (job := parent.jobs.get(job_id)) is not None
    assert job.status == "running"
    await parent.dispose()
    assert (job := parent.jobs.get(job_id)) is not None
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_hands_off_final_descendant_usage_after_disposal(tmp_path, monkeypatch):
    """Cancellation joins descendant settlement before detaching its ledger.

    This is the production ordering from the review probe: the running row first
    exposes four tokens, cancellation cleanup finalizes six, and zero retention
    removes the row immediately. The parent must keep six without retaining the
    disposed child Session.
    """
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    descendant_started = asyncio.Event()
    child_refs: list[weakref.ReferenceType[Session]] = []
    child_managers: list[AsyncJobManager] = []
    orig_build = subagent_mod._build_child_session

    async def build_with_running_descendant(**kwargs):
        child = await orig_build(**kwargs)
        child.jobs = AsyncJobManager(retention_ms=0)
        child_refs.append(weakref.ref(child))
        child_managers.append(child.jobs)

        async def descendant(job_id, signal, report_progress):
            row = child.jobs.get(job_id)
            assert row is not None
            row.usage = Usage(input_tokens=4)
            descendant_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                # Models a provider call that reports its final receipt while
                # unwinding cancellation, after the pre-dispose live snapshot.
                row.usage = Usage(input_tokens=6)
                raise

        child.jobs.register("task", "nested", descendant)
        return child

    class _HangStream:
        def __call__(self, request, signal):
            async def gen():
                await asyncio.Future()
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    monkeypatch.setattr(subagent_mod, "_build_child_session", build_with_running_descendant)
    parent = make_session(tmp_path, _HangStream())
    job_id = parent._launch_subagent(label="slow", prompt="never finish")
    await asyncio.wait_for(descendant_started.wait(), timeout=5.0)
    row = parent.jobs.get(job_id)
    assert row is not None
    live_child_jobs = row.child_jobs
    assert isinstance(live_child_jobs, AsyncJobManager)
    assert sum(item.input_tokens for item in live_child_jobs.accounting_components()) == 4

    assert await parent.jobs.cancel(job_id) is True
    assert row.status == "cancelled"
    assert row.child_jobs is None
    assert sum(item.input_tokens for item in row.descendant_usage) == 6
    assert child_managers[0].list() == []
    assert sum(item.input_tokens for item in parent.jobs.accounting_components()) == 6
    # The finalizer must finish before the durable ledger snapshot too, not just
    # before the retained row is painted. A restart must own the final six once.
    await parent._await_subagent_roster_writer()
    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR

    checkpoint = json.loads((parent._transcript.directory / SUBAGENT_ROSTER_SIDECAR).read_text())
    restored_manager = AsyncJobManager()
    restored_manager.restore_accounting(
        [Usage.model_validate(row) for row in checkpoint["accounting"]]
    )
    assert sum(item.input_tokens for item in restored_manager.accounting_components()) == 6

    # The probe kept the manager only to inspect zero-retention settlement; drop
    # that artificial reference before proving the production parent edge is gone.
    child_managers.clear()
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)
    assert child_refs[0]() is None
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_inherits_parent_compaction_settings(tmp_path, monkeypatch):
    """A long-running review child must not bypass the operator's compaction
    budget. Live finding: a one-shot child ran 48 requests / 1.5M tokens on the
    default 600k threshold while the parent's absolute trigger was 250k. The
    child Session must receive the parent's compaction settings."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness import subagent as subagent_mod

    capped = CompactionSettings(threshold_tokens=250_000)
    stream = OneShotStream()
    parent = make_session(tmp_path, stream, compaction_settings=capped)

    # Capture the child session constructed by the runner.
    built_children: list[Session] = []
    orig_build = subagent_mod._build_child_session

    async def captured_build(*a, **kw):
        child = await orig_build(*a, **kw)
        built_children.append(child)
        return child

    subagent_mod._build_child_session = captured_build

    job_id = parent._launch_subagent(label="sub", prompt="review")
    await wait_for(lambda: parent.jobs.get(job_id) is not None)
    await wait_for(
        lambda: (job := parent.jobs.get(job_id)) is not None and job.status == "completed"
    )

    assert built_children, "the runner must construct a child session"
    child = built_children[0]
    assert child._compaction_settings is not None
    assert child._compaction_settings.threshold_tokens == 250_000
    assert child._compaction_settings == capped
    await parent.dispose()


# --- session parity: what a child is actually constructed with -----------------------
#
# Reported live: a subagent "worked around its lack of MCP access by using your
# stored OAuth token directly" — it could not call the Linear MCP tools its
# parent had, so it improvised with raw credentials. These tests pin the
# inheritance decisions that failure exposed.


class FakeMcpManager:
    """The manager surface ``_child_mcp_wiring`` and the ``mcp://`` resolver use.

    Real ``McpManager`` construction needs the SDK, transports and a config
    pass; every behaviour under test here is about which tools land on which
    session, so the fake carries exactly the lookup surface both consume.
    """

    def __init__(self, tools: dict[str, list[str]]) -> None:
        async def never_called(tool_call_id, args, signal, on_update, context):
            raise AssertionError("these tests never dispatch an MCP call")

        self._by_server = {
            server: [
                AgentTool(
                    name=f"mcp__{server}_{raw}",
                    description=f"{raw} on {server}",
                    parameters={"type": "object", "properties": {}},
                    execute=never_called,
                )
                for raw in raws
            ]
            for server, raws in tools.items()
        }
        self._meta = {
            tool.name: {"server_name": server, "mcp_tool_name": raw, "deferred": False}
            for server, raws in tools.items()
            for raw, tool in zip(raws, self._by_server[server])
        }

        self.disconnected = 0

    async def disconnect_all(self) -> None:
        """The real manager tears every server down here. A child calling it
        would end MCP for the rest of the PARENT's session."""
        self.disconnected += 1

    def get_all_server_names(self) -> list[str]:
        return sorted(self._by_server)

    def get_connection_status(self, name: str) -> str:
        return "connected"

    def get_server_config(self, name: str) -> object | None:
        return None

    def get_server_tools(self, name: str) -> list[AgentTool]:
        return sorted(self._by_server.get(name, ()), key=lambda tool: tool.name)

    def rebuild_tools(self) -> None:
        """Replace every AgentTool with a fresh, equivalent object, which is
        what a reconnect's ``_register_tools`` does to the real manager."""
        self.__init__(
            {
                s: [self._meta[t.name]["mcp_tool_name"] for t in ts]
                for s, ts in self._by_server.items()
            }
        )

    def get_tools(self) -> list[AgentTool]:
        out = [tool for tools in self._by_server.values() for tool in tools]
        return sorted(out, key=lambda tool: tool.name)

    def get_tool_meta(self, tool_name: str):
        return self._meta.get(tool_name)


async def build_child(parent, model_spec=None, job_id="job-1", agent="task", prompt="do the thing"):
    from local_operator.harness import subagent as subagent_mod

    return await subagent_mod._build_child_session(
        label="sub",
        prompt=prompt,
        parent_session=parent,
        model_spec=model_spec,
        job_id=job_id,
        agent=agent,
    )


def attach_manager(parent, manager) -> None:
    """What ``attach_mcp_dispose`` does for the manager handle, without the SDK:
    the parent HOLDS the manager and the child borrows it off that attribute."""
    parent.mcp_manager = manager


def knowledge_tail(session) -> str:
    """The system prompt's volatile tail — where the MCP catalogue rides."""
    return session._system_blocks_provider()[3]


def resolve(session, url: str) -> str | None:
    """Read one internal URL exactly as the ``read`` tool would."""
    resolver = session._skill_resolver
    assert resolver is not None, "the child must always get an internal-URL resolver"
    return resolver(url)


@pytest.mark.asyncio
async def test_child_gets_the_parents_mcp_manager_and_catalogue(tmp_path, monkeypatch):
    """The reported gap: a child had no MCP at all. It must borrow the parent's
    LIVE manager (never run a second discovery pass) and see the catalogue that
    tells it ``read mcp://<server>`` is available."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)

    child = await build_child(parent, prompt="list the Linear issues")

    assert child.mcp_manager is manager
    tail = knowledge_tail(child)
    assert "<mcps>" in tail
    assert "- linear:" in tail
    # BORROWED, not owned. `attach_mcp_dispose` sets `mcp_manager` AND hangs
    # `disconnect_all` on dispose; the obvious "make the child symmetric with
    # the parent" edit would therefore have the first subagent to finish tear
    # down every server for the rest of the parent's session.
    assert not [hook for hook in child._dispose_hooks if getattr(hook, "__name__", "") == "close"]
    # The child owns its web pool and config subscription, but never the
    # borrowed MCP manager. Assert ownership directly rather than hook count.
    assert manager.disconnect_all not in child._dispose_hooks
    await child.dispose()
    assert manager.disconnected == 0
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_starts_with_the_tools_the_parent_already_activated(tmp_path, monkeypatch):
    """Lazy activation is a token-budget decision the PARENT already made for
    the task it is delegating; the child should not have to re-read the same
    server to get back to where its parent was."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)
    active = manager.get_server_tools("linear")[1]  # mcp__linear_list_teams
    parent.refresh_tools(list(parent._tools) + [active])

    child = await build_child(parent)

    child_mcp = sorted(t.name for t in child._tools if t.name.startswith("mcp__"))
    assert child_mcp == ["mcp__linear_list_teams"]
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_activation_lands_on_the_child_not_the_parent(tmp_path, monkeypatch):
    """``read mcp://linear/list_issues`` inside a child must enable that schema
    on the CHILD. The parent's own resolver chain ends in an activate() bound to
    the parent's inventory, so a child that simply reused it would enable the
    tool on the wrong session and see nothing appear in its own."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.mcp.resources import make_mcp_resolver

    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    # Wired the way the factory wires it: the parent's chain ENDS in an mcp
    # resolver whose activate() targets the parent. Without this the parent's
    # resolver is None, a parent-first chain falls through to the child's, and
    # an inversion of the documented order is undetectable.
    parent_activations: list[tuple[str, str]] = []
    parent = make_session(
        tmp_path,
        OneShotStream(),
        skill_resolver=make_mcp_resolver(
            manager, lambda server, tool: parent_activations.append((server, tool))
        ),
    )
    attach_manager(parent, manager)
    parent_tools_before = [t.name for t in parent._tools]

    child = await build_child(parent)
    assert [t.name for t in child._tools if t.name.startswith("mcp__")] == []

    rendered = resolve(child, "mcp://linear/list_issues")

    assert rendered is not None and "mcp__linear_list_issues" in rendered
    assert [t.name for t in child._tools if t.name.startswith("mcp__")] == [
        "mcp__linear_list_issues"
    ]
    # The parent is untouched: a delegated read must not spend the parent's
    # prompt budget on a schema the parent never asked for, and its activate()
    # must not have been reached at all.
    assert parent_activations == []
    assert [t.name for t in parent._tools] == parent_tools_before
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_resolver_falls_through_to_the_parents_knowledge(tmp_path, monkeypatch):
    """``skill://`` and ``guide://`` are pure lookups over indexes the parent
    has already built. The child does not re-run knowledge SELECTION, but a
    prompt that names a skill must still be able to read it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(
        tmp_path,
        OneShotStream(),
        skill_resolver=lambda url: "SKILL BODY" if url == "skill://deploy" else None,
    )
    attach_manager(parent, FakeMcpManager({"linear": ["list_teams"]}))

    child = await build_child(parent)

    assert resolve(child, "skill://deploy") == "SKILL BODY"
    assert resolve(child, "skill://missing") is None
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_without_parent_mcp_still_builds(tmp_path, monkeypatch):
    """MCP unconfigured, SDK missing, or a host that never wired a manager: the
    child must build with an empty knowledge tail rather than degrade to an
    error, exactly as it did before MCP inheritance existed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    assert parent.mcp_manager is None

    child = await build_child(parent)

    assert child.mcp_manager is None
    assert knowledge_tail(child) == "<skills/>"
    await child.dispose()
    await parent.dispose()


# --- credential inheritance --------------------------------------------------
#
# The live failure (session 835fbcafdc27) had a second half nobody had tested:
# a credential stored on the parent must reach a delegated child, or the child
# improvises with the wrong credential the way the MCP-parity failure above
# showed children improvising with raw tokens. The store is shared BY
# REFERENCE, so these tests pin that decision end to end: system blocks at
# spawn, bash env injection at execution, and a store that moved AFTER the
# child was built.


@pytest.mark.asyncio
async def test_child_system_blocks_advertise_parent_credentials(tmp_path, monkeypatch):
    """A credential stored on the parent before the spawn is named in the
    child's system-prompt tail — names only, never the value."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.variables import VariableStore

    parent = make_session(tmp_path, OneShotStream())
    parent._variables = VariableStore(cwd=str(tmp_path), env={})
    parent._variables.store_credential("OSWORLD_OPENAI_API_KEY", "sk-child-secret-xyz", "command")

    child = await build_child(parent)

    tail = knowledge_tail(child)
    assert "<session-credentials>" in tail
    assert "OSWORLD_OPENAI_API_KEY" in tail
    # The value must never reach any system block. The provider is typed
    # possibly-async (``Callable[..., list[str]] | Callable[..., Awaitable...]]``);
    # the child's is sync, so cast rather than widen the production type.
    from typing import cast

    blocks = cast(list[str], child._system_blocks_provider("test/m"))
    for block in blocks:
        assert "sk-child-secret-xyz" not in block
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_bash_injects_parent_credential_env(tmp_path, monkeypatch):
    """The child's bash tool reads ``context.variables`` — which must be the
    PARENT's store, so a delegated child can use a secret its parent holds
    without either of them ever printing it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools.builtin import execute_bash
    from local_operator.variables import VariableStore

    parent = make_session(tmp_path, OneShotStream())
    parent._variables = VariableStore(cwd=str(tmp_path), env={})
    parent._variables.store_credential("PARENT_TOOL_KEY", "sk-inherited-secret-1", "command")

    child = await build_child(parent)

    # The child's per-turn tool context carries the parent's store — the same
    # object, not a copy, so bash injection and list_variables agree with the
    # parent's live state.
    ctx = child._build_tool_context()
    assert ctx.variables is parent._variables
    result = await execute_bash(
        "bash-1",
        # Presence probe only: the value is redacted from output by design,
        # so the test asserts the env var EXISTS for the child process.
        {"command": 'test -n "$PARENT_TOOL_KEY" && echo present || echo missing'},
        None,
        None,
        ctx,
    )
    assert not result.is_error, result.text
    assert "present" in result.text
    assert "sk-inherited-secret-1" not in result.text
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_sees_a_credential_stored_after_its_spawn(tmp_path, monkeypatch):
    """The store is shared by reference: an operator who stores a credential
    while a long-running child is mid-task must have it reach that child's
    LATER turns without a respawn. Proven end to end through bash."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools.builtin import execute_bash
    from local_operator.variables import VariableStore

    parent = make_session(tmp_path, OneShotStream())
    parent._variables = VariableStore(cwd=str(tmp_path), env={})

    child = await build_child(parent)
    ctx_before = child._build_tool_context()
    result_before = await execute_bash(
        "bash-1",
        {"command": 'test -n "$LATE_KEY" && echo present || echo missing'},
        None,
        None,
        ctx_before,
    )
    assert "missing" in result_before.text

    # Stored AFTER the child exists — the operator's mid-task /credential.
    parent._variables.store_credential("LATE_KEY", "sk-late-secret-2", "command")

    # A later turn rebuilds the tool context; the SAME store (by reference)
    # now injects the new credential.
    ctx_after = child._build_tool_context()
    result_after = await execute_bash(
        "bash-2",
        {"command": 'test -n "$LATE_KEY" && echo present || echo missing'},
        None,
        None,
        ctx_after,
    )
    assert "present" in result_after.text
    assert "sk-late-secret-2" not in result_after.text
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_inherits_a_mid_session_model_override(tmp_path, monkeypatch):
    """``/model`` swaps the spec through ``set_model``. A child spawned after
    that must run the model the operator is now on, not the one the session
    booted with — otherwise a delegated task silently uses a different (and
    differently priced) model than everything around it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    switched = ModelSpec(provider="test", model_id="switched", context_window=42_000)
    parent.set_model(switched)

    child = await build_child(parent)

    assert child.model == switched
    # An explicit per-launch override still wins over the session's spec.
    override = ModelSpec(provider="test", model_id="override", context_window=7_000)
    other = await build_child(parent, model_spec=override)
    assert other.model == override
    await other.dispose()
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_of_top_level_session_keeps_delegation_but_not_wake(tmp_path, monkeypatch):
    """Depth is two: a child of a TOP-LEVEL session keeps task/wait/jobs (its
    own job manager is observable through its own tools and is disposed with
    it, so grandchildren cannot outlive their lineage), while ``wake`` never
    crosses any boundary — a child session ends after one prompt, so a wake
    armed there would be silently lost."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    child = await build_child(parent)

    names = {tool.name for tool in child._tools}
    assert {"task", "wait", "jobs"} <= names
    assert "wake" not in names
    # The prune must not take the ordinary inventory with it.
    assert {"bash", "read", "edit", "write"} <= names
    # ...and the loop sees the same list the session does.
    assert [t.name for t in child._context.tools] == [t.name for t in child._tools]
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_grandchild_cannot_fan_out_but_polls_its_own_background_bash(tmp_path, monkeypatch):
    """One level deeper the SPAWN/PERSIST tools go: a grandchild's children
    would register on a job manager nothing observes and that dies mid-turn.
    But ``jobs`` stays, because the grandchild keeps ``bash`` with
    ``background`` — so it can still poll and cancel the background command it
    is told to (the bash receipt advertises ``jobs(op='peek')``), and without
    ``jobs`` that advice loops forever on ``Tool not found: jobs``. The
    invariant: ``jobs`` survives IFF ``bash`` can still background."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    # A parent that is itself a child is recognisable by its _job_id.
    parent._job_id = "job-parent"

    child = await build_child(parent)

    names = {tool.name for tool in child._tools}
    # No fan-out and no cross-boundary persistence from a grandchild.
    assert names.isdisjoint({"task", "wait", "wake"})
    # ...but it can observe/cancel its OWN background job.
    assert "jobs" in names
    assert {"bash", "read", "edit", "write"} <= names
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_restricted_child_may_discover_mcp_but_never_activate_it(tmp_path, monkeypatch):
    """A restricted role reads the MCP catalogue and cannot enable a new tool.

    The boundary an allowlist draws is around CHANGE, not around reach, so
    discovery (a pure read) resolves. Activation does not: a server's tools are
    minted ``approval_tier="exec"`` because their side effects are unknowable
    from here, so a restricted child must not be able to widen its own surface
    past the parent that delegated to it. It is told WHY rather than handed a
    bare ``None``, because a child that only sees "no" re-reads the same URL.

    The parent's own resolver chain also ends in an MCP resolver bound to the
    PARENT's inventory; falling through to it would activate on the wrong
    session as well as route around this denial, so ``mcp://`` never falls
    through while ``guide://`` still does."""
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod
    from local_operator.mcp.resources import make_mcp_resolver

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"slack": ["post_message"]})
    parent_activations: list[tuple[str, str]] = []

    def parent_resolver(url: str) -> str | None:
        if url == "guide://safe":
            return "safe guide"
        return make_mcp_resolver(
            manager, lambda server, tool: parent_activations.append((server, tool))
        )(url)

    parent = make_session(
        tmp_path,
        OneShotStream(),
        skill_resolver=parent_resolver,
    )
    attach_manager(parent, manager)
    profile = AgentProfile(
        name="reviewer",
        description="reviews without MCP access",
        tools=("read", "grep"),
    )
    child = await subagent_mod._build_child_session(
        label="review",
        prompt="review the Slack integration",
        parent_session=parent,
        model_spec=None,
        job_id="job-restricted",
        agent="reviewer",
        profile=profile,
    )

    assert resolve(child, "guide://safe") == "safe guide"
    # Discovery answers: reading the catalogue enables nothing.
    index = resolve(child, "mcp://slack")
    assert index is not None and "post_message" in index
    # Activation is refused, with the reason, and enables nothing anywhere.
    denied = resolve(child, "mcp://slack/post_message")
    assert denied is not None and "cannot enable new ones" in denied
    assert parent_activations == []
    assert all(not tool.name.startswith("mcp__") for tool in child._tools)
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_restricted_child_inherits_the_mcp_tools_the_parent_enabled(tmp_path, monkeypatch):
    """What the restricted child DOES get: the parent's already-active tools.

    Withholding MCP wholesale from every allowlisted role cost a reviewer or
    scout the reads its role is made of — the ticket, the design doc, the log
    are frequently only reachable through a server. Inheriting is the honest
    line: the parent is unrestricted, it activated these tools for the very
    task it is now delegating, and it stays accountable for them. The child
    still cannot ADD to the set (asserted above)."""
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_issues", "create_issue"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)
    # Only ``list_issues`` is live on the parent, which is what "the set the
    # parent already enabled" means — the child must not receive the rest.
    parent.refresh_tools(
        list(parent._tools) + [t for t in manager.get_tools() if t.name.endswith("list_issues")]
    )

    child = await subagent_mod._build_child_session(
        label="scout",
        prompt="find the ticket",
        parent_session=parent,
        model_spec=None,
        job_id="job-inherit",
        agent="scout",
        profile=AgentProfile(name="scout", tools=("read", "grep")),
    )

    names = {tool.name for tool in child._tools}
    assert "mcp__linear_list_issues" in names
    assert "mcp__linear_create_issue" not in names
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_grandchild_cannot_activate_what_its_restricted_ancestor_was_refused(
    tmp_path, monkeypatch
):
    """The MCP denial is STICKY DOWNWARD, checked at depth 2.

    Review round 1 (R1). Restriction used to be computed from the child's own
    profile alone, so a delegating restricted role had a trivial escape: the
    packaged ``manager`` carries an allowlist AND ``delegate: yes``, keeps
    ``task``, and is handed the parent's live MCP manager. Its child rebuilt
    with ``profile=None``, computed ``restricted=False``, and activated freely
    into that shared manager -- so a manager refused ``delete_issue`` could
    spawn a plain child and have IT enable the tool, an ``approval_tier="exec"``
    write obtained one hop below the boundary that refused it.

    Driven off the REAL packaged seed rather than a synthetic profile, because
    the finding is that a SHIPPED role reaches the escape: a hand-written
    profile in this test could drift from what the seed actually grants and the
    regression would silently stop being covered."""
    from local_operator.agent_profiles import load_seed
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager_seed = load_seed("manager")
    assert manager_seed is not None
    # The two properties that make this role the vector. If a future edit drops
    # either, this test would pass while covering nothing.
    assert manager_seed.tools, "the manager seed must carry an allowlist"
    assert manager_seed.may_delegate, "the manager seed must be able to delegate"

    manager = FakeMcpManager({"linear": ["list_issues", "delete_issue"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)
    # The top-level parent enabled ONLY list_issues; delete_issue is the write.
    parent.refresh_tools(
        list(parent._tools) + [t for t in manager.get_tools() if t.name.endswith("list_issues")]
    )

    child = await subagent_mod._build_child_session(
        label="mgr",
        prompt="coordinate the work",
        parent_session=parent,
        model_spec=None,
        job_id="job-mgr",
        agent="manager",
        profile=manager_seed,
    )
    denied = resolve(child, "mcp://linear/delete_issue")
    assert denied is not None and "cannot enable new ones" in denied
    # It really can delegate -- otherwise there is no depth 2 to test.
    assert "task" in {tool.name for tool in child._tools}

    # The manager delegates an ordinary full child: no profile, no role.
    grandchild = await subagent_mod._build_child_session(
        label="worker",
        prompt="do the thing",
        parent_session=child,
        model_spec=None,
        job_id="job-worker",
        agent="task",
        profile=None,
    )

    escalated = resolve(grandchild, "mcp://linear/delete_issue")
    assert escalated is not None and "cannot enable new ones" in escalated
    names = {tool.name for tool in grandchild._tools}
    assert "mcp__linear_delete_issue" not in names
    # It still INHERITS what the lineage legitimately held.
    assert "mcp__linear_list_issues" in names
    await grandchild.dispose()
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_resumed_child_keeps_the_denial_it_inherited(tmp_path, monkeypatch):
    """The denial survives a resume, end to end (review round 2, R5).

    ``hub op='resume'`` rebuilds a settled child against the comms-owning ROOT
    session rather than the child's real parent, and the child that leaks is a
    plain ``task`` grandchild whose own role says "unrestricted" -- so neither
    the role nor the parent session can re-derive the denial and it has to be
    carried forward from the persisted record. Before the carry, a grandchild
    correctly refused ``delete_issue`` while live came back from a resume with
    that ``approval_tier="exec"`` write in its inventory.

    This asserts the CONSEQUENCE on a really-built session; the plumbing that
    feeds ``restricted`` in from the record is asserted at the comms seam in
    ``tests/unit/harness/test_comms.py``."""
    from local_operator.agent_profiles import load_seed
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager_seed = load_seed("manager")
    assert manager_seed is not None

    manager = FakeMcpManager({"linear": ["list_issues", "delete_issue"]})
    root = make_session(tmp_path, OneShotStream())
    attach_manager(root, manager)
    root.refresh_tools(
        list(root._tools) + [t for t in manager.get_tools() if t.name.endswith("list_issues")]
    )

    mgr = await subagent_mod._build_child_session(
        label="mgr",
        prompt="coordinate",
        parent_session=root,
        model_spec=None,
        job_id="job-mgr",
        agent="manager",
        profile=manager_seed,
    )
    grandchild = await subagent_mod._build_child_session(
        label="worker",
        prompt="work",
        parent_session=mgr,
        model_spec=None,
        job_id="job-gc",
        agent="task",
        profile=None,
    )
    assert getattr(grandchild, subagent_mod.MCP_DENIED_ATTR, False) is True
    resume_dir = grandchild._transcript.directory

    # The resume: rebuilt against the ROOT (unrestricted) with no profile,
    # exactly as ``SubagentComms.resume`` does, carrying only the flag the
    # record persisted.
    resumed = await subagent_mod._build_child_session(
        label="worker",
        prompt="continue",
        parent_session=root,
        model_spec=None,
        job_id="job-gc2",
        resume_dir=resume_dir,
        agent="task",
        profile=None,
        restricted=True,
    )

    denied = resolve(resumed, "mcp://linear/delete_issue")
    assert denied is not None and "cannot enable new ones" in denied
    names = {tool.name for tool in resumed._tools}
    assert "mcp__linear_delete_issue" not in names
    assert "mcp__linear_list_issues" in names
    # The restriction is re-stamped, so a further delegation from the resumed
    # child inherits it too rather than starting clean.
    assert getattr(resumed, subagent_mod.MCP_DENIED_ATTR, False) is True
    await resumed.dispose()
    await grandchild.dispose()
    await mgr.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_an_unrestricted_lineage_still_activates_at_depth_two(tmp_path, monkeypatch):
    """The counter-check to the sticky denial: it must not over-apply.

    Restriction propagates from a restricted ancestor only. A plain ``task``
    lineage has no allowlist anywhere in it, so activation must keep working at
    every depth -- otherwise the R1 fix would have closed the escape by
    breaking ordinary delegation, which no test above would have caught."""
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)

    child = await subagent_mod._build_child_session(
        label="c",
        prompt="p",
        parent_session=parent,
        model_spec=None,
        job_id="job-c",
        agent="task",
        profile=None,
    )
    grandchild = await subagent_mod._build_child_session(
        label="g",
        prompt="p",
        parent_session=child,
        model_spec=None,
        job_id="job-g",
        agent="task",
        profile=None,
    )

    rendered = resolve(grandchild, "mcp://linear/list_issues")
    assert rendered is not None and "Enabled MCP tool" in rendered
    assert "mcp__linear_list_issues" in {tool.name for tool in grandchild._tools}
    await grandchild.dispose()
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_scout_child_is_read_only_and_cannot_delegate(tmp_path, monkeypatch):
    """The scout tier: tool inventory filtered to retrieval (allowlist, not
    tier — no bash, no eval-style execution, no delegation), its prompt
    stamped with the scout preamble, and the capability tools pruned entirely:
    a read-only agent that delegates autonomous work is not read-only.

    Read-only means it CHANGES nothing, not that it reaches nothing. The web
    tools are in the surface because a scout launched to research a question
    on the web otherwise reports that it has no network access and greps the
    local disk for facts that were never on it — the failure this tier exists
    to perform well."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    child = await build_child(parent, agent="scout")

    names = {tool.name for tool in child._tools}
    # hub survives the allowlist: it is the child's ONLY way to answer a
    # parent question, and it cannot edit, write or execute anything.
    assert names <= {
        "read",
        "glob",
        "grep",
        "list_variables",
        "read_variable",
        "web_search",
        "web_fetch",
        "hub",
    }
    assert "hub" in names
    assert {"web_search", "web_fetch"} <= names
    assert {"bash", "edit", "write", "task", "wait", "jobs", "wake"}.isdisjoint(names)
    assert (
        "scout mode" in getattr(child._context.messages[0], "text", "")
        if child._context.messages
        else True
    )
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_tool_restricted_role_keeps_hub_so_it_can_answer(tmp_path, monkeypatch):
    """A role allowlist that does not name ``hub`` must not silence the child.

    The installed reviewer profile restricts tools to read/glob/grep/bash/todo
    and used to lose ``hub`` to the filter, so every ``ask`` to a reviewer
    timed out by design: the child saw the question, found no tool to answer
    with, and the parent burned its budget. ``hub`` is a messaging surface,
    not a capability — sparing it weakens no boundary the allowlist draws.
    """
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    profile = AgentProfile(
        name="reviewer",
        description="reviews, never edits",
        tools=("read", "glob", "grep", "bash", "todo"),
    )
    child = await subagent_mod._build_child_session(
        label="sub",
        prompt="review the diff",
        parent_session=parent,
        model_spec=None,
        job_id="job-1",
        agent="reviewer",
        profile=profile,
    )

    names = {tool.name for tool in child._tools}
    assert names <= {
        "read",
        "glob",
        "grep",
        "bash",
        "todo",
        "hub",
        "jobs",
        # Floored in regardless of what the allowlist names — see
        # ``test_a_stale_role_allowlist_still_gets_the_network_floor``.
        "web_search",
        "web_fetch",
    }
    assert "hub" in names
    # It keeps ``bash`` with ``background``, so it must keep ``jobs`` to poll
    # and cancel that background job — otherwise the bash receipt's advice to
    # ``jobs(op='peek')`` loops on ``Tool not found: jobs``.
    assert "jobs" in names
    # The boundary itself is intact: nothing the allowlist denies survived,
    # and a role that must not fan out still has no spawn/persist tools.
    assert {"edit", "write", "eval", "task", "wait", "wake"}.isdisjoint(names)
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_stale_role_allowlist_still_gets_the_network_floor(tmp_path, monkeypatch):
    """The registry-row case, which is the one that actually broke a machine.

    ``resolve_profile`` is REGISTRY-FIRST, and installing a role freezes its
    allowlist into a ``tools:a,b,c`` tag. A ``scout`` installed before the web
    tools joined the read-only surface therefore keeps an allowlist that names
    neither of them, and editing the packaged seed reaches it never. The floor
    is applied at child construction so such a row regains the network without
    the harness rewriting the operator's profile behind their back.

    The profile below is spelled out as the literal pre-fix tag list rather
    than derived from a constant: the point of the test is that a list frozen
    on disk in the past still works, so it must not track today's constant."""
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    stale = AgentProfile(
        name="scout",
        description="installed before the network tools existed",
        tools=("read", "glob", "grep", "list_variables", "read_variable"),
    )
    child = await subagent_mod._build_child_session(
        label="research",
        prompt="what does the vendor's API return for a 429?",
        parent_session=parent,
        model_spec=None,
        job_id="job-stale",
        agent="scout",
        profile=stale,
    )

    names = {tool.name for tool in child._tools}
    assert {"web_search", "web_fetch"} <= names
    # A FLOOR, not a widening: every write/execution denial the stale row drew
    # is still in force, which is what keeps a reviewer unable to edit its diff.
    assert {"edit", "write", "bash", "eval", "browser", "task"}.isdisjoint(names)
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_network_floor_stays_empty_when_web_tools_are_disabled(tmp_path, monkeypatch):
    """The floor re-admits tools from the session's OWN inventory, so a machine
    with web search/fetch turned off in config contributes nothing to floor and
    the child simply has no network tools — the floor must not mint a tool the
    operator disabled."""
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod

    config = tmp_path / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.yml").write_text(
        "version: '1.0'\nvalues:\n  web_search:\n    enabled: false\n"
        "  web_fetch:\n    enabled: false\n"
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    parent = make_session(tmp_path, OneShotStream())

    child = await subagent_mod._build_child_session(
        label="research",
        prompt="research the thing",
        parent_session=parent,
        model_spec=None,
        job_id="job-nonet",
        agent="scout",
        profile=AgentProfile(name="scout", tools=("read", "grep")),
    )

    names = {tool.name for tool in child._tools}
    assert {"web_search", "web_fetch"}.isdisjoint(names)
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_non_delegating_role_keeps_jobs_to_poll_its_background_bash(tmp_path, monkeypatch):
    """The bug this fixes: a non-delegating role (coder/reviewer) that
    backgrounds a long ``bash`` command spun forever emitting ``Tool not
    found: jobs``. Such a role loses the SPAWN/PERSIST capability tools
    (``task``/``wait``/``wake``) — a slice that fans out is a fan-out nobody
    watches — but it keeps ``bash`` with ``background``, so it must keep
    ``jobs`` to observe and cancel the job that ``bash`` receipt tells it to
    poll. Invariant: ``jobs`` survives IFF ``bash`` can still background."""
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    # A coder-shaped profile: full toolset (no allowlist), does not delegate.
    profile = AgentProfile(
        name="coder",
        description="implements a slice",
        may_delegate=False,
    )
    child = await subagent_mod._build_child_session(
        label="sub",
        prompt="implement the slice",
        parent_session=parent,
        model_spec=None,
        job_id="job-1",
        agent="coder",
        profile=profile,
    )

    names = {tool.name for tool in child._tools}
    assert "bash" in names
    assert "jobs" in names
    assert {"task", "wait", "wake"}.isdisjoint(names)
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_inherits_cwd_approval_handler_and_never_yolo(tmp_path, monkeypatch):
    """The approval HANDLER carries over (the parent's UI surface is the only
    place a human can answer), the ``yolo`` FLAG never does. A parent running
    ``--yolo`` still auto-approves inside the child, because that is what its
    handler returns — the flag's job is to stop the child from bypassing a gate
    the parent is subject to, not to re-gate what the operator already opened."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    async def approve(tool_name: str, description: str) -> bool:
        return True

    parent = make_session(
        tmp_path, OneShotStream(), cwd="/tmp/workspace", request_approval=approve, yolo=True
    )

    child = await build_child(parent)

    assert child._cwd == "/tmp/workspace"
    assert child._request_approval is approve
    assert child._yolo is False
    assert child._build_tool_context().request_approval is approve
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_carries_the_sessions_standing_goal(tmp_path, monkeypatch):
    """``/goal`` is a constraint the operator set on the whole session, and a
    delegated slice is exactly where an unstated constraint gets violated. It
    is read off the parent's live holder on every provider call, so an edit
    reaches a child that is ALREADY RUNNING — a snapshot taken at build time
    would pass every weaker version of this test."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    parent.set_goal("never touch production")

    child = await build_child(parent)
    assert "never touch production" in knowledge_tail(child)

    parent.set_goal("staging only")
    # The child built BEFORE the edit sees it too: that is the live read, and
    # it is the property a build-time snapshot would fail.
    assert "staging only" in knowledge_tail(child)
    later = await build_child(parent)
    assert "staging only" in knowledge_tail(later)
    await later.dispose()
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_childs_approval_reaches_the_host_tagged_with_its_job(tmp_path, monkeypatch):
    """A background child's approval must be distinguishable from the parent
    turn's. Reproduced live before this existed: a subagent running past the
    end of its parent's turn inherited that turn's approval state and had its
    tools denied with no prompt shown to anyone, because the host had no way to
    tell the two apart. Driven through the real loop and the real ``write``
    tool, so the provenance is checked on the path a tool call actually takes,
    not on the context the session was constructed with."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    target = tmp_path / "child-wrote-this.txt"
    asks: list[tuple[str, str | None]] = []

    async def gate(tool_name: str, description: str, job_id: str | None) -> bool:
        asks.append((tool_name, job_id))
        return False  # denial is the interesting answer: it must be scopeable

    class WriteOnce:
        """One write attempt, then a plain answer."""

        def __init__(self) -> None:
            self.turn = 0

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.turn += 1
            first = self.turn == 1

            async def gen():
                if first:
                    yield StreamToolCallDelta(
                        index=0,
                        id="call_1",
                        name="write",
                        argument_delta=json.dumps({"path": str(target), "content": "x"}),
                    )
                    yield StreamEndEvent(stop_reason="toolUse")
                else:
                    yield StreamTextDelta(delta="denied, stopping")
                    yield StreamEndEvent(stop_reason="stop")

            return gen()

    parent = make_session(tmp_path, WriteOnce(), request_approval=gate)
    job_id = parent._launch_subagent(label="writer", prompt="write the file")
    await wait_for(
        lambda: (job := parent.jobs.get(job_id)) is not None
        and job.status in ("completed", "failed")
    )

    assert asks == [("write", job_id)], f"the host saw {asks}"
    assert not target.exists(), "a denied write must not have run"
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_reconnect_swaps_the_managers_tools_under_a_running_child(tmp_path, monkeypatch):
    """The child deliberately does NOT install ``set_on_tools_changed`` — it is
    a single slot the parent holds, and taking it would freeze the parent's
    inventory for the rest of the session. The cost is that a reconnect leaves
    the child holding stale ``AgentTool`` objects. That is only acceptable
    because a stale object still routes (its execute closes over the manager
    and the server/tool pair, not over a connection) and because the child's
    NEXT activation picks up the fresh schemas. Both pinned here; confirmed
    live against the real Linear server before this test existed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)
    child = await build_child(parent)
    resolve(child, "mcp://linear/list_teams")
    stale = next(t for t in child._tools if t.name == "mcp__linear_list_teams")

    manager.rebuild_tools()  # what a reconnect / tools_list_changed does
    assert manager.get_tools()[0] is not stale, "the fake must actually swap objects"
    # The child keeps the object it already had: nothing refreshed it, and
    # nothing had to — the call still routes through the manager.
    assert next(t for t in child._tools if t.name == "mcp__linear_list_teams") is stale
    # The next activation rebuilds the selection from the manager's CURRENT
    # tools, so the child converges on the fresh schemas without a callback.
    resolve(child, "mcp://linear/list_issues")
    refreshed = {t.name: t for t in child._tools if t.name.startswith("mcp__")}
    assert sorted(refreshed) == ["mcp__linear_list_issues", "mcp__linear_list_teams"]
    assert refreshed["mcp__linear_list_teams"] is not stale
    assert [t.name for t in parent._tools if t.name.startswith("mcp__")] == []
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_parent_with_no_configured_servers_gives_an_empty_tail(tmp_path, monkeypatch):
    """A manager exists but discovery found nothing. The child carries only
    the compact discovery escape hatch, and ``mcp://`` still answers rather
    than raising because the resolver is installed either way."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, FakeMcpManager({}))

    child = await build_child(parent)

    assert knowledge_tail(child) == (
        "<mcps>Find MCP tools: `mcp://?search=terms`; list: `mcp://`.</mcps>"
    )
    assert resolve(child, "mcp://nope") == "Unknown MCP server: nope. Available: (none)"
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_launch_prompt_is_recorded_on_the_job(tmp_path, monkeypatch):
    """A reader opening a child's transcript needs the instruction it was given,
    and `label` is only a short summary the launcher wrote for a status line.
    Nothing else can supply it: `Session.prompt` feeds the text straight into
    the turn pipeline without emitting an event, so it never reaches
    `job.trajectory`. Recorded at registration, so a job still QUEUED behind
    the capacity gate — which has no trajectory and may never run — can still
    say what it was asked to do."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    instruction = "review the diff at /tmp/x.diff and report blockers only"

    job_id = parent._launch_subagent(label="review", prompt=instruction)
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.prompt == instruction
    assert job.label == "review", "the summary must stay a separate field"
    # Verbatim, not summarised or truncated: the point is the exact instruction.
    assert job.prompt is not None and len(job.prompt) == len(instruction)

    await wait_for(
        lambda: (settled := parent.jobs.get(job_id)) is not None and settled.status == "completed"
    )
    settled = parent.jobs.get(job_id)
    assert settled is not None and settled.prompt == instruction, "must survive settlement"
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_launch_role_and_effort_are_recorded_on_the_job(tmp_path, monkeypatch):
    """The child's ROLE and effort TIER are stamped on the job at registration,
    on the same rule as `prompt`: the title and the status band name what kind
    of child this is and at what level, and a job still QUEUED behind the
    capacity gate (which never entered its runner) must still be able to say
    both. Effort is recorded here rather than derived from the resolved model
    spec because a tier does not survive that resolution — two tiers can point
    at one model, and a child on the parent's own model still ran at a chosen
    level the band should name.

    The tier is CONFIGURED here because a launch that names a tier now fails
    closed when nothing is configured for it (see
    ``test_pinned_subagent_model``): an ``effort="hi"`` that silently ran on
    the parent's model is the incident that rule exists to prevent, so this
    test can no longer rely on it. The parent's own selector is used so the
    child still runs through the same ``OneShotStream``."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yml").write_text(
        f"values:\n  subagents:\n    models:\n      hi: {MODEL.provider}/{MODEL.model_id}\n"
    )
    parent = make_session(tmp_path, OneShotStream())

    job_id = parent._launch_subagent(
        label="research", prompt="map the repo", agent="scout", effort="hi"
    )
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.agent_role == "scout"
    assert job.effort == "hi"

    await wait_for(
        lambda: (settled := parent.jobs.get(job_id)) is not None and settled.status == "completed"
    )
    settled = parent.jobs.get(job_id)
    # Both survive settlement, like `prompt`: the page stays honest after the
    # child finishes.
    assert settled is not None
    assert settled.agent_role == "scout" and settled.effort == "hi"
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_task_launched_without_an_effort_records_none(tmp_path, monkeypatch):
    """The None-means-not-recorded convention: a plain task with no tier records
    `effort=None`, distinct from a recorded value, so the band's inherit-if-
    same-model fallback (not a printed level) applies."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    job_id = parent._launch_subagent(label="plain", prompt="do the work")
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.agent_role == "task"
    assert job.effort is None
    await wait_for(
        lambda: (settled := parent.jobs.get(job_id)) is not None and settled.status == "completed"
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_scout_preamble_reaches_the_provider_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = OneShotStream()
    parent = make_session(tmp_path, stream)
    job_id = parent._launch_subagent(label="research", prompt="Map the repo.", agent="scout")

    def settled() -> bool:
        job = parent.jobs.get(job_id)
        return job is not None and job.status != "running"

    await wait_for(settled)
    assert stream.requests
    first_user = next(message for message in stream.requests[0].messages if message.role == "user")
    # The ROLE framing reaches the turn ahead of the task, and the task itself
    # survives verbatim. Asserted by behaviour rather than by the seed's exact
    # wording: the scout guidance now lives in an editable profile
    # (``local_operator/agent_seeds/scout.md``), so pinning its prose here
    # would make every improvement to it a test failure.
    assert first_user.text.startswith("[role: scout]"), first_user.text[:80]
    assert "READ-ONLY" in first_user.text
    assert "Map the repo." in first_user.text
    assert first_user.text.index("Map the repo.") > first_user.text.index("READ-ONLY")
    await parent.dispose()


class LongStream:
    """A child turn that streams many deltas, yielding the loop between each.

    A real provider suspends on network I/O between chunks. Reproducing that
    suspension is what makes this a concurrency test rather than a CPU
    benchmark: without it the generator never yields and nothing else could
    run no matter how the harness behaved.
    """

    def __init__(self, deltas: int = 2000, chunk_chars: int = 1200) -> None:
        self.deltas = deltas
        self.chunk = "lorem ipsum dolor " * max(1, chunk_chars // 18)

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            for _ in range(self.deltas):
                await asyncio.sleep(0)
                yield StreamTextDelta(delta=self.chunk)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


@pytest.mark.asyncio
async def test_the_loop_stays_responsive_while_several_subagents_run(tmp_path, monkeypatch):
    """Concurrent children must not stall the event loop they share.

    THE regression this guards. One asyncio loop serves the parent session,
    every subagent and the TUI's repaint, and the per-turn compaction gate
    used to tokenize the whole history INLINE on it. With several children
    streaming at once that made one child's threshold check freeze all of
    them: measured at 2.5 s of maximum loop stall (worst single stall 860 ms,
    with 116 of 121 stall samples inside the tokenizer). The user-visible
    symptom was an agent reporting that its subagents "only run when I yield".

    WHY THIS IS A STRUCTURAL SPY AND NOT A TIME BOUND.

    Two attempts to bound a watchdog failed in the same way. #418 replaced a
    flat 1.0 s wall-clock bound with a calibrated one; #373 is that calibrated
    bound still going red on unmodified ``main`` (reproduced here: 3 of 12
    under six CPU spinners, margins 1.07-1.22x). Converting the watchdog to
    ``time.thread_time`` (the statistic ``test_loop_liveness.py`` adopted in
    #136) removed the *load* sensitivity — wall lateness under 8 CPU hogs
    exploded to 943-2034 ms while loop-thread CPU stayed at 393-492 ms — but
    not the *core-speed* sensitivity. A slower CI core burns more CPU-seconds
    on the same tokenizer pass: ubuntu-latest 3.13 reported 1056 ms and
    1156 ms of loop-thread CPU against a 950 ms bound, on an unmodified
    healthy tree, while the same SHA's parallel run passed. No portable
    numeric bound can sit above CI's healthy 1156 ms and below this box's
    1315 ms regression.

    The contract this test can actually pin, load-immune and core-speed-
    immune, is the one ``test_loop_liveness.py`` already pins for the image
    path: the function that moved OFF the loop is observed running on a
    thread that is not the loop's. ``Session._offloaded`` hops
    ``estimate_messages_tokens`` / ``find_cut_point`` through
    ``asyncio.to_thread`` once the history crosses ``OFFLOAD_MIN_CHARS``.
    Spying those two names on ``local_operator.compaction.api`` (the module
    ``_offloaded`` resolves by name) records the thread they ran on; the
    assertion is that every call of a history large enough to offload ran
    off-loop. Putting the rulers back inline (lifting ``OFFLOAD_MIN_CHARS``
    above any history this workload builds — the 70e66526 seam reversed) is
    the reproduction: the spy then sees the loop thread and the test dies.

    The workload is calibrated, not arbitrary. It has to build a history big
    enough that the gate's tokenizer pass is expensive AND crosses the
    offload threshold, because that is the stretch under test; at a smaller
    size both variants pass and the test proves nothing.
    """
    import local_operator.compaction.api as compaction_api

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, LongStream())

    loop_thread = threading.get_ident()
    seen: dict[str, list[int]] = {
        "estimate_messages_tokens": [],
        "find_cut_point": [],
    }

    def _wrap(name: str, real):
        def spy(*args, **kwargs):
            seen[name].append(threading.get_ident())
            return real(*args, **kwargs)

        return spy

    monkeypatch.setattr(
        compaction_api,
        "estimate_messages_tokens",
        _wrap("estimate_messages_tokens", compaction_api.estimate_messages_tokens),
    )
    monkeypatch.setattr(
        compaction_api,
        "find_cut_point",
        _wrap("find_cut_point", compaction_api.find_cut_point),
    )

    job_ids = [parent._launch_subagent(label=f"c{i}", prompt="do the work") for i in range(6)]

    def all_settled() -> bool:
        jobs = [parent.jobs.get(job_id) for job_id in job_ids]
        return all(job is not None and job.status != "running" for job in jobs)

    # Waits on the children settling, never on a frame or time budget: the
    # timeout is a deadlock guard, and a run that reaches it has no result
    # to assert on rather than a slow one.
    await wait_for(all_settled, timeout=120.0)

    # At least one ruler must have run — otherwise the workload no longer
    # crosses the offload threshold and this test is watching nothing.
    total = sum(len(idents) for idents in seen.values())
    assert total, (
        "neither compaction ruler ran — the history this workload builds is "
        "no longer large enough to exercise the offload seam, so the "
        "assertion below would pass vacuously"
    )
    on_loop = {
        name: sum(1 for ident in idents if ident == loop_thread) for name, idents in seen.items()
    }
    assert all(n == 0 for n in on_loop.values()), (
        f"a compaction ruler ran on the event-loop thread "
        f"({on_loop}; {total} calls total) — the asyncio.to_thread hop is "
        "gone and a child's tokenizer pass is starving its siblings"
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_launched_subagent_stays_out_of_the_resume_picker(tmp_path, monkeypatch):
    """The reported bug, end to end: `/resume` listed the machine's own runs.

    A child session is an ephemeral directory under ``sessions/`` with exactly
    the shape of a real conversation, so nothing on disk told them apart and
    the picker named each delegated run by its role preamble. This drives the
    production launch path and then asks the same function the picker calls,
    rather than asserting on the marker file — the marker is the mechanism, and
    the row count is the promise.
    """
    from local_operator.resume import (
        ORIGIN_SUBAGENT,
        recent_session_rows,
        session_origin,
    )

    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

    # The parent is a user session: an ephemeral directory under sessions/,
    # which is what session_factory hands a TUI run.
    parent_dir = config / "sessions" / "parentsession"
    transcript = Transcript(parent_dir)
    await transcript.append_message(Message.user("fix the resume picker"))
    parent = Session(
        model=MODEL,
        stream_fn=OneShotStream(),
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable", "env"],
    )

    events: list[AgentEvent] = []
    parent.subscribe(events.append)
    parent._launch_subagent(label="reviewer", prompt="[role: reviewer] review the diff")
    await wait_for(lambda: any(e.type == "subagent_end" for e in events))

    sessions = sorted(p.name for p in (config / "sessions").iterdir())
    assert len(sessions) == 2, f"expected the parent and its child on disk, got {sessions}"
    child_dir = next(p for p in (config / "sessions").iterdir() if p.name != "parentsession")
    # The nonzero control: the child really did write a transcript, so an empty
    # picker row set below cannot be a directory that was never created.
    assert (child_dir / "transcript.jsonl").is_file()

    # The promise first, then the mechanism that delivers it: a reader who
    # only trusts one line should trust the row set.
    rows = recent_session_rows(config, limit=50)
    assert [row.id for row in rows] == ["parentsession"]
    assert [row.name for row in rows] == ["fix the resume picker"]
    assert session_origin(child_dir) == ORIGIN_SUBAGENT

    await parent.dispose()


@pytest.mark.asyncio
async def test_hub_ask_reaches_a_child_under_the_eager_task_factory(tmp_path, monkeypatch):
    """The observed wedge, end to end. Textual installs
    ``asyncio.eager_task_factory`` on its loop (``textual/app.py``), which makes
    ``ensure_future`` execute a new coroutine synchronously up to its first
    suspension. The subagent runner registered inside ``jobs_manager.register``
    can therefore build its child and call ``comms.attach`` BEFORE ``register``
    returns to ``run_subagent`` — which then calls ``record_launch`` on an
    already-attached record. The pre-fix ``record_launch`` REPLACED that
    record, discarding the live child, the reply watcher and the session
    directory: every later ``hub`` send/steer/ask buffered into ``pending`` on
    a record whose flush (attach) had already happened and would never happen
    again. Live, a healthy reviewer worked 41 minutes while two ``hub ask``
    status checks never reached it, it was cancelled as wedged, and the roster
    reported the settled child as "never started, so it has no transcript".

    This drives the real launch path on an eager loop with a child that stays
    in a slow tool call (the wedged reviewer's shape), asks it mid-run, and
    asserts the question reaches the child's next provider request, the
    child's prose reply resolves the ask, and the roster keeps the transcript.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    loop = asyncio.get_running_loop()
    old_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)

    try:
        from local_operator.harness import subagent as subagent_mod
        from local_operator.harness.types import AgentTool, TextContent, ToolResult

        slow_runs = [0]

        async def slow_execute(tool_call_id, args, signal=None, on_update=None, context=None):
            slow_runs[0] += 1
            await asyncio.sleep(0.4)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name="slow",
                content=[TextContent(text="ok")],
            )

        slow_tool = AgentTool(
            name="slow",
            label="slow",
            description="slow test tool",
            parameters={"type": "object", "properties": {}},
            execute=slow_execute,
        )

        class SlowToolChildStream:
            """One slow-tool turn, then a final text turn; records whether a
            parent message ever reached the child's context."""

            def __init__(self) -> None:
                self.calls = 0
                self.parent_message_seen = False

            def __call__(self, request: ChatRequest, signal):
                self.calls += 1
                for m in request.messages:
                    if "parent-message" in m.text:
                        self.parent_message_seen = True
                n = self.calls

                async def gen():
                    if n == 1:
                        yield StreamToolCallDelta(
                            index=0, id="tc1", name="slow", argument_delta="{}"
                        )
                        yield StreamEndEvent(stop_reason="toolUse")
                    else:
                        yield StreamTextDelta(delta="child done, all fine")
                        yield StreamEndEvent(stop_reason="stop")

                return gen()

        child_stream = SlowToolChildStream()
        orig_build = subagent_mod._build_child_session

        async def build_with_slow_tool(**kwargs):
            child = await orig_build(**kwargs)
            child._tools = [slow_tool]
            child._context.tools = [slow_tool]
            child._stream_fn = child_stream
            return child

        monkeypatch.setattr(subagent_mod, "_build_child_session", build_with_slow_tool)

        parent = make_session(tmp_path, child_stream)
        job_id = parent._launch_subagent(label="reviewer", prompt="review the diff")
        comms = parent.subagent_comms

        # The eager runner attached before record_launch ran; the fix must have
        # merged, so the live child is still addressable right now.
        record = comms._records[job_id]
        assert record.child is not None, "attach was clobbered by record_launch"
        assert record.session_dir is not None

        # The child is parked in its slow tool call; ask it mid-run, exactly
        # the parent's status-check pattern that used to time out.
        reply = await comms.ask(job_id, "Status check: where are you?", 10_000)
        assert reply.timed_out is False
        assert reply.error is None
        assert "child done" in (reply.text or "")
        assert child_stream.parent_message_seen, "the question never reached the child"
        assert slow_runs[0] == 1

        # The settled child keeps its transcript on the roster, so resume and
        # the roster's "never started" lie are both gone.
        def _settled() -> bool:
            job = parent.jobs.get(job_id)
            return job is None or job.status != "running"

        await wait_for(_settled)
        [row] = [info for info in comms.roster() if info.job_id == job_id]
        assert row.session_id == record.session_dir.name
        assert "never started" not in (row.detail or "")

        await parent.dispose()
    finally:
        loop.set_task_factory(old_factory)


@pytest.mark.asyncio
async def test_a_childs_tool_context_carries_its_label_and_its_parents_live_title(
    tmp_path, monkeypatch
):
    """The browser tab group of a delegated child must name the work, not `Session`.

    A subagent never generates a conversation title — naming runs in the TUI
    host and the owned-session runtime and a one-shot child passes through
    neither — so its ONLY display identity is the label its parent launched it
    under plus its parent's title. Both have to survive the trip to the
    EXECUTE-time context (``_build_tool_context``), which is rebuilt per turn
    and is the one a tool actually sees; the construction-time context that
    ``create_tools`` inspects is not it.

    The parent's title is asserted through a rename performed AFTER the child
    was built, because that is the real sequence: a parent is named a second or
    two into its first turn while its children are launched later, so a value
    snapshotted at child construction would be empty for the child's whole life.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools import builtin

    stream = OneShotStream()
    # A real cwd, because the unnamed-parent half of the label falls back to
    # its basename and the Session default (".") has none.
    parent = make_session(tmp_path, stream, cwd="/Users/damian/local-operator")
    child = await build_child(parent)

    context = child._build_tool_context()
    assert context.job_label == "sub"
    assert context.job_id == "job-1"
    # Its own identity is untouched: the borrowed NAME must never become a
    # borrowed identity.
    assert context.session_id == child.session_id != parent.session_id

    # Unnamed parent: the child is still distinguishable from its siblings by
    # its own label, where before it fell back to the cwd every sibling shares.
    assert builtin._browser_session_label(context).endswith("› sub")

    # The rename case, end to end through the real holder the child was handed.
    parent.set_conversation_name("Fix tab groups")
    assert builtin._browser_session_label(child._build_tool_context()) == "Fix tab groups › sub"

    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_top_level_session_is_never_composed_as_a_subagent(tmp_path):
    """The discriminator must not misfire on the operator's own session."""
    from local_operator.tools import builtin

    stream = OneShotStream()
    parent = make_session(tmp_path, stream)
    parent.set_conversation_name("Debug browser extension port binding issue")

    context = parent._build_tool_context()
    assert context.job_id is None and context.job_label == ""
    # The exact label the operator's pill should have read.
    assert builtin._browser_session_label(context) == "Debug browser extension…"
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_grandchilds_pill_names_the_conversation_not_the_shared_cwd(tmp_path, monkeypatch):
    """Depth 2 must resolve to the TOP-LEVEL title, not fall back to the cwd.

    Delegation nests: a child of a top-level session keeps ``task``/``wait``/
    ``jobs``, so a manager fanning out to workers is depth 2 and is the shape
    the operator actually runs. The middle child has no title of its own and
    can never grow one, so handing a grandchild the middle child's title
    HOLDER yielded "" forever and the pill fell through to the parent cwd that
    every session in the repo shares. Resolving the parent's DISPLAY name
    walks past the untitled middle to the conversation that owns the work.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools import builtin

    top = make_session(tmp_path, OneShotStream(), cwd="/Users/damian/local-operator")
    top.set_conversation_name("Fix browser tab group naming")
    middle = await build_child(top)
    grandchild = await build_child(middle, job_id="job-2")

    assert middle.conversation_name == "", "a subagent never holds a title of its own"
    assert (
        builtin._browser_session_label(grandchild._build_tool_context())
        == "Fix browser tab group… › sub"
    )

    await grandchild.dispose()
    await middle.dispose()
    await top.dispose()


@pytest.mark.asyncio
async def test_grandchildren_of_different_conversations_do_not_collide(tmp_path, monkeypatch):
    """The reported collision, one level deeper than the direct-child fix.

    Two managers under two different conversations each fan out to a worker
    called ``qa``. Before the transitive resolution both rendered
    ``local-operator › qa`` — the same class of indistinguishable pill the
    direct-child case was fixed for.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools import builtin

    async def qa_pill(title: str, session_dir: str) -> str:
        top = make_session(
            tmp_path / session_dir, OneShotStream(), cwd="/Users/damian/local-operator"
        )
        top.set_conversation_name(title)
        manager = await build_child(top)
        worker = await build_child(manager, job_id="job-2")
        pill = builtin._browser_session_label(worker._build_tool_context())
        await worker.dispose()
        await manager.dispose()
        await top.dispose()
        return pill

    first = await qa_pill("Fix browser tab group naming", "a")
    second = await qa_pill("Add Radient OAuth PKCE sign-in", "b")

    assert first != second
    assert first.endswith("› sub") and second.endswith("› sub")


@pytest.mark.asyncio
async def test_a_late_rename_of_the_grandparent_reaches_the_grandchild(tmp_path, monkeypatch):
    """The live-resolution property has to survive the extra hop.

    A title normally lands a second or two into the top-level session's first
    turn, while children — and their children — are launched later. Each hop
    resolves on read, so the rename reaches depth 2 on the grandchild's next
    command exactly as it reaches depth 1.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools import builtin

    top = make_session(tmp_path, OneShotStream(), cwd="/Users/damian/local-operator")
    middle = await build_child(top)
    grandchild = await build_child(middle, job_id="job-2")

    # Unnamed grandparent: the cwd stands in, and the label still separates it
    # from its siblings.
    assert builtin._browser_session_label(grandchild._build_tool_context()) == (
        "local-operator › sub"
    )

    top.set_conversation_name("Fix tab groups")
    assert (
        builtin._browser_session_label(grandchild._build_tool_context()) == "Fix tab groups › sub"
    )

    await grandchild.dispose()
    await middle.dispose()
    await top.dispose()


@pytest.mark.asyncio
async def test_the_parent_name_resolver_holds_its_parent_weakly(tmp_path, monkeypatch):
    """The display-name resolver must not add a child→parent retention edge.

    A detached child can outlive the session that launched it, and a strong
    reference back would pin the parent's whole graph — transcript, tools, MCP
    manager — for that child's lifetime. Asserted on the resolver itself
    rather than through a live child, because a child already shares its
    parent's ``SubagentComms`` and that edge (not this one) governs when a
    parent is actually collectable; this test pins the property THIS code owns.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.harness import subagent as subagent_mod

    parent = make_session(tmp_path, OneShotStream(), cwd="/Users/damian/local-operator")
    parent.set_conversation_name("Fix tab groups")
    resolve = subagent_mod._parent_display_name_resolver(parent)
    assert resolve() == "Fix tab groups"

    parent_ref = weakref.ref(parent)
    await parent.dispose()
    del parent
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)

    # The resolver alone never kept it alive, and a dead parent lends no name.
    assert parent_ref() is None
    assert resolve() == ""


@pytest.mark.asyncio
async def test_a_swept_child_keeps_its_durable_identity_on_the_roster(tmp_path, monkeypatch):
    """The retention sweep releases execution evidence, never identity.

    ``AsyncJobManager._sweep_due`` deletes a settled row from ``_jobs`` five
    minutes after it settles (``DEFAULT_RETENTION_MS``), which is what makes a
    long session's memory bounded. The row must nevertheless keep riding
    ``state.jobs``, because that projection is the ONLY thing a follower sees:
    the comms registry cannot cross the socket, so a child whose row vanished
    would lose the ``session_id``/``session_dir`` its page needs to reach a
    ``transcript.jsonl`` that outlives the process forever.

    It survives through ``_ChildRecord.job_ref`` — the registry holds a live
    reference the sweep's ``del`` cannot free — which ``comms.job_rows()``
    re-adds to the roster. Pinned here because that is a load-bearing
    consequence of an unrelated-looking field, and a future sweep that also
    dropped the comms record would silently reintroduce the defect (a settled
    child rendering "no saved transcript" over a fully intact transcript).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    try:
        job_id = parent._launch_subagent(label="swept", prompt="go do a thing")
        await asyncio.wait_for(parent.jobs.settled_event(job_id).wait(), timeout=10)
        child_dir = parent.subagent_comms.session_dir_of(job_id)
        assert child_dir is not None and (child_dir / "transcript.jsonl").exists()

        # Exactly what the five-minute retention does, without waiting for it.
        parent.jobs._retention_ms = 0
        parent.jobs._sweep_due()
        assert parent.jobs.get(job_id) is None, "the execution row must be swept"

        parent.refresh_frontend_state()
        row = next((row for row in parent.frontend_state.jobs if row.id == job_id), None)
        assert row is not None, "a swept child must not vanish from the roster"
        assert row.session_id == child_dir.name
        assert row.session_dir == str(child_dir)
        assert row.status == "completed"
        # The owner's own page reads the live registry, which also survives.
        assert parent.subagent_comms.session_dir_of(job_id) == child_dir
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_restored_swept_child_still_points_at_its_transcript(tmp_path, monkeypatch):
    """The state the operator's machine was actually in.

    A session that is RESTARTED after a child was swept persists an asymmetric
    roster: ``_persist_subagent_roster`` writes ``jobs`` from ``jobs.list()``
    (the swept row is gone) but ``records`` from the comms snapshot (kept, so
    resume works). Restore therefore rebuilds those children from the comms
    graph alone, as ``restored`` rows with an EMPTY trajectory — the honest
    result, since in-memory execution evidence is precisely what the sweep
    released.

    What must NOT be lost is the way back to the durable transcript. Observed
    live on a nine-child session that persisted one job row: the eight
    reconstructed children carried ``session_id`` but no directory, so the
    follower's page painted "no saved transcript" over a 1153-row file.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    try:
        job_id = parent._launch_subagent(label="swept", prompt="go do a thing")
        await asyncio.wait_for(parent.jobs.settled_event(job_id).wait(), timeout=10)
        child_dir = parent.subagent_comms.session_dir_of(job_id)
        snapshot = parent.subagent_comms.snapshot()
    finally:
        await parent.dispose()

    # The restart: comms records survive to disk, the swept job row does not.
    resumed = make_session(tmp_path / "resumed", OneShotStream())
    try:
        resumed.subagent_comms.restore(snapshot)
        assert resumed.jobs.get(job_id) is None, "no execution row survives a restart"

        resumed.refresh_frontend_state()
        row = next((row for row in resumed.frontend_state.jobs if row.id == job_id), None)
        assert row is not None, "the comms graph must still reconstruct the child"
        assert row.restored is True
        assert not row.trajectory, "a swept child has no in-memory trajectory to serve"
        # The one fact the page cannot do without.
        assert row.session_id == child_dir.name
        assert row.session_dir == str(child_dir)
    finally:
        await resumed.dispose()
