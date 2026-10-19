"""TeamMember ``tier``/``effort`` overrides: what a member launch RESOLVES to.

Design note layer 2 (``agent-model-tiers-design.md``), plan PR A. Every test
asserts something an operator or a child OBSERVES — the model and reasoning
depth the child's own provider request carried, the refusal text, the bytes on
disk, the receipt line — never an internal call. The point of the feature is
that one roster can pin ONE member's model without moving the rest, so the
observable is always "which model did that child run on", and the negative half
("which model did the OTHER child run on") is asserted beside it in the arms
where an over-broad lookup would pass silently.

The rig is a real ``Session`` with a real ``TeamRegistry`` and a real launch:
``_launch_subagent`` → a job row → the request the child actually sent. Resumes
go through ``hub op='resume'`` on the same rig, because a resume is a SECOND
LAUNCH and the roster has to be re-derived from the child's record.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from local_operator.agents import AgentRegistry
from local_operator.harness.subagent import SubagentModelUnavailable, TeamLaunchError
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolContext,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tools.agent_tool import AgentParams, write_profile
from local_operator.tools.builtin import execute_hub, execute_task
from local_operator.tools.team_tool import execute_team

#: The launching session's own model, with a real ladder so an inherited EFFORT
#: is observable on the child's request rather than only a model label.
BIRTH = ModelSpec(
    provider="openrouter",
    model_id="anthropic/claude-opus-5-5",
    context_window=100_000,
    max_output_tokens=4_096,
    reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
    reasoning_effort="high",
    reasoning_default_effort="medium",
)

#: A second model WITH a ladder, so a member tier that moves the MODEL can be
#: told apart from one that only moves the depth (Opus 4.5 stops at ``high``).
OTHER = "anthropic/claude-opus-4-5"
#: Models the catalogue does not know at all: the launch still builds a spec,
#: which is exactly what a tier pointing at a new provider looks like.
LO = "test/lo-model"
MED = "test/med-model"
HI = "test/hi-model"

#: The role these roster tests run on when they are about the ROSTER and not
#: about a role's own pin. It is deliberately NOT ``coder``: the packaged coder
#: seed carries the inherit sentinel ("always run on the launching session's
#: model"), and that pin outranks the roster, so a roster's own mechanics — the
#: slot's tier moving the model, an unconfigured slot tier refusing the launch,
#: the receipt naming the team as the decider — are exercised on a role that
#: pins nothing. The interaction between the two has its own tests.

RESUME_PROMPT = "carry on from where you stopped"


class RecordingStream:
    """Answers every call at once and records the request that asked."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    def _text(self, request: ChatRequest) -> str:
        return " ".join(
            block.text
            for message in request.messages
            for block in (getattr(message, "content", None) or [])
            if isinstance(block, TextContent)
        )

    def requests_for(self, needle: str) -> list[ChatRequest]:
        """The requests whose conversation carries ``needle``.

        The parent and every child share this stream, so a launch's own calls
        are picked out by a marker only that launch's prompt contains.
        """
        return [request for request in self.requests if needle in self._text(request)]

    def model_for(self, needle: str) -> str:
        (request,) = self.requests_for(needle)
        return f"{request.model.provider}/{request.model.model_id}"

    def effort_for(self, needle: str) -> str | None:
        (request,) = self.requests_for(needle)
        return request.model.reasoning_effort


class HoldingStream(RecordingStream):
    """Holds ONE launch's first turn open so the child stays a live session.

    Needed wherever a child has to launch a child of its own: a child that has
    answered its only turn is settled, and a settled session cannot launch.
    """

    def __init__(self, hold: str) -> None:
        super().__init__()
        self.hold = hold
        self.release = asyncio.Event()

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        holding = self.hold in self._text(request) and not self.release.is_set()

        async def gen():
            if holding:
                await self.release.wait()
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    def model_for(self, needle: str) -> str:
        (request,) = self.requests_for(needle)
        return f"{request.model.provider}/{request.model.model_id}"

    def effort_for(self, needle: str) -> str | None:
        (request,) = self.requests_for(needle)
        return request.model.reasoning_effort


async def wait_for(predicate, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


def write_config(config_dir: Path, **values: Any) -> None:
    """Write ``values.subagents.*`` verbatim, the way ``lop config`` does."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yml").write_text(yaml.safe_dump({"values": {"subagents": values}}))


def write_tiers(config_dir: Path, **tiers: Any) -> None:
    write_config(config_dir, models=tiers)


def make_team(registry: TeamRegistry, name: str, members: list[TeamMember], **kwargs: Any) -> Any:
    return registry.create_team(
        TeamEditFields(
            name=name,
            manager=kwargs.pop("manager", "manager"),
            members=members,
            instructions=kwargs.pop("instructions", "Work together."),
            project=kwargs.pop("project", "widgets"),
            **kwargs,
        )
    )


def pin_role(registry: AgentRegistry, name: str, effort: str) -> None:
    """A role row carrying an ``effort:`` tier pin — the layer under the roster."""
    write_profile(
        registry,
        AgentParams(
            op="create",
            name=name,
            description=f"{name} role",
            instructions=f"You are {name}.",
            effort=effort,
        ),
        creating=True,
    )


def make_session(
    tmp_path: Path,
    registry: TeamRegistry,
    stream: RecordingStream,
    *,
    team: str | None = None,
    agents: AgentRegistry | None = None,
    transcript: Path | None = None,
) -> Session:
    session = Session(
        model=BIRTH,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(transcript or (tmp_path / "sess")),
        system_blocks_provider=lambda: ["stable"],
        team_registry=registry,
    )
    if agents is not None:
        session.agent_registry = agents
    if team is not None:
        found = registry.get_team_by_name(team)
        assert found is not None
        session.attach_team(found)
    return session


def job_row(session: Session, job_id: str) -> Any:
    return session.subagent_comms.job(job_id)


def completed(session: Session, job_id: str) -> bool:
    row = job_row(session, job_id)
    return row is not None and row.status == "completed"


async def launch(session: Session, stream: RecordingStream, marker: str, **kwargs: Any) -> Any:
    """Launch one child on ``marker``, wait for it to finish, return its row."""
    job_id = session._launch_subagent(label=kwargs.pop("label", "child"), prompt=marker, **kwargs)
    await wait_for(lambda: completed(session, job_id))
    return job_row(session, job_id)


async def hub_resume(session: Session, job_id: str) -> str:
    result = await execute_hub(
        "call-resume",
        {"op": "resume", "to": [job_id], "message": RESUME_PROMPT},
        None,
        None,
        session._build_tool_context(),
    )
    assert not result.is_error, result
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def resumed_id(session: Session, text: str) -> str:
    marker = "resumed as job "
    assert marker in text, text
    return text.split(marker, 1)[1].split()[0]


# ---------------------------------------------------------------------------
# 1-3: resolution, and each field falling through on its own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_member_tier_moves_only_the_role_the_roster_names(tmp_path, monkeypatch):
    """Plan A.1. The override belongs to ONE slot of ONE roster.

    The same role launched from a team that does not name it, and from a
    session with no team at all, must stay on the row's own ``effort:`` tag —
    a lookup that read the wrong roster (or ignored the roster entirely) would
    move all three.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO, "med": MED, "hi": HI})
    agents = AgentRegistry(tmp_path / "agents")
    pin_role(agents, "coder", "med")
    registry = TeamRegistry(config)
    make_team(registry, "pinned", [TeamMember(role="coder", tier="hi")])
    make_team(registry, "plain", [TeamMember(role="coder")])

    stream = RecordingStream()
    pinned = make_session(tmp_path, registry, stream, team="pinned", agents=agents)
    plain = make_session(tmp_path, registry, stream, team="plain", agents=agents)
    rootless = make_session(tmp_path, registry, stream, agents=agents)
    try:
        assert (await launch(pinned, stream, "run-under-pinned", agent="coder")).model_label == HI
        assert (await launch(plain, stream, "run-under-plain", agent="coder")).model_label == MED
        assert (
            await launch(rootless, stream, "run-under-nothing", agent="coder")
        ).model_label == MED
        # And the request itself, not only the label the row carries.
        assert stream.model_for("run-under-pinned") == HI
    finally:
        await pinned.dispose()
        await plain.dispose()
        await rootless.dispose()


@pytest.mark.asyncio
async def test_a_member_effort_alone_keeps_the_row_tiers_model(tmp_path, monkeypatch):
    """Plan A.2, first bullet. The two fields fall through INDEPENDENTLY: an
    ``effort`` with no ``tier`` is not a model choice, so the row's tier still
    decides the model and the member decides the depth."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"med": "anthropic/claude-opus-5-5"})
    agents = AgentRegistry(tmp_path / "agents")
    pin_role(agents, "coder", "med")
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="coder", effort="high")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork", agents=agents)
    try:
        row = await launch(session, stream, "effort-only", agent="coder")
        assert row.model_label == "anthropic/claude-opus-5-5"
        assert stream.effort_for("effort-only") == "high"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_member_tier_alone_uses_that_tiers_own_effort(tmp_path, monkeypatch):
    """Plan A.2, second bullet. ``tier: hi`` with no effort takes the tier map's
    effort (``xhigh``), not the model's own default (``medium``)."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(
        config, model_choice="model", models={"hi": {"model": BIRTH.model_id, "effort": "xhigh"}}
    )
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        await launch(session, stream, "tier-only", agent="worker")
        assert stream.effort_for("tier-only") == "xhigh"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_auto_overrides_the_tier_map_effort_to_the_models_default(tmp_path, monkeypatch):
    """Plan A.2, third bullet, and the reason ``auto`` is not ``None``: a team
    takes tier ``hi`` WITHOUT inheriting the ``xhigh`` the operator configured
    on that tier. Both arms run in one test, on one tier map, so the difference
    is the member's word and nothing else."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(
        config, model_choice="model", models={"hi": {"model": BIRTH.model_id, "effort": "xhigh"}}
    )
    registry = TeamRegistry(config)
    make_team(
        registry,
        "teamwork",
        [
            TeamMember(role="worker", tier="hi"),
            TeamMember(role="reviewer", tier="hi", effort="auto"),
        ],
    )

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        await launch(session, stream, "inherit-the-map", agent="worker")
        await launch(session, stream, "take-the-default", agent="reviewer")
        assert stream.effort_for("inherit-the-map") == "xhigh"
        assert stream.effort_for("take-the-default") == "medium"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_launch_tier_beats_the_member_tier_and_the_member_effort_survives(
    tmp_path, monkeypatch
):
    """Plan A.3, first two clauses. A caller's ``effort=`` is the operator's own
    argument and outranks the roster; the member's EFFORT still applies, because
    the fields are independent."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"med": BIRTH.model_id, "hi": OTHER})
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", tier="hi", effort="high")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        row = await launch(session, stream, "launch-wins", agent="worker", effort="med")
        assert row.model_label == BIRTH.model_id
        assert stream.effort_for("launch-wins") == "high"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_explicit_model_keeps_the_members_own_effort(tmp_path, monkeypatch):
    """Plan A.3, third clause. ``model=`` moves the MODEL; a member's ``effort``
    is not a model statement, so it still applies to the chosen spec."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO})
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", effort="high")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        # Credentials are outside this clause: the scripted provider only needs
        # the validated pin spec, never a live API key (the same allowance
        # ``test_pinned_subagent_model`` makes for a per-run ``model=``).
        monkeypatch.setattr(
            "local_operator.mobile.peer_model.validate_subagent_selection",
            lambda provider, model_id: ModelSpec(
                provider=provider,
                model_id=model_id,
                context_window=100_000,
                # A ladder that does NOT start at ``high``, so the member's
                # effort is observable as a CHANGE rather than as a value
                # the spec already carried.
                reasoning_efforts=("low", "medium", "high", "xhigh", "max"),
                reasoning_effort="low",
                reasoning_default_effort="medium",
            ),
        )
        await launch(session, stream, "explicit-model", agent="worker", model=OTHER)
        assert stream.model_for("explicit-model") == OTHER
        assert stream.effort_for("explicit-model") == "high"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_inherit_runs_on_the_parents_model_and_depth(tmp_path, monkeypatch):
    """Plan A.3, last clause: ``model='inherit'`` is the explicit statement that
    the child runs AS ITS PARENT, so a member override does not reach it."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"hi": OTHER})
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", tier="hi", effort="low")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        row = await launch(session, stream, "inherits", agent="worker", model="inherit")
        assert stream.model_for("inherits") == f"{BIRTH.provider}/{BIRTH.model_id}"
        assert stream.effort_for("inherits") == BIRTH.reasoning_effort
        assert row.owns_model is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_member_tier_that_is_not_configured_refuses_the_launch(tmp_path, monkeypatch):
    """Plan A.4. Configured-ness is checked at LAUNCH, and the refusal NAMES the
    tier: silently inheriting the parent's model is the self-review collapse the
    strict launch path exists to prevent."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO})  # no `hi`
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        with pytest.raises(SubagentModelUnavailable) as caught:
            session._launch_subagent(label="child", prompt="go", agent="worker")
        assert caught.value.tier == "hi"
        assert "no model configured at subagents.models.hi" in caught.value.reason
        assert not session.jobs.list()
        assert stream.requests == []
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# 5: nested teams
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_team_slot_pins_the_sub_teams_manager_not_its_members(tmp_path, monkeypatch):
    """Plan A.5. On ``team:pod`` the override belongs to the role THIS roster
    launches — pod's manager — while pod's own members keep reading pod's
    ``team.yml``, which pins the worker to ``lo`` here. A lookup keyed on the
    resolved ROLE (pod's manager) instead of the SLOT would move neither.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO, "hi": HI})
    registry = TeamRegistry(config)
    make_team(registry, "pod", [TeamMember(role="worker", tier="lo")], manager="manager")
    make_team(
        registry,
        "org",
        [TeamMember(role="pod", kind="team", tier="hi")],
    )

    stream = RecordingStream()
    root = make_session(tmp_path, registry, stream, team="org")
    try:
        row = await launch(root, stream, "launch-the-pod", agent="team:pod")
        assert row.model_label == HI
        assert stream.requests, "the pod manager took a turn"

        # Inside pod, the same roster's own slot decides: `lo`, not `hi`.
        inner = make_session(tmp_path, registry, stream, team="pod")
        try:
            assert (await launch(inner, stream, "worker-inside", agent="worker")).model_label == LO
        finally:
            await inner.dispose()
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_a_member_tier_on_an_unknown_sub_team_is_not_silently_ignored(tmp_path, monkeypatch):
    """A ``team:`` slot naming a team that does not exist is still the launch
    error it always was; the override must not turn it into an inherited child."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"hi": HI})
    registry = TeamRegistry(config)
    make_team(registry, "org", [TeamMember(role="ghost", kind="team", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="org")
    try:
        with pytest.raises(TeamLaunchError):
            session._launch_subagent(label="child", prompt="go", agent="team:ghost")
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# 6: resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resumed_child_comes_back_on_its_member_overrides_model(tmp_path, monkeypatch):
    """Plan A.6, first clause, end to end: launch → settle → ``hub op='resume'``,
    asserted on the request the resumed child SENT. The roster is RE-DERIVED
    from the child's recorded lineage, because nothing about the override is
    persisted on the child."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO, "med": MED})
    agents = AgentRegistry(tmp_path / "agents")
    pin_role(agents, "worker", "med")
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", tier="lo")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork", agents=agents)
    try:
        row = await launch(session, stream, "do the work", agent="worker")
        assert row.model_label == LO
        child_id = row.id

        text = await hub_resume(session, child_id)
        new_id = resumed_id(session, text)
        await wait_for(lambda: completed(session, new_id))
        assert stream.model_for(RESUME_PROMPT) == LO
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_resume_follows_an_override_edited_since_the_launch(tmp_path, monkeypatch):
    """Plan A.6, second clause — the documented resume rule (a resume follows a
    tier edit), now for a roster slot. The row pin stays where it was, so only
    the roster half can produce the new model."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO, "hi": HI, "med": MED})
    agents = AgentRegistry(tmp_path / "agents")
    pin_role(agents, "worker", "med")
    registry = TeamRegistry(config)
    team = make_team(registry, "teamwork", [TeamMember(role="worker", tier="lo")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork", agents=agents)
    try:
        row = await launch(session, stream, "do the work", agent="worker")
        assert row.model_label == LO

        # The operator re-pins the slot, as the desktop editor or the team tool
        # would. Nothing on the child's record changes.
        registry.update_team(
            team.id, TeamEditFields(members=[TeamMember(role="worker", tier="hi")])
        )

        text = await hub_resume(session, row.id)
        new_id = resumed_id(session, text)
        await wait_for(lambda: completed(session, new_id))
        assert stream.model_for(RESUME_PROMPT) == HI
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_worker_of_a_settled_override_pinned_manager_takes_the_pin(tmp_path, monkeypatch):
    """Plan A.6, last clause: the ``_owned_model`` walk. A manager pinned by a
    ROSTER (no row pin, no ``effort`` argument) settles; its worker is resumed
    after a snapshot restart and after the tier the override names was
    RE-POINTED. The worker must land on the tier's NEW model — the pin is
    re-resolved from current config, not read off the manager's recorded label,
    which is what a roster-less walk would fall back to."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO})
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="lead", tier="lo")])

    stream = HoldingStream(hold="manage it")
    root = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        lead_id = root._launch_subagent(label="lead", prompt="manage it", agent="lead")
        await wait_for(lambda: (r := root.subagent_comms._record(lead_id)) is not None and r.child)
        lead = root.subagent_comms._record(lead_id).child
        assert lead.model_label == LO
        worker_id = lead._launch_subagent(label="worker", prompt="do the work")
        await wait_for(lambda: completed(lead, worker_id))
        assert stream.model_for("do the work") == LO
        stream.release.set()
        await wait_for(lambda: completed(root, lead_id))
        await root._persist_subagent_roster()
    finally:
        stream.release.set()
        await root.dispose()

    # Restart. The operator re-points the tier the override names.
    write_tiers(config, lo=HI)
    after = RecordingStream()
    revived = make_session(tmp_path, registry, after, team="teamwork")
    try:
        text = await hub_resume(revived, worker_id)
        new_id = resumed_id(revived, text)
        await wait_for(lambda: completed(revived, new_id))
        assert after.model_for(RESUME_PROMPT) == HI
    finally:
        await revived.dispose()


# ---------------------------------------------------------------------------
# 7: persistence
# ---------------------------------------------------------------------------


def test_a_roster_with_overrides_survives_a_reload(tmp_path):
    """Plan A.7, first clause."""
    registry = TeamRegistry(tmp_path / "config")
    make_team(
        registry,
        "teamwork",
        [TeamMember(role="coder", tier="hi", effort="xhigh"), TeamMember(role="pod", kind="team")],
    )

    reloaded = TeamRegistry(tmp_path / "config").get_team_by_name("teamwork")
    assert reloaded is not None
    assert (reloaded.members[0].tier, reloaded.members[0].effort) == ("hi", "xhigh")
    assert reloaded.members[1].is_default()


def test_a_roster_without_overrides_keeps_its_bytes_and_its_revision(tmp_path):
    """Plan A.7, second clause. ``tier: null, effort: null`` written into every
    member would move ``definition_revision`` for every team on its next save
    and turn the next save from every open editor into a spurious 409."""
    from local_operator.hub_sync.definition import definition_revision

    registry = TeamRegistry(tmp_path / "config")
    team = make_team(registry, "teamwork", [TeamMember(role="coder")])
    path = tmp_path / "config" / "teams" / team.id / "team.yml"
    before_bytes = path.read_text()
    before_revision = definition_revision("team", str(team.id), team.model_dump(mode="json"))

    # A second save of the loaded row is the shape that would churn.
    reloaded = registry.get_team_by_name("teamwork")
    assert reloaded is not None
    registry.save_team(reloaded)
    after = registry.get_team_by_name("teamwork")
    assert after is not None
    assert path.read_text() == before_bytes
    assert definition_revision("team", str(after.id), after.model_dump(mode="json")) == (
        before_revision
    )


def test_a_hand_edited_bad_override_word_drops_the_field_not_the_team(tmp_path, caplog):
    """Plan A.7, third clause. ``_load`` skips any row that fails validation, so
    a typo in ``team.yml`` must degrade to "no override" with a warning — a
    vanished team is a far worse cost than an ignored word."""
    import logging

    registry = TeamRegistry(tmp_path / "config")
    team = make_team(registry, "teamwork", [TeamMember(role="coder")])
    path = tmp_path / "config" / "teams" / team.id / "team.yml"
    payload = yaml.safe_load(path.read_text())
    payload["members"] = [{"role": "coder", "count": 1, "kind": "agent", "tier": "high"}]
    path.write_text(yaml.safe_dump(payload, default_flow_style=False, sort_keys=False))

    with caplog.at_level(logging.WARNING):
        loaded = TeamRegistry(tmp_path / "config").get_team_by_name("teamwork")
    assert loaded is not None, "the team itself must survive"
    assert [member.role for member in loaded.members] == ["coder"]
    assert loaded.members[0].tier is None
    assert any("ignoring the override" in record.message for record in caplog.records)


def test_a_bad_override_word_is_refused_on_every_write_path(tmp_path):
    """Plan A.7, last clause. Strict at the DATA layer, which every writer
    reaches: the model constructor, and the registry that the route and the
    tool both go through."""
    from pydantic import ValidationError

    # ``model_validate`` rather than the constructor so the payload is the one a
    # route or a hand-edited ``team.yml`` actually produces.
    with pytest.raises(ValidationError):
        TeamMember.model_validate({"role": "coder", "tier": "high"})
    with pytest.raises(ValidationError):
        TeamMember.model_validate({"role": "coder", "effort": "extreme"})
    with pytest.raises(ValidationError):
        TeamMember.model_validate({"role": "coder", "effort": 3})

    registry = TeamRegistry(tmp_path / "config")
    make_team(registry, "teamwork", [TeamMember(role="coder")])
    # A conflicting duplicate is refused at the SAVE choke point, which is the
    # only place a raw route roster can be caught: a launch reads the first
    # match, so the second slot's overrides would be silently inert.
    with pytest.raises(ValueError, match="different overrides"):
        registry.create_team(
            TeamEditFields(
                name="clashing",
                manager="manager",
                members=[TeamMember(role="coder"), TeamMember(role="coder", tier="hi")],
            )
        )


# ---------------------------------------------------------------------------
# 8-9: the writers that rebuild a roster, and the operator gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_members_update_keeps_the_overrides_and_inherit_clears_one(tmp_path, monkeypatch):
    """Plan A.8. ``members=`` replaces the roster wholesale, so a roster edit
    that silently dropped the overrides would put that child back on the model
    the operator moved it off. ``inherit`` is the way to take one off."""
    registry = TeamRegistry(tmp_path)
    context = ToolContext(cwd=".", team_registry=registry)
    write_config(tmp_path, model_choice="model", models={"lo": LO, "hi": HI})
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    created = await execute_team(
        "tc",
        {
            "op": "create",
            "name": "teamwork",
            "manager": "manager",
            "members": ["coder", "reviewer"],
            "overrides": [
                {"member": "coder", "tier": "hi", "effort": "xhigh"},
                {"member": "reviewer", "tier": "lo"},
            ],
        },
        None,
        None,
        context,
    )
    assert not created.is_error, created.text

    updated = await execute_team(
        "tc",
        {"op": "update", "name": "teamwork", "members": ["coder", "reviewer", "scout"]},
        None,
        None,
        context,
    )
    assert not updated.is_error, updated.text
    team = registry.get_team_by_name("teamwork")
    assert team is not None
    by_role = {member.role: member for member in team.members}
    assert (by_role["coder"].tier, by_role["coder"].effort) == ("hi", "xhigh")
    assert by_role["reviewer"].tier == "lo"
    assert by_role["scout"].is_default()

    cleared = await execute_team(
        "tc",
        {"op": "update", "name": "teamwork", "overrides": [{"member": "coder", "tier": "inherit"}]},
        None,
        None,
        context,
    )
    assert not cleared.is_error, cleared.text
    team = registry.get_team_by_name("teamwork")
    assert team is not None
    by_role = {member.role: member for member in team.members}
    assert by_role["coder"].tier is None
    assert by_role["coder"].effort == "xhigh", "the other field is left alone"

    shown = await execute_team("tc", {"op": "show", "name": "teamwork"}, None, None, context)
    assert "effort xhigh" in shown.text
    assert "tier lo" in shown.text

    unknown = await execute_team(
        "tc",
        {"op": "update", "name": "teamwork", "overrides": [{"member": "ghost", "tier": "lo"}]},
        None,
        None,
        context,
    )
    assert unknown.is_error and "no roster slot named 'ghost'" in unknown.text

    # Configured-ness is checked at LAUNCH, not at write -- config moves
    # independently of teams -- so the receipt says it instead, while the edit is
    # still on screen rather than on the launch that would refuse.
    unconfigured = await execute_team(
        "tc",
        {"op": "update", "name": "teamwork", "overrides": [{"member": "scout", "tier": "med"}]},
        None,
        None,
        context,
    )
    assert not unconfigured.is_error, unconfigured.text
    assert "No model is configured for tier 'med'" in unconfigured.text
    assert "will refuse to launch" in unconfigured.text
    # And the write still went through: a warning, never a refusal.
    warned = registry.get_team_by_name("teamwork")
    assert warned is not None and warned.members[-1].tier == "med"


@pytest.mark.asyncio
async def test_the_team_tool_refuses_overrides_until_the_operator_delegates(tmp_path, monkeypatch):
    """Plan A.9. The same gate the ``agent`` tool puts on a role pin: with the
    shipped ``operator`` default a model cannot pin a member's model, and the
    refusal says who can."""
    registry = TeamRegistry(tmp_path)
    context = ToolContext(cwd=".", team_registry=registry)
    args = {
        "op": "create",
        "name": "teamwork",
        "manager": "manager",
        "members": ["coder"],
        "overrides": [{"member": "coder", "tier": "hi"}],
    }

    write_config(tmp_path, models={"hi": HI})  # model_choice absent: the shipped default
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    refused = await execute_team("tc", args, None, None, context)
    assert refused.is_error
    assert "subagents.model_choice" in refused.text
    assert registry.get_team_by_name("teamwork") is None

    # The same call under `model`, on the same rig, is accepted: the gate reads
    # the LIVE config, so this proves it is the setting and nothing else.
    write_config(tmp_path, model_choice="model", models={"hi": HI})
    allowed = await execute_team("tc", args, None, None, context)
    assert not allowed.is_error, allowed.text
    team = registry.get_team_by_name("teamwork")
    assert team is not None and team.members[0].tier == "hi"


@pytest.mark.asyncio
async def test_the_receipt_says_the_team_decided_the_model(tmp_path, monkeypatch):
    """Plan A.11. ``[selection: team]`` is how a delegating model — and the
    operator reading ``hub``/``jobs`` — can tell a roster override from a row
    pin, which is the difference between "the operator pinned this role" and
    "this roster pinned this one slot"."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"hi": HI})
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="worker", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        result = await execute_task(
            "call-1",
            {"context": "goal", "tasks": [{"label": "pinned", "prompt": "go", "agent": "worker"}]},
            None,
            None,
            session._build_tool_context(),
        )
        text = "".join(block.text for block in result.content if isinstance(block, TextContent))
        assert "[selection: team]" in text
        details = result.details or {}
        row = session.jobs.get(details["jobs"][0]["job_id"])
        assert row is not None and row.model_label == HI
    finally:
        await session.dispose()


def test_the_tier_vocabulary_is_re_exported_and_the_literal_matches_it():
    """``CANONICAL_EFFORT_TIERS`` moved to the leaf module and ``subagent``
    re-exports it, while the ``TeamMember.tier`` Literal is written out (a
    Literal cannot be built from a tuple). This pins the two spellings equal so
    widening the constant cannot leave a stale annotation behind."""
    from typing import get_args

    from local_operator.harness.subagent import CANONICAL_EFFORT_TIERS as re_exported
    from local_operator.model.effort import CANONICAL_EFFORT_TIERS

    field_type = TeamMember.model_fields["tier"].annotation
    literal = next(arg for arg in get_args(field_type) if arg is not type(None))
    assert re_exported is CANONICAL_EFFORT_TIERS
    assert set(get_args(literal)) == set(CANONICAL_EFFORT_TIERS)


def test_the_display_clamp_is_the_launch_clamp_where_a_ladder_exists():
    """``/effective``'s reported effort must be the level the LAUNCH would run at.

    Pinned as an EQUIVALENCE rather than as a table, because a table is two
    copies of the ladder rules by the time the next model lands. ``auto`` is in
    the list on purpose: it is the member's "this model's own default", not a
    rung, and the two sides have to agree about that too.

    Only models WITH a ladder are compared here. Where there is no ladder the
    contract deliberately DIVERGES from ``_with_tier_effort``: the launch leaves
    such a spec untouched (there is nothing to clamp), while the frozen §1 rule
    for the endpoint is that ``effort.value`` "equals ``requested``" when the
    model is not known. That half is asserted through the endpoint itself, in
    ``test_the_effective_endpoint_reports_the_requested_effort...``, because it is
    the response shape P3 consumes rather than this helper.
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.server.utils.desktop_profiles import _clamped_effort
    from local_operator.session.session import (
        _member_reasoning_effort,
        _with_tier_effort,
    )

    for provider, model_id in (
        ("anthropic", "claude-opus-4-5"),  # ladder stops at ``high``
        ("anthropic", "claude-opus-5-5"),  # full ladder, default ``medium``
    ):
        spec = build_model_spec(provider, model_id)
        assert spec.reasoning_efforts, f"precondition: {model_id} has a ladder"
        for requested in (None, "auto", "none", "low", "medium", "high", "xhigh", "max"):
            # Exactly the route's expression for a known model: with nothing
            # requested, the value is what the spec already carries.
            reported = (
                spec.reasoning_effort if requested is None else _clamped_effort(spec, requested)
            )
            launched = _with_tier_effort(
                spec, _member_reasoning_effort(spec, requested)
            ).reasoning_effort
            assert reported == launched, (model_id, requested)


# ---------------------------------------------------------------------------
# Remediation round 1: I1/M2 (fresh roster), M1/I2/Q1 (tool strictness),
# I5 (config-read error), m1 (hand-edited conflict), m4 (row pin on a
# team:pod ancestor), Q3 (one warning per team/value)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fresh_launch_uses_an_override_edited_mid_session(tmp_path, monkeypatch):
    """I1/M2, D8. The row the model came from is the one the NEXT launch sees.

    ``Session.active_team`` is assigned once, at ``attach_team``, and nothing
    refreshes it -- so a launch that read it would keep using the roster as it
    was when the session attached the team, while ``/effective`` and the resume
    (both registry reads) showed the edit. The P3 editor's whole loop is "edit
    the override, then the next launch uses it", so the launch has to read the
    same place they do.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"lo": LO, "hi": HI})
    registry = TeamRegistry(config)
    team = make_team(registry, "teamwork", [TeamMember(role="worker", tier="lo")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        first = await launch(session, stream, "first", agent="worker")
        assert first.model_label == LO

        # The operator edits the slot; the session is NOT re-attached.
        registry.update_team(
            team.id, TeamEditFields(members=[TeamMember(role="worker", tier="hi")])
        )
        active = session.active_team
        assert active is not None and active.members[0].tier == "lo", (
            "precondition: the attached object is a snapshot, which is why it "
            "cannot be the roster a launch resolves against"
        )

        second = await launch(session, stream, "second", agent="worker")
        assert second.model_label == HI
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_team_tool_refuses_a_bad_override_word_on_both_ops(tmp_path, monkeypatch):
    """M1/I2/Q1, D9. ``model_copy`` skipped the validators, so the tool saved a
    word that the next load silently dropped -- the receipt claimed a pin that
    did not exist, and the child ran on a model the operator was not told about.

    ``high`` is the natural slip (a valid EFFORT word offered while the model is
    choosing a TIER), which is why the refused value is that one and not
    something absurd.
    """
    registry = TeamRegistry(tmp_path)
    context = ToolContext(cwd=".", team_registry=registry)
    write_config(tmp_path, model_choice="model", models={"hi": HI})
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    async def call(**args: Any) -> Any:
        return await execute_team("tc", args, None, None, context)

    created = await call(op="create", name="clean", manager="m", members=["worker"])
    assert not created.is_error, created.text

    for bad in ({"member": "worker", "tier": "high"}, {"member": "worker", "effort": "warp"}):
        refused = await call(op="update", name="clean", overrides=[bad])
        assert refused.is_error, refused.text
        assert "invalid arguments" in refused.text
        # Nothing landed -- not in the row, and not in the file a reload reads.
        stored = registry.get_team_by_name("clean")
        assert stored is not None and stored.members[0].is_default()
        written = next((tmp_path / "teams").glob("*/team.yml")).read_text()
        assert "high" not in written and "warp" not in written

    refused_create = await call(
        op="create",
        name="dirty",
        manager="m",
        members=["worker"],
        overrides=[{"member": "worker", "tier": "high"}],
    )
    assert refused_create.is_error, refused_create.text
    assert registry.get_team_by_name("dirty") is None

    # The valid path still works, so the refusal is the word and not the feature.
    accepted = await call(op="update", name="clean", overrides=[{"member": "worker", "tier": "hi"}])
    assert not accepted.is_error, accepted.text
    stored = registry.get_team_by_name("clean")
    assert stored is not None and stored.members[0].tier == "hi"


@pytest.mark.asyncio
async def test_a_config_read_error_is_named_as_the_config_not_the_registry(tmp_path, monkeypatch):
    """I5. The tier-targets read moved into the resolver, so its failure used to
    be reported as an unreadable AGENT REGISTRY -- two different problems, with
    two different remedies. It is a reason again, and the lenient path (session
    naming) must not die in a retry that reads the same broken config."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_tiers(tmp_path / "config", lo=LO)
    registry = TeamRegistry(tmp_path / "config")
    session = make_session(tmp_path, registry, RecordingStream())

    def boom() -> Any:
        raise OSError("config.yml unreadable")

    monkeypatch.setattr("local_operator.harness.subagent.read_effort_tier_targets", boom)
    try:
        with pytest.raises(SubagentModelUnavailable) as caught:
            session._launch_subagent(label="x", prompt="go", effort="lo")
        assert "config could not be read" in caught.value.reason
        assert "agent registry" not in caught.value.reason
        assert not session.jobs.list()

        # Lenient: warns, then answers "inherit". No traceback, no raised error.
        assert session._resolve_subagent_model("task", "lo") is None

        # And a MEMBER tier reaches the same reason rather than the registry's.
        make_team(registry, "teamwork", [TeamMember(role="worker", tier="lo")])
        team = registry.get_team_by_name("teamwork")
        assert team is not None
        session.attach_team(team)
        with pytest.raises(SubagentModelUnavailable) as member_caught:
            session._launch_subagent(label="w", prompt="go", agent="worker")
        assert "config could not be read" in member_caught.value.reason
    finally:
        await session.dispose()


def test_a_hand_edited_conflict_does_not_block_an_unrelated_save(tmp_path):
    """m1. The conflict check is DELTA-based: a row already on disk with two
    disagreeing slots of one name still loads (the load is lenient by design), so
    refusing every later write would brick a row whose only fault is a hand edit
    the operator has not got to. A write that INTRODUCES or CHANGES the conflict
    is still refused, because that write is the one that would leave one of the
    two slots silently inert."""
    config = tmp_path / "config"
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="coder")])
    path = next((config / "teams").glob("*/team.yml"))
    path.write_text(
        path.read_text().replace(
            "- role: coder\n  count: 1\n  kind: agent\n",
            "- role: coder\n  count: 1\n  kind: agent\n  tier: hi\n"
            "- role: Coder\n  count: 1\n  kind: agent\n  tier: lo\n",
        )
    )

    reloaded = TeamRegistry(config)
    conflicted = reloaded.get_team_by_name("teamwork")
    assert (
        conflicted is not None and len(conflicted.members) == 2
    ), "precondition: the hand-edited row still loads"

    saved = reloaded.update_team(conflicted.id, TeamEditFields(description="tidy"))
    assert saved.description == "tidy"

    with pytest.raises(ValueError, match="different overrides"):
        reloaded.update_team(
            conflicted.id,
            TeamEditFields(
                members=[
                    TeamMember(role="coder", tier="hi"),
                    TeamMember(role="Coder", tier="med"),
                ]
            ),
        )


@pytest.mark.asyncio
async def test_a_worker_of_a_settled_team_launch_manager_takes_the_managers_row_pin(
    tmp_path, monkeypatch
):
    """m4. A ``team:pod`` launch runs as pod's MANAGER, so that manager's own row
    pin is what the launch and the resume resolve with -- but the two walk
    readers used the recorded ``team:pod`` TOKEN as the role, found no row of that
    name, and fell back to the manager's recorded label. With the tier re-pointed
    across a restart, that fallback answers the OLD model, which is exactly what
    the walk exists to avoid."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, models={"med": MED})
    agents = AgentRegistry(tmp_path / "agents")
    write_profile(
        agents,
        AgentParams(
            op="create",
            name="lead",
            description="leads a pod",
            instructions="Lead the pod.",
            effort="med",
            delegate=True,
        ),
        creating=True,
    )
    registry = TeamRegistry(config)
    make_team(registry, "pod", [TeamMember(role="worker")], manager="lead")
    make_team(registry, "org", [TeamMember(role="pod", kind="team")])

    stream = HoldingStream(hold="manage it")
    root = make_session(tmp_path, registry, stream, team="org", agents=agents)
    try:
        lead_id = root._launch_subagent(label="pod", prompt="manage it", agent="team:pod")
        await wait_for(lambda: (r := root.subagent_comms._record(lead_id)) is not None and r.child)
        lead = root.subagent_comms._record(lead_id).child
        assert lead.model_label == MED
        worker_id = lead._launch_subagent(label="worker", prompt="do the work")
        await wait_for(lambda: completed(lead, worker_id))
        stream.release.set()
        await wait_for(lambda: completed(root, lead_id))
        await root._persist_subagent_roster()
    finally:
        stream.release.set()
        await root.dispose()

    # Restart, and the operator re-points the tier the manager's ROW pin names.
    write_tiers(config, med=HI)
    after = RecordingStream()
    revived = make_session(tmp_path, registry, after, team="org", agents=agents)
    try:
        text = await hub_resume(revived, worker_id)
        new_id = resumed_id(revived, text)
        await wait_for(lambda: completed(revived, new_id))
        assert after.model_for(RESUME_PROMPT) == HI
    finally:
        await revived.dispose()


def test_a_bad_override_word_warns_once_per_team_and_value(tmp_path, caplog):
    """Q3. The registry re-validates every row on every refresh, so an undeduped
    warning repeated for as long as the bad word sat in ``team.yml`` (QA measured
    four identical pairs inside one request burst). The message still names the
    fault, once per (team, field, value)."""
    import logging
    import uuid

    config = tmp_path / "config"
    registry = TeamRegistry(config)
    make_team(registry, "teamwork", [TeamMember(role="coder")])
    path = next((config / "teams").glob("*/team.yml"))
    # A word no other test can have used, so the once-per-key dedupe cannot be
    # satisfied by an earlier test in the same worker.
    bad = f"x{uuid.uuid4().hex[:6]}"
    path.write_text(
        path.read_text().replace(
            "- role: coder\n  count: 1\n  kind: agent\n",
            f"- role: coder\n  count: 1\n  kind: agent\n  tier: {bad}\n",
        )
    )

    # TWO registry instances, because the dedupe has to survive the reads a
    # running process makes over time -- and a single instance would not even
    # re-read inside its refresh interval, which is how a test can pass without
    # exercising the repeat.
    first_reader = TeamRegistry(config)
    with caplog.at_level(logging.WARNING, logger="local_operator.teams"):
        assert first_reader.list_teams(), "the row still loads"
        first = [r for r in caplog.records if "ignoring the override" in r.getMessage()]
        assert len(first) == 1, [r.getMessage() for r in first]
        TeamRegistry(config).list_teams()
        again = [r for r in caplog.records if "ignoring the override" in r.getMessage()]
        assert len(again) == 1, "the next read must not repeat it"


# ---------------------------------------------------------------------------
# the inherit sentinel against the roster: the absolute pin outranks the slot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_roster_slot_tier_does_not_move_a_role_that_pins_the_session_model(
    tmp_path, monkeypatch
):
    """The absolute pin outranks the roster too, and the roster is why.

    A roster slot's ``tier`` is NOT a safe place to leave above the role's
    pin, because the ``team`` tool lets a DELEGATING MODEL set one (the
    ``overrides`` list, available whenever ``subagents.model_choice=model``).
    A pin a model could move by writing the roster would not be a pin: the
    role would run on whatever the delegating model last asked for, which is
    the failure the sentinel exists to close.

    So the precedence is: the role's own inherit pin > the roster slot's tier >
    the launch tier > the role's own TIER pin. The roster is the operator's
    layer for every other kind of pin, and it stays that way — what it cannot
    do is move a role that has declared which model it must run on.
    """
    from local_operator.agent_profiles import install_seed

    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"hi": HI})
    agents = AgentRegistry(tmp_path / "agents")
    # The PACKAGED seed is where the pin travels: the published definition of
    # the role, and the only channel a teammate pulling it gets.
    assert install_seed("coder", registry=agents) is not None
    registry = TeamRegistry(config)
    make_team(registry, "pinned", [TeamMember(role="coder", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="pinned", agents=agents)
    try:
        row = await launch(session, stream, "run-under-pinned", agent="coder")
        assert row.model_label == f"{BIRTH.provider}/{BIRTH.model_id}"
        # And the request itself, not only the label the row carries.
        assert stream.model_for("run-under-pinned") == f"{BIRTH.provider}/{BIRTH.model_id}"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_receipt_names_the_role_not_the_team_for_an_inherit_pinned_slot(
    tmp_path, monkeypatch
):
    """``[selection: role_inherit]`` — the layer that actually decided.

    A receipt that said ``team`` would send an operator looking at the roster
    for a setting that did not decide anything, and would hide the one that
    did. The narrowness matters as much: a roster slot whose role pins an
    ordinary TIER still reads ``team``, because the ordinary tier is the role's
    DEFAULT and the roster is free to move it.
    """
    from local_operator.agent_profiles import install_seed

    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    write_config(config, model_choice="model", models={"hi": HI})
    agents = AgentRegistry(tmp_path / "agents")
    assert install_seed("coder", registry=agents) is not None
    registry = TeamRegistry(config)
    make_team(registry, "pinned", [TeamMember(role="coder", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="pinned", agents=agents)
    try:
        result = await execute_task(
            "call-1",
            {"context": "goal", "tasks": [{"label": "pinned", "prompt": "go", "agent": "coder"}]},
            None,
            None,
            session._build_tool_context(),
        )
        text = "".join(block.text for block in result.content if isinstance(block, TextContent))
        assert "[selection: role_inherit]" in text
        assert "[selection: team]" not in text
    finally:
        await session.dispose()
