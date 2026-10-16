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
    make_team(registry, "teamwork", [TeamMember(role="coder", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        await launch(session, stream, "tier-only", agent="coder")
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
            TeamMember(role="coder", tier="hi"),
            TeamMember(role="reviewer", tier="hi", effort="auto"),
        ],
    )

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        await launch(session, stream, "inherit-the-map", agent="coder")
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
    make_team(registry, "teamwork", [TeamMember(role="coder", tier="hi", effort="high")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        row = await launch(session, stream, "launch-wins", agent="coder", effort="med")
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
    make_team(registry, "teamwork", [TeamMember(role="coder", effort="high")])

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
        await launch(session, stream, "explicit-model", agent="coder", model=OTHER)
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
    make_team(registry, "teamwork", [TeamMember(role="coder", tier="hi", effort="low")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        row = await launch(session, stream, "inherits", agent="coder", model="inherit")
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
    make_team(registry, "teamwork", [TeamMember(role="coder", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        with pytest.raises(SubagentModelUnavailable) as caught:
            session._launch_subagent(label="child", prompt="go", agent="coder")
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
    make_team(registry, "teamwork", [TeamMember(role="coder", tier="hi")])

    stream = RecordingStream()
    session = make_session(tmp_path, registry, stream, team="teamwork")
    try:
        result = await execute_task(
            "call-1",
            {"context": "goal", "tasks": [{"label": "pinned", "prompt": "go", "agent": "coder"}]},
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


def test_the_display_clamp_is_the_launch_clamp():
    """``/effective``'s reported effort must be the level the LAUNCH would run at.

    Pinned as an EQUIVALENCE rather than as a table, because a table is two
    copies of the ladder rules by the time the next model lands. ``auto`` is in
    the list on purpose: it is the member's "this model's own default", not a
    rung, and the two sides have to agree about that too.
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
        ("test", "unknown-model"),  # no ladder at all
    ):
        spec = build_model_spec(provider, model_id)
        for requested in (None, "auto", "none", "low", "medium", "high", "xhigh", "max"):
            # Exactly the route's expression: with nothing requested, the value
            # is what the spec already carries.
            reported = (
                spec.reasoning_effort if requested is None else _clamped_effort(spec, requested)
            )
            launched = _with_tier_effort(
                spec, _member_reasoning_effort(spec, requested)
            ).reasoning_effort
            assert reported == launched, (model_id, requested)
