"""Nested-team launches: resolution, propagation, errors and the depth cap.

BEN-7 decisions D1-D4 and D7, scored against BEN-1 S3 gates N0-N3:

* N0 — a depth-1 member launch keeps origin/main's exact bytes;
* N1 — every launch at depth 2 and 3 carries the right team's text;
* N2 — the chain-of-command line reaches every launch below depth 1;
* N3 — an unknown team, a cycle or a launch past the cap is an ERROR, never a
  silent generic child.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.agent_profiles import AgentProfile, load_seed
from local_operator.harness import subagent as subagent_mod
from local_operator.harness.comms import SubagentComms
from local_operator.harness.subagent import (
    DEFAULT_MAX_TEAM_DEPTH,
    TOP_SESSION_REPORTS_TO,
    CarriedLaunch,
    LaunchTarget,
    TeamLaunchError,
    _effective_prompt,
    read_max_team_depth,
    resolve_launch_target,
)
from local_operator.harness.types import ModelSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import (
    TeamEditFields,
    TeamMember,
    TeamRegistry,
    escalation_preamble,
)
from tests.unit.harness.test_comms import ChangeSignal, wait_for

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The exact depth-1 launch bytes for a plain ``task`` member of this
#: fixture's ``org`` team, FROZEN so a change to depth-1 launch text fails here
#: by name. Originally captured from origin/main 5473ef38e (BEN-1 N0, which
#: proved the nested-team refactor did not move these bytes).
#:
#: RE-BASELINED by the role-word address fix: the team brief's
#: "The manager is <role>." line was replaced with the escalation preamble's
#: hub phrasing ("You report to <role>, through hub."), because the old line
#: modelled the manager's ROLE NAME as something to address — and a sender
#: reading a roster did exactly that, typing ``target="manager"`` into a
#: resolver whose substring tier then landed on any session whose title
#: contained the word. Update deliberately, never to admit a cosmetic edit.
FROZEN_ORG_TASK_BYTES = (
    "[team: org]\n\nYou are task on this team. You report to manager, through hub.\n\n"
    "Teammates:\n- manager: manager (you, when this team is invoked)\n- coder\n"
    "- pod (team)\n\nCollaboration:\nReview before merge.\n\nProject:\nwidgets\n\n"
    "implement the button"
)


class NoStream:
    """A stream the tests never reach: they build sessions, not turns."""

    def __call__(self, request, signal):  # pragma: no cover - never called
        raise AssertionError("no provider turn expected")


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "config"
    directory.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(directory))
    return directory


@pytest.fixture()
def teams(config: Path) -> TeamRegistry:
    """``org`` (manager + coder + team:pod) and ``pod`` (manager + coder)."""
    registry = TeamRegistry(config)
    registry.create_team(
        TeamEditFields(
            name="pod",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="Pod rules.",
            project="pod project",
        )
    )
    registry.create_team(
        TeamEditFields(
            name="org",
            manager="manager",
            members=[TeamMember(role="coder"), TeamMember(role="pod", kind="team")],
            instructions="Review before merge.",
            project="widgets",
        )
    )
    return registry


def make_root(tmp_path: Path, registry: TeamRegistry | None, team: str | None = "org") -> Session:
    session = Session(
        model=MODEL,
        stream_fn=NoStream(),
        tools=[],
        transcript=Transcript(tmp_path / "root"),
        system_blocks_provider=lambda: ["stable", "env"],
        team_registry=registry,
    )
    if team is not None and registry is not None:
        found = registry.get_team_by_name(team)
        assert found is not None
        session.attach_team(found)
    return session


async def build(parent: Session, agent: str, job_id: str, prompt: str = "work"):
    """Launch-resolve and build a child exactly as ``run_subagent`` wires it."""
    target = resolve_launch_target(agent, parent)
    text, profile = _effective_prompt(prompt, agent, parent, target)
    # The comms row ``run_subagent`` writes: a grandchild names its parent's
    # role in the chain-of-command line off it.
    parent.subagent_comms.record_launch(
        job_id,
        job_id,
        parent_job_id=getattr(parent, "_job_id", None),
        agent_role=f"team:{target.team.name}" if target.is_team_launch else agent,
        team_name=str(getattr(target.team, "name", "") or ""),
        team_lineage=target.team_lineage,
        depth=target.depth,
    )
    child = await subagent_mod._build_child_session(
        label=job_id,
        prompt=text,
        parent_session=parent,
        model_spec=None,
        job_id=job_id,
        agent=target.role,
        profile=profile,
        target=target,
    )
    return child, target, text


# -- N0: depth-1 bytes ---------------------------------------------------------


def test_depth_one_member_bytes_are_frozen(tmp_path, teams):
    root = make_root(tmp_path, teams)
    target = resolve_launch_target("task", root)
    assert (target.depth, target.is_team_launch) == (1, False)
    text, _ = _effective_prompt("implement the button", "task", root, target)
    assert text == FROZEN_ORG_TASK_BYTES
    # ...and identical to the legacy no-target path, for a role too.
    for role in ("task", "coder", "reviewer", "scout"):
        with_target, _ = _effective_prompt(
            "implement the button", role, root, resolve_launch_target(role, root)
        )
        legacy, _ = _effective_prompt("implement the button", role, root)
        assert with_target == legacy
        assert "[chain of command]" not in with_target
    coder, _ = _effective_prompt("x", "coder", root, resolve_launch_target("coder", root))
    seed = load_seed("coder")
    assert seed is not None
    org = teams.get_team_by_name("org")
    assert org is not None
    assert coder == org.member_preamble("coder") + seed.preamble + "x"


def test_a_plain_child_with_no_team_is_unchanged(tmp_path, config):
    root = make_root(tmp_path, None, team=None)
    target = resolve_launch_target("task", root)
    assert target.team is None and target.team_lineage == () and target.depth == 1
    assert _effective_prompt("go", "task", root, target)[0] == "go"


# -- D1 / N1: propagation ------------------------------------------------------


@pytest.mark.asyncio
async def test_depth_two_and_three_carry_the_member_preamble(tmp_path, teams):
    root = make_root(tmp_path, teams)
    org = teams.get_team_by_name("org")
    assert org is not None
    mgr, t1, _ = await build(root, "manager", "job-mgr")
    assert mgr.active_team is not None and mgr.active_team.name == "org"
    assert (t1.depth, mgr._delegation_depth) == (1, 1)
    assert mgr._team_lineage == (org.id,)
    worker, t2, text2 = await build(mgr, "task", "job-w")
    assert t2.depth == 2 and t2.team is not None and t2.team.name == "org"
    assert text2.startswith(org.member_preamble("task"))
    grand, t3, text3 = await build(worker, "task", "job-g")
    assert t3.depth == 3
    assert text3.startswith(org.member_preamble("task"))
    # Never the parent's system-tail brief (D1 (a), (c)).
    for child in (mgr, worker, grand):
        assert child._goal_state.team_brief == ""
    for child in (grand, worker, mgr, root):
        await child.dispose()


# -- D2 / N1 / N3: team launches -----------------------------------------------


@pytest.mark.asyncio
async def test_a_team_launch_runs_as_the_sub_teams_manager(tmp_path, teams):
    root = make_root(tmp_path, teams)
    pod = teams.get_team_by_name("pod")
    org = teams.get_team_by_name("org")
    assert pod is not None and org is not None
    lead, target, text = await build(root, "team:pod", "job-lead", prompt="fix it")
    assert target.is_team_launch and target.role == "manager" and target.depth == 1
    assert text.startswith(pod.manager_preamble() + escalation_preamble(TOP_SESSION_REPORTS_TO))
    assert "You are the manager of this team" in text and text.endswith("fix it")
    assert lead.active_team is not None and lead.active_team.id == pod.id
    assert lead._team_lineage == (org.id, pod.id)
    assert "task" in {tool.name for tool in lead._tools}
    await lead.dispose()
    await root.dispose()


def test_a_bare_name_with_only_a_team_slot_is_a_team_launch(tmp_path, teams):
    root = make_root(tmp_path, teams)
    assert resolve_launch_target("pod", root).is_team_launch
    assert resolve_launch_target("TEAM:Pod", root).is_team_launch


def test_a_bare_name_with_an_agent_slot_too_stays_an_agent_launch(tmp_path, teams):
    org = teams.get_team_by_name("org")
    assert org is not None
    teams.update_team(
        org.id,
        TeamEditFields(members=[*org.members, TeamMember(role="pod")]),
    )
    root = make_root(tmp_path, teams)
    target = resolve_launch_target("pod", root)
    assert not target.is_team_launch and target.role == "pod"


@pytest.mark.asyncio
async def test_the_recorded_role_is_team_prefixed(tmp_path, teams):
    root = make_root(tmp_path, teams)
    job_id = subagent_mod.run_subagent(
        label="lead", prompt="go", parent_session=root, jobs_manager=root.jobs, agent="pod"
    )
    job = root.jobs.get(job_id)
    assert job is not None and job.agent_role == "team:pod"
    node = root.subagent_comms.node(job_id)
    assert node is not None and node.agent_role == "team:pod"


@pytest.mark.asyncio
async def test_a_team_launch_stamps_the_team_prefixed_agent_on_job_and_origin(tmp_path, teams):
    """The field the S3 N3 scorer keys on, pinned at both write sites.

    ``run_subagent`` rewrites the launch agent to ``team:<name>`` exactly once
    (subagent.py's target resolution), and that ONE value feeds both the job
    row and ``mark_session_origin``'s ``agent=`` (the ``origin.json`` the bench
    reads). Pinning both stops a future edit from stamping the bare role on
    disk while the row says the team — the silent-fallback shape N3 exists to
    catch — and stops the row drifting from the file the scorer reads.
    """
    root = make_root(tmp_path, teams)
    job_id = subagent_mod.run_subagent(
        label="lead", prompt="go", parent_session=root, jobs_manager=root.jobs, agent="pod"
    )
    job = root.jobs.get(job_id)
    assert job is not None and job.agent_role == "team:pod"

    comms = root.subagent_comms

    def origin_stamped() -> bool:
        session_dir = comms.session_dir_of(job_id)
        return session_dir is not None and (session_dir / "origin.json").exists()

    # Wait on the event, never on the clock (AGENTS.md): ride the child's own
    # publications, both sources watched exactly as test_comms.py's integration
    # cases do — the parent stream carries the attach that makes
    # ``session_dir_of`` non-None, the detail registry every durable transcript
    # append. The attach runs strictly after the builder's
    # ``mark_session_origin`` stamp, so no publication can wake this wait
    # before ``origin.json`` exists.
    signal = ChangeSignal().watch_comms(comms).watch_session(root)
    try:
        await wait_for(origin_stamped, signal=signal)
    finally:
        signal.close()
    session_dir = comms.session_dir_of(job_id)
    assert session_dir is not None, "the child never stamped origin.json"
    origin_path = session_dir / "origin.json"
    origin = json.loads(origin_path.read_text())
    assert origin["origin"] == "subagent"
    assert origin["agent"] == "team:pod"
    assert origin["label"] == "lead"
    await root.dispose()


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        ("team:nosuch", "unknown team 'nosuch'"),
        ("team:pod:2", "per copy"),
        ("team:", "no team name"),
    ],
)
def test_an_unresolvable_team_launch_is_an_error_and_registers_nothing(
    tmp_path, teams, agent, message
):
    root = make_root(tmp_path, teams)
    with pytest.raises(TeamLaunchError, match=message):
        root._launch_subagent(label="x", prompt="go", agent=agent)
    assert root.jobs.list() == []


def test_no_registry_is_an_unknown_team(tmp_path, config):
    root = make_root(tmp_path, None, team=None)
    with pytest.raises(TeamLaunchError, match="unknown team"):
        resolve_launch_target("team:pod", root)


@pytest.mark.parametrize("manager", ["scout", "quiet"])
def test_a_manager_that_cannot_delegate_is_an_error(tmp_path, teams, manager, monkeypatch):
    pod = teams.get_team_by_name("pod")
    assert pod is not None
    teams.update_team(pod.id, TeamEditFields(manager=manager))
    quiet = AgentProfile(name="quiet", description="d", may_delegate=False)
    real = subagent_mod._resolve_role
    monkeypatch.setattr(
        subagent_mod,
        "_resolve_role",
        lambda agent, parent: quiet if agent == "quiet" else real(agent, parent),
    )
    root = make_root(tmp_path, teams)
    with pytest.raises(TeamLaunchError, match="cannot delegate"):
        resolve_launch_target("team:pod", root)


# -- D3 / N3: cycles and the cap -----------------------------------------------


@pytest.mark.asyncio
async def test_a_cycle_is_an_error(tmp_path, teams):
    pod = teams.get_team_by_name("pod")
    assert pod is not None
    teams.update_team(
        pod.id,
        TeamEditFields(members=[TeamMember(role="coder"), TeamMember(role="org", kind="team")]),
    )
    root = make_root(tmp_path, teams)
    with pytest.raises(TeamLaunchError, match=r"cycle: team 'org' is already above"):
        resolve_launch_target("team:org", root)  # A -> A
    lead, _, _ = await build(root, "team:pod", "job-lead")
    with pytest.raises(TeamLaunchError, match=r"cycle: team 'org'.*\(org > pod\)"):
        resolve_launch_target("team:org", lead)  # A -> B -> A
    await lead.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_launch_past_the_cap_is_an_error(tmp_path, teams, config):
    (config / "config.yml").write_text("values:\n  subagents:\n    max_team_depth: 2\n")
    root = make_root(tmp_path, teams)
    child, _, _ = await build(root, "manager", "job-1")
    grand, target, _ = await build(child, "task", "job-2")
    assert target.depth == 2
    with pytest.raises(
        TeamLaunchError,
        match="depth cap: this launch would be depth 3; subagents.max_team_depth is 2",
    ):
        resolve_launch_target("task", grand)
    for session in (grand, child, root):
        await session.dispose()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, DEFAULT_MAX_TEAM_DEPTH), ("20", 8), ("0", 1), ("junk", 3), ("true", 3)],
)
def test_the_cap_reader_clamps_and_never_raises(config, raw, expected):
    if raw is not None:
        (config / "config.yml").write_text(f"values:\n  subagents:\n    max_team_depth: {raw}\n")
    assert read_max_team_depth() == expected


@pytest.mark.asyncio
async def test_a_tree_with_no_team_is_not_capped(tmp_path, config):
    (config / "config.yml").write_text("values:\n  subagents:\n    max_team_depth: 1\n")
    root = make_root(tmp_path, None, team=None)
    child, _, _ = await build(root, "task", "job-1")
    grand, target, text = await build(child, "task", "job-2")
    deeper = resolve_launch_target("task", grand)
    assert (target.depth, deeper.depth) == (2, 3)
    assert text == "work", "no team, no stamp at any depth"
    for session in (grand, child, root):
        await session.dispose()


# -- D4 / N2: the chain of command ---------------------------------------------


@pytest.mark.asyncio
async def test_every_launch_below_depth_one_carries_the_escalation_line(tmp_path, teams):
    root = make_root(tmp_path, teams)
    pod = teams.get_team_by_name("pod")
    assert pod is not None
    lead, _, lead_text = await build(root, "team:pod", "job-lead")
    assert "[chain of command] You report to the operator's top session" in lead_text
    worker, target, worker_text = await build(lead, "coder", "job-w")
    assert target.team is not None and target.team.name == "pod" and target.depth == 2
    assert worker_text.startswith(
        pod.member_preamble("coder") + escalation_preamble("pod manager (job job-lead)")
    )
    assert "Do not push, merge, deploy, release, delete data, or print secrets." in worker_text
    # Depth-1 member bytes do not carry it (N0).
    member, _ = _effective_prompt("x", "coder", root, resolve_launch_target("coder", root))
    assert "[chain of command]" not in member
    await worker.dispose()
    await lead.dispose()
    await root.dispose()


def test_the_escalation_text_is_frozen():
    assert escalation_preamble("X") == (
        "[chain of command] You report to X, through hub. Do not push, merge, deploy, "
        "release, delete data, or print secrets. When the work needs one of those, stop "
        "that step and escalate it to X through hub with what you would run and why.\n\n"
    )


# -- frozen rule: non-delegating roles never delegate ---------------------------


@pytest.mark.asyncio
async def test_non_delegating_roles_hold_no_task_at_any_depth_inside_a_sub_team(tmp_path, teams):
    root = make_root(tmp_path, teams)
    lead, _, _ = await build(root, "team:pod", "job-lead")
    mgr2, _, _ = await build(lead, "manager", "job-m2")
    sessions = [lead, mgr2]
    for parent, depth, job in ((root, 1, "a"), (lead, 2, "b"), (mgr2, 3, "c")):
        for role in ("scout", "coder"):
            child, target, _ = await build(parent, role, f"job-{job}-{role}")
            assert target.depth == depth
            assert "task" not in {tool.name for tool in child._tools}
            sessions.append(child)
    for session in reversed(sessions):
        await session.dispose()
    await root.dispose()


# -- D1 / D7: resume -----------------------------------------------------------


def _resume_spy(monkeypatch) -> dict[str, object]:
    seen: dict[str, object] = {}

    def spy(**kwargs):
        seen.update(kwargs)
        target = kwargs["target"]
        text, _ = _effective_prompt(
            kwargs["prompt"], kwargs["agent"], kwargs["parent_session"], target
        )
        seen["effective_prompt"] = text
        return "job-new"

    monkeypatch.setattr("local_operator.harness.subagent.run_subagent", spy)
    return seen


def _settle(comms: SubagentComms, job_id: str, tmp_path: Path) -> None:
    directory = tmp_path / job_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    comms._records[job_id].session_dir = directory


def test_a_depth_one_resume_message_is_the_frozen_bytes(tmp_path, teams, monkeypatch):
    root = make_root(tmp_path, teams)
    comms = root.subagent_comms
    comms.record_launch(
        "job-1",
        "w",
        agent_role="task",
        team_name="org",
        team_lineage=(teams.get_team_by_name("org").id,),  # type: ignore[union-attr]
        depth=1,
    )
    _settle(comms, "job-1", tmp_path)
    seen = _resume_spy(monkeypatch)
    new_id, error = comms.resume("job-1", "implement the button")
    assert error is None and new_id == "job-new"
    assert seen["effective_prompt"] == FROZEN_ORG_TASK_BYTES


@pytest.mark.asyncio
async def test_a_resumed_pod_worker_keeps_the_pod_team(tmp_path, teams, monkeypatch):
    """Resumed under its LIVE lead (D5), stamped with pod's text, not org's.

    The carried facts are also what survive a restart: the snapshot/restore
    leg re-reads them off the sidecar rather than the live record."""
    root = make_root(tmp_path, teams)
    org = teams.get_team_by_name("org")
    pod = teams.get_team_by_name("pod")
    assert org is not None and pod is not None
    lead, _, _ = await build(root, "team:pod", "job-lead")
    comms = root.subagent_comms
    comms.attach("job-lead", lead, tmp_path / "job-lead")
    comms.record_launch(
        "job-w",
        "w",
        parent_job_id="job-lead",
        agent_role="coder",
        team_name="pod",
        team_lineage=(org.id, pod.id),
        depth=2,
    )
    _settle(comms, "job-w", tmp_path)
    seen = _resume_spy(monkeypatch)
    new_id, error = comms.resume("job-w", "carry on")
    assert error is None and new_id == "job-new"
    assert seen["parent_session"] is lead
    target = seen["target"]
    assert isinstance(target, LaunchTarget)
    assert target.team is not None and target.team.name == "pod"
    assert target.depth == 2 and target.team_lineage == (org.id, pod.id)
    text = seen["effective_prompt"]
    assert isinstance(text, str)
    assert text.startswith(
        pod.member_preamble("coder") + escalation_preamble("pod manager (job job-lead)")
    )
    assert "[team: org]" not in text

    restored = SubagentComms(root)
    restored.restore(comms.snapshot())
    carried = restored._carried_launch(restored._records["job-w"])
    assert (carried.team_name, carried.team_lineage, carried.depth) == (
        "pod",
        (org.id, pod.id),
        2,
    )
    assert carried.reports_to == "pod manager (job job-lead)"
    await lead.dispose()
    await root.dispose()


def test_an_orphaned_pod_worker_resumes_on_the_root_but_keeps_its_pod(tmp_path, teams, monkeypatch):
    """D5 addendum 2 (manager ruling 2026-09-26): the fallback, not a refusal.

    A resume whose parent is gone re-parents the JOB onto the root — refusing
    would delete the recovery ``_inherited_model`` exists for — while the child
    keeps the team, lineage and depth its own record carries. So a pod worker
    resumed after its lead died comes back under pod's preamble, not org's.
    """
    root = make_root(tmp_path, teams)
    org = teams.get_team_by_name("org")
    pod = teams.get_team_by_name("pod")
    assert org is not None and pod is not None
    comms = root.subagent_comms
    comms.record_launch("job-lead", "lead", agent_role="team:pod", team_name="pod")
    comms.record_launch(
        "job-w",
        "w",
        parent_job_id="job-lead",
        agent_role="coder",
        team_name="pod",
        team_lineage=(org.id, pod.id),
        depth=2,
    )
    _settle(comms, "job-lead", tmp_path)
    _settle(comms, "job-w", tmp_path)
    seen = _resume_spy(monkeypatch)

    new_id, error = comms.resume("job-w", "carry on")

    assert error is None and new_id == "job-new"
    # The job is the root's; the team is still pod's.
    assert seen["parent_session"] is root
    target = seen["target"]
    assert isinstance(target, LaunchTarget)
    assert target.team is not None and target.team.name == "pod"
    assert target.depth == 2 and target.team_lineage == (org.id, pod.id)
    text = seen["effective_prompt"]
    assert isinstance(text, str)
    assert text.startswith(pod.member_preamble("coder"))
    assert "[team: org]" not in text


def test_a_legacy_record_resumes_under_the_roots_team(tmp_path, teams):
    root = make_root(tmp_path, teams)
    comms = root.subagent_comms
    comms.record_launch("job-1", "w", agent_role="task")
    carried = comms._carried_launch(comms._records["job-1"])
    assert isinstance(carried, CarriedLaunch)
    assert (carried.team_name, carried.depth) == ("org", 1)
