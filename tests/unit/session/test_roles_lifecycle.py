"""Conversation-scoped roles through the S3 lifecycle surfaces (LOH-9).

S1 built the composite and S2 the writes; this file is the S3 VERIFICATION
half. Everything asserted here already resolves through the session's composite
on ``main`` — the tests exist so a future reader can see that it does, and so a
regression in one of the four surfaces is a red test rather than a silent
fallback to the account store:

* ``task(agent=…)`` and ``task(agent="team:…")`` (``harness/subagent.py``),
  including the deliberate degradation of an unknown name to a full child;
* ``/agent`` and ``/team`` (the handlers call ``attach_agent_profile`` and
  ``attach_team`` with a registry lookup, and those are what is driven here);
* the COLD viewer path (``AttachedSession``), which is the historical ``/team``
  regression;
* name shadowing, delete/retention/archive, resume and the agent-hint index.

The lifecycle facts these tests pin come from the design's §3.5 table. Two of
them are stated more carefully than the pre-S3 document did: an ARCHIVED
session is protected only while it is one of the picker's ten most recent — the
sweep's guard is that picker, not the archive flag — and the device-level stash
(``<config>/dropped-session-roles``) is never a source or a target of any of
these paths.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.agent_profiles import resolve_profile
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.subagent import (
    _effective_prompt,
    _resolve_role,
    lookup_team,
    resolve_launch_target,
)
from local_operator.profile_scope import scope_session_registries
from local_operator.resume import TRANSCRIPT_NAME
from local_operator.session.archived import set_archived
from local_operator.session.attached import AttachedSession
from local_operator.session.cleanup import (
    CleanupPolicy,
    delete_session,
    mark_store,
    run_cleanup,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import TeamEditFields, TeamRegistry

from .test_session import MODEL, ScriptedStream

PARENT_ID = "aaaaaaaaaaaa"
DAY = 86400.0


def _role_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` spelled out in full (it validates in strict mode)."""
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


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, keyed by relative path, with its bytes."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _registries(config_dir: Path, session_id: str) -> tuple[Any, Any]:
    """The session's registries, scoped the way ``session_factory`` scopes them."""
    return scope_session_registries(
        AgentRegistry(config_dir), TeamRegistry(config_dir), config_dir / "sessions" / session_id
    )


def _session(config_dir: Path, session_id: str) -> Session:
    agents, teams = _registries(config_dir, session_id)
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(config_dir / "sessions" / session_id),
        system_blocks_provider=lambda: [],
        agent_registry=agents,
        team_registry=teams,
    )


def _agents(session: Session) -> Any:
    """The session's agent registry, narrowed for the type checker.

    ``Session.agent_registry`` is ``Any | None`` because a host may carry none;
    every session built by the factory (and by :func:`_session` here) has both,
    and a test that silently ran against ``None`` would be a false pass.
    """
    registry = session.agent_registry
    assert registry is not None
    return registry


def _teams(session: Session) -> Any:
    registry = session.team_registry
    assert registry is not None
    return registry


def _transcript(directory: Path, rows: int = 1) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TRANSCRIPT_NAME).write_text(
        "".join(
            f'{{"id": "m{i}", "role": "user", "content": "question {i}"}}\n' for i in range(rows)
        ),
        encoding="utf-8",
    )


@pytest.fixture
def config(tmp_path: Path) -> Path:
    """A store with one parent session carrying a scoped role and a scoped team."""
    root = tmp_path / "config"
    _transcript(root / "sessions" / PARENT_ID)
    mark_store(root / "sessions")
    agents, teams = _registries(root, PARENT_ID)
    row = agents.create_agent(
        _role_fields(
            name="release-check",
            description="Checks a release",
            tags=["role"],
        ),
        scope="session",
    )
    agents.set_agent_system_prompt(row.id, "Check every release in order.")
    teams.create_team(
        TeamEditFields(
            name="audit",
            description="audits",
            manager="manager",
            members=[],
            instructions="Review in order, then ship.",
            project="local-operator",
        ),
        scope="session",
    )
    return root


def _session_role_id(config_dir: Path) -> str:
    agents, _ = _registries(config_dir, PARENT_ID)
    row = agents.get_agent_by_name("release-check")
    assert row is not None
    return str(row.id)


def _account_rows(config_dir: Path) -> None:
    """One account-wide agent and team, so "the global store is untouched" is measurable."""
    agents = AgentRegistry(config_dir)
    row = agents.create_agent(_role_fields(name="keeper", description="durable", tags=["role"]))
    agents.set_agent_system_prompt(row.id, "Durable instructions.")
    TeamRegistry(config_dir).create_team(
        TeamEditFields(
            name="keepers",
            description="durable roster",
            manager="manager",
            members=[],
            instructions="Ship reviewed work.",
            project="",
        )
    )


def _age(directory: Path, *, days: float) -> None:
    """Backdate a session's activity clock, which is what the sweep ranks on."""
    stamp = time.time() - days * DAY
    os.utime(directory / TRANSCRIPT_NAME, (stamp, stamp))
    os.utime(directory, (stamp, stamp))


class TestTheLaunchPathsResolveSessionRows:
    def test_task_agent_resolves_a_session_role(self, config: Path) -> None:
        """``task(agent=<session role>)``, through the real launch resolver."""
        session = _session(config, PARENT_ID)

        profile = _resolve_role("release-check", session)
        assert profile is not None and profile.name == "release-check"
        target = resolve_launch_target("release-check", session)
        assert target.role == "release-check" and target.team is None
        prompt, resolved = _effective_prompt("do the work", "release-check", session)
        assert resolved == profile
        assert "Check every release in order." in prompt

    def test_an_unknown_role_name_still_degrades_to_a_full_child(self, config: Path) -> None:
        """EXISTING behaviour, pinned: a typo must not cost the delegation.

        ``task`` treats an unresolvable name as "no role" and launches a full
        child rather than failing (``harness/subagent.py:_resolve_role``).
        """
        session = _session(config, PARENT_ID)

        assert _resolve_role("not-a-role", session) is None
        target = resolve_launch_target("not-a-role", session)
        assert target.role == "not-a-role"
        prompt, resolved = _effective_prompt("do the work", "not-a-role", session)
        assert resolved is None
        assert prompt == "do the work"

    def test_task_agent_team_resolves_a_session_team(self, config: Path) -> None:
        """``task(agent="team:<session team>")`` picks the session roster."""
        session = _session(config, PARENT_ID)

        team = lookup_team(session, "audit")
        assert team is not None and team.name == "audit"
        target = resolve_launch_target("team:audit", session)
        assert target.team is not None and target.team.name == "audit"
        assert target.is_team_launch is True
        prompt, _ = _effective_prompt("do the work", "team:audit", session, target)
        assert "Review in order, then ship." in prompt

    def test_agent_attach_resolves_a_session_role(self, config: Path) -> None:
        """The `/agent <name>` seam: the handler's registry lookup + attach."""
        session = _session(config, PARENT_ID)
        row = _agents(session).get_agent_by_name("release-check")
        assert row is not None

        assert session.attach_agent_profile("release-check") == "release-check"
        assert session.active_agent == "release-check"
        assert "Check every release in order." in session._goal_state.agent_brief

    def test_team_attach_resolves_a_session_team(self, config: Path) -> None:
        """The `/team <name>` seam, including the manager slot the team claims."""
        session = _session(config, PARENT_ID)
        team = _teams(session).get_team_by_name("audit")
        assert team is not None

        session.attach_team(team)

        assert session.active_team is not None and session.active_team.name == "audit"
        assert "Review in order, then ship." in session._goal_state.team_brief

    def test_the_cold_viewer_resolves_the_same_rows(self, config: Path) -> None:
        """The historical `/team` regression: a viewer must see session rows.

        ``AttachedSession`` builds both registries lazily from
        ``<config>/sessions/<id>/roles``; reading them without a bound runtime is
        exactly what the cold path does.
        """
        viewer = AttachedSession(
            config_dir=config,
            session_id=PARENT_ID,
            takeover_factory=lambda: None,
        )

        assert viewer.agent_registry is not None
        assert viewer.agent_registry.get_agent_by_name("release-check") is not None
        assert viewer.team_registry is not None
        assert viewer.team_registry.get_team_by_name("audit") is not None


class TestNameShadowing:
    def test_a_session_row_shadows_the_account_row_everywhere(self, config: Path) -> None:
        """Session wins: the composite lists session rows first and resolves there."""
        global_row = AgentRegistry(config).create_agent(
            _role_fields(name="release-check", description="account copy", tags=["role"])
        )
        AgentRegistry(config).set_agent_system_prompt(global_row.id, "Account instructions.")
        session = _session(config, PARENT_ID)
        session_role = _session_role_id(config)

        assert _agents(session).get_agent_by_name("release-check").id == session_role
        profile = resolve_profile("release-check", registry=_agents(session))
        assert profile is not None
        assert "Check every release in order." in profile.preamble
        assert "Account instructions." not in profile.preamble

        # The shared resolver behind /agent attach, a team's manager and the org
        # chart, so one assertion covers those three readers.
        from local_operator.agent_profiles import resolve_profile_or_specialist

        _name, resolved, _specialist, _seed = resolve_profile_or_specialist(
            "release-check", registry=_agents(session)
        )
        assert resolved is not None
        assert "Check every release in order." in resolved.preamble

    def test_a_session_team_shadows_the_account_team(self, config: Path) -> None:
        """The team twin: the picker and both launch lookups answer session-first."""
        global_row = TeamRegistry(config).create_team(
            TeamEditFields(
                name="audit",
                description="account roster",
                manager="manager",
                members=[],
                instructions="Account instructions.",
                project="",
            )
        )
        session = _session(config, PARENT_ID)
        session_row = _teams(session).session_registry.get_team_by_name("audit")
        assert session_row is not None and session_row.id != global_row.id

        assert _teams(session).get_team_by_name("audit").id == session_row.id
        assert lookup_team(session, "audit").id == session_row.id
        target = resolve_launch_target("team:audit", session)
        assert target.team is not None and target.team.id == session_row.id
        prompt, _ = _effective_prompt("do the work", "team:audit", session, target)
        assert "Review in order, then ship." in prompt
        assert "Account instructions." not in prompt

    def test_the_account_row_still_wins_in_a_session_that_does_not_hold_the_name(
        self, config: Path
    ) -> None:
        global_row = AgentRegistry(config).create_agent(
            _role_fields(name="release-check", description="account copy", tags=["role"])
        )
        AgentRegistry(config).set_agent_system_prompt(global_row.id, "Account instructions.")
        other = config / "sessions" / "bbbbbbbbbbbb"
        _transcript(other)
        session = _session(config, "bbbbbbbbbbbb")

        assert _agents(session).get_agent_by_name("release-check").id == global_row.id
        profile = _resolve_role("release-check", session)
        assert profile is not None
        assert "Account instructions." in profile.preamble


class TestDeleteRetentionArchive:
    def test_deleting_the_session_takes_its_roles_and_leaves_the_account_stores_alone(
        self, config: Path
    ) -> None:
        _account_rows(config)
        agents_before = _tree(config / "agents")
        teams_before = _tree(config / "teams")

        outcome = delete_session(config, PARENT_ID, actor="test")

        assert outcome.deleted is True
        assert not (config / "sessions" / PARENT_ID).exists()
        assert _tree(config / "agents") == agents_before
        assert _tree(config / "teams") == teams_before

    def test_the_sweep_takes_an_eligible_roles_bearing_session(self, config: Path) -> None:
        """A session outside the picker's recent-10 goes, roles and all.

        The REGISTRY-WRITTEN tree is what is measured here, not a marker file:
        the roles-bearing session is the fixture's parent, aged past the guard.
        """
        parent = config / "sessions" / PARENT_ID
        roles = _tree(parent / "roles")
        assert "agents" in "".join(roles)  # the fixture's role really is on disk
        _account_rows(config)
        agents_before = _tree(config / "agents")
        teams_before = _tree(config / "teams")
        _age(parent, days=40)
        # Ten fresher sessions, so the target is outside the recent-10 guard and
        # the store is one over ``max_sessions``.
        for index in range(10):
            _transcript(config / "sessions" / f"fresh{index:06d}", rows=2)

        result = run_cleanup(config, CleanupPolicy(enabled=True, max_sessions=10))

        assert PARENT_ID in [candidate.session for candidate in result.chosen]
        assert not parent.exists()
        assert _tree(config / "agents") == agents_before
        assert _tree(config / "teams") == teams_before

    def test_an_archived_session_outside_the_recent_window_is_still_eligible(
        self, config: Path
    ) -> None:
        """DOCUMENTED behaviour: archive hides, it does not declare disposable.

        The sweep's protection is the picker's recent-10 window, and an archived
        row still ranks there; once it falls outside, it is swept like any other
        old session. Pinned as-is — S3 does not change the archive contract.
        """
        parent = config / "sessions" / PARENT_ID
        assert set_archived(config, PARENT_ID, True) is True
        _age(parent, days=40)
        for index in range(10):
            _transcript(config / "sessions" / f"fresh{index:06d}", rows=2)

        result = run_cleanup(config, CleanupPolicy(enabled=True, max_sessions=10))

        # The archive hid it from every list; it did not make it immortal.
        assert PARENT_ID in [candidate.session for candidate in result.chosen]
        assert not parent.exists()

    def test_unarchiving_keeps_the_rows(self, config: Path) -> None:
        roles = config / "sessions" / PARENT_ID / "roles"
        before = _tree(roles)

        assert set_archived(config, PARENT_ID, True) is True
        assert _tree(roles) == before
        # An unarchive of a row that is not archived is a no-op returning False;
        # what matters here is that neither direction touches the tree.
        set_archived(config, PARENT_ID, False)
        assert _tree(roles) == before

    def test_the_stash_is_untouched_by_delete_sweep_and_archive(self, config: Path) -> None:
        stash = config / "dropped-session-roles" / "gone00000001"
        stash.mkdir(parents=True)
        (stash / "stash.json").write_text('{"definitions": ["kept"]}', encoding="utf-8")
        before = _tree(stash)

        set_archived(config, PARENT_ID, True)
        set_archived(config, PARENT_ID, False)
        run_cleanup(config, CleanupPolicy(enabled=True, max_inactive_days=1))
        delete_session(config, PARENT_ID, actor="test")

        assert _tree(stash) == before


class TestResumeAndTheHintIndex:
    def test_a_resumed_session_resolves_the_rows_and_the_persona(self, config: Path) -> None:
        first = _session(config, PARENT_ID)
        # The attach path journals the sidecar; this is the state a resume meets.
        first.attach_agent_profile("release-check")

        resumed = _session(config, PARENT_ID)

        assert _agents(resumed).get_agent_by_name("release-check") is not None
        assert _teams(resumed).get_team_by_name("audit") is not None
        assert resumed.active_agent == "release-check"
        assert resumed.attachment_restore_notice == ""
        assert "Check every release in order." in resumed._goal_state.agent_brief

    def test_a_new_session_in_the_same_config_cannot_see_the_rows(self, config: Path) -> None:
        _transcript(config / "sessions" / "cccccccccccc")
        other = _session(config, "cccccccccccc")

        assert _agents(other).get_agent_by_name("release-check") is None
        assert _teams(other).get_team_by_name("audit") is None

    def test_the_agent_hint_index_points_at_the_sessions_own_tree(self, config: Path) -> None:
        """The recommender must index the COPY, not ``<config>/agents``.

        ``_registered_agent_hints`` reads the directory from the registry's
        ``agent_dir`` seam for exactly this reason; a derived global path would
        point every session row at a directory that does not hold it.
        """
        from local_operator.session_factory import _registered_agent_hints

        session = _session(config, PARENT_ID)
        row = _agents(session).get_agent_by_name("release-check")
        assert row is not None

        hints = _registered_agent_hints(_agents(session))

        hint = next(h for h in hints if h.name == f"registered-agent-{row.id}")
        assert hint.base_dir == config / "sessions" / PARENT_ID / "roles" / "agents" / row.id
        assert hint.file_path == hint.base_dir / "agent.yml"
