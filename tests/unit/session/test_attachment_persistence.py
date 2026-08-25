"""An attached ``/team``, ``/agent`` and ``/goal`` have to survive a resume.

The team and agent briefs ride the VOLATILE TAIL of the system prompt, and the
holder that tail is built from (``GoalState``) is constructed EMPTY by
``session_factory`` on every session. Nothing in the transcript reproduced it,
so a ``--resume`` did not merely blank the status band's team segment: it
dropped the persona from the model's instructions entirely, and the
conversation carried on as an ordinary session while the band honestly reported
nothing attached.

The fix journals the attachment to a sidecar beside the transcript and
re-resolves it at construction. Four properties are load-bearing and each has a
test below:

* **The BRIEF comes back, not just the name.** A band segment that names a team
  whose instructions are not in the prompt is the same bug wearing a label, so
  the round-trip test asserts on the tail's content.
* **Names are re-resolved, never replayed.** The stored value is a NAME; a team
  edited between sessions must resume with its CURRENT briefs.
* **A stale name degrades, and says so.** A team the operator deleted must not
  make the conversation unopenable, must not leave the band naming it, and must
  not be silent.
* **A restore does not overwrite the sidecar.** It is a read of state already on
  disk, and a partial restore that wrote back would erase the half it could not
  resolve.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.resume import ATTACHMENT_SIDECAR_NAME, read_session_attachment
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import Team, TeamRegistry

from .test_session import MODEL, ScriptedStream


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


def _team(name: str, *, instructions: str = "Ship reviewed work.") -> Team:
    return Team(
        id=f"t-{name}",
        name=name,
        created_date=datetime.now(timezone.utc),
        manager="manager",
        instructions=instructions,
        project="local-operator",
    )


@pytest.fixture
def registries(tmp_path: Path) -> tuple[AgentRegistry, TeamRegistry]:
    agents = AgentRegistry(tmp_path)
    agents.create_agent(_role_fields(name="auditor", description="Audits", tags=["role"]))
    teams = TeamRegistry(tmp_path)
    teams.save_team(_team("lopdev"))
    return agents, teams


def _session(tmp_path: Path, registries: tuple[AgentRegistry, TeamRegistry]) -> Session:
    """A session over ``tmp_path/sess`` — reopening one RESUMES it, which is
    exactly the shape both ``--resume <id>`` and the ``/resume`` picker build."""
    agents, teams = registries
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        agent_registry=agents,
        team_registry=teams,
    )


def _sidecar(tmp_path: Path) -> dict[str, Any]:
    raw = (tmp_path / "sess" / ATTACHMENT_SIDECAR_NAME).read_text(encoding="utf-8")
    return json.loads(raw)


class TestTheRoundTrip:
    def test_an_attached_team_and_agent_come_back_after_a_resume(self, tmp_path, registries):
        """THE bug: a resumed session opened with the persona gone."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))
        first.attach_agent_profile("auditor")
        first.set_goal("keep the band honest")

        resumed = _session(tmp_path, registries)
        assert resumed.active_team_name == "lopdev"
        assert resumed.active_agent == "auditor"
        assert resumed.goal == "keep the band honest"

    def test_the_team_brief_is_back_in_the_prompt_tail(self, tmp_path, registries):
        """The NAME alone would be a display fix. What the model is told is the
        thing that was actually lost, so the brief itself has to be asserted."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        resumed = _session(tmp_path, registries)
        assert "Ship reviewed work." in resumed._goal_state.team_brief

    def test_the_active_team_object_is_restored_not_just_its_name(self, tmp_path, registries):
        """``active_team`` is what every ``task`` child inherits its briefs
        from, so a restore that set only the band segment would leave a resumed
        manager delegating without the roster."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        resumed = _session(tmp_path, registries)
        assert resumed.active_team is not None
        assert resumed.active_team.name == "lopdev"

    def test_a_detach_survives_the_resume_too(self, tmp_path, registries):
        """A sidecar left naming a profile the user just dropped would silently
        re-attach it on the next resume."""
        first = _session(tmp_path, registries)
        first.attach_agent_profile("auditor")
        first.clear_agent_profile()

        resumed = _session(tmp_path, registries)
        assert resumed.active_agent == ""
        assert resumed._goal_state.agent_brief == ""

    def test_a_session_that_attached_nothing_writes_no_attachment(self, tmp_path, registries):
        """An unused feature must not litter every session directory."""
        _session(tmp_path, registries)
        assert not (tmp_path / "sess" / ATTACHMENT_SIDECAR_NAME).exists()


class TestNamesNotBriefs:
    def test_only_names_are_stored(self, tmp_path, registries):
        """Briefs are large and go stale; the sidecar holds names."""
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))
        session.attach_agent_profile("auditor")

        payload = _sidecar(tmp_path)
        assert payload == {"team": "lopdev", "agent": "auditor", "goal": ""}
        assert "Ship reviewed work." not in json.dumps(payload)

    def test_an_edited_team_resumes_with_its_current_briefs(self, tmp_path, registries):
        """Why re-resolving by NAME is the deliberate choice: the operator edits
        a team between sessions, and a stored brief would resume them onto
        instructions that no longer exist anywhere."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        edited = teams.get_team_by_name("lopdev")
        assert edited is not None
        edited.instructions = "Ship reviewed work, and always capture frames."
        teams.save_team(edited)

        resumed = _session(tmp_path, registries)
        assert "always capture frames" in resumed._goal_state.team_brief


class TestTheStaleName:
    def test_a_deleted_team_degrades_to_unattached_and_reports_it(self, tmp_path, registries):
        """Renamed or deleted: the resume must open, the band must not name it,
        and the user must be told once."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        # The registry the resumed session sees no longer has the team.
        empty_teams = TeamRegistry(tmp_path / "elsewhere")
        agents, _ = registries
        resumed = Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=agents,
            team_registry=empty_teams,
        )
        assert resumed.active_team_name == ""
        assert resumed._goal_state.team_brief == ""
        assert "lopdev" in resumed.attachment_restore_error
        assert "without it" in resumed.attachment_restore_error

    def test_a_deleted_agent_degrades_the_same_way(self, tmp_path, registries):
        first = _session(tmp_path, registries)
        first.attach_agent_profile("auditor")

        teams = registries[1]
        resumed = Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=AgentRegistry(tmp_path / "elsewhere"),
            team_registry=teams,
        )
        assert resumed.active_agent == ""
        assert "auditor" in resumed.attachment_restore_error

    def test_a_successful_restore_reports_nothing(self, tmp_path, registries):
        """The notice is for a real miss; a clean resume must stay quiet."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        resumed = _session(tmp_path, registries)
        assert resumed.attachment_restore_error == ""

    def test_a_partial_restore_does_not_erase_the_name_it_could_not_resolve(
        self, tmp_path, registries
    ):
        """A restore is a READ. Writing back through the mutators it calls would
        persist ``team=""`` for a team that was only momentarily unresolvable
        (registry not wired on this host, a team directory not yet synced),
        turning a recoverable miss into permanent data loss."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))
        first.attach_agent_profile("auditor")

        agents, _ = registries
        Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=agents,
            team_registry=TeamRegistry(tmp_path / "elsewhere"),
        )
        # The team name is still on disk, so a session opened once the registry
        # is available again resumes it.
        assert _sidecar(tmp_path)["team"] == "lopdev"
        recovered = _session(tmp_path, registries)
        assert recovered.active_team_name == "lopdev"


class TestTheSidecarIsRobust:
    def test_a_corrupt_sidecar_does_not_break_the_resume(self, tmp_path, registries):
        """Losing an attachment is a notice; losing the conversation is not
        survivable. A file cut inside a multi-byte character decodes with
        ``errors="replace"`` rather than raising past an ``except OSError``."""
        session_dir = tmp_path / "sess"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / ATTACHMENT_SIDECAR_NAME).write_bytes(b'{"team": "lop\xff\xfe')

        resumed = _session(tmp_path, registries)
        assert resumed.active_team_name == ""
        assert read_session_attachment(session_dir) is None

    def test_a_missing_sidecar_reads_as_nothing_attached(self, tmp_path):
        assert read_session_attachment(tmp_path / "nope") is None

    def test_writing_preserves_the_directory_mtime(self, tmp_path, registries):
        """Recency ranks the ``/resume`` picker by the transcript's mtime, but
        other readers look at the directory. Attaching a team is bookkeeping
        ABOUT a session, never activity IN it, so it must not reorder anything."""
        import os

        session = _session(tmp_path, registries)
        session_dir = tmp_path / "sess"
        os.utime(session_dir, (1_000_000, 1_000_000))

        session.set_goal("do not touch my mtime")
        assert session_dir.stat().st_mtime == 1_000_000
