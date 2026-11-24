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
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.types import Message, StreamEndEvent
from local_operator.resume import (
    ATTACHMENT_SIDECAR_NAME,
    read_session_attachment,
    write_session_attachment,
)
from local_operator.session.goal import GoalState
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript, TranscriptEntry
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


def _session(
    tmp_path: Path, registries: tuple[AgentRegistry, TeamRegistry], *, directory: Path | None = None
) -> Session:
    """A session over ``tmp_path/sess`` — reopening one RESUMES it, which is
    exactly the shape both ``--resume <id>`` and the ``/resume`` picker build."""
    agents, teams = registries
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[]]),
        tools=[],
        transcript=Transcript(directory or tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        agent_registry=agents,
        team_registry=teams,
    )


def _sidecar(tmp_path: Path) -> dict[str, Any]:
    raw = (tmp_path / "sess" / ATTACHMENT_SIDECAR_NAME).read_text(encoding="utf-8")
    return json.loads(raw)


def _store_sidecar(tmp_path: Path, *, team: str = "", agent: str = "", goal: str = "") -> None:
    """Write the sidecar DIRECTLY, the way a build that predates the team rule did.

    Issue #2014 made ``/team`` + ``/agent`` unreachable through the mutators (a
    team owns the agent slot), so a fixture that needs a STORED PAIR — the
    legacy shape this file has to keep reading, and the widest miss notice —
    writes the file instead of reaching for the two attaches that would now
    raise. This is not a workaround around the rule: the pair on disk is exactly
    what an older build left behind, which is the input the restore must define
    behaviour for.
    """
    (tmp_path / "sess").mkdir(parents=True, exist_ok=True)
    write_session_attachment(tmp_path / "sess", team=team, agent=agent, goal=goal)


class TestTheRoundTrip:
    def test_an_attached_team_comes_back_with_its_manager_as_the_speaker(
        self, tmp_path, registries
    ):
        """THE bug: a resumed session opened with the persona gone.

        A team is the ONE persona a session carries now (issue #2014): attaching
        it claims the agent slot for its manager, so what a resume has to bring
        back is the team, its brief, and that claim — not a team/agent pair that
        was never a state this session could hold.
        """
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))
        first.set_goal("keep the band honest")

        resumed = _session(tmp_path, registries)
        assert resumed.active_team_name == "lopdev"
        assert resumed.active_agent == "manager"
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
    def test_a_name_with_repeated_spaces_round_trips(self, tmp_path, registries):
        """R2: the stored value is a LOOKUP KEY, so it must not be normalised.

        Agent-profile names are free-form and ``AgentRegistry.create_agent``
        does not collapse whitespace, but every resolver matches with ``strip``
        only. Collapsing on write meant a profile named ``"Deep  Auditor"``
        attached live, was written as ``"Deep Auditor"``, then failed to resolve
        on resume and told the user it had been renamed or deleted. Team names
        cannot hit this (``_NAME_RE`` forbids spaces), so this is agent-only.
        """
        agents, teams = registries
        agents.create_agent(
            _role_fields(name="Deep  Auditor", description="Audits deeply", tags=["role"])
        )
        first = _session(tmp_path, registries)
        assert first.attach_agent_profile("Deep  Auditor") == "Deep  Auditor"
        assert _sidecar(tmp_path)["agent"] == "Deep  Auditor"

        resumed = _session(tmp_path, registries)
        assert resumed.active_agent == "Deep  Auditor"
        assert resumed.attachment_restore_notice == ""

    def test_surrounding_whitespace_is_still_stripped(self, tmp_path, registries):
        """Strip yes, collapse no: a stored key with stray edge whitespace would
        not match either, and stripping is what the resolvers themselves do."""
        from local_operator.resume import (
            read_session_attachment,
            write_session_attachment,
        )

        write_session_attachment(tmp_path, team="  lopdev ", agent=" a  b ", goal="  g ")
        stored = read_session_attachment(tmp_path)
        assert stored is not None
        assert (stored.team, stored.agent, stored.goal) == ("lopdev", "a  b", "g")

    def test_only_names_are_stored(self, tmp_path, registries):
        """Briefs are large and go stale; the sidecar holds names.

        A team's stored agent slot is the MANAGER it claimed (issue #2014), not
        a second profile: the slot is the team's, so the name on disk is the
        identity the team put there. The brief itself never reaches the file —
        only the two names and the goal.
        """
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))

        payload = _sidecar(tmp_path)
        assert payload == {"team": "lopdev", "agent": "manager", "goal": ""}
        assert "Ship reviewed work." not in json.dumps(payload)

    def test_an_agent_only_session_stores_its_profile_name(self, tmp_path, registries):
        """The other half of the same contract, and the shape a session keeps
        when no team is attached: the display name, nothing else."""
        session = _session(tmp_path, registries)
        session.attach_agent_profile("auditor")

        assert _sidecar(tmp_path) == {"team": "", "agent": "auditor", "goal": ""}

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
        assert "lopdev" in resumed.attachment_restore_notice
        # Names the recovery step, the way every sibling miss-notice does.
        assert "/team" in resumed.attachment_restore_notice
        assert "re-attach" in resumed.attachment_restore_notice

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
        assert "auditor" in resumed.attachment_restore_notice

    def test_a_successful_restore_reports_nothing(self, tmp_path, registries):
        """The notice is for a real miss; a clean resume must stay quiet."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        resumed = _session(tmp_path, registries)
        assert resumed.attachment_restore_notice == ""

    def test_the_cause_is_only_claimed_when_it_has_been_established(self, tmp_path, registries):
        """A missing or unloaded definition must not be reported as deleted."""
        _, teams = registries
        agents, _ = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        def resumed_with(team_registry: Any) -> Session:
            return Session(
                model=MODEL,
                stream_fn=ScriptedStream([[]]),
                tools=[],
                transcript=Transcript(tmp_path / "sess"),
                system_blocks_provider=lambda: [],
                agent_registry=agents,
                team_registry=team_registry,
            )

        class Raises:
            def get_team_by_name(self, name: str, *, require_loaded=False) -> Any:
                raise RuntimeError("registry down")

        for registry in (TeamRegistry(tmp_path / "elsewhere"), None, Raises()):
            session = resumed_with(registry)
            assert session.active_team is None
            assert session.attachment_restore_notice == (
                "lopdev: team definition is unavailable; re-attach via /team"
            )

    def test_the_notice_fits_one_line_on_a_standard_terminal(self, tmp_path, registries):
        """It wrapped to two lines at 100 columns, which is where it is read."""
        agents, _ = registries
        # The legacy pair is not an authority record for either of its slots.
        _store_sidecar(tmp_path, team="lopdev", agent="auditor")

        resumed = Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=AgentRegistry(tmp_path / "elsewhere"),
            team_registry=TeamRegistry(tmp_path / "elsewhere"),
        )
        assert "lopdev" in resumed.attachment_restore_notice
        assert "re-attach via /team" in resumed.attachment_restore_notice
        assert resumed.active_agent == ""
        assert len(resumed.attachment_restore_notice) <= 96, resumed.attachment_restore_notice

    def test_a_partial_restore_does_not_erase_the_name_it_could_not_resolve(
        self, tmp_path, registries
    ):
        """A restore is a READ. Writing back through the mutators it calls would
        persist ``team=""`` for a team that was only momentarily unresolvable
        (registry not wired on this host, a team directory not yet synced),
        turning a recoverable miss into permanent data loss."""
        agents, _ = registries
        # A stored pair whose TEAM is the half that cannot resolve. The agent
        # half resolves, and that is the point: a restore that wrote back
        # through ``attach_agent_profile`` would journal the empty team slot.
        _session(tmp_path, registries).attach_team(registries[1].get_team_by_name("lopdev"))
        _store_sidecar(tmp_path, team="lopdev", agent="auditor")
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

    def test_an_unresolved_name_survives_later_use_of_the_session(self, tmp_path, registries):
        """R1: the recovery must outlive the restore, not just the restore call.

        Suppressing the write DURING the restore only protected the stored name
        until the resumed session was used at all: ``active_team`` is ``None``
        for the half that missed, so the next ordinary mutation journalled that
        emptiness over the surviving name. A plain ``/goal`` was enough, and the
        team was then gone for good even once the registry came back — the exact
        scenario the suppression was written for, one command later. The
        previous version of this test stopped before any mutation, which is why
        it passed.
        """
        _, teams = registries
        agents, _ = registries
        # The stored pair, because the R1 scenario is a session that was carrying
        # one when the team registry went away mid-life.
        _session(tmp_path, registries).attach_team(registries[1].get_team_by_name("lopdev"))
        _store_sidecar(tmp_path, team="lopdev", agent="auditor")

        # Transient miss: the team registry is not wired on this host yet.
        resumed = Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=agents,
            team_registry=TeamRegistry(tmp_path / "elsewhere"),
        )
        assert resumed.active_team_name == ""

        # Ordinary use of the resumed session, of each kind that journals.
        resumed.set_goal("carry on working")
        resumed.attach_agent_profile("auditor")
        assert _sidecar(tmp_path)["team"] == "lopdev"

        # And the recovery still works once the registry is back.
        recovered = _session(tmp_path, registries)
        assert recovered.active_team_name == "lopdev"
        assert recovered.goal == "carry on working"

    def test_attaching_another_team_retires_the_carried_name(self, tmp_path, registries):
        """The carried name is a recovery hint, not a permanent shadow.

        A user who resumes into a miss and then attaches something else has
        answered the question, so the old name must not survive to re-attach
        itself at the next resume.
        """
        _, teams = registries
        agents, _ = registries
        teams.save_team(_team("other", instructions="Different work."))
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        resumed = Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=agents,
            team_registry=TeamRegistry(tmp_path / "elsewhere"),
        )
        assert resumed.active_team_name == ""
        # The user picks a different team; the registry is reachable again.
        resumed.team_registry = teams
        resumed.attach_team(teams.get_team_by_name("other"))
        assert _sidecar(tmp_path)["team"] == "other"

    def test_an_explicit_detach_retires_the_carried_agent_name(self, tmp_path, registries):
        """``/agent clear`` after a miss must actually clear."""
        agents, _ = registries
        first = _session(tmp_path, registries)
        first.attach_agent_profile("auditor")

        resumed = Session(
            model=MODEL,
            stream_fn=ScriptedStream([[]]),
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: [],
            agent_registry=AgentRegistry(tmp_path / "elsewhere"),
            team_registry=registries[1],
        )
        assert resumed.active_agent == ""
        resumed.clear_agent_profile()
        assert _sidecar(tmp_path)["agent"] == ""


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


def _late_adoption_session(
    tmp_path: Path,
    registries: tuple[AgentRegistry, TeamRegistry],
) -> Session:
    """A session over ``tmp_path/sess`` constructed with NOTHING on disk yet.

    The warm-then-create ordering in miniature: construction runs while no
    sidecar exists (what the desktop's first-keystroke warm does), and the
    sidecar is written only afterwards (what ``create`` does on send). The
    provider reads the session's own ``GoalState`` at call time and renders
    through the REAL ``build_system_blocks``, so a tail asserted on it is the
    tail the model would receive.
    """
    agents, teams = registries
    goal_state = GoalState()

    def provider() -> list[str]:
        from local_operator.prompts_api import build_system_blocks

        return build_system_blocks(
            [],
            "<skills/>",
            "test environment",
            "2026-09-26",
            team_brief=goal_state.team_brief,
            agent_brief=goal_state.agent_brief,
        )

    setattr(provider, "append_only_state", True)
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=provider,
        goal_state=goal_state,
        agent_registry=agents,
        team_registry=teams,
    )


def _system_prefix_snapshots(session: Session) -> list[TranscriptEntry]:
    """Every frozen ``system_prefix`` row on the session's own journal."""
    return [
        entry
        for entry in session._transcript.entries()
        if entry.payload.get("custom_type") == "system_prefix"
    ]


class TestLateAdoptionBeforeTheFirstTurn:
    """The desktop's warm-then-create ordering must still brief the first turn.

    Since the draft pre-engage pair (runtime #1607 / app #531) the new-chat
    flow is warm-then-create: the runtime — and so the Session — is constructed
    on the first keystroke, while ``attachment.json`` is written by ``create``
    only when the user sends. ``__init__`` was the sidecar's only reader, so the
    warm runtime read nothing and the first message ran bare on a session the
    user had launched with a team. The fix re-attempts the restore at the last
    safe moment — the top of the first ``_prepare_system_blocks``, before the
    first ``system_prefix`` snapshot is frozen — and these tests pin it, plus
    the guard that keeps it from ever firing on an answered conversation.
    """

    @pytest.mark.asyncio
    async def test_a_sidecar_written_after_construction_is_adopted_on_the_first_turn(
        self, tmp_path, registries
    ):
        """THE regression pin: warm construction + sidecar-on-send = briefed turn."""
        session = _late_adoption_session(tmp_path, registries)
        # The warm: constructed while NOTHING was on disk...
        assert not (tmp_path / "sess" / ATTACHMENT_SIDECAR_NAME).exists()
        # ...and the sidecar lands only now, the way ``create`` writes it.
        from local_operator.resume import bind_session_attachment

        assert bind_session_attachment(tmp_path / "sess", team="lopdev", agent="", root=tmp_path)

        await session.prompt("open the team")
        try:
            assert session.active_team_name == "lopdev"
            assert "Ship reviewed work." in session._goal_state.team_brief
            snapshots = _system_prefix_snapshots(session)
            assert len(snapshots) == 1, "the first turn froze more than one prefix"
            tail = snapshots[0].payload["details"]["blocks"][-1]
            # The brief itself reaches the FIRST snapshot's tail, and the
            # manager marker rides it the way ``manager_preamble`` stamps it.
            assert "<team>" in tail
            assert "[team: lopdev]" in tail
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_an_agent_target_is_adopted_by_the_same_hook(self, tmp_path, registries):
        """The agent slot rides the same late sidecar and the same hook."""
        agents, _ = registries
        row = agents.get_agent_by_name("auditor")
        assert row is not None
        agents.set_agent_system_prompt(row.id, "Audit the work carefully.")
        session = _late_adoption_session(tmp_path, registries)
        write_session_attachment(tmp_path / "sess", team="", agent="auditor", goal="")

        await session.prompt("audit this")
        try:
            assert session.active_agent == "auditor"
            assert "[role: auditor]" in session._goal_state.agent_brief
            snapshots = _system_prefix_snapshots(session)
            assert len(snapshots) == 1
            tail = snapshots[0].payload["details"]["blocks"][-1]
            assert "<agent>" in tail
            assert "[role: auditor]" in tail
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_an_answered_conversation_never_adopts_late(self, tmp_path, registries):
        """The guard: never retro-attach mid-life, however late the sidecar lands.

        The assistant row is written STRAIGHT to the journal — no turn runs
        first — so that the one-shot adoption attempt has NOT yet been consumed
        when the sidecar appears: what rejects the adoption here is the
        assistant-history guard itself, not the attempt having already run.
        """
        session = _late_adoption_session(tmp_path, registries)
        await session._transcript.append_message(Message.assistant("already answered"))
        write_session_attachment(tmp_path / "sess", team="lopdev", agent="", goal="")
        # The sidecar IS on disk and resolvable: the negative below is the
        # guard's doing, not a missing attachment.
        assert read_session_attachment(tmp_path / "sess") is not None

        await session.prompt("carry on")
        try:
            assert session.active_team_name == ""
            assert session._goal_state.team_brief == ""
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_a_live_pre_turn_attach_is_not_reverted_by_a_stale_sidecar(
        self, tmp_path, registries, monkeypatch
    ):
        """F1 (review round 1): the attach the operator JUST made wins over a stale file.

        The sidecar write is best-effort (``write_session_attachment``
        swallows failures by contract), so a live ``/team`` / ``/agent`` /
        ``/goal`` made before the first turn can sit beside a file still
        holding the PREVIOUS life's values. The late adoption must not revert
        those live slots.

        This is the CROSS-SLOT case (issue #2014): the stale file names a team
        AND an agent, the live session has an agent attached and no team, and
        the two slots are mutually exclusive now. "The live value wins" may not
        mean "adopt the stored team and let it claim the slot" — that would drop
        the profile this life just attached, the precise revert F1 forbids. The
        stored team is skipped with the rest of the stale half.
        """
        from local_operator import resume as resume_module

        agents, teams = registries
        # The stale sidecar's values RESOLVE, so a guard-less adoption would
        # actually re-attach them and clobber the live ones — the failure F1
        # names. A stale value that merely fails to resolve would not.
        teams.save_team(_team("architects"))
        agents.create_agent(_role_fields(name="scribe", description="Writes", tags=["role"]))
        session = _late_adoption_session(tmp_path, registries)

        # The previous life left its values on disk...
        resume_module.write_session_attachment(
            tmp_path / "sess", team="architects", agent="scribe", goal="the stale goal"
        )
        # ...and the live attach below runs with the journal write failing
        # SILENTLY (a no-op writer is the observable end state of the real
        # best-effort contract), so the file keeps those old values.
        monkeypatch.setattr(resume_module, "write_session_attachment", lambda *a, **k: None)
        session.attach_agent_profile("auditor")
        session.set_goal("the live goal")

        await session.prompt("first")
        try:
            # Per slot: the live value survives, and the stored team does not
            # engage at all — there is no pair to form, which is the rule.
            assert session.active_team_name == ""
            assert session.active_agent == "auditor"
            assert session.goal == "the live goal"
            # The stale values never engaged at all — no carries, no notice.
            assert session._unresolved_team == ""
            assert session._unresolved_agent == ""
            assert session.attachment_restore_notice == ""
        finally:
            await session.dispose()


@pytest.mark.parametrize("damage", ["missing", "unreadable", "manager-pins"])
def test_moved_team_reopen_refuses_unverified_authority(tmp_path, registries, damage):
    from shutil import copyfile

    from local_operator.network.move_team import (
        KEY,
        TeamIdentityRefused,
        check_arrival,
        source_offer,
    )
    from local_operator.session.placement import MeshStamp, write_stamp_into

    source = _session(tmp_path, registries)
    source.attach_team(registries[1].get_team_by_name("lopdev"))
    directory = tmp_path / "arrived"
    directory.mkdir()
    copyfile(tmp_path / "sess" / ATTACHMENT_SIDECAR_NAME, directory / ATTACHMENT_SIDECAR_NAME)
    offer = source_offer(tmp_path, tmp_path / "sess", supported=True)
    binding = check_arrival(tmp_path, directory, offer)
    assert binding is not None
    write_stamp_into(directory, MeshStamp(session_id="000000000000", origin={KEY: binding}))

    if damage == "missing":
        (directory / ATTACHMENT_SIDECAR_NAME).unlink()
        reason = "lopdev: attachment evidence is unavailable"
    elif damage == "unreadable":
        (directory / ATTACHMENT_SIDECAR_NAME).write_text("{", encoding="utf-8")
        reason = "lopdev: attachment evidence is unreadable"
    else:
        agents, _ = registries
        agents.create_agent(_role_fields(name="manager", tags=["role", "tools:edit"]))
        reason = "lopdev: would grant tool pins: edit"

    with pytest.raises(TeamIdentityRefused, match=reason):
        _session(tmp_path, registries, directory=directory)


def test_local_restore_does_not_reauthorize_manager_pins(tmp_path, registries):
    agents, teams = registries
    manager = agents.create_agent(_role_fields(name="lead", tags=["role", "tools:read"]))
    teams.save_team(_team("ceiling").model_copy(update={"manager": "lead"}))
    _session(tmp_path, registries).attach_team(teams.get_team_by_name("ceiling"))
    agents.update_agent(manager.id, _role_fields(tags=["role", "tools:read,edit"]))

    resumed = _session(tmp_path, registries)
    assert resumed.active_team is None
    assert "edit" not in resumed._deferral_pins
    assert "would grant tool pins: edit" in resumed.attachment_restore_notice


def _legacy_session_dir(tmp_path: Path, *, team: str = "lopdev", sid: str = "000000000001") -> Path:
    """A team session as a pre-binding build left it: no binding, no markers."""
    from local_operator.resume import TRANSCRIPT_NAME

    directory = tmp_path / "sessions" / sid
    directory.mkdir(parents=True)
    (directory / TRANSCRIPT_NAME).write_text(
        f'{{"id":"head-{sid}","role":"user","content":"hi"}}\n', encoding="utf-8"
    )
    write_session_attachment(directory, team=team, agent="", goal="keep")
    return directory


def _cold_team(tmp_path: Path, directory: Path) -> str:
    from types import SimpleNamespace

    from local_operator.session.attached import AttachedSession

    return AttachedSession._cold_bound_team(
        SimpleNamespace(_config_dir=tmp_path, _session_id=directory.name)
    )


def _assert_refused(tmp_path, registries, directory: Path) -> None:
    from local_operator.network.move_team import LOCAL_BINDING_NAME

    before = (
        (directory / LOCAL_BINDING_NAME).read_bytes()
        if (directory / LOCAL_BINDING_NAME).exists()
        else None
    )
    assert _cold_team(tmp_path, directory) == ""
    reopened = _session(tmp_path, registries, directory=directory)
    assert reopened.active_team is None
    assert "recorded authority is unavailable" in reopened.attachment_restore_notice
    assert reopened.goal == "keep"
    after = (
        (directory / LOCAL_BINDING_NAME).read_bytes()
        if (directory / LOCAL_BINDING_NAME).exists()
        else None
    )
    assert after == before, "a refused restore must not record authority"


def test_legacy_local_team_session_is_grandfathered_on_first_load(tmp_path, registries):
    # The v0.112.10 regression: every pre-binding team session refused every send.
    from local_operator.network.move_team import (
        LOCAL_BINDING_NAME,
        was_authorized_locally,
    )

    directory = _legacy_session_dir(tmp_path)
    # Cold preview reports what the warm path will do, without writing.
    assert _cold_team(tmp_path, directory) == "lopdev"
    assert not (directory / LOCAL_BINDING_NAME).exists()

    first = _session(tmp_path, registries, directory=directory)
    assert first.active_team_name == "lopdev"
    assert first.attachment_restore_notice == ""
    assert was_authorized_locally(directory)

    # Second load takes the ordinary bound path against the stamped binding.
    second = _session(tmp_path, registries, directory=directory)
    assert second.active_team_name == "lopdev"
    assert _cold_team(tmp_path, directory) == "lopdev"


def test_legacy_local_session_whose_team_is_missing_stays_refused(tmp_path, registries):
    _assert_refused(tmp_path, registries, _legacy_session_dir(tmp_path, team="ghost"))


def test_mesh_stamp_blocks_grandfathering(tmp_path, registries):
    # A readable stamp without team identity: the session moved here.
    from local_operator.session.placement import MeshStamp, write_stamp_into

    directory = _legacy_session_dir(tmp_path)
    write_stamp_into(directory, MeshStamp(session_id=directory.name, origin={}))
    _assert_refused(tmp_path, registries, directory)


def test_unreadable_mesh_stamp_still_refuses_construction(tmp_path, registries):
    from local_operator.network.move_team import LOCAL_BINDING_NAME, TeamIdentityRefused

    directory = _legacy_session_dir(tmp_path)
    (directory / "mesh.json").write_text("{", encoding="utf-8")
    assert _cold_team(tmp_path, directory) == ""
    with pytest.raises(TeamIdentityRefused):
        _session(tmp_path, registries, directory=directory)
    assert not (directory / LOCAL_BINDING_NAME).exists()


def test_fork_boundary_blocks_grandfathering_durably(tmp_path, registries):
    # The boundary is consumed on first construction; the refusal must outlive it.
    from local_operator.fork import FORK_BOUNDARY_NAME

    directory = _legacy_session_dir(tmp_path)
    (directory / FORK_BOUNDARY_NAME).write_text("{}", encoding="utf-8")
    assert _cold_team(tmp_path, directory) == ""
    first = _session(tmp_path, registries, directory=directory)
    assert first.active_team is None
    assert not (directory / FORK_BOUNDARY_NAME).exists()
    _assert_refused(tmp_path, registries, directory)


def test_reauthorize_binding_blocks_grandfathering(tmp_path, registries):
    from local_operator.network.move_team import require_replica_reauthorization

    directory = _legacy_session_dir(tmp_path)
    require_replica_reauthorization(directory)
    _assert_refused(tmp_path, registries, directory)


def test_held_replica_id_blocks_grandfathering(tmp_path, registries):
    from local_operator.network.sync import replica_dir

    directory = _legacy_session_dir(tmp_path)
    replica_dir(tmp_path, directory.name).mkdir(parents=True)
    _assert_refused(tmp_path, registries, directory)


def test_fresh_id_copy_of_a_held_replica_blocks_grandfathering(tmp_path, registries):
    # Pre-0.112.10 recovery copied a replica into a FRESH id; an interrupted run
    # could leave it unmarked. Its transcript head still matches the replica.
    from shutil import copyfile

    from local_operator.network.sync import replica_dir
    from local_operator.resume import TRANSCRIPT_NAME

    replica = replica_dir(tmp_path, "0000000000aa")
    replica.mkdir(parents=True)
    (replica / TRANSCRIPT_NAME).write_text(
        '\n{"id":"replica-head","role":"user","content":"hi"}\n{"id":"r2"}\n', encoding="utf-8"
    )
    directory = _legacy_session_dir(tmp_path, sid="0000000000bb")
    copyfile(replica / TRANSCRIPT_NAME, directory / TRANSCRIPT_NAME)
    _assert_refused(tmp_path, registries, directory)


def test_unreadable_replica_does_not_block_unrelated_legacy_session(tmp_path, registries):
    from local_operator.network.sync import replica_dir
    from local_operator.resume import TRANSCRIPT_NAME

    replica = replica_dir(tmp_path, "0000000000aa")
    replica.mkdir(parents=True)
    (replica / (TRANSCRIPT_NAME + ".zst")).write_bytes(b"not zstd")
    directory = _legacy_session_dir(tmp_path)
    assert _session(tmp_path, registries, directory=directory).active_team_name == "lopdev"


def test_legacy_grandfather_write_failure_authorizes_nothing(tmp_path, registries, monkeypatch):
    from local_operator.network import move_team

    directory = _legacy_session_dir(tmp_path)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(move_team, "_write_local_binding", fail)
    reopened = _session(tmp_path, registries, directory=directory)
    assert reopened.active_team is None
    assert "recorded authority is unavailable" in reopened.attachment_restore_notice
    assert not (directory / move_team.LOCAL_BINDING_NAME).exists()


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX mode bits, non-root")
def test_unwritable_legacy_session_refuses_warm_and_cold(tmp_path, registries):
    # The real write failure: the cold preview must not advertise a team the
    # warm restore cannot stamp.
    directory = _legacy_session_dir(tmp_path)
    directory.chmod(0o555)
    try:
        _assert_refused(tmp_path, registries, directory)
    finally:
        directory.chmod(0o755)


def test_fork_of_legacy_parent_inherits_the_team(tmp_path, registries):
    from local_operator.fork import fork_session
    from local_operator.network.move_team import LOCAL_BINDING_NAME

    parent = _legacy_session_dir(tmp_path)
    child = parent.parent / fork_session(tmp_path, parent.name)
    assert not (parent / LOCAL_BINDING_NAME).exists(), "forking must not write the parent"
    assert _session(tmp_path, registries, directory=child).active_team_name == "lopdev"


def test_fork_of_unresolvable_legacy_parent_stays_refused(tmp_path, registries):
    from local_operator.fork import fork_session

    parent = _legacy_session_dir(tmp_path, team="ghost")
    child = parent.parent / fork_session(tmp_path, parent.name)
    reopened = _session(tmp_path, registries, directory=child)
    assert reopened.active_team is None
    assert "recorded authority is unavailable" in reopened.attachment_restore_notice


def _write_origin(directory: Path, **payload: Any) -> None:
    # Written directly, not via mark_session_origin: these fixtures model what
    # OLD builds left on disk, and the marker writer also touches lineage state.
    from local_operator.resume import ORIGIN_NAME

    (directory / ORIGIN_NAME).write_text(json.dumps(payload), encoding="utf-8")


def test_keep_copy_stamp_with_team_identity_survives_its_fork_boundary(tmp_path, registries):
    # R1: a mesh keep-copy carries authority in mesh.json, not a binding file,
    # beside a fork boundary. The seal must not turn it into a refusal.
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.network.move_team import (
        KEY,
        LOCAL_BINDING_NAME,
        check_arrival,
        source_offer,
    )
    from local_operator.session.placement import MeshStamp, write_stamp_into

    source = _session(tmp_path, registries)
    source.attach_team(registries[1].get_team_by_name("lopdev"))
    directory = _legacy_session_dir(tmp_path, sid="0000000000cc")
    offer = source_offer(tmp_path, tmp_path / "sess", supported=True)
    binding = check_arrival(tmp_path, directory, offer)
    assert binding is not None
    write_stamp_into(
        directory, MeshStamp(session_id=directory.name, origin={"kind": "fork", KEY: binding})
    )
    (directory / FORK_BOUNDARY_NAME).write_text("{}", encoding="utf-8")

    first = _session(tmp_path, registries, directory=directory)
    assert first.active_team_name == "lopdev"
    assert first.attachment_restore_notice == ""
    record = json.loads((directory / LOCAL_BINDING_NAME).read_text("utf-8"))
    assert "reauthorize" not in record[KEY]
    assert _session(tmp_path, registries, directory=directory).active_team_name == "lopdev"


def test_failed_seal_keeps_the_fork_boundary(tmp_path, registries, monkeypatch):
    # S2: a transient seal failure must not consume the only arrival marker.
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.network import move_team

    directory = _legacy_session_dir(tmp_path)
    (directory / FORK_BOUNDARY_NAME).write_text("{}", encoding="utf-8")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(move_team, "_write_local_binding", fail)
        assert _session(tmp_path, registries, directory=directory).active_team is None
    assert (directory / FORK_BOUNDARY_NAME).exists(), "the boundary is the only refusal left"
    # The failure cleared: the next load seals instead of grandfathering.
    assert _session(tmp_path, registries, directory=directory).active_team is None
    _assert_refused(tmp_path, registries, directory)


def test_old_build_fork_of_a_moved_parent_is_refused(tmp_path, registries):
    # S1: forks from de83a610ac47..v0.112.10 dropped mesh.json and wrote no
    # binding; origin.json naming the moved parent is the only evidence.
    from local_operator.session.placement import MeshStamp, write_stamp_into

    parent = _legacy_session_dir(tmp_path, sid="0000000000a1")
    write_stamp_into(parent, MeshStamp(session_id=parent.name, origin={"kind": "moved"}))
    child = _legacy_session_dir(tmp_path, sid="0000000000a2")
    _write_origin(child, origin="fork", parent=parent.name)
    _assert_refused(tmp_path, registries, child)


def test_fork_chain_reaching_a_moved_ancestor_is_refused(tmp_path, registries):
    from local_operator.fork import FORK_BOUNDARY_NAME

    grandparent = _legacy_session_dir(tmp_path, sid="0000000000b1")
    (grandparent / FORK_BOUNDARY_NAME).write_text("{}", encoding="utf-8")
    parent = _legacy_session_dir(tmp_path, sid="0000000000b2")
    _write_origin(parent, origin="fork", parent=grandparent.name)
    child = _legacy_session_dir(tmp_path, sid="0000000000b3")
    _write_origin(child, origin="fork", parent=parent.name)
    _assert_refused(tmp_path, registries, child)


@pytest.mark.parametrize(
    "origin",
    [
        {"origin": "fork", "parent": "0000000000c0", "source_device": "peer"},
        "malformed",
        {"origin": "fork", "parent": "../escape"},
        "cycle",
    ],
)
def test_arrival_or_unreadable_origin_is_refused(tmp_path, registries, origin):
    from local_operator.resume import ORIGIN_NAME

    child = _legacy_session_dir(tmp_path, sid="0000000000c1")
    if origin == "malformed":
        (child / ORIGIN_NAME).write_text("{", encoding="utf-8")
    elif origin == "cycle":
        other = _legacy_session_dir(tmp_path, sid="0000000000c2")
        _write_origin(other, origin="fork", parent=child.name)
        _write_origin(child, origin="fork", parent=other.name)
    else:
        _write_origin(child, **origin)
    _assert_refused(tmp_path, registries, child)


def test_legacy_local_fork_chain_still_grandfathers(tmp_path, registries):
    # A local chain, including a fork whose original was deleted (an ordinary
    # gesture), carries no arrival evidence and must keep working.
    grandparent = _legacy_session_dir(tmp_path, sid="0000000000d1")
    parent = _legacy_session_dir(tmp_path, sid="0000000000d2")
    _write_origin(parent, origin="fork", parent=grandparent.name)
    child = _legacy_session_dir(tmp_path, sid="0000000000d3")
    _write_origin(child, origin="fork", parent=parent.name)
    orphan = _legacy_session_dir(tmp_path, sid="0000000000d4")
    _write_origin(orphan, origin="fork", parent="0000000000ff")  # parent deleted
    for directory in (child, orphan):
        assert _cold_team(tmp_path, directory) == "lopdev"
        assert _session(tmp_path, registries, directory=directory).active_team_name == "lopdev"


def test_incomplete_registry_previews_no_legacy_team_cold(tmp_path, registries, monkeypatch):
    # S4/R2: the cold preview must refuse where the warm grandfather refuses.
    from local_operator.session.errors import ProfileRegistryUnavailable

    directory = _legacy_session_dir(tmp_path)
    assert _cold_team(tmp_path, directory) == "lopdev"

    def incomplete(_self: Any) -> None:
        raise ProfileRegistryUnavailable()

    monkeypatch.setattr(AgentRegistry, "require_complete_metadata", incomplete)
    assert _cold_team(tmp_path, directory) == ""


def test_fork_of_legacy_parent_on_incomplete_registry_is_not_grandfathered(
    tmp_path, registries, monkeypatch
):
    # R3: the fork grandfather runs the same completeness guard.
    from local_operator.fork import fork_session
    from local_operator.network.move_team import KEY, LOCAL_BINDING_NAME
    from local_operator.session.errors import ProfileRegistryUnavailable

    def incomplete(_self: Any) -> None:
        raise ProfileRegistryUnavailable()

    from local_operator.network import move_team

    authorized: list[str] = []
    real_authorize = move_team.authorize_named_team

    def spy(*args: Any, **kwargs: Any) -> str:
        authorized.append(args[-1])
        return real_authorize(*args, **kwargs)

    parent = _legacy_session_dir(tmp_path)
    monkeypatch.setattr(AgentRegistry, "require_complete_metadata", incomplete)
    monkeypatch.setattr(move_team, "authorize_named_team", spy)
    child = parent.parent / fork_session(tmp_path, parent.name)
    record = json.loads((child / LOCAL_BINDING_NAME).read_text("utf-8"))
    assert record == {KEY: {"reauthorize": True}}
    # The guard refuses BEFORE any authorization is attempted, not by a
    # downstream resolution failure that happens to look the same.
    assert authorized == []


def test_undecodable_transcript_is_judged_only_against_held_replicas(tmp_path):
    # Q1: with nothing held there is nothing to be a copy of, so a transcript
    # this runtime cannot decode (zstd on 3.12/3.13) is not refused for it.
    # Once a replica IS held, an undecodable head fails closed.
    from local_operator.network.move_team import is_legacy_local_attachment
    from local_operator.network.sync import (
        NETWORK_DIRNAME,
        REPLICA_DIRNAME,
        replica_dir,
    )
    from local_operator.resume import TRANSCRIPT_NAME

    directory = _legacy_session_dir(tmp_path)
    (directory / TRANSCRIPT_NAME).unlink()
    (directory / (TRANSCRIPT_NAME + ".zst")).write_bytes(b"not zstd")
    assert is_legacy_local_attachment(directory)
    (tmp_path / NETWORK_DIRNAME / REPLICA_DIRNAME).mkdir(parents=True)
    assert is_legacy_local_attachment(directory), "an empty replica store holds nothing"
    replica = replica_dir(tmp_path, "0000000000aa")
    replica.mkdir()
    (replica / TRANSCRIPT_NAME).write_text('{"id":"r"}\n', encoding="utf-8")
    assert not is_legacy_local_attachment(directory)


@pytest.mark.asyncio
async def test_generic_child_of_moved_team_cannot_gain_tools_or_pins(tmp_path, registries):
    from local_operator.harness.subagent import (
        TeamLaunchError,
        _build_child_session,
        resolve_launch_target,
    )
    from local_operator.harness.types import AgentTool
    from local_operator.network.move_team import KEY, check_arrival, source_offer
    from local_operator.session.placement import MeshStamp, write_stamp_into

    agents, teams = registries
    agents.create_agent(_role_fields(name="lead", tags=["role", "tools:read"]))
    teams.save_team(_team("ceiling").model_copy(update={"manager": "lead"}))
    _session(tmp_path, registries).attach_team(teams.get_team_by_name("ceiling"))
    directory = tmp_path / "arrived"
    write_session_attachment(directory, team="ceiling", agent="", goal="")
    offer = source_offer(tmp_path, tmp_path / "sess", supported=True)
    binding = check_arrival(tmp_path, directory, offer)
    write_stamp_into(directory, MeshStamp(session_id="000000000000", origin={KEY: binding}))
    moved = _session(tmp_path, registries, directory=directory)

    async def not_called(*args, **kwargs):
        raise AssertionError("no tools are executed")

    moved.refresh_tools([AgentTool(name="read", execute=not_called)])
    child = await _build_child_session(
        label="bound",
        prompt="inspect",
        parent_session=moved,
        model_spec=None,
        job_id="job-generic",
        target=resolve_launch_target("task", moved),
    )
    try:
        assert {tool.name for tool in child._tools} == {"read"}
        child.set_tool_deferral(pins={"edit"})
        assert child._deferral_pins <= moved._deferral_pins == {"read"}
        with pytest.raises(TeamLaunchError, match="must inherit its parent's model"):
            moved._launch_subagent("bound", "inspect", effort="hi")
        assert not moved.jobs.list()
    finally:
        await child.dispose()
        await moved.dispose()
