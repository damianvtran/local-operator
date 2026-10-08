"""Issue #2014: a team OWNS the session's agent slot, and says so.

The bug was a session that could carry BOTH a team and an agent: the model was
told it was the team's manager ("you coordinate; you do not implement") and then
handed an unrelated profile's brief, with prompt ORDER as the only precedence
rule, no membership check anywhere, and no way for a user or a client to tell
which of the two was answering.

The rule these tests pin is option (i) from the issue, the one the teams guide
already promised ("the current agent becomes the manager of that roster"):

* attaching a team claims the agent slot for that team's MANAGER, replacing
  whatever profile was adopted before it;
* ``/agent`` (attach or clear) is refused while a team is attached, with the
  reason and the way out (``/team clear``);
* ``/team clear`` detaches the team and frees the slot again;
* the resulting identity is published as ``effective_identity`` so a client
  never has to infer it.

Every cell here drives the REAL ``Session`` (or the real routed handler over
one) against real registries — none of them assert on a double's bookkeeping.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from local_operator.agent_profiles import install_seed
from local_operator.agents import AgentRegistry
from local_operator.prompts_api import build_system_blocks
from local_operator.resume import write_session_attachment
from local_operator.session.errors import AgentSlotOwnedByTeam
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.teams import Team, TeamMember, TeamRegistry

from .test_session import MODEL, ScriptedStream

MANAGER = "manager"


def _team(name: str = "lopdev", *, manager: str = MANAGER) -> Team:
    return Team(
        id=f"t-{name}",
        name=name,
        created_date=datetime.now(timezone.utc),
        manager=manager,
        members=[TeamMember(role="coder")],
        instructions="Ship reviewed work.",
        project="local-operator",
    )


@pytest.fixture
def registries(tmp_path: Path) -> tuple[AgentRegistry, TeamRegistry]:
    agents = AgentRegistry(tmp_path)
    # The PACKAGED seed, not a hand-made role row: a seed carries a real
    # preamble, and a role with an empty one is the A2 "hollow" profile the
    # front end reports separately — not the shape this file is about.
    installed = install_seed("scout", registry=agents)
    assert installed is not None
    teams = TeamRegistry(tmp_path)
    teams.save_team(_team())
    return agents, teams


def _session(tmp_path: Path, registries: tuple[AgentRegistry, TeamRegistry]) -> Session:
    """A session over ``tmp_path/sess``; making a second one RESUMES it."""
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


def _tail(session: Session) -> str:
    """The volatile tail exactly as the model receives it (one persona, or two)."""
    return build_system_blocks(
        [],
        "",
        "",
        "",
        team_brief=session._goal_state.team_brief,
        agent_brief=session._goal_state.agent_brief,
    )[-1]


class TestTheTeamClaimsTheSlot:
    def test_attaching_a_team_names_its_manager_as_the_speaker(self, tmp_path, registries):
        """One attach, one speaker — and it is the team's manager, not an empty
        slot the model has to guess about."""
        _, teams = registries
        session = _session(tmp_path, registries)

        session.attach_team(teams.get_team_by_name("lopdev"))

        assert session.active_team_name == "lopdev"
        assert session.active_agent == MANAGER
        # The manager's instructions live in the TEAM brief (the attach lays its
        # profile in front of the roster), so the slot holds the name and no
        # second copy of the same persona.
        assert session.agent_brief == ""
        assert "You are the manager of this team." in session._goal_state.team_brief

    def test_the_tail_carries_exactly_one_persona(self, tmp_path, registries):
        """The bug was TWO briefs on the tail with order as the rule. There is
        now one: ``<team>`` and no ``<agent>`` element at all."""
        _, teams = registries
        session = _session(tmp_path, registries)

        session.attach_team(teams.get_team_by_name("lopdev"))
        tail = _tail(session)

        assert "<team>" in tail and "</team>" in tail
        assert "<agent>" not in tail

    def test_attaching_a_team_replaces_a_profile_attached_before_it(self, tmp_path, registries):
        """The team wins the slot outright, so the mixed state cannot be reached
        by attaching in the other order either."""
        _, teams = registries
        session = _session(tmp_path, registries)
        assert session.attach_agent_profile("scout") == "scout"
        assert session.agent_brief != ""

        session.attach_team(teams.get_team_by_name("lopdev"))

        assert session.active_agent == MANAGER
        assert session.agent_brief == ""
        assert "scout" not in session.agent_brief
        assert "<agent>" not in _tail(session)

    def test_the_managers_identity_survives_a_resume(self, tmp_path, registries):
        """Claimed at attach time, restored with the team: a resumed session
        must not come back with the roster but no speaker."""
        _, teams = registries
        first = _session(tmp_path, registries)
        first.attach_team(teams.get_team_by_name("lopdev"))

        resumed = _session(tmp_path, registries)

        assert resumed.active_team_name == "lopdev"
        assert resumed.active_agent == MANAGER
        assert resumed.attachment_restore_notice == ""


class TestTheSlotIsClosed:
    def test_agent_attach_is_refused_and_names_the_way_out(self, tmp_path, registries):
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))

        with pytest.raises(AgentSlotOwnedByTeam) as refusal:
            session.attach_agent_profile("scout")

        message = str(refusal.value)
        assert message == (
            "team lopdev owns this session: manager is the speaker, so /agent is "
            "closed. Run /team clear to detach the team first."
        )
        # Refused BEFORE resolution and before any mutation: the slot still holds
        # the manager, and no scout brief reached the tail.
        assert session.active_agent == MANAGER
        assert session.agent_brief == ""
        assert "<agent>" not in _tail(session)

    def test_an_unknown_name_is_refused_as_the_rule_not_as_a_typo(self, tmp_path, registries):
        """With a team attached the reason is the team, whatever was typed:
        reporting "no agent named 'nonsense'" would bury the actual rule."""
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))

        with pytest.raises(AgentSlotOwnedByTeam) as refusal:
            session.attach_agent_profile("nonsense")

        assert "no agent named" not in str(refusal.value)
        assert "/team clear" in str(refusal.value)

    def test_agent_clear_is_refused_for_the_same_reason(self, tmp_path, registries):
        """Clearing would leave the team attached with nobody named as the
        speaker, so the detach verb for that state is ``/team clear``.

        The sentence is pinned VERBATIM: it is the copy a user reads, and it has
        to name the team, the manager and the verb that actually moves the state
        in one grammatical line.
        """
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))

        with pytest.raises(AgentSlotOwnedByTeam) as refusal:
            session.clear_agent_profile()

        assert str(refusal.value) == (
            "team lopdev owns this session's profile (manager is the speaker), so "
            "there is nothing to detach here. Run /team clear to detach the team."
        )
        # Refused BEFORE mutating: the manager is still the speaker.
        assert session.active_agent == MANAGER

    def test_detaching_the_team_frees_the_slot(self, tmp_path, registries):
        """ "Detaching the team frees the agent slot" — the issue's own words,
        and the reason it cannot live only in the mutators that refuse."""
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))

        session.attach_team(None)

        assert session.active_team_name == ""
        assert session.active_agent == ""
        assert session.agent_brief == ""
        assert _tail(session) == "<skills/>"
        # And the ordinary verb works again.
        assert session.attach_agent_profile("scout") == "scout"
        assert "<agent>" in _tail(session)


class TestTheIdentityIsPublished:
    def test_no_attachment_is_an_empty_statement(self, tmp_path, registries):
        session = _session(tmp_path, registries)

        assert session.effective_identity == {"speaker": "", "team": "", "role_of_speaker": ""}

    def test_a_profile_is_the_speaker(self, tmp_path, registries):
        session = _session(tmp_path, registries)

        session.attach_agent_profile("scout")

        assert session.effective_identity == {
            "speaker": "scout",
            "team": "",
            "role_of_speaker": "",
        }

    def test_a_team_publishes_its_manager_as_the_speaker(self, tmp_path, registries):
        _, teams = registries
        session = _session(tmp_path, registries)

        session.attach_team(teams.get_team_by_name("lopdev"))

        assert session.effective_identity == {
            "speaker": MANAGER,
            "team": "lopdev",
            "role_of_speaker": "manager",
        }

    def test_a_nameless_team_still_names_a_speaker(self, tmp_path, registries):
        """A reduced double must not produce a field that reads as "nobody":
        the team name is the fallback, and it is a true statement."""
        session = _session(tmp_path, registries)

        # A raw double, because the real ``Team`` model defaults ``manager`` to
        # a name — this is the reduced-facade case the fallback exists for.
        session.attach_team(SimpleNamespace(name="bare", manager=""))

        assert session.effective_identity["speaker"] == "bare"
        assert session.effective_identity["role_of_speaker"] == "manager"
        # And the slot is still the team's, so no profile can layer over it.
        with pytest.raises(AgentSlotOwnedByTeam):
            session.attach_agent_profile("scout")

    def test_the_frontend_state_carries_it_verbatim(self, tmp_path, registries):
        """The field clients read, not the session property: this is the surface
        the companion UI issue paints, so it is asserted on the wire model."""
        _, teams = registries
        session = _session(tmp_path, registries)
        session.attach_team(teams.get_team_by_name("lopdev"))

        session.refresh_frontend_state()
        state = session._frontend_state_store.state

        assert state.effective_identity == {
            "speaker": MANAGER,
            "team": "lopdev",
            "role_of_speaker": "manager",
        }
        assert state.active_agent == MANAGER
        assert state.active_team == "lopdev"


class TestTheLegacyPairMigrates:
    def test_a_stored_team_and_agent_resumes_as_the_team_manager(self, tmp_path, registries):
        """A sidecar written before this rule named both slots (the reporter's
        own session did). It must come back as ONE identity — the team's manager
        — with no "agent did not come back" notice, because nothing failed: the
        rule superseded the stored profile."""
        (tmp_path / "sess").mkdir(parents=True, exist_ok=True)
        write_session_attachment(tmp_path / "sess", team="lopdev", agent="scout", goal="")

        resumed = _session(tmp_path, registries)

        assert resumed.active_team_name == "lopdev"
        assert resumed.active_agent == MANAGER
        assert resumed.agent_brief == ""
        assert resumed.attachment_restore_notice == ""
        assert resumed._unresolved_agent == ""

    def test_a_live_profile_is_not_replaced_by_a_stale_stored_team(self, tmp_path, registries):
        """A sidecar left by the LAST life must not claim this one's slot.

        The live attach journals its own state, so the stale pair is gone from
        disk before any restore can read it: the property is that the ordinary
        verb stays ordinary, and no restore path can reach in and install a
        manager over a profile the operator just chose.
        """
        session = _session(tmp_path, registries)
        # The previous life left a pair on disk (the journal write below is the
        # ordinary one an attach makes, so nothing here is a special case)...
        write_session_attachment(tmp_path / "sess", team="lopdev", agent="scout", goal="")
        # ...and this life attaches a profile instead.
        session.attach_agent_profile("scout")

        assert session.active_team_name == ""
        assert session.active_agent == "scout"
        resumed = _session(tmp_path, registries)
        assert resumed.active_team_name == ""
        assert resumed.active_agent == "scout"


@pytest_asyncio.fixture
async def routed(tmp_path: Path, registries) -> Any:
    """The ROUTED handler over a REAL session — the seam a live terminal uses.

    ``/team`` and ``/agent`` are authoritative-session commands, so on every
    attached session the user's words reach ``ServingSessionHandle`` rather than
    the TUI-local handler. Driving the handle is therefore the difference
    between "the rule is implemented" and "the command the user types obeys it".
    """
    session = _session(tmp_path, registries)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        yield handle, session
    finally:
        await session.dispose()


class TestTheRoutedCommands:
    @pytest.mark.asyncio
    async def test_slash_team_claims_the_slot(self, routed) -> None:
        handle, session = routed

        outcome = await handle.run_slash_authoritative("team", "lopdev", [])

        assert outcome["kind"] == "notice"
        assert "lopdev" in outcome["text"]
        assert session.active_agent == MANAGER

    @pytest.mark.asyncio
    async def test_slash_agent_is_refused_with_the_teams_sentence(self, routed) -> None:
        handle, session = routed
        await handle.run_slash_authoritative("team", "lopdev", [])

        outcome = await handle.run_slash_authoritative("agent", "scout", [])

        assert outcome["kind"] == "notice"
        assert outcome["style"] == "warning"
        assert "team lopdev" in outcome["text"]
        assert "/team clear" in outcome["text"]
        assert session.active_agent == MANAGER

    @pytest.mark.asyncio
    async def test_slash_agent_clear_is_refused_the_same_way(self, routed) -> None:
        handle, session = routed
        await handle.run_slash_authoritative("team", "lopdev", [])

        outcome = await handle.run_slash_authoritative("agent", "clear", [])

        assert outcome["style"] == "warning"
        assert "/team clear" in outcome["text"]
        assert session.active_agent == MANAGER

    @pytest.mark.asyncio
    async def test_slash_team_clear_detaches_and_frees_the_slot(self, routed) -> None:
        handle, session = routed
        await handle.run_slash_authoritative("team", "lopdev", [])

        outcome = await handle.run_slash_authoritative("team", "clear", [])

        assert outcome["kind"] == "notice"
        assert session.active_team_name == ""
        assert session.active_agent == ""
        # And the slot is genuinely usable again through the same seam.
        outcome = await handle.run_slash_authoritative("agent", "scout", [])
        assert session.active_agent == "scout"
        assert outcome["kind"] == "notice"

    @pytest.mark.asyncio
    async def test_slash_team_clear_with_a_request_is_a_mistyped_attach(self, routed) -> None:
        """Only the bare verb detaches — ``/team clear <text>`` stays a lookup
        of a team named ``clear``, exactly as ``/agent clear <text>`` does."""
        handle, session = routed
        await handle.run_slash_authoritative("team", "lopdev", [])

        outcome = await handle.run_slash_authoritative("team", "clear fix this", [])

        assert "no team named" in outcome["text"]
        assert session.active_team_name == "lopdev"

    @pytest.mark.asyncio
    async def test_slash_team_announces_the_profile_it_replaced(self, routed) -> None:
        """Issue #2014 + the UI lane (companion #866): the drop is ANNOUNCED.

        Attaching a team claims the agent slot, so a profile the user chose is
        replaced. A persona that vanishes without a word is the confusion this
        rule exists to remove, and the desktop half of this change tells users the
        runtime says so — so the clause is pinned BYTE FOR BYTE here, and built by
        the one shared builder (``teams.replaced_profile_clause``) in all three
        seams that paint a team receipt.
        """
        handle, session = routed
        await handle.run_slash_authoritative("agent", "scout", [])
        assert session.active_agent == "scout"

        outcome = await handle.run_slash_authoritative("team", "lopdev", [])

        assert outcome["text"].endswith(" Replaced profile scout; manager now speaks.")
        assert session.active_agent == MANAGER

    @pytest.mark.asyncio
    async def test_slash_team_announces_nothing_when_nothing_was_replaced(self, routed) -> None:
        """No clause when there was no profile to replace — a notice that fires
        on every attach is noise, and noise is how a real replacement gets read
        as boilerplate."""
        handle, _ = routed

        outcome = await handle.run_slash_authoritative("team", "lopdev", [])

        assert "Replaced profile" not in outcome["text"]
