"""CLI org sharing (design §8.3): ``agents push/pull --org`` and ``teams push/pull``.

What this pins: the parser contract (``--org`` is additive; teams push/pull are
new subcommands), the dispatch, the credential rule (org calls act as the
signed-in PERSON through the stored OAuth access token; a missing or key-only
login gets the re-login remedy), the org picker (no interactive prompt --
memberships are printed and the flag demanded), and that the without-``--org``
paths stay on the public code.

The hub is faked at the two seams the CLI owns -- ``RadientClient`` and the
``radient_credentials`` resolvers -- so these are dispatch/transport tests; the
wire behavior is the client suite's and the integration run's.
"""

from __future__ import annotations

import time
import types
from pathlib import Path
from typing import Any

import pytest

from local_operator.cli import build_cli_parser, main


@pytest.fixture
def parser():
    return build_cli_parser()


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect Path.home() so no test touches the real ~/.local-operator."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def quiet_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)


class _FakeOrgHub:
    """The org-surface calls the commands make, and what they were asked for."""

    def __init__(self) -> None:
        self.memberships: list[dict[str, Any]] = [
            {
                "tenant_id": "org-a",
                "tenant_name": "Org A",
                "role": "admin",
                "status": "active",
                "is_home": False,
                "plan": {"status": "active", "seats": 2},
            }
        ]
        self.published_teams: list[tuple[dict[str, Any], str]] = []
        self.published_agents: list[tuple[dict[str, Any], Any, Any]] = []
        self.pulled_teams: list[str] = []
        self.team_documents: dict[str, dict[str, Any]] = {}
        #: Rows ``get_agent`` answers, and every (id, with_credential) it was asked for.
        self.agent_rows: dict[str, dict[str, Any]] = {
            "hub-agent-1": {"tenant_id": "org-a", "visibility": "org"}
        }
        self.agent_get_calls: list[tuple[str, bool]] = []

    def list_memberships(self) -> list[dict[str, Any]]:
        return self.memberships

    def publish_team_document(self, document: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.published_teams.append((document, tenant_id))
        return {
            "team": {"id": "hub-team-1", "name": document["name"], "version": document["version"]}
        }

    def get_team(self, team_id: str, **_kw: Any) -> dict[str, Any]:
        self.pulled_teams.append(team_id)
        return self.team_documents[team_id]

    def get_agent(self, agent_id: str, *, with_credential: bool = False) -> dict[str, Any] | None:
        self.agent_get_calls.append((agent_id, with_credential))
        row = self.agent_rows.get(agent_id)
        if row is None:
            return None
        # The real hub envelopes this read; the client returns the envelope
        # as-is (legacy shape), so the fake mirrors it.
        return {"msg": "Agent retrieved successfully", "result": row}

    def publish_agent_instruction_set(
        self, document: dict[str, Any], *, visibility: Any = None, tenant_id: Any = None
    ) -> dict[str, Any]:
        self.published_agents.append((document, visibility, tenant_id))
        return {"agent_id": "hub-agent-1", "name": document["name"], "version": document["version"]}


@pytest.fixture
def org_hub(tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch) -> _FakeOrgHub:
    """A fake org hub behind a signed-in account (OAuth row), wired at the seams."""
    from local_operator.providers import radient_credentials

    hub = _FakeOrgHub()
    monkeypatch.setattr("local_operator.clients.radient.RadientClient", lambda **kwargs: hub)
    monkeypatch.setattr(
        radient_credentials,
        "resolve_radient_oauth_access_sync",
        lambda *args, **kwargs: types.SimpleNamespace(access_token="jwt-fixture", kind="oauth"),
    )
    return hub


def _make_team(name: str = "release-crew"):
    from local_operator.paths import config_dir
    from local_operator.teams import TeamEditFields, TeamRegistry, parse_members

    registry = TeamRegistry(config_dir())
    return registry.create_team(
        TeamEditFields(
            name=name,
            description="Ships it.",
            manager="manager",
            members=parse_members(["coder:2"]),
            instructions="You ship.",
            project="rad-1",
        )
    )


# --- Parser contract -----------------------------------------------------------


def test_agents_push_org_flag_is_additive(parser) -> None:
    assert parser.parse_args(["agents", "push", "--name", "X", "--org", "org-a"]).org == "org-a"
    # Absent by default, so the public path is what runs without the flag.
    assert parser.parse_args(["agents", "push", "--name", "X"]).org is None


def test_agents_pull_org_flag_is_additive(parser) -> None:
    args = parser.parse_args(["agents", "pull", "--id", "Y", "--org", "org-a"])
    assert (args.id, args.org) == ("Y", "org-a")
    assert parser.parse_args(["agents", "pull", "--id", "Y"]).org is None


def test_teams_push_and_pull_parse(parser) -> None:
    push = parser.parse_args(["teams", "push", "--org", "org-a", "crew"])
    assert (push.teams_command, push.name, push.org) == ("push", "crew", "org-a")
    pull = parser.parse_args(["teams", "pull", "team-1", "--org", "org-a"])
    assert (pull.teams_command, pull.team_id, pull.org) == ("pull", "team-1", "org-a")


# --- teams push ----------------------------------------------------------------


def test_teams_push_publishes_the_local_team(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    out = capsys.readouterr().out
    assert len(org_hub.published_teams) == 1
    document, tenant = org_hub.published_teams[0]
    assert tenant == "org-a"
    assert document["name"] == "release-crew"
    assert document["members"] == [{"role": "coder", "kind": "agent", "count": 2}]
    assert document["instructions"] == "You ship."
    assert "Successfully pushed team 'release-crew' to organization 'org-a'" in out


def test_teams_push_refuses_an_oversize_document_locally(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The preflight reports HERE, in the command's own refusal shape: no
    upload is spent on a document the hub would refuse."""
    from local_operator.paths import config_dir
    from local_operator.teams import TeamEditFields, TeamRegistry

    TeamRegistry(config_dir()).create_team(
        TeamEditFields(
            name="release-crew",
            description="d" * 2001,
            manager="manager",
            instructions="You ship.",
        )
    )
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert (
        "cannot push this team: description must be at most 2000 characters (submitted 2001)" in out
    )
    assert org_hub.published_teams == []


def test_teams_push_refuses_a_blank_brief_locally(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 0-byte brief cannot be published (the hub's 'must not be empty'): the
    refusal lands here, in the command's own shape, with no upload spent."""
    from local_operator.paths import config_dir
    from local_operator.teams import TeamEditFields, TeamRegistry

    TeamRegistry(config_dir()).create_team(
        TeamEditFields(name="helpdesk", manager="manager", instructions="")
    )
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "helpdesk"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "cannot push this team: instructions must not be empty" in out
    assert org_hub.published_teams == []


def test_teams_push_sends_a_brief_over_the_old_cap(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A brief between the old local bound (8000) and the hub's (32768) is a
    valid upload and reaches the client."""
    from local_operator.paths import config_dir
    from local_operator.teams import TeamEditFields, TeamRegistry

    brief = "y" * 9_000
    TeamRegistry(config_dir()).create_team(
        TeamEditFields(name="release-crew", manager="manager", instructions=brief)
    )
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    out = capsys.readouterr().out
    assert len(org_hub.published_teams) == 1
    document, tenant = org_hub.published_teams[0]
    assert document["instructions"] == brief
    assert tenant == "org-a"
    assert "Successfully pushed team 'release-crew' to organization 'org-a'" in out


def test_teams_push_without_org_prints_the_memberships_and_requires_the_flag(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The org picker (design §8.3): print, demand the flag, never guess."""
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "pass --org <tenant_id>" in out
    assert "tenant_id: org-a" in out
    assert org_hub.published_teams == []


def test_teams_push_without_memberships_explains_the_invite_path(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    org_hub.memberships = []
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "not a member of one" in out
    assert org_hub.published_teams == []


def test_teams_push_renders_the_hub_code(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A refusal keeps its code and the detail a user can act on."""
    from local_operator.clients._http import APIError

    _make_team()

    def refuse(document: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        raise APIError(
            "The name is already held.",
            status_code=409,
            code="name_taken",
            details={"existing_team_id": "hub-team-9"},
        )

    org_hub.publish_team_document = refuse  # type: ignore[method-assign]
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "The name is already held." in out
    assert "hub-team-9" in out
    assert "name_taken" in out


# --- teams pull ----------------------------------------------------------------


def test_teams_pull_reconstructs_a_local_team(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    org_hub.team_documents["hub-team-1"] = {
        "id": "hub-team-1",
        "tenant_id": "org-a",
        "name": "incident-crew",
        "description": "Responds.",
        "manager": "manager",
        "members": [{"role": "coder", "kind": "agent", "count": 1}],
        "instructions": "You respond.",
        "project": "rad-9",
        "version": "1.0.0",
    }
    monkeypatch.setattr("sys.argv", ["program", "teams", "pull", "hub-team-1", "--org", "org-a"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "Successfully pulled team 'incident-crew'" in out
    assert org_hub.pulled_teams == ["hub-team-1"]

    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    stored = TeamRegistry(config_dir()).get_team_by_name("incident-crew")
    assert stored is not None
    assert stored.instructions == "You respond."
    assert stored.project == "rad-9"


def test_teams_pull_refuses_a_team_from_another_organization(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    org_hub.team_documents["hub-team-9"] = {
        "id": "hub-team-9",
        "tenant_id": "org-b",
        "name": "crew",
        "members": [],
        "instructions": "You ship.",
        "version": "1.0.0",
    }
    monkeypatch.setattr("sys.argv", ["program", "teams", "pull", "hub-team-9", "--org", "org-a"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "belongs to organization 'org-b'" in out


# --- agents org flows -----------------------------------------------------------


def test_agents_push_org_publishes_an_instruction_set(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.paths import config_dir

    registry = AgentRegistry(config_dir())
    agent = registry.create_agent(
        AgentEditFields.model_validate({"name": "OrgCoder", "description": "Writes."})
    )
    registry.set_agent_system_prompt(agent.id, "You write code.")
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "push", "--name", "OrgCoder", "--org", "org-a"]
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert len(org_hub.published_agents) == 1
    document, visibility, tenant = org_hub.published_agents[0]
    assert visibility == "org"
    assert tenant == "org-a"
    assert document["name"] == "OrgCoder"
    assert document["instructions"] == "You write code."
    assert "Successfully pushed agent 'OrgCoder' to organization 'org-a'" in out


def test_agents_pull_org_downloads_through_the_person_client(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from local_operator.agents import AgentRegistry

    seen: dict[str, Any] = {}

    def fake_download(
        self, radient_client, agent_id, *, with_credential=False, tenant_id=None, **_kw
    ):
        seen["client"] = radient_client
        seen["agent_id"] = agent_id
        seen["with_credential"] = with_credential
        pulled = types.SimpleNamespace(name="Pulled", id="local-1")
        return types.SimpleNamespace(agent=pulled, renamed_from=None, model_notice=None)

    monkeypatch.setattr(AgentRegistry, "download_agent_from_radient", fake_download)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", "org-a"]
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert seen["client"] is org_hub
    assert seen["agent_id"] == "hub-agent-1"
    # The org pull must prove membership at the wire: the pre-flight read and
    # the download both carry the bearer (the public pull stays anonymous, R-6).
    assert org_hub.agent_get_calls == [("hub-agent-1", True)]
    assert seen["with_credential"] is True
    assert "Successfully pulled agent 'Pulled'" in out
    assert "from organization 'org-a'" in out


def test_agents_pull_org_refuses_a_row_from_another_organization(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Provenance is verified, teams-style, BEFORE anything is written (Q-1)."""
    from local_operator.agents import AgentRegistry

    org_hub.agent_rows["hub-agent-1"] = {"tenant_id": "org-b", "visibility": "org"}

    def fail_download(*args: Any, **kwargs: Any):
        raise AssertionError("nothing may be downloaded for a mismatched row")

    monkeypatch.setattr(AgentRegistry, "download_agent_from_radient", fail_download)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", "org-a"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "belongs to organization 'org-b', not 'org-a'" in out
    assert "Check --org" in out


def test_agents_pull_org_answers_a_missing_row_like_the_non_member_404(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The hub's 404 (missing id == non-member, §8.2) is reported curated."""
    from local_operator.agents import AgentRegistry

    org_hub.agent_rows.pop("hub-agent-1")

    def fail_download(*args: Any, **kwargs: Any):
        raise AssertionError("nothing may be downloaded for a missing row")

    monkeypatch.setattr(AgentRegistry, "download_agent_from_radient", fail_download)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", "org-a"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "agent not found" in out
    assert "not a member" in out


def test_agents_pull_org_refuses_a_public_row_for_an_org_pull(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A public row is not an org's: the flag's provenance claim would be false.

    Public agents pull anonymously, without the flag; under ``--org`` the row
    must be that organization's (uniform with the mismatch refusal above).
    """
    from local_operator.agents import AgentRegistry

    org_hub.agent_rows["hub-agent-1"] = {"tenant_id": "home-x", "visibility": "public"}

    def fail_download(*args: Any, **kwargs: Any):
        raise AssertionError("nothing may be downloaded for a non-org row")

    monkeypatch.setattr(AgentRegistry, "download_agent_from_radient", fail_download)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", "org-a"]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "is not shared with organization 'org-a'" in out


def test_empty_org_value_refuses_instead_of_falling_back_to_public(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--org ""` is a PASSED flag: refuse it, never run the public path (R1-1)."""
    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.paths import config_dir

    registry = AgentRegistry(config_dir())
    registry.create_agent(AgentEditFields.model_validate({"name": "PushProbe"}))
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "push", "--name", "PushProbe", "--org", ""]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "needs an organization tenant id" in out
    assert "RADIENT_API_KEY" not in out  # the public path's refusal, not ours
    assert org_hub.published_agents == []

    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", ""]
    )

    assert main() == 1

    out = capsys.readouterr().out
    assert "needs an organization tenant id" in out
    assert org_hub.agent_get_calls == []


def test_org_calls_refuse_a_non_canonical_hub_by_default(
    tmp_home: Path,
    quiet_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The person's bearer does not travel to a non-canonical configured hub.

    Security round 1 (S-1/R1-2): the org resolver shares the public resolver's
    destination boundary, so the command refuses with a remedy that names the
    cause and never constructs a client; the documented opt-in
    (``RADIENT_ORG_ALLOW_NONCANONICAL_BASE``) lets a local/QA hub receive it.
    """
    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(config_dir() / "auth.db", config_dir=config_dir())
    try:
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "access": "jwt-fixture",
                "refresh": "refresh-fixture",
                "expires": int(time.time() * 1000) + 3600000,
            },
        )
    finally:
        store.close()

    monkeypatch.setenv("RADIENT_API_BASE_URL", "http://127.0.0.1:9/v1")
    constructed: list[dict[str, Any]] = []

    def note_client(**kwargs: Any) -> _FakeOrgHub:
        constructed.append(kwargs)
        return _FakeOrgHub()

    monkeypatch.setattr("local_operator.clients.radient.RadientClient", note_client)
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "not the Radient cloud API" in out
    # The canonical destination is QUOTED at runtime (cli.py must stay free of
    # URL-shaped Radient literals -- the single-reader guard), so pin that the
    # sentence still names it.
    from local_operator.env import DEFAULT_RADIENT_API_BASE_URL

    assert DEFAULT_RADIENT_API_BASE_URL in out
    assert "RADIENT_ORG_ALLOW_NONCANONICAL_BASE=1" in out
    assert constructed == []
    # And byte-equal to the server's rendered sentence: one definition, two
    # surfaces (agent review round 1, MINOR-2).
    from local_operator.providers.radient_credentials import (
        org_destination_refused_sentence,
    )

    assert (
        org_destination_refused_sentence("http://127.0.0.1:9/v1", DEFAULT_RADIENT_API_BASE_URL)
        in out
    )

    # The explicit opt-in (local/QA hubs) lets the same command reach it.
    monkeypatch.setenv("RADIENT_ORG_ALLOW_NONCANONICAL_BASE", "1")
    hub = _FakeOrgHub()
    monkeypatch.setattr("local_operator.clients.radient.RadientClient", lambda **kwargs: hub)

    assert main() == 0

    assert len(hub.published_teams) == 1
    assert hub.published_teams[0][1] == "org-a"


# --- the credential rule --------------------------------------------------------


def test_org_commands_need_a_signed_in_account(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Missing login -> the re-login remedy, not a raw failure."""
    from local_operator.providers import radient_credentials

    monkeypatch.setattr(
        radient_credentials, "resolve_radient_oauth_access_sync", lambda *a, **k: None
    )
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "crew"])

    assert main() == 1

    out = capsys.readouterr().out
    # The printed sentence is the ONE object both surfaces render (agent review
    # round 1, MINOR-2): a copy on either side breaks this.
    from local_operator.providers.radient_credentials import ORG_LOGIN_REMEDY

    assert ORG_LOGIN_REMEDY in out
    # A device in no network keeps the single sentence: the ask-the-holder clause
    # needs a mesh to ask (design review round 1, D3).
    assert "lop network credential share radient --with <this device>" not in out


def test_the_org_remedy_names_the_holder_share_on_a_mesh_member(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A member device's missing-login remedy gains the ask-the-holder clause (D3).

    The GUIDE says never to run a login on a peer, so where a paired device could
    hold the org login the remedy must name the share path a peer can act on; the
    cell above pins the no-network shape unchanged.
    """
    from local_operator.network import identity as identity_mod
    from local_operator.network import store as network_store
    from local_operator.network.types import MemberRecord, NetworkRecord
    from local_operator.providers import radient_credentials
    from local_operator.providers.radient_credentials import ORG_LOGIN_REMEDY

    identity = identity_mod.load_or_mint()
    record = NetworkRecord(
        network_id="net_org_remedy", name="home", self_device_id=identity.device_id
    )
    record.members.append(MemberRecord(device_id=identity.device_id, name="this-device"))
    record.members.append(MemberRecord(device_id="d_" + "b" * 32, name="cloud-node-1"))
    network_store.save(record)

    monkeypatch.setattr(
        radient_credentials, "resolve_radient_oauth_access_sync", lambda *a, **k: None
    )
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert ORG_LOGIN_REMEDY in out
    assert (
        "Or ask the device that holds it to run "
        "`lop network credential share radient --with <this device>`." in out
    ), out


def test_the_local_server_does_not_restate_the_remedy_sentences() -> None:
    """One definition, two surfaces: the server imports its remedy (MINOR-2).

    The route module used to hold its own ``ORG_LOGIN_REMEDY`` /
    ``ORG_DESTINATION_REFUSED`` bodies pinned only against themselves; this
    catches a second copy growing back.
    """
    from local_operator.server.routes import agents as agents_module

    assert "ORG_LOGIN_REMEDY" not in vars(agents_module)
    assert "ORG_DESTINATION_REFUSED" not in vars(agents_module)


def test_agents_push_without_org_never_runs_the_org_resolver(
    tmp_home: Path, quiet_env: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Without the flag the public code runs, byte-for-byte (design §11 R-6)."""
    from pydantic import SecretStr

    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.paths import config_dir
    from local_operator.providers import radient_credentials

    calls = {"org": 0}

    def explode(*args, **kwargs):
        calls["org"] += 1
        raise AssertionError("the org resolver must not run without --org")

    class _PublicHub:
        def get_agent(self, agent_id: str):
            return None

        def upload_agent_to_marketplace(self, zip_path: Path) -> str:
            return "hub-listing-1"

        def overwrite_agent_in_marketplace(self, agent_id: str, zip_path: Path) -> None:
            raise AssertionError("nothing to overwrite")

    monkeypatch.setattr(radient_credentials, "resolve_radient_oauth_access_sync", explode)
    monkeypatch.setattr(
        radient_credentials, "resolve_radient_credential_sync", lambda *a, **k: SecretStr("k")
    )
    monkeypatch.setattr(
        "local_operator.clients.radient.RadientClient", lambda **kwargs: _PublicHub()
    )

    registry = AgentRegistry(config_dir())
    agent = registry.create_agent(AgentEditFields.model_validate({"name": "PushProbe"}))
    monkeypatch.setattr("sys.argv", ["program", "agents", "push", "--id", agent.id])

    assert main() == 0
    assert calls["org"] == 0
    assert "New agent ID: hub-listing-1" in capsys.readouterr().out


def test_teams_push_with_an_empty_org_names_the_empty_value(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A passed-but-empty value must not read like an omitted flag (R1-1).

    The picker still runs (the empty value cannot name a tenant), but its
    first line says WHAT was wrong rather than implying the flag was missing.
    """
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "", "release-crew"])

    assert main() == 1

    out = capsys.readouterr().out
    assert "the value was empty" in out
    assert "tenant_id: org-a" in out
    assert org_hub.published_teams == []


# --- hub model suggestions (§3.2 push, §4.3 notices) -----------------------------


def test_teams_push_derives_from_the_manager_when_an_agents_store_exists(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The manager row's pair is the derived suggestion, with zero new UI (§3.2)."""

    from local_operator.agents import AgentEditFields, AgentRegistry
    from local_operator.paths import config_dir

    agents = AgentRegistry(config_dir())
    manager_agent = agents.create_agent(
        AgentEditFields.model_validate({"name": "manager", "description": "Runs it."})
    )
    # Every field spelled out (the tree's convention for ``AgentEditFields`` —
    # pyright requires them, no defaults): ``update_agent`` skips ``None``
    # values, so the pair is all this sets.
    agents.update_agent(
        manager_agent.id,
        AgentEditFields(
            name=None,
            security_prompt=None,
            hosting="openrouter",
            model="vendor/model",
            description=None,
            tags=None,
            categories=None,
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
        ),
    )
    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    document, tenant = org_hub.published_teams[0]
    assert tenant == "org-a"
    assert document["model_suggestion"] == {"hosting": "openrouter", "model": "vendor/model"}
    assert "Successfully pushed" in capsys.readouterr().out


def test_teams_push_without_an_agents_store_derives_nothing(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``agents_store_present`` guard: a storeless machine must not create
    one just to derive — and pushes succeed, just with no suggestion."""

    from local_operator.paths import config_dir

    _make_team()
    monkeypatch.setattr("sys.argv", ["program", "teams", "push", "--org", "org-a", "release-crew"])

    assert main() == 0

    document, _tenant = org_hub.published_teams[0]
    assert "model_suggestion" not in document
    # The guard held: no agents store was created by the push.
    assert not (config_dir() / "agents").exists()


def test_agents_pull_org_prints_the_model_suggestion_notice(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A skipped suggestion is a yellow line, never an error (§4.3)."""

    from local_operator.agents import AgentRegistry
    from local_operator.model.suggestion import ModelNotice

    notice = ModelNotice(reason="unknown_provider", requested_hosting="nope", requested_model="m")

    def fake_download(
        self, radient_client, agent_id, *, with_credential=False, auth_store=None, tenant_id=None
    ):
        pulled = types.SimpleNamespace(name="Carrier", id="local-9")
        return types.SimpleNamespace(agent=pulled, renamed_from=None, model_notice=notice)

    monkeypatch.setattr(AgentRegistry, "download_agent_from_radient", fake_download)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", "org-a"]
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert "Successfully pulled agent 'Carrier'" in out
    assert "Model suggestion 'm' (hosting 'nope') was not applied" in out
    assert "Using your default model instead." in out


def test_teams_pull_prints_where_a_stored_suggestion_applies(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stored ⇒ a line saying where it applies, not a silence (§4.3)."""

    monkeypatch.setattr(
        "local_operator.model.discovery.offered_model_ids",
        lambda provider_id, *, cache_dir=None: None,
    )
    org_hub.team_documents["hub-team-2"] = {
        "id": "hub-team-2",
        "tenant_id": "org-a",
        "name": "carrier-crew",
        "members": [],
        "instructions": "You ship.",
        "version": "1.0.0",
        # The mock provider needs no credential, so the CLI's own short-lived
        # store resolves it as usable with no seeding -- this is the plain
        # "the suggestion runs here" arm.
        "model_suggestion": {"hosting": "test", "model": "mock-1"},
    }
    monkeypatch.setattr("sys.argv", ["program", "teams", "pull", "hub-team-2", "--org", "org-a"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "Model suggestion stored" in out
    assert "lop exec --team carrier-crew" in out

    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    stored = TeamRegistry(config_dir()).get_team_by_name("carrier-crew")
    assert stored is not None
    assert stored.model_suggestion is not None
    assert stored.model_suggestion.model == "mock-1"


def test_teams_pull_prints_the_notice_when_the_suggestion_cannot_run(
    org_hub: _FakeOrgHub,
    tmp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unavailable ⇒ the row omits it and the notice line names the reason."""

    org_hub.team_documents["hub-team-3"] = {
        "id": "hub-team-3",
        "tenant_id": "org-a",
        "name": "skipping-crew",
        "members": [],
        "instructions": "You ship.",
        "version": "1.0.0",
        "model_suggestion": {"hosting": "no-such-provider", "model": "m"},
    }
    monkeypatch.setattr("sys.argv", ["program", "teams", "pull", "hub-team-3", "--org", "org-a"])

    assert main() == 0

    out = capsys.readouterr().out
    assert "was not applied" in out
    assert "no-such-provider" in out

    from local_operator.paths import config_dir
    from local_operator.teams import TeamRegistry

    stored = TeamRegistry(config_dir()).get_team_by_name("skipping-crew")
    assert stored is not None
    assert stored.model_suggestion is None
