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

    def list_memberships(self) -> list[dict[str, Any]]:
        return self.memberships

    def publish_team_document(self, document: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        self.published_teams.append((document, tenant_id))
        return {
            "team": {"id": "hub-team-1", "name": document["name"], "version": document["version"]}
        }

    def get_team(self, team_id: str) -> dict[str, Any]:
        self.pulled_teams.append(team_id)
        return self.team_documents[team_id]

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

    def fake_download(self, radient_client, agent_id, *, with_credential=False):
        seen["client"] = radient_client
        seen["agent_id"] = agent_id
        seen["with_credential"] = with_credential
        return types.SimpleNamespace(name="Pulled", id="local-1"), None

    monkeypatch.setattr(AgentRegistry, "download_agent_from_radient", fake_download)
    monkeypatch.setattr(
        "sys.argv", ["program", "agents", "pull", "--id", "hub-agent-1", "--org", "org-a"]
    )

    assert main() == 0

    out = capsys.readouterr().out
    assert seen["client"] is org_hub
    assert seen["agent_id"] == "hub-agent-1"
    # The org pull must prove membership at the wire: the bearer travels with
    # the download (the public pull stays anonymous, §11 R-6).
    assert seen["with_credential"] is True
    assert "Successfully pulled agent 'Pulled'" in out


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
    assert "lop login radient" in out


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
