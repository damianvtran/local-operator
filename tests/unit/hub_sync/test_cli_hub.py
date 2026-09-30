"""The CLI surface: flags, the deprecated --force alias, teams sync/link, hub status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from local_operator import cli
from local_operator.agents import AgentRegistry
from local_operator.hub_sync import provenance as prov
from local_operator.teams import TeamRegistry
from tests.unit.hub_sync.test_service import BASE, Hub, _pull_agent


@pytest.fixture()
def parser() -> argparse.ArgumentParser:
    return cli.build_cli_parser()


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / ".local-operator"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    hub = Hub()
    monkeypatch.setattr(
        "local_operator.agents._fetch_hub_profile", lambda _c, h, **_k: hub.agents[h]
    )
    monkeypatch.setattr(
        "local_operator.hub_sync.service.build_clients_sync",
        lambda _cm: ((lambda _t: hub.client()), "ok"),
    )
    return root, hub, AgentRegistry(root)


def test_new_flags_and_subcommands_parse(parser) -> None:
    a = parser.parse_args(
        [
            "agents",
            "sync",
            "--name",
            "x",
            "--check",
            "--prefer",
            "local",
            "--accept-unknown-baseline",
            "--dry-run",
            "--json",
        ]
    )
    assert (a.check, a.prefer, a.accept_unknown_baseline, a.dry_run, a.json) == (
        True,
        "local",
        True,
        True,
        True,
    )
    t = parser.parse_args(["teams", "sync", "--all", "--replace", "--yes"])
    assert t.teams_command == "sync" and t.replace and t.yes
    link = parser.parse_args(
        ["teams", "link", "release", "hub-id", "--org", "o", "--accept-unknown-baseline"]
    )
    assert (link.name, link.hub_team_id, link.org) == ("release", "hub-id", "o")
    assert parser.parse_args(["hub", "status", "--json"]).hub_command == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["agents", "sync", "--prefer", "both"])


def _run(parser, argv, root, registry):
    args = parser.parse_args(argv)
    return cli.agents_sync_command(args, registry, root)


def test_a_default_sync_merges_and_names_what_it_did(rig, parser, capsys) -> None:
    root, hub, agents = rig
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, BASE.replace("\n\n## Legacy\nOld advice.", ""))
    hub.agents["h1"] = (BASE + "\n\n## New\nAdded upstream.", "d")
    assert _run(parser, ["agents", "sync", "--name", "coder"], root, agents) == 0
    out = capsys.readouterr().out
    assert "coder: updated from the hub" in out and "overwritten" not in out
    text = agents.get_agent_system_prompt(row.id)
    assert "Added upstream" in text and "Legacy" not in text


def test_check_only_reports_and_changes_nothing(rig, parser, capsys) -> None:
    root, hub, agents = rig
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nX.", "d")
    assert _run(parser, ["agents", "sync", "--name", "coder", "--check"], root, agents) == 0
    assert "hub update available" in capsys.readouterr().out
    assert "## New" not in agents.get_agent_system_prompt(row.id)


def test_dry_run_writes_nothing_and_json_is_the_route_shape(rig, parser, capsys) -> None:
    root, hub, agents = rig
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nX.", "d")
    assert (
        _run(parser, ["agents", "sync", "--name", "coder", "--dry-run", "--json"], root, agents)
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["reports"][0]["outcome"] == "would-merge"
    assert "## New" not in agents.get_agent_system_prompt(row.id)


def test_replace_needs_yes_and_force_is_a_deprecated_alias_that_echoes(rig, parser, capsys) -> None:
    root, hub, agents = rig
    row = _pull_agent(agents, hub, root)
    agents.set_agent_system_prompt(row.id, "MY TEXT")
    hub.agents["h1"] = ("HUB TEXT", "d")
    assert _run(parser, ["agents", "sync", "--name", "coder", "--replace"], root, agents) == 1
    assert agents.get_agent_system_prompt(row.id) == "MY TEXT"
    assert _run(parser, ["agents", "sync", "--name", "coder", "--force"], root, agents) == 0
    captured = capsys.readouterr()
    assert "--force is deprecated" in captured.err
    assert "your instructions was" in captured.out or "MY TEXT" in captured.out
    assert agents.get_agent_system_prompt(row.id).strip() == "HUB TEXT"


def test_check_refuses_to_combine_with_write_flags(rig, parser, capsys) -> None:
    root, hub, agents = rig
    assert _run(parser, ["agents", "sync", "--check", "--prefer", "local"], root, agents) == 1


def test_hub_status_prints_the_store_and_json_matches_the_route_payload(
    rig, parser, capsys
) -> None:
    root, hub, agents = rig
    _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nX.", "d")
    _run(parser, ["agents", "sync", "--check"], root, agents)
    capsys.readouterr()
    assert cli.hub_status_command(parser.parse_args(["hub", "status"]), root) == 0
    assert "coder: available" in capsys.readouterr().out
    cli.hub_status_command(parser.parse_args(["hub", "status", "--json"]), root)
    assert set(json.loads(capsys.readouterr().out)) == {
        "generated_at",
        "credential",
        "settings",
        "counts",
        "items",
    }


def test_teams_link_refuses_a_differing_copy_until_accepted_and_then_never_deletes(
    rig, parser, capsys, monkeypatch
) -> None:
    root, hub, _ = rig
    teams = TeamRegistry(root)
    from local_operator.teams import TeamEditFields, TeamMember

    team = teams.create_team(
        TeamEditFields(
            name="release", manager="m", members=[TeamMember(role="coder")], instructions="mine"
        )
    )
    hub.teams["ht"] = {
        "id": "ht",
        "tenant_id": "org",
        "name": "release",
        "description": "",
        "manager": "m",
        "members": [{"role": "coder", "kind": "agent", "count": 1}],
        "instructions": "theirs",
        "project": "",
    }
    monkeypatch.setattr(cli, "_org_target_or_picker", lambda _a, _b: (hub.client(), "org"))
    args = parser.parse_args(["teams", "link", "release", "ht", "--org", "org"])
    assert cli.teams_link_command(args, teams, root) == 1
    assert prov.read_baseline(root, "team", team.id) is None
    args = parser.parse_args(
        ["teams", "link", "release", "ht", "--org", "org", "--accept-unknown-baseline"]
    )
    assert cli.teams_link_command(args, teams, root) == 0
    record = prov.read_baseline(root, "team", team.id)
    assert record and record.recorded_by == prov.UNKNOWN_BASELINE
    # Linked-but-unknown: a sync must ask for the acknowledgement, never delete.
    sync = parser.parse_args(["teams", "sync", "--name", "release"])
    assert cli.teams_sync_command(sync, teams, root) == 0
    assert teams.get_team(team.id).instructions == "mine"
