"""The CLI surface: flags, the deprecated --force alias, teams sync/link, hub status."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

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


def test_hub_status_survives_an_unreadable_credential_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, caplog
) -> None:
    """A corrupt credential store must not turn a read-only status read into a crash.

    Before Q-4 the command never opened ``auth.db``: resolving the credential so
    ``login ok|none`` is a checked fact rather than a dataclass default added
    this failure mode (review round 2, M-8). A store that cannot be opened means
    no usable credential, so the honest degradation is ``login none`` — not a
    traceback, and not the pre-fix unchecked "ok". This runs the REAL resolver
    (no ``build_clients_sync`` stub) against a genuinely unreadable file, and it
    asserts the degradation is LOGGED, so a future silent failure is not the
    only evidence it happened (review round 3, N-5).
    """

    root = tmp_path / ".local-operator"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    root.mkdir(parents=True, exist_ok=True)
    (root / "auth.db").write_bytes(b"not a sqlite database!")

    with caplog.at_level(logging.WARNING):
        rc = cli.hub_status_command(cli.build_cli_parser().parse_args(["hub", "status"]), root)

    assert rc == 0
    assert "login none" in capsys.readouterr().out
    assert any("credential store could not be read" in r.message for r in caplog.records)


def test_hub_status_does_not_swallow_an_unexpected_resolution_bug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corrupt-store fallback is scoped to store-open failures, not to everything.

    An unconditional ``except Exception`` around the credential read would make a
    genuine bug inside ``sync_context`` present forever as the checked fact
    "login none" — the same instrument-returns-a-reading shape the Q-4 fix
    removed. Only the store-open class degrades; anything else is left to raise
    (review round 3, N-5).
    """

    from local_operator.hub_sync import service as svc

    def boom(_cm: object) -> object:
        raise ValueError("a genuine bug, not an unreadable store")

    monkeypatch.setattr(svc, "sync_context", boom)
    root = tmp_path / ".local-operator"
    root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        cli.hub_status_command(cli.build_cli_parser().parse_args(["hub", "status"]), root)


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


def test_hub_links_and_link_set_parse(parser) -> None:
    links = parser.parse_args(["hub", "links", "--kind", "agent", "--json"])
    assert links.hub_command == "links" and links.kind == "agent" and links.json

    setp = parser.parse_args(
        ["hub", "link", "set", "abc", "--subscribe", "false", "--auto-push", "true"]
    )
    assert setp.hub_command == "link" and setp.hub_link_command == "set"
    assert setp.link_id == "abc" and setp.subscribe == "false" and setp.auto_push == "true"


def test_hub_links_command_lists_a_local_link(rig, capsys) -> None:
    root, hub, agents = rig
    _pull_agent(agents, hub, root)

    rc = cli.hub_links_command(argparse.Namespace(kind=None, json=True), root)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capability"] == "hub_links:1"
    # A legacy link is under `unresolved` until its source is confirmed.
    assert payload["links"] == [] and len(payload["unresolved"]) == 1
    assert payload["unresolved"][0]["hub_id"] == "h1"


def test_hub_link_set_edits_subscribe_and_rejects_a_stale_revision(rig, capsys) -> None:
    root, hub, agents = rig
    _pull_agent(agents, hub, root)

    # Resolve the legacy link to a confirmed identity so it has a policy revision.
    from local_operator.config import ConfigManager
    from local_operator.hub_sync import service as svc

    cm = ConfigManager(root)
    ctx = svc.HubSyncContext(
        config_dir=root, config_manager=cm, client_for_tenant=lambda _t: None, credential="ok"
    )
    record = svc.resolve_source(
        ctx,
        kind="agent",
        name=agents.list_agents()[0].name,
        hub_source="https://api.radienthq.com/v1",
        visibility="public",
        tenant_id=None,
    )
    link_id = record.identity.link_id

    rc = cli.hub_link_set_command(
        argparse.Namespace(
            link_id=link_id,
            subscribe="false",
            auto_push=None,
            expected_revision=None,
            confirm_public_egress=False,
            json=True,
        ),
        root,
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["policy"]["subscribe"] is False and out["policy"]["auto_push"] is False

    # A WRONG explicit revision is refused (CAS), not applied.
    bad = cli.hub_link_set_command(
        argparse.Namespace(
            link_id=link_id,
            subscribe="true",
            auto_push=None,
            expected_revision="not-it",
            confirm_public_egress=False,
            json=True,
        ),
        root,
    )
    assert bad == 1
    assert "error:" in capsys.readouterr().out


def _seed_public_link(root: Path, *, lineage: str = "clean", recorded_by: str = "pull") -> str:
    """A CONFIRMED public link the way a real pull + capability probe leaves it.

    A baseline record with a source-scoped remote revision and known scope, plus
    the persisted revision-CAS attestation — the facts ``_eligibility`` reads.
    ``lineage``/``recorded_by`` are varied to exercise the earlier gates.
    """

    import dataclasses

    from local_operator.config import ConfigManager
    from local_operator.hub_sync import provenance as prov
    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.identity import LinkIdentity

    source = "https://api.radienthq.com/v1"
    identity = LinkIdentity("agent", "synthetic-local", source, "public", None, "h1")
    record = prov.make_record(
        "agent", "synthetic-local", "h1", None, {"instructions": "x"}, recorded_by
    )
    record = dataclasses.replace(
        record,
        identity=identity,
        remote_revision='"rev-1"',
        lineage=lineage,
        scope_known=True,
        local_alias="coder",
    )
    prov.write_baseline(root, record)
    cm = ConfigManager(root)
    cm.set_config_value(
        "hub", {"links": {identity.link_id: {"subscribe": True, "auto_push": False}}}
    )
    if recorded_by != "adopt-unknown":
        svc.record_server_capabilities(
            svc.HubSyncContext(
                config_dir=root, config_manager=cm, client_for_tenant=lambda _t: None
            ),
            {"revision_cas": True},
        )
    return identity.link_id


def _set(root: Path, link_id: str, **over: Any) -> int:
    base: dict[str, Any] = dict(
        link_id=link_id,
        subscribe=None,
        auto_push=None,
        expected_revision=None,
        confirm_public_egress=False,
        json=True,
    )
    base.update(over)
    # The parser hands the command STRINGS ("true"/"false"); mirror that here.
    for key in ("subscribe", "auto_push"):
        if isinstance(base[key], bool):
            base[key] = "true" if base[key] else "false"
    return cli.hub_link_set_command(argparse.Namespace(**base), root)


def _grant_written(root: Path, link_id: str) -> bool:
    from local_operator.config import ConfigManager

    hub = ConfigManager(root).get_config().values.get("hub", {})
    return link_id in (hub.get("public_egress_grants") or {})


def test_cli_can_mint_the_first_public_grant_with_confirm(rig, capsys) -> None:
    """The cross-surface fix: the REAL CLI can create a public link's first grant.

    ``can_enable_auto_push`` is False for a public origin with no grant (reason
    ``unsafe-publication``); the carve-out inside ``update_policy`` mints the
    grant only when ``--confirm-public-egress`` is given.
    """

    root, _hub, _agents = rig
    link_id = _seed_public_link(root)
    # WITHOUT consent: refused, naming the missing public consent, NO grant.
    assert _set(root, link_id, auto_push=True) == 1
    assert "public publication consent is required" in capsys.readouterr().out
    assert not _grant_written(root, link_id)
    # WITH consent: rc 0 and the grant persisted.
    assert _set(root, link_id, auto_push=True, confirm_public_egress=True) == 0
    assert _grant_written(root, link_id)


def test_cli_refuses_when_an_earlier_gate_fails_even_with_consent(rig, capsys) -> None:
    """No CAS attestation ⇒ refused ``unsupported-precondition``, NO grant (consent and all)."""

    root, _hub, _agents = rig
    link_id = _seed_public_link(root, recorded_by="adopt-unknown")
    assert _set(root, link_id, auto_push=True, confirm_public_egress=True) == 1
    assert "not eligible for this origin" in capsys.readouterr().out
    assert not _grant_written(root, link_id)
