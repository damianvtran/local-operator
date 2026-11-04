"""``lop hub org``: hidden unless eligible, the one sentence otherwise, and the real flows.

The gate is seeded the way slice B's refresh writes it (``update_record``), so
these tests read the same record shape production does. No network: the
client comes from a monkeypatched resolver.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator import cli
from local_operator.config import ConfigManager
from local_operator.hub_org import cli as hub_org_cli
from local_operator.hub_org import eligibility
from local_operator.providers.radient_credentials import configured_radient_base_url

from .hub_org.test_pull import SKILL_ID, TENANT, FakeOrgHub, _skill_doc


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    # A closed gate stays closed: the forced refresh must not reach a network.
    monkeypatch.setattr(eligibility, "refresh", lambda *a, **k: [])
    return root


def _seed(root: Path, orgs: list[dict[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = configured_radient_base_url(ConfigManager(root))

    def write(data: dict[str, Any]) -> None:
        data.update(base_url=base, checked_at=time.time(), orgs=orgs, account_id="acct")

    eligibility.update_record(root, write)


MEMBER = [{"tenant_id": TENANT, "tenant_name": "Minerva", "role": "member"}]


def _hub_help() -> str:
    parser = cli.build_cli_parser()
    sub = next(a for a in parser._actions if a.dest == "subcommand")
    choices: Any = sub.choices
    return choices["hub"].format_help()


def test_help_hides_org_when_signed_out(_isolated: Path) -> None:
    assert "org" not in _hub_help().split("positional arguments")[-1].split()


def test_help_hides_org_with_no_team_plan(_isolated: Path) -> None:
    _seed(_isolated, [])
    assert "Radient Team" not in _hub_help()
    assert " org " not in _hub_help()


def test_help_lists_org_for_a_team_member(_isolated: Path) -> None:
    _seed(_isolated, MEMBER)
    assert "Radient Team" in _hub_help()


def test_ineligible_invocation_prints_the_one_sentence(_isolated: Path, capsys) -> None:
    args = cli.build_cli_parser().parse_args(["hub", "org"])
    rc = hub_org_cli.command(args, _isolated)
    err = capsys.readouterr().err
    assert rc == 1
    assert err.strip() == (
        "Organization catalogues are a Radient Team feature; this account isn’t in a "
        "team-plan organization."
    )


@pytest.fixture
def member_hub(_isolated: Path, monkeypatch: pytest.MonkeyPatch) -> FakeOrgHub:
    _seed(_isolated, MEMBER)
    hub = FakeOrgHub(configured_radient_base_url(ConfigManager(_isolated)))
    monkeypatch.setattr(cli, "_resolve_org_client", lambda _base: hub)
    monkeypatch.setattr(
        "local_operator.hub_sync.service.build_clients_sync",
        lambda _m: ((lambda _t: hub), "ok"),
    )
    return hub


def test_list_marks_unlinked_local_copies(_isolated: Path, member_hub, capsys) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("minerva-platform-deployments", "x\n")
    (_isolated / "skills" / "minerva-platform-deployments").mkdir(parents=True)
    args = cli.build_cli_parser().parse_args(["hub", "org", "list"])
    assert hub_org_cli.command(args, _isolated) == 0
    out = capsys.readouterr().out
    assert "minerva-platform-deployments" in out and "conflict" in out
    assert "unlinked local copies: minerva-platform-deployments" in out


def test_pull_without_yes_prints_the_plan_and_writes_nothing(
    _isolated: Path, member_hub, capsys
) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("deploys", "x\n")
    args = cli.build_cli_parser().parse_args(["hub", "org", "pull"])
    assert hub_org_cli.command(args, _isolated) == 0
    assert "re-run with --yes" in capsys.readouterr().out
    assert not (_isolated / "skills" / "deploys").exists()


def test_pull_yes_by_name_prefix_resolves_and_installs(_isolated: Path, member_hub, capsys) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("minerva-platform-deployments", "x\n")
    args = cli.build_cli_parser().parse_args(
        ["hub", "org", "pull", "--name", "minerva-plat", "--yes"]
    )
    assert hub_org_cli.command(args, _isolated) == 0
    captured = capsys.readouterr()
    assert f"resolved “minerva-plat” → {SKILL_ID}" in captured.err
    assert "pulled" in captured.out
    assert (_isolated / "skills" / "minerva-platform-deployments" / "SKILL.md").is_file()


def test_ambiguous_prefix_refuses_with_candidates(_isolated: Path, member_hub, capsys) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("minerva-a", "x\n")
    member_hub.skills["55555555-5555-4555-8555-555555555555"] = _skill_doc("minerva-b", "y\n")
    args = cli.build_cli_parser().parse_args(["hub", "org", "pull", "--name", "minerva", "--yes"])
    assert hub_org_cli.command(args, _isolated) == 1
    err = capsys.readouterr().err
    assert "matches several items" in err and "minerva-a" in err and "minerva-b" in err


def test_skills_pull_org_accepts_a_name(_isolated: Path, member_hub, capsys) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("minerva-platform-deployments", "x\n")
    args = cli.build_cli_parser().parse_args(
        ["skills", "pull", "minerva-platform-deployments", "--org", TENANT, "--yes"]
    )
    assert cli.skills_command(args, _isolated) == 0
    assert f"→ {SKILL_ID}" in capsys.readouterr().err
    assert (_isolated / "skills" / "minerva-platform-deployments").is_dir()


def test_skills_pull_yes_chains_document_adoption(_isolated: Path, member_hub, capsys) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("deploys", "Hub body.\n")
    folder = _isolated / "skills" / "deploys"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: deploys\ndescription: deploys guidance\n---\n\nMine.\n", encoding="utf-8"
    )
    args = cli.build_cli_parser().parse_args(
        ["skills", "pull", SKILL_ID, "--org", TENANT, "--yes", "--strategy", "take-hub", "--json"]
    )
    assert cli.skills_command(args, _isolated) == 0
    assert "Hub body." in (folder / "SKILL.md").read_text()
    assert list((_isolated / "hub" / "backups").glob("skill-adopt-deploys-*"))


def test_teams_search_public_output_is_unchanged_when_ineligible(
    _isolated: Path, monkeypatch, capsys
) -> None:
    from local_operator.clients import radient

    rows = [{"id": "t1", "name": "lopdev", "version": "1", "description": "d", "manager": "m"}]
    monkeypatch.setattr(
        radient.RadientClient,
        "list_public_teams",
        lambda self, **_k: {"records": rows, "page": 1, "total_pages": 1, "total_records": 1},
    )
    args = cli.build_cli_parser().parse_args(["teams", "search"])
    assert cli.teams_search_command(args, _isolated) == 0
    ineligible = capsys.readouterr().out
    assert "[org:" not in ineligible

    _seed(_isolated, MEMBER)
    hub = FakeOrgHub("x")
    hub.teams["t2"] = {"id": "t2", "name": "minerva-support", "description": "support"}
    monkeypatch.setattr(
        "local_operator.hub_org.client.build_org_client",
        lambda _m: type("B", (), {"client": hub})(),
    )
    assert cli.teams_search_command(args, _isolated) == 0
    eligible = capsys.readouterr().out
    assert eligible.startswith(ineligible.rstrip("\n").split("showing page")[0])
    assert "[org: Minerva] minerva-support — id t2" in eligible
