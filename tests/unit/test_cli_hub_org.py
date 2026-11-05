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
    text = _hub_help()
    # Neither the command row nor the usage metavar may name it.
    assert "org" not in text.split("positional arguments")[-1].split()
    assert "{status,links,link}" in text and ",org}" not in text


def test_help_hides_org_with_no_team_plan(_isolated: Path) -> None:
    _seed(_isolated, [])
    assert "Radient Team" not in _hub_help()
    assert " org " not in _hub_help()


def test_help_lists_org_for_a_team_member(_isolated: Path) -> None:
    _seed(_isolated, MEMBER)
    text = _hub_help()
    assert "Radient Team" in text and "{status,links,link,org}" in text


def test_ineligible_invocation_prints_the_one_sentence(_isolated: Path, capsys) -> None:
    args = cli.build_cli_parser().parse_args(["hub", "org"])
    rc = hub_org_cli.command(args, _isolated)
    err = capsys.readouterr().err
    assert rc == 1
    # Signed out (no record): the cause and the way forward (UX review U5).
    assert err.strip() == (
        "Organization catalogues are a Radient Team feature; you’re not signed in to "
        "Radient (lop login radient)."
    )
    _seed(_isolated, [])  # signed in, no team plan
    assert hub_org_cli.command(args, _isolated) == 1
    assert capsys.readouterr().err.strip() == (
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
    # The public header/footer stay, and the org rows are their own block (D2/Q4).
    assert "showing page 1 of 1" in eligible
    assert "Minerva (organization)" in eligible
    assert "minerva-support — id t2" in eligible
    assert eligible.index("showing page") < eligible.index("Minerva (organization)")

    query = cli.build_cli_parser().parse_args(["teams", "search", "support"])
    assert cli.teams_search_command(query, _isolated) == 0
    searched = capsys.readouterr().out
    assert "No public team matches 'support'." in searched
    assert "minerva-support — id t2" in searched


def test_empty_plan_does_not_ask_for_yes(_isolated: Path, member_hub, capsys) -> None:
    args = cli.build_cli_parser().parse_args(["hub", "org", "pull"])
    assert hub_org_cli.command(args, _isolated) == 0
    out = capsys.readouterr().out
    assert "Everything is already linked" in out and "--yes" not in out


def test_plan_names_what_yes_does_to_a_conflict(_isolated: Path, member_hub, capsys) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("deploys", "x\n")
    (_isolated / "skills" / "deploys").mkdir(parents=True)
    args = cli.build_cli_parser().parse_args(["hub", "org", "pull"])
    assert hub_org_cli.command(args, _isolated) == 0
    assert "conflict → adopt-link" in capsys.readouterr().out


@pytest.mark.parametrize("ref", ["Minerva", "minerva", "Min", TENANT])
def test_org_flag_accepts_a_name_prefix_or_id(_isolated: Path, member_hub, capsys, ref) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("deploys", "x\n")
    args = cli.build_cli_parser().parse_args(["hub", "org", "--org", ref])
    assert hub_org_cli.command(args, _isolated) == 0
    assert f"Minerva ({TENANT})" in capsys.readouterr().out


def test_unknown_org_exits_non_zero_with_the_choices(_isolated: Path, member_hub, capsys) -> None:
    for argv in (["hub", "org", "--org", "Acme"], ["hub", "org", "pull", "--org", "Acme"]):
        assert hub_org_cli.command(cli.build_cli_parser().parse_args(argv), _isolated) == 1
        err = capsys.readouterr().err
        assert "matches “Acme”" in err and TENANT in err


def test_pull_yes_prints_each_item_as_it_lands(
    _isolated: Path, member_hub, monkeypatch, capsys
) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("deploys", "x\n")
    from local_operator.hub_org import pull as pull_mod

    real = pull_mod.pull_all
    seen: dict[str, Any] = {}

    def spy(*a: Any, **k: Any) -> Any:
        seen["progress"] = k.get("progress")
        return real(*a, **k)

    monkeypatch.setattr(pull_mod, "pull_all", spy)
    args = cli.build_cli_parser().parse_args(["hub", "org", "pull", "--yes"])
    assert hub_org_cli.command(args, _isolated) == 0
    assert callable(seen["progress"])
    out = capsys.readouterr().out
    assert out.count("deploys") == 1 and "pulled" in out


def test_adopt_link_via_the_cli_keeps_the_local_bytes(_isolated: Path, member_hub) -> None:
    """U1 through the real CLI path: the default --on-conflict, model allowed."""
    member_hub.skills[SKILL_ID] = _skill_doc("deploys", "Hub body.\n")
    folder = _isolated / "skills" / "deploys"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: deploys\ndescription: mine\n---\n\nMine.\n", encoding="utf-8"
    )
    before = (folder / "SKILL.md").read_bytes()
    args = cli.build_cli_parser().parse_args(["hub", "org", "pull", "--yes", "--json"])
    assert hub_org_cli.command(args, _isolated) == 0
    assert (folder / "SKILL.md").read_bytes() == before


def test_last_sync_time_is_humanized(_isolated: Path, member_hub, capsys) -> None:
    stamp = time.mktime((2026, 11, 4, 9, 30, 0, 0, 0, -1))

    def write(data: dict[str, Any]) -> None:
        data["last_enrol"] = {TENANT: {"at": stamp, "pulled": 2, "adopted": []}}

    eligibility.update_record(_isolated, write)
    args = cli.build_cli_parser().parse_args(["hub", "org"])
    assert hub_org_cli.command(args, _isolated) == 0
    out = capsys.readouterr().out
    assert "Last automatic sync: 2026-11-04 09:30, 2 items pulled" in out


def test_local_column_aligns_ids_and_drops_a_repeated_name(
    _isolated: Path, member_hub, capsys
) -> None:
    member_hub.skills[SKILL_ID] = _skill_doc("a-very-long-skill-name-indeed", "x\n")
    member_hub.skills["55555555-5555-4555-8555-555555555555"] = _skill_doc("b", "y\n")
    (_isolated / "skills" / "a-very-long-skill-name-indeed").mkdir(parents=True)
    args = cli.build_cli_parser().parse_args(["hub", "org"])
    assert hub_org_cli.command(args, _isolated) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if " skill " in line]
    assert len(rows) == 2 and " as " not in "".join(rows)
    assert len({line.rindex(" ") for line in rows}) == 1
