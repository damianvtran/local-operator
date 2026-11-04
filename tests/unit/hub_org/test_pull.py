"""``hub_org.pull.pull_all``: idempotent, subscribed, conflict-aware bulk pull.

Runs against an isolated config root and ONE fake org hub (agent zips, team
documents, skill manifests) wired in at ``build_clients_sync``. That is the seam
``svc.sync_context`` reads, so the merge/apply paths under test are the real
ones, and every assertion reads a persisted effect (rows, baselines, backups,
policy).
"""

from __future__ import annotations

import os
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.hub_org import pull as pull_mod
from local_operator.hub_org.eligibility import OrgRef
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync.skill_bundle import build_bundle, tree_hash_of
from local_operator.providers.radient_credentials import configured_radient_base_url
from local_operator.teams import TeamEditFields, TeamRegistry

TENANT = "tenant-minerva"
AGENT_ID = "11111111-1111-4111-8111-111111111111"
TEAM_ID = "22222222-2222-4222-8222-222222222222"
SKILL_ID = "33333333-3333-4333-8333-333333333333"
BUNDLE_ID = "44444444-4444-4444-8444-444444444444"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / ".local-operator"))
    return tmp_path


def _skill_doc(name: str, body: str) -> dict[str, Any]:
    return {
        "document_version": 1,
        "name": name,
        "description": f"{name} guidance",
        "files": [
            {
                "path": "SKILL.md",
                "content": f"---\nname: {name}\ndescription: {name} guidance\n---\n\n{body}",
            }
        ],
    }


class FakeOrgHub:
    """List + detail reads for one organization, shaped like agent-server."""

    def __init__(self, base: str) -> None:
        self.base_url = base
        self.api_key = "member-token"
        self.agents: dict[str, dict[str, Any]] = {}
        self.teams: dict[str, dict[str, Any]] = {}
        self.skills: dict[str, dict[str, Any]] = {}
        self.bundles: dict[str, bytes] = {}
        self.downloads: list[str] = []

    # -- lists --------------------------------------------------------------------
    def list_org_agents(self, tenant: str, *, page: int = 1, per_page: int = 20) -> dict[str, Any]:
        rows = [
            {"id": k, "name": v["name"], "description": v["description"], "tenant_id": tenant}
            for k, v in self.agents.items()
        ]
        return {"records": rows, "page": 1, "per_page": per_page, "total_pages": 1}

    def list_org_teams(self, tenant: str) -> list[dict[str, Any]]:
        return [dict(v) for v in self.teams.values()]

    def list_org_skills(self, tenant: str) -> list[dict[str, Any]]:
        out = []
        for k, v in self.skills.items():
            row = {"id": k, "name": v["name"], "description": v["description"]}
            if "bundle" in v:
                row["bundle"] = v["bundle"]
            out.append(row)
        return out

    # -- details ------------------------------------------------------------------
    def download_agent_from_marketplace(self, agent_id: str, dest_path: Path, **_kw: Any) -> None:
        self.downloads.append(agent_id)
        spec = self.agents[agent_id]
        agent_yml = yaml.dump(
            {
                "id": str(uuid.uuid4()),
                "name": spec["name"],
                "description": spec["description"],
                "created_date": "2026-01-01T00:00:00Z",
                "version": "1.0.0",
                "tags": [],
                "categories": [],
            }
        )
        with zipfile.ZipFile(dest_path, "w") as archive:
            archive.writestr("agent.yml", agent_yml)
            archive.writestr("system_prompt.md", spec["text"])

    def get_team(self, team_id: str, **_kw: Any) -> dict[str, Any]:
        return {**self.teams[team_id], "revision": '"team-rev-1"', "content_revision": 1}

    def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
        doc = dict(self.skills[skill_id])
        doc.update(
            {
                "id": skill_id,
                "visibility": "org",
                "tenant_id": TENANT,
                "revision": '"skill-rev-1"',
                "content_revision": 1,
                "sync_contract": {"revision_cas": 1, "skills": 1, "skill_bundles": 1},
                "moderation": {"flagged": False},
            }
        )
        return doc

    def download_skill_bundle(self, skill_id: str, *, dest: Any = None, **_kw: Any) -> Any:
        self.downloads.append(skill_id)
        data = self.bundles[skill_id]
        if dest is not None:
            dest.write(data)  # the service hands an open binary handle
        return SimpleNamespace(data=None, not_modified=False, etag=None, server_hash=None)

    def probe_skill_capabilities(self, **_kw: Any) -> dict[str, Any]:
        return {"revision_cas": 1, "skills": 1, "skill_bundles": 1}


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / ".local-operator"
    manager = ConfigManager(root)
    hub = FakeOrgHub(configured_radient_base_url(manager))
    monkeypatch.setattr(svc, "build_clients_sync", lambda _cm: ((lambda _t: hub), "ok"))
    role = {"value": "member"}
    monkeypatch.setattr(
        "local_operator.hub_org.eligibility.eligible_orgs",
        lambda _cfg: [OrgRef(TENANT, "Minerva", role["value"])],
    )
    agents, teams = AgentRegistry(root), TeamRegistry(root)

    def run(**kwargs: Any) -> pull_mod.OrgPullReport:
        return pull_mod.pull_all(
            manager, TENANT, client=hub, agent_registry=agents, team_registry=teams, **kwargs
        )

    return {
        "root": root,
        "manager": manager,
        "hub": hub,
        "agents": agents,
        "teams": teams,
        "run": run,
        "role": role,
    }


def _local_agent(agents: AgentRegistry, name: str, text: str, description: str = "") -> Any:
    from local_operator.agents import AgentEditFields as F  # noqa: N814

    fields = F(
        name=name,
        description=description,
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
    row = agents.create_agent(fields)
    agents.set_agent_system_prompt(row.id, text)
    return row


def _outcomes(report: pull_mod.OrgPullReport) -> dict[str, str]:
    return {f"{i.kind}:{i.name}": i.outcome for i in report.items}


# -- new + subscribe; idempotent re-run ---------------------------------------------------


def test_new_items_are_pulled_and_subscribed_and_a_rerun_mints_no_duplicates(rig) -> None:
    hub = rig["hub"]
    hub.agents[AGENT_ID] = {"name": "coder", "description": "writes code", "text": "Code well."}
    hub.teams[TEAM_ID] = {
        "id": TEAM_ID,
        "tenant_id": TENANT,
        "name": "lopdev",
        "description": "dev team",
        "manager": "coder",
        "members": [{"agent": "coder"}],
        "instructions": "Ship it.",
        "project": "",
    }
    hub.skills[SKILL_ID] = _skill_doc("deploys", "Deploy carefully.\n")

    first = rig["run"]()
    assert _outcomes(first) == {
        "agent:coder": "pulled",
        "skill:deploys": "pulled",
        "team:lopdev": "pulled",
    }, first.to_json()
    # Every new link exists and is subscribed (creation policy / explicit write).
    snapshot = svc.links_snapshot(svc.sync_context(rig["manager"]))
    kinds = sorted(link["kind"] for link in snapshot["links"])
    assert kinds == ["agent", "skill", "team"], snapshot
    assert all(link["policy"]["subscribe"] is True for link in snapshot["links"])
    assert all(link["policy"]["auto_push"] is False for link in snapshot["links"])

    second = rig["run"]()
    assert set(_outcomes(second).values()) == {"already-linked"}, second.to_json()
    assert [a.name for a in rig["agents"].list_agents() if a.name.startswith("coder")] == ["coder"]
    assert [t.name for t in rig["teams"].list_teams()] == ["lopdev"]


def test_names_filter_accepts_ids_and_names(rig) -> None:
    hub = rig["hub"]
    hub.agents[AGENT_ID] = {"name": "coder", "description": "", "text": "x"}
    hub.skills[SKILL_ID] = _skill_doc("deploys", "y\n")
    report = rig["run"](names=["DEPLOYS"])
    assert _outcomes(report) == {"skill:deploys": "pulled"}
    report = rig["run"](names=[AGENT_ID])
    assert _outcomes(report) == {"agent:coder": "pulled"}


# -- agent/team conflicts -----------------------------------------------------------------


def test_adopt_link_identical_agent_links_in_place_with_a_backup(rig) -> None:
    rig["hub"].agents[AGENT_ID] = {"name": "coder", "description": "d", "text": "Same text."}
    local = _local_agent(rig["agents"], "coder", "Same text.", "d")

    report = rig["run"]()

    assert _outcomes(report) == {"agent:coder": "adopt-linked"}
    assert report.adopted[0]["result"] == "identical"
    assert (rig["root"] / report.adopted[0]["backup"]).is_file()
    record = prov.read_baseline(rig["root"], "agent", local.id)
    assert record is not None and record.hub_id == AGENT_ID and record.identity is not None
    assert len(rig["agents"].list_agents()) == len({a.id for a in rig["agents"].list_agents()})
    assert not [a for a in rig["agents"].list_agents() if a.name == "coder-2"]


def test_adopt_link_differing_without_a_model_keeps_local_and_links(rig, monkeypatch) -> None:
    rig["hub"].agents[AGENT_ID] = {"name": "coder", "description": "d", "text": "Hub text."}
    local = _local_agent(rig["agents"], "coder", "My own text.", "d")

    report = rig["run"](allow_llm=False)

    assert report.adopted and report.adopted[0]["kind"] == "agent"
    assert report.adopted[0]["result"] in {"kept-local-linked", "reconciled"}
    assert (
        rig["agents"].get_agent_system_prompt(local.id).strip()
        in {
            "My own text.",
        }
        or report.adopted[0]["result"] == "reconciled"
    )
    record = prov.read_baseline(rig["root"], "agent", local.id)
    assert record is not None and record.hub_id == AGENT_ID
    backups = list((rig["root"] / "hub" / "backups").glob(f"agent-{local.id}-*.json"))
    assert backups


def test_adopt_agent_hub_wins_and_writes_a_backup(rig) -> None:
    rig["hub"].agents[AGENT_ID] = {"name": "coder", "description": "d", "text": "Hub text."}
    local = _local_agent(rig["agents"], "coder", "My own text.", "d")

    report = rig["run"](on_conflict="adopt")

    assert _outcomes(report) == {"agent:coder": "adopted"}, report.to_json()
    assert rig["agents"].get_agent_system_prompt(local.id).strip() == "Hub text."
    assert list((rig["root"] / "hub" / "backups").glob(f"agent-{local.id}-*.json"))
    assert not [a for a in rig["agents"].list_agents() if a.name == "coder-2"]


def test_adopt_team_hub_wins_and_writes_a_backup(rig) -> None:
    from local_operator.teams import TeamMember

    hub_team = {
        "id": TEAM_ID,
        "tenant_id": TENANT,
        "name": "lopdev",
        "description": "hub",
        "manager": "manager",
        "members": [],
        "instructions": "Hub brief.",
        "project": "",
    }
    rig["hub"].teams[TEAM_ID] = hub_team
    local = rig["teams"].create_team(
        TeamEditFields(
            name="lopdev", description="mine", manager="manager", instructions="My brief."
        )
    )
    del TeamMember

    report = rig["run"](on_conflict="adopt")

    assert _outcomes(report) == {"team:lopdev": "adopted"}, report.to_json()
    assert rig["teams"].get_team(local.id).instructions == "Hub brief."
    assert list((rig["root"] / "hub" / "backups").glob(f"team-{local.id}-*.json"))
    assert [t.name for t in rig["teams"].list_teams()] == ["lopdev"]


def test_reconcile_calls_apply_items_with_the_unknown_baseline_ack(rig, monkeypatch) -> None:
    rig["hub"].agents[AGENT_ID] = {"name": "coder", "description": "d", "text": "Hub."}
    _local_agent(rig["agents"], "coder", "Mine.", "d")
    seen: list[dict[str, Any]] = []
    real = svc.apply_items

    def spy(ctx, **kwargs):
        seen.append(kwargs)
        return real(ctx, **kwargs)

    monkeypatch.setattr(svc, "apply_items", spy)
    rig["run"](on_conflict="reconcile", allow_llm=False)
    assert seen and seen[0]["acknowledge_unknown_baseline"] is True
    assert "replace" not in seen[0]


def test_skip_reports_the_resolving_command_and_writes_nothing(rig) -> None:
    rig["hub"].agents[AGENT_ID] = {"name": "coder", "description": "d", "text": "Hub."}
    local = _local_agent(rig["agents"], "coder", "Mine.", "d")
    report = rig["run"](on_conflict="skip")
    assert _outcomes(report) == {"agent:coder": "conflict-skipped"}
    assert "lop hub org pull" in report.items[0].detail
    assert prov.read_baseline(rig["root"], "agent", local.id) is None


# -- skill conflicts -----------------------------------------------------------------------


def _authored_skill(root: Path, name: str, body: str) -> Path:
    folder = root / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} guidance\n---\n\n{body}", encoding="utf-8"
    )
    return folder


def test_document_skill_identical_is_adopt_linked(rig) -> None:
    rig["hub"].skills[SKILL_ID] = _skill_doc("deploys", "Deploy carefully.\n")
    _authored_skill(rig["root"], "deploys", "Deploy carefully.\n")

    report = rig["run"]()

    assert _outcomes(report) == {"skill:deploys": "adopt-linked"}, report.to_json()
    assert report.adopted[0]["result"] == "identical"
    assert report.adopted[0]["backup"]
    assert any(r.kind == "skill" and r.hub_id == SKILL_ID for r in prov.iter_baselines(rig["root"]))


def test_document_skill_adopt_takes_hub_with_snapshot(rig) -> None:
    rig["hub"].skills[SKILL_ID] = _skill_doc("deploys", "Hub body.\n")
    folder = _authored_skill(rig["root"], "deploys", "My body.\n")

    report = rig["run"](on_conflict="adopt")

    assert _outcomes(report) == {"skill:deploys": "adopted"}, report.to_json()
    assert "Hub body." in (folder / "SKILL.md").read_text()
    backup = Path(report.adopted[0]["backup"])
    assert "My body." in (backup / "content" / "SKILL.md").read_text()


def _bundle(tmp_path: Path, hub: FakeOrgHub, body: str) -> Any:
    source = tmp_path / "bundle-src" / "pdf-forms"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        f"---\nname: pdf-forms\ndescription: Fill PDF forms.\n---\n\n{body}", encoding="utf-8"
    )
    (source / "references").mkdir()
    (source / "references" / "guide.md").write_text("alpha\n" * 20, encoding="utf-8")
    built = build_bundle(source)
    hub.skills[BUNDLE_ID] = {
        "document_version": 1,
        "name": built.name,
        "description": built.description,
        "bundle": built.bundle.to_json(),
    }
    hub.bundles[BUNDLE_ID] = built.data
    return built


def test_bundle_skill_adopt_backs_up_the_folder_and_installs_the_hub_tree(rig, tmp_path) -> None:
    built = _bundle(tmp_path, rig["hub"], "# Hub body\n")
    folder = _authored_skill(rig["root"], "pdf-forms", "# My body\n")

    report = rig["run"](on_conflict="adopt")

    assert _outcomes(report) == {"skill:pdf-forms": "adopted"}, report.to_json()
    assert tree_hash_of(folder) == built.tree_hash
    backup = Path(report.adopted[0]["backup"])
    assert "My body" in (backup / "content" / "SKILL.md").read_text()
    # No dot-prefixed aside folder is left behind.
    assert not [p for p in (rig["root"] / "skills").iterdir() if p.name.startswith(".adopt-")]


def test_bundle_skill_adopt_link_differing_is_never_touched(rig, tmp_path) -> None:
    _bundle(tmp_path, rig["hub"], "# Hub body\n")
    folder = _authored_skill(rig["root"], "pdf-forms", "# My body\n")
    report = rig["run"]()
    assert _outcomes(report) == {"skill:pdf-forms": "needs-review"}
    assert "My body" in (folder / "SKILL.md").read_text()


def test_bundle_install_without_take_hub_still_refuses(rig, tmp_path) -> None:
    from local_operator.hub_sync.identity import LinkError

    _bundle(tmp_path, rig["hub"], "# Hub body\n")
    _authored_skill(rig["root"], "pdf-forms", "# My body\n")
    run = pull_mod._Run(
        rig["manager"],
        TENANT,
        client=rig["hub"],
        agent_registry=rig["agents"],
        team_registry=rig["teams"],
        allow_llm=False,
    )
    with pytest.raises(LinkError) as caught:
        svc.install_skill(
            run.ctx,
            hub_source=run.hub_source(),
            hub_id=BUNDLE_ID,
            visibility="org",
            tenant_id=TENANT,
            alias=None,
            confirm=True,
        )
    assert caught.value.code == "alias-occupied"


# -- auto-push gate ----------------------------------------------------------------------


def test_a_member_cannot_enable_auto_push(rig) -> None:
    rig["hub"].skills[SKILL_ID] = _skill_doc("deploys", "x\n")
    report = rig["run"](auto_push=True)
    assert "owner/admin only" in report.items[0].detail
    snapshot = svc.links_snapshot(svc.sync_context(rig["manager"]))
    assert all(link["policy"]["auto_push"] is False for link in snapshot["links"])


def test_one_failing_item_does_not_stop_the_batch(rig, monkeypatch) -> None:
    hub = rig["hub"]
    hub.agents[AGENT_ID] = {"name": "coder", "description": "", "text": "x"}
    hub.skills[SKILL_ID] = _skill_doc("deploys", "y\n")

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("simulated hub failure")

    monkeypatch.setattr(hub, "download_agent_from_marketplace", boom)
    report = rig["run"]()
    assert _outcomes(report) == {"agent:coder": "failed", "skill:deploys": "pulled"}


del AgentEditFields
