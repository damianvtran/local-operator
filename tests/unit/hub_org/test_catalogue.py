"""``hub_org.catalogue``: the per-item local column and name/prefix resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.hub_org import catalogue as cat
from local_operator.hub_sync import provenance as prov

from .test_pull import (  # noqa: F401
    AGENT_ID,
    SKILL_ID,
    TEAM_ID,
    TENANT,
    FakeOrgHub,
    _skill_doc,
    rig,
)


def test_local_column_linked_conflict_absent(rig) -> None:  # noqa: F811
    hub: FakeOrgHub = rig["hub"]
    hub.agents[AGENT_ID] = {"name": "coder", "description": "", "text": "x"}
    hub.skills[SKILL_ID] = _skill_doc("deploys", "x\n")
    hub.teams[TEAM_ID] = {"id": TEAM_ID, "name": "lopdev", "description": "d"}
    (rig["root"] / "skills" / "Deploys").mkdir(parents=True)
    prov.write_baseline(
        rig["root"], prov.make_record("team", "local-team", TEAM_ID, TENANT, {}, "pull")
    )

    items = cat.list_catalogue(
        hub,
        TENANT,
        config_dir=rig["root"],
        agent_registry=rig["agents"],
        team_registry=rig["teams"],
    )

    by = {(i.kind, i.name): i for i in items}
    assert by[("agent", "coder")].local == "absent"
    # Casefolded: the registries' own collision rule.
    assert by[("skill", "deploys")].local == "conflict"
    assert by[("skill", "deploys")].local_name == "Deploys"
    assert by[("skill", "deploys")].form == "document"
    assert by[("team", "lopdev")].local == "linked"
    # Agents first, then skills, then teams: rosters name agents.
    assert [i.kind for i in items] == ["agent", "skill", "team"]


def test_one_kind_failing_is_reported_without_hiding_the_rest(  # noqa: F811
    rig, monkeypatch  # noqa: F811
) -> None:
    hub: FakeOrgHub = rig["hub"]
    hub.skills[SKILL_ID] = _skill_doc("deploys", "x\n")

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("403 team_plan_required")

    monkeypatch.setattr(hub, "list_org_teams", boom)
    result = cat.catalogue_result(
        hub,
        TENANT,
        config_dir=rig["root"],
        agent_registry=rig["agents"],
        team_registry=rig["teams"],
    )
    assert [i.name for i in result.items] == ["deploys"]
    assert "team_plan_required" in result.errors["team"]


def _item(name: str, hub_id: str, kind: Any = "skill") -> cat.CatalogueItem:
    return cat.CatalogueItem(kind, hub_id, name, "", "", "absent", None)


ITEMS = [
    _item("minerva-platform-deployments", "aaaaaaaa-0000-4000-8000-000000000001"),
    _item("minerva-support-workspace", "aaaaaaaa-0000-4000-8000-000000000002"),
    _item("minerva-credentials", "aaaaaaaa-0000-4000-8000-000000000003"),
    _item("coder", "aaaaaaaa-0000-4000-8000-000000000004", "agent"),
]


def test_resolve_by_id_exact_name_and_unique_prefix() -> None:
    assert cat.resolve_reference(ITEMS, "aaaaaaaa-0000-4000-8000-000000000003").name == (
        "minerva-credentials"
    )
    assert cat.resolve_reference(ITEMS, "Minerva-Platform-Deployments").hub_id.endswith("1")
    assert cat.resolve_reference(ITEMS, "minerva-sup").name == "minerva-support-workspace"


def test_an_ambiguous_prefix_refuses_with_the_candidates() -> None:
    with pytest.raises(cat.ReferenceError_) as caught:
        cat.resolve_reference(ITEMS, "minerva-", kind="skill")
    assert len(caught.value.candidates) == 3


def test_no_match_and_kind_filter() -> None:
    with pytest.raises(cat.ReferenceError_) as caught:
        cat.resolve_reference(ITEMS, "coder", kind="skill")
    assert caught.value.candidates == ()


def test_list_org_skills_reads_the_unpaged_envelope(monkeypatch) -> None:
    from local_operator.clients import radient

    seen: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return {"skills": [{"id": "s1", "name": "deploys"}], "sync_contract": {"skills": 1}}

    def fake_get(url: str, **kw: Any) -> _Resp:
        seen["url"] = url
        return _Resp()

    monkeypatch.setattr(radient.requests, "get", fake_get)
    client = radient.RadientClient(api_key=None, base_url="https://api.example.test/v1")
    monkeypatch.setattr(client, "_get_headers", lambda **_k: {})
    monkeypatch.setattr(client, "_publication_result", lambda r, action: r.json())
    assert client.list_org_skills("t1") == [{"id": "s1", "name": "deploys"}]
    assert seen["url"].endswith("/tenants/t1/skills")


del Path
