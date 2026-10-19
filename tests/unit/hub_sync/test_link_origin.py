"""The origin a link records: a pull, a publication and the repair that follows.

WHY THESE TESTS EXIST. A link used to record a hub id and nothing else, so the
links inventory listed every pulled row under "unresolved" with the reason
"source-unresolved" -- the desktop's "Source not confirmed" and its "?" badge
-- for a row this machine had pulled from a source the operator trusts. The
origin is what the SERVING ARM proved, so these tests pin the arm's scope
(public names no organization, an organization names its own), the identity's
digest, and the refusals that keep a repair from being a relink.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync.check import ItemCheck
from local_operator.hub_sync.identity import link_origin

SOURCE = "https://api.radienthq.com/v1"
TENANT = "5b72d9f4-8ce7-4821-ac8a-f92c365ed5b3"


def _team(local_id: str) -> Any:
    return SimpleNamespace(
        id=local_id,
        name="crew",
        description="d",
        manager="manager",
        members=[],
        instructions="i",
        project="",
    )


def _ctx(tmp_path: Path) -> Any:
    cm = ConfigManager(tmp_path)
    return svc.HubSyncContext(
        config_dir=cm.config_dir,
        config_manager=cm,
        client_for_tenant=lambda _t: None,
    )


def _pull(ctx: Any, local_id: str, hub_id: str, *, tenant: str, origin: Any) -> Any:
    return prov.record_team_pull(
        ctx.config_dir,
        _team(local_id),
        {"id": hub_id, "name": "crew"},
        tenant_id=tenant,
        origin=origin,
    )


def _rows(snapshot: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {row["local_id"]: row for row in snapshot[key]}


def test_a_pulled_team_is_a_confirmed_link_and_not_an_unresolved_row(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    _pull(ctx, "team-1", "hub-1", tenant="", origin=link_origin(SOURCE, "public"))

    snapshot = svc.links_snapshot(ctx)
    assert list(_rows(snapshot, "links")) == ["team-1"]
    assert snapshot["unresolved"] == []
    row = _rows(snapshot, "links")["team-1"]
    assert row["link_id"]
    assert row["hub_source"] == SOURCE
    assert row["visibility"] == "public"
    assert row["tenant_id"] is None


def test_the_public_arm_and_the_organization_arm_mint_different_links(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    _pull(ctx, "team-public", "hub-1", tenant="", origin=link_origin(SOURCE, "public"))
    _pull(ctx, "team-org", "hub-2", tenant=TENANT, origin=link_origin(SOURCE, "org", TENANT))

    snapshot = svc.links_snapshot(ctx)
    links = _rows(snapshot, "links")
    public = links["team-public"]
    org = links["team-org"]
    # The public arm names NO organization: the publisher's tenant on a public
    # row is not this machine's scope, and an identity that named one would
    # put a public row past the public egress gate.
    assert public["visibility"] == "public" and public["tenant_id"] is None
    assert org["visibility"] == "org" and org["tenant_id"] == TENANT
    # The link id is a digest of the whole tuple, so the two arms differ.
    assert public["link_id"] != org["link_id"]


def test_a_merge_does_not_drop_the_identity(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _pull(ctx, "team-1", "hub-1", tenant=TENANT, origin=link_origin(SOURCE, "org", TENANT))
    before = _rows(svc.links_snapshot(ctx), "links")["team-1"]["link_id"]

    svc._advance_baseline(
        ctx,
        ItemCheck(
            "team",
            "team-1",
            "crew",
            "hub-1",
            TENANT,
            "up-to-date",
            remote={
                "description": "d",
                "manager": "manager",
                "members": [],
                "instructions": "i",
                "project": "",
            },
        ),
        "merge-pull",
    )

    after = _rows(svc.links_snapshot(ctx), "links")["team-1"]["link_id"]
    assert after == before


def test_the_repair_links_a_row_that_was_pulled_before_the_fix(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # A record written before links carried an origin: a hub id, a tenant, no
    # identity. The inventory reports it as unresolved.
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id=TENANT)
    before = svc.links_snapshot(ctx)
    assert before["links"] == []
    assert _rows(before, "unresolved")["team-1"]["reason"] == "source-unresolved"

    attached = svc.attach_served_origin(
        ctx,
        ItemCheck("team", "team-1", "crew", "hub-1", TENANT, "up-to-date", served_by=SOURCE),
        source=SOURCE,
    )

    assert attached is True
    after = svc.links_snapshot(ctx)
    assert after["unresolved"] == []
    row = _rows(after, "links")["team-1"]
    assert row["link_id"]
    assert row["visibility"] == "org" and row["tenant_id"] == TENANT


def test_the_repair_leaves_a_row_the_hub_no_longer_serves_unresolved(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id=TENANT)

    attached = svc.attach_served_origin(
        ctx,
        ItemCheck(
            "team",
            "team-1",
            "crew",
            "hub-1",
            TENANT,
            "unavailable",
            reason="hub-item-missing",
            served_by=SOURCE,
        ),
        source=SOURCE,
    )

    assert attached is False
    snapshot = svc.links_snapshot(ctx)
    assert snapshot["links"] == []
    assert _rows(snapshot, "unresolved")["team-1"]["reason"] == "source-unresolved"


def test_the_repair_refuses_a_row_another_base_served(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id=TENANT)

    attached = svc.attach_served_origin(
        ctx,
        ItemCheck(
            "team",
            "team-1",
            "crew",
            "hub-1",
            TENANT,
            "up-to-date",
            served_by="https://elsewhere.test/v1",
        ),
        source=SOURCE,
    )

    assert attached is False
    assert svc.links_snapshot(ctx)["unresolved"]


def test_auto_push_stays_refused_for_a_repaired_public_row(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id="")
    attached = svc.attach_served_origin(
        ctx,
        ItemCheck("team", "team-1", "crew", "hub-1", None, "up-to-date", served_by=SOURCE),
        source=SOURCE,
    )
    assert attached is True

    snapshot = svc.links_snapshot(ctx)
    row = _rows(snapshot, "links")["team-1"]
    eligibility = row["eligibility"]
    assert eligibility["can_auto_push"] is False
    # A NAMED reason, never "eligible": the repaired row has no attested
    # revision-CAS for its source, so an unattended push is refused.
    assert eligibility["reason"]
    assert eligibility["reason"] != "eligible"


def test_a_public_link_still_needs_the_public_egress_grant(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    source = prov.canonical_hub_source(ctx.config_manager)
    origin = link_origin(source, "public")
    assert origin is not None
    record = prov.make_record(
        "team",
        "team-1",
        "hub-1",
        None,
        {
            "description": "d",
            "manager": "manager",
            "members": [],
            "instructions": "i",
            "project": "",
        },
        "pull",
    )
    record = prov.attach_origin(ctx.config_dir, record, origin)
    assert record is not None and record.identity is not None

    # The row is a public link, so the grant is what an unattended push would
    # need; without it the refusal is named rather than silent.
    def attest(doc: dict[str, Any]) -> None:
        server = doc.setdefault("server", {})
        sources = server.setdefault("sources", {})
        sources[source] = {
            "revision_cas": 1,
            "skills": 1,
            "attested_at": "2026-10-19T00:00:00Z",
        }

    assert ctx.store().mutate(attest) is True

    fresh = prov.read_baseline(ctx.config_dir, "team", "team-1")
    assert fresh is not None and fresh.identity is not None
    # Every OTHER precondition is met, so the grant is the one thing left to
    # refuse: a public link is never published unattended without a deliberate
    # egress consent bound to this exact link.
    prov.write_baseline(
        ctx.config_dir,
        dataclasses.replace(fresh, remote_revision='"rev-1"', lineage="clean"),
    )

    snapshot = svc.links_snapshot(ctx)
    row = _rows(snapshot, "links")["team-1"]
    assert row["eligibility"]["can_auto_push"] is False
    assert row["eligibility"]["reason"] == "unsafe-publication"


def test_a_publication_records_the_origin_the_user_chose(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # The row was linked into an organization before the fix: a hub id, the
    # owning tenant, and no identity.
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id=TENANT)
    assert svc.links_snapshot(ctx)["unresolved"]

    recorded = svc.record_publish_origin(
        ctx.config_dir,
        "team",
        "team-1",
        hub_id="hub-1",
        origin=link_origin(SOURCE, "org", TENANT),
    )

    assert recorded is True
    snapshot = svc.links_snapshot(ctx)
    assert snapshot["unresolved"] == []
    row = _rows(snapshot, "links")["team-1"]
    assert row["link_id"]
    assert row["visibility"] == "org" and row["tenant_id"] == TENANT


def test_a_publication_into_another_scope_is_refused_not_relinked(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id=TENANT)

    recorded = svc.record_publish_origin(
        ctx.config_dir,
        "team",
        "team-1",
        hub_id="hub-1",
        origin=link_origin(SOURCE, "public"),
    )

    assert recorded is False
    snapshot = svc.links_snapshot(ctx)
    assert snapshot["links"] == []
    assert _rows(snapshot, "unresolved")["team-1"]["reason"] == "source-unresolved"


def test_a_public_publication_keeps_the_record_free_of_an_organization(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    prov.record_team_pull(ctx.config_dir, _team("team-1"), {"id": "hub-1"}, tenant_id="")

    recorded = svc.record_publish_origin(
        ctx.config_dir,
        "team",
        "team-1",
        hub_id="hub-1",
        origin=link_origin(SOURCE, "public"),
    )

    assert recorded is True
    record = prov.read_baseline(ctx.config_dir, "team", "team-1")
    assert record is not None
    assert record.tenant_id is None
    assert record.identity is not None
    assert record.identity.visibility == "public"
    assert record.identity.tenant_id is None


def test_an_origin_that_cannot_be_stated_leaves_the_row_unresolved(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    # A public scope that names an organization is refused by the origin
    # itself, and a pull must never fail over that bookkeeping.
    assert link_origin(SOURCE, "public", TENANT) is None
    assert link_origin(SOURCE, "org") is None
    assert link_origin("", "public") is None

    record = _pull(ctx, "team-1", "hub-1", tenant=TENANT, origin=None)
    assert record is not None
    assert record.identity is None
    assert svc.links_snapshot(ctx)["unresolved"]


@pytest.mark.parametrize(
    "visibility,tenant,expected_visibility,expected_tenant",
    [
        ("public", None, "public", None),
        ("org", TENANT, "org", TENANT),
    ],
)
def test_the_link_id_is_a_digest_of_the_whole_origin(
    tmp_path: Path,
    visibility: str,
    tenant: str | None,
    expected_visibility: str,
    expected_tenant: str | None,
) -> None:
    ctx = _ctx(tmp_path)
    _pull(
        ctx,
        "team-1",
        "hub-1",
        tenant=tenant or "",
        origin=link_origin(SOURCE, visibility, tenant),
    )

    row = _rows(svc.links_snapshot(ctx), "links")["team-1"]
    assert row["visibility"] == expected_visibility
    assert row["tenant_id"] == expected_tenant
    expected = prov.link_identity(
        "team",
        "team-1",
        "hub-1",
        origin=link_origin(SOURCE, visibility, tenant),
    )
    assert expected is not None
    assert row["link_id"] == expected.link_id
