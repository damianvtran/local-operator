"""The five hub routes: shapes, receipts replay, status vocabulary, the desktop gate."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.hub_sync import service as svc_module
from local_operator.server.routes import capabilities, desktop_hub, desktop_sessions
from local_operator.teams import TeamRegistry
from tests.unit.hub_sync.test_service import (
    BASE,
    Hub,
    _fields,
    _make_legacy,
    _pull_agent,
    _pull_team,
)

pytestmark = pytest.mark.asyncio

AUTH = {"Authorization": "Bearer test-token"}


def mutation(**fields: Any) -> dict[str, Any]:
    return {"request_id": str(uuid.uuid4()), **fields}


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "test-token")
    hub = Hub()
    monkeypatch.setattr(
        "local_operator.agents._fetch_hub_profile", lambda _c, h, **_k: hub.agents[h]
    )

    async def fake_clients(_cm: Any, _store: Any = None):
        return (lambda _t: hub.client()), "ok"

    monkeypatch.setattr("local_operator.hub_sync.service.build_clients", fake_clients)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    # The rig's hub IS this machine's configured hub: the source-resolution
    # route approves the CONFIGURED base, so a rig that answered from another
    # one could not confirm the origin a pull recorded.
    app.state.config_manager.set_config_value("radient_base_url", hub.base_url)
    app.state.env_config = EnvConfig()
    app.include_router(desktop_hub.router)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    app.dependency_overrides[desktop_hub.get_provider_auth_store] = lambda: None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=AUTH
    ) as client:
        yield client, root, hub
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def _seed(root: Path, hub: Hub) -> Any:
    agents = AgentRegistry(root)
    row = _pull_agent(agents, hub, root)
    hub.agents["h1"] = (BASE + "\n\n## New\nAdded upstream.", "d")
    return agents, row


async def test_the_updates_route_is_a_network_free_store_read_with_the_documented_shape(
    api,
) -> None:
    client, root, hub = api
    _seed(root, hub)
    empty = (await client.get("/v1/desktop/hub/updates")).json()["result"]
    assert empty["items"] == [] and empty["counts"]["available"] == 0
    assert set(empty) == {"generated_at", "credential", "settings", "counts", "items"}

    checked = await client.post("/v1/desktop/hub/updates/check", json=mutation())
    assert checked.status_code == 200
    body = checked.json()["result"]
    assert body["reports"][0]["outcome"] == "available" and "status" in body
    (item,) = (await client.get("/v1/desktop/hub/updates")).json()["result"]["items"]
    assert (
        item["state"] == "available" and item["auto_will_apply"] is True and item["kind"] == "agent"
    )


async def test_check_applies_nothing_even_with_auto_on(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    await client.post("/v1/desktop/hub/updates/check", json=mutation())
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)


async def test_apply_merges_one_item_and_returns_the_fresh_snapshot(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    res = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="coder")
    )
    assert res.status_code == 200
    result = res.json()["result"]
    assert result["reports"][0]["outcome"] == "merged" and result["reports"][0]["applied"] is True
    assert result["status"]["counts"]["applied"] == 1
    assert "Added upstream" in agents.get_agent_system_prompt(row.id)


async def test_dry_run_previews_without_writing(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    res = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="coder", dry_run=True)
    )
    assert res.json()["result"]["reports"][0]["outcome"] == "would-merge"
    assert "Added upstream" not in agents.get_agent_system_prompt(row.id)


async def test_a_lost_response_is_replayed_not_re_run(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    body = mutation(kind="agent", name="coder")
    first = await client.post("/v1/desktop/hub/updates/apply", json=body)
    hub.agents["h1"] = (BASE + "\n\n## New\nAdded upstream.\n\n## Newer\nAgain.", "d")
    second = await client.post("/v1/desktop/hub/updates/apply", json=body)
    # The replay is the stored first answer: the hub's newer edit ("Again") was not
    # fetched or applied by the second request.
    replayed = dict(second.json()["result"])
    assert replayed.pop("replayed") is True
    assert replayed == first.json()["result"]
    assert "Again" not in agents.get_agent_system_prompt(row.id)


async def test_replace_needs_confirmation_and_cannot_combine_with_prefer(api) -> None:
    client, root, hub = api
    _seed(root, hub)
    unconfirmed = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="coder", replace="remote")
    )
    assert unconfirmed.status_code == 422 and "confirm_replace" in unconfirmed.json()["detail"]
    both = await client.post(
        "/v1/desktop/hub/updates/apply",
        json=mutation(
            kind="agent", name="coder", replace="remote", confirm_replace=True, prefer="local"
        ),
    )
    assert both.status_code == 422


async def test_replace_echoes_what_it_discarded(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    agents.set_agent_system_prompt(row.id, "MY OWN TEXT")
    res = await client.post(
        "/v1/desktop/hub/updates/apply",
        json=mutation(kind="agent", name="coder", replace="remote", confirm_replace=True),
    )
    report = res.json()["result"]["reports"][0]
    assert report["applied"] is True and report["replaced"]["instructions"] == "MY OWN TEXT"
    assert report["backup"]


async def test_unknown_item_is_404_and_extra_fields_are_refused(api) -> None:
    client, *_ = api
    assert (
        await client.post(
            "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="ghost")
        )
    ).status_code == 404
    assert (
        await client.post("/v1/desktop/hub/updates/check", json=mutation(bogus=1))
    ).status_code == 422


async def test_apply_all_and_retry(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    res = await client.post("/v1/desktop/hub/updates/apply-all", json=mutation())
    assert res.status_code == 200 and res.json()["result"]["reports"][0]["outcome"] == "merged"
    retried = await client.post(
        "/v1/desktop/hub/updates/retry", json=mutation(kind="agent", name="coder")
    )
    assert retried.status_code == 200


async def test_a_manual_retry_leaves_the_row_ready_to_update_not_retryable_again(
    api, monkeypatch: pytest.MonkeyPatch
) -> None:
    """U10: the retry's own answer says "press it again to update" - so the row must offer it.

    Manual mode makes a retry a DRY RUN: it proves the update computes and writes
    nothing. Before this, the ``hub-error`` the item failed with stayed on the row,
    so the mark kept saying Retry and the update was unreachable from the row.
    """

    client, root, hub = api
    _seed(root, hub)
    ConfigManager(root).set_config_value("hub", {"auto_update": {"agents": False}})
    # The hub went down after the item was found (one failed check), then came back.
    checked = await client.post("/v1/desktop/hub/updates/check", json=mutation())
    assert checked.json()["result"]["status"]["items"][0]["state"] == "available"

    def boom(_client: Any, _hub_id: str, **_kw: Any) -> tuple[str, str]:
        raise RuntimeError("500 Server Error")

    # "The hub went down" is BOTH reads failing: the check reads the DETAIL
    # first (one consistent content+revision snapshot, round-3 R18) and only
    # falls back to the zip, so a live detail with a dead zip is not an outage.
    original_get_agent = hub.get_agent

    def boom_detail(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("500 Server Error")

    monkeypatch.setattr("local_operator.agents._fetch_hub_profile", boom)
    monkeypatch.setattr(hub, "get_agent", boom_detail)
    failed = await client.post("/v1/desktop/hub/updates/check", json=mutation())
    body = failed.json()["result"]["status"]
    (item,) = body["items"]
    # QA F3: a carried error is reported as failed and counted, so a consumer
    # that reads only ``state``/``counts`` does not miss the outage.
    assert item["error_class"] == "hub-error" and item["state"] == "failed"
    assert body["counts"]["failed"] == 1 and body["counts"]["available"] == 0

    monkeypatch.setattr(
        "local_operator.agents._fetch_hub_profile", lambda _c, h, **_k: hub.agents[h]
    )
    monkeypatch.setattr(hub, "get_agent", original_get_agent)
    retried = await client.post(
        "/v1/desktop/hub/updates/retry", json=mutation(kind="agent", name="coder")
    )
    body = retried.json()["result"]
    assert body["reports"][0]["outcome"] == "would-merge"
    (after,) = body["status"]["items"]
    assert after["error_class"] is None and after["state"] == "available"


async def test_a_conflict_is_needs_review_over_the_wire_and_prefer_settles_it(api) -> None:
    client, root, hub = api
    agents, row = _seed(root, hub)
    agents.set_agent_system_prompt(row.id, "## Rules\nBe brief.\n\n## Tools\nUse tools carefully.")
    hub.agents["h1"] = (BASE.replace("Old advice.", "Older, better advice."), "d")
    res = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="coder")
    )
    assert res.json()["result"]["reports"][0]["outcome"] == "needs-review"
    (item,) = res.json()["result"]["status"]["items"]
    assert item["error_class"] == "merge-refused" and item["auto_will_apply"] is False
    ok = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="coder", prefer="remote")
    )
    assert ok.json()["result"]["reports"][0]["applied"] is True
    assert "Older, better advice." in agents.get_agent_system_prompt(row.id)


async def test_the_desktop_gate_refuses_an_unauthenticated_caller(api) -> None:
    client, *_ = api
    res = await client.get("/v1/desktop/hub/updates", headers={"Authorization": "Bearer wrong"})
    assert res.status_code in (401, 403)


async def test_capabilities_advertise_hub_updates() -> None:
    from local_operator.server.features import feature_flags

    assert feature_flags()["hub_updates"] == 1


async def test_links_inventory_lists_a_healthy_link_with_capability(api):
    """The inventory enumerates a healthy link (identity+policy+health), not just pending."""
    client, root, hub = api
    agents = AgentRegistry(root)
    _pull_agent(agents, hub, root)  # a real pulled link with a baseline

    # Capability advertised.
    caps = (await client.get("/v1/capabilities")).json()["result"]
    assert caps["features"]["hub_links"] == 1

    payload = (await client.get("/v1/desktop/hub/links")).json()["result"]
    assert payload["capability"] == "hub_links:1"
    # A pull records the origin it was served under, so the link is CONFIRMED
    # and is listed with its link id -- not an unresolved row.
    assert payload["unresolved"] == []
    assert len(payload["links"]) == 1
    row = payload["links"][0]
    assert row["kind"] == "agent" and row["hub_id"] == "h1"
    assert row["link_id"] and row["visibility"] == "public"


async def test_links_inventory_lists_a_legacy_row_as_unresolved(api):
    """A row pulled BEFORE the fix carries no origin, and is listed as unresolved."""
    client, root, hub = api
    agents = AgentRegistry(root)
    row = _pull_agent(agents, hub, root)
    _make_legacy(root, "agent", row.id)

    payload = (await client.get("/v1/desktop/hub/links")).json()["result"]
    assert payload["links"] == []
    assert len(payload["unresolved"]) == 1
    row = payload["unresolved"][0]
    assert row["kind"] == "agent" and row["hub_id"] == "h1"
    assert row["reason"] == "source-unresolved"


async def test_link_policy_patch_is_revision_checked(api):
    """A stale revision is a named 409; the correct one writes both values independently."""
    client, root, hub = api
    agents = AgentRegistry(root)
    _pull_agent(agents, hub, root)

    link_id = await _resolved_link_id(client, root, hub.base_url)

    stale = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(subscribe=False, expected_policy_revision="not-the-revision"),
    )
    assert stale.status_code == 409, stale.text

    payload = (await client.get("/v1/desktop/hub/links")).json()["result"]["links"][0]
    revision = payload["policy"]["revision"]

    ok = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(subscribe=False, expected_policy_revision=revision),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["result"]["policy"]["subscribe"] is False
    assert ok.json()["result"]["policy"]["auto_push"] is False


async def test_link_policy_patch_refuses_enabling_auto_push_without_eligibility(api):
    """Enabling auto_push demands a server eligibility attestation; refused, not faked."""
    client, root, hub = api
    agents = AgentRegistry(root)
    _pull_agent(agents, hub, root)
    link_id = await _resolved_link_id(client, root, hub.base_url)
    payload = next(
        row
        for row in (await client.get("/v1/desktop/hub/links")).json()["result"]["links"]
        if row["link_id"] == link_id
    )

    refused = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(auto_push=True, expected_policy_revision=payload["policy"]["revision"]),
    )
    assert refused.status_code == 422, refused.text


async def test_link_policy_patch_404s_an_unknown_link(api):
    client, root, hub = api
    response = await client.patch(
        "/v1/desktop/hub/links/deadbeef",
        json=mutation(subscribe=True, expected_policy_revision="x"),
    )
    assert response.status_code == 404


async def _resolved_link_id(client: Any, root: Path, hub_source: str) -> str:
    """Resolve the pulled agent's link to a confirmed identity, return its id.

    The source named is the CONFIGURED one (the route approves nothing else),
    which is the hub the pull was served by -- so the resolution confirms the
    origin the pull already recorded instead of pointing the link elsewhere.
    """
    resolved = await client.post(
        "/v1/desktop/hub/updates/apply",
        json=mutation(
            kind="agent",
            name="coder",
            resolve_source=True,
            hub_source=hub_source,
            visibility="public",
        ),
    )
    assert resolved.status_code == 200, resolved.text
    return resolved.json()["result"]["link"]["link_id"]


def _seed_public_link(
    root: Path, *, lineage: str = "clean", recorded_by: str = "pull", cas: bool = True
) -> str:
    """A CONFIRMED public link as a real pull + capability probe leaves it.

    Baseline record with a source-scoped remote revision and known scope, plus
    (unless ``cas`` is False) the persisted revision-CAS attestation — the facts
    ``_eligibility`` reads. ``lineage``/``recorded_by``/``cas`` are varied to
    reach each earlier gate in order: capability first (unsupported-
    precondition), then the baseline (baseline-unknown), then lineage.
    """

    import dataclasses

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
    if cas:
        svc.record_server_capabilities(
            svc.HubSyncContext(
                config_dir=root, config_manager=cm, client_for_tenant=lambda _t: None
            ),
            {"revision_cas": 1},
            source=source,
        )
    return identity.link_id


def _grant_written(root: Path, link_id: str) -> bool:
    hub = ConfigManager(root).get_config().values.get("hub", {})
    return link_id in (hub.get("public_egress_grants") or {})


def _auto_push_persisted(root: Path, link_id: str) -> bool:
    """N1: the persisted auto_push value (a refusal must leave it False)."""

    hub = ConfigManager(root).get_config().values.get("hub", {})
    return bool(((hub.get("links") or {}).get(link_id) or {}).get("auto_push"))


async def _revision(client: Any, link_id: str) -> str:
    payload = (await client.get("/v1/desktop/hub/links")).json()["result"]["links"]
    return next(row["policy"]["revision"] for row in payload if row["link_id"] == link_id)


async def test_desktop_patch_mints_the_first_public_grant_only_with_consent(api):
    """The cross-surface fix on the REAL ASGI app: consent mints the first grant."""

    client, root, _hub = api
    link_id = _seed_public_link(root)
    # WITHOUT confirm: 422, no grant.
    refused = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=False,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert refused.status_code == 422, refused.text
    assert not _grant_written(root, link_id)
    assert _auto_push_persisted(root, link_id) is False
    # WITH confirm: 200 and the grant persisted.
    ok = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=True,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["result"]["policy"]["auto_push"] is True
    assert _grant_written(root, link_id)


async def test_desktop_patch_refuses_when_an_earlier_gate_fails_even_with_consent(api):
    """An unknown baseline refuses even with consent — no grant, 422."""

    client, root, _hub = api
    link_id = _seed_public_link(root, recorded_by="adopt-unknown")
    refused = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=True,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["code"] == "baseline-unknown"
    assert not _grant_written(root, link_id)
    assert _auto_push_persisted(root, link_id) is False
    assert _auto_push_persisted(root, link_id) is False


async def test_desktop_patch_refuses_mixed_lineage_with_consent(api):
    """D34: a MIXED-LINEAGE public origin refuses even with confirm_public_egress."""

    client, root, _hub = api
    link_id = _seed_public_link(root, lineage="mixed-lineage")
    refused = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=True,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"]["code"] == "mixed-lineage"
    assert not _grant_written(root, link_id)
    assert _auto_push_persisted(root, link_id) is False


async def test_desktop_second_write_after_the_grant_uses_the_served_revision(api):
    """F1: after a grant is minted, the NEXT write on the SERVED revision succeeds.

    The served revision omitted the egress grant from the digest while
    ``update_policy`` compared the grant-aware one, so the first write minted the
    grant and EVERY later write (here: a subscribe flip) 409'd "link policy
    changed; reload before saving" — forever, on all three surfaces.
    """

    client, root, _hub = api
    link_id = _seed_public_link(root)
    minted = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=True,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert minted.status_code == 200, minted.text
    assert _grant_written(root, link_id)
    # SECOND write, with the revision the inventory serves AFTER the grant.
    second = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            subscribe=False,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert second.status_code == 200, second.text
    stored = _policy_values_desktop(root, link_id)
    assert stored["subscribe"] is False and stored["auto_push"] is True


def _policy_values_desktop(root: Path, link_id: str) -> dict[str, Any]:
    hub = ConfigManager(root).get_config().values.get("hub", {})
    return dict((hub.get("links") or {}).get(link_id) or {})


async def test_policy_patch_no_confirm_refusal_is_typed_unsafe_publication(api):
    """F3: the no-consent refusal carries detail.code == "unsafe-publication".

    The renderer reads `detail.code` (a declared code always wins in
    desktop-api.ts) and the desktop panel branches on `write.error.code`; the
    old string detail dropped the category entirely.
    """

    client, root, _hub = api
    link_id = _seed_public_link(root)
    refused = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=False,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert refused.status_code == 422, refused.text
    detail = refused.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail["code"] == "unsafe-publication", detail
    assert isinstance(detail["message"], str) and detail["message"].strip()


async def test_policy_patch_stale_revision_refusal_is_typed(api):
    """F3: a stale revision is a 409 whose detail carries policy-revision-changed."""

    client, root, _hub = api
    link_id = _seed_public_link(root)
    stale = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(subscribe=False, expected_policy_revision="not-the-revision"),
    )
    assert stale.status_code == 409, stale.text
    detail = stale.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail["code"] == "policy-revision-changed", detail
    assert isinstance(detail["message"], str) and detail["message"].strip()


async def test_policy_patch_no_cas_refusal_is_typed_unsupported_precondition(api):
    """F3: the FIRST gate's refusal carries its own code even with consent."""

    client, root, _hub = api
    link_id = _seed_public_link(root, cas=False)
    refused = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json=mutation(
            auto_push=True,
            confirm_public_egress=True,
            expected_policy_revision=await _revision(client, link_id),
        ),
    )
    assert refused.status_code == 422, refused.text
    detail = refused.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail["code"] == "unsupported-precondition", detail
    assert isinstance(detail["message"], str) and detail["message"].strip()


# -- addressing one item by its local id (`hub_items_by_id`) -----------------------------
#
# Display names are not unique: `create_agent` refuses only an EXACT duplicate and
# the hub checks match names case-insensitively, so `name: "coder"` reaches both
# `Coder` and `coder`. The desktop's per-row actions send `local_id` instead, and
# these tests pin that it touches exactly the addressed row of exactly that kind.


def _seed_twins(root: Path, hub: Hub) -> tuple[AgentRegistry, Any, Any]:
    """Two hub-linked agents whose names differ only by case, both with an update."""

    agents, lower = _seed(root, hub)  # `coder`, linked to h1
    upper = agents.create_agent(_fields(name="Coder", description="d"))
    agents.set_agent_system_prompt(upper.id, BASE)
    hub.agents["h2"] = (BASE + "\n\n## Twin\nAdded upstream.", "d")
    stamped = agents._stamp_hub_provenance(agents.get_agent(upper.id), "h2")
    agents._record_hub_baseline(stamped, "h2", None, hub.client())
    return agents, lower, stamped


def _ids(response: Any) -> list[str]:
    return [r["local_id"] for r in response.json()["result"]["reports"]]


async def test_the_name_arm_still_reaches_every_case_twin(api) -> None:
    """The unchanged name contract, and the reason `local_id` exists."""

    client, root, hub = api
    agents, lower, upper = _seed_twins(root, hub)
    res = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="coder")
    )
    assert res.status_code == 200
    assert sorted(_ids(res)) == sorted([lower.id, upper.id])
    assert "Added upstream" in agents.get_agent_system_prompt(lower.id)
    assert "Added upstream" in agents.get_agent_system_prompt(upper.id)


async def test_check_and_apply_by_local_id_touch_exactly_one_twin(api) -> None:
    client, root, hub = api
    agents, lower, upper = _seed_twins(root, hub)

    checked = await client.post(
        "/v1/desktop/hub/updates/check", json=mutation(kind="agent", local_id=lower.id)
    )
    assert checked.status_code == 200 and _ids(checked) == [lower.id]

    applied = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", local_id=lower.id)
    )
    assert applied.status_code == 200 and _ids(applied) == [lower.id]
    assert applied.json()["result"]["reports"][0]["outcome"] == "merged"
    assert "Added upstream" in agents.get_agent_system_prompt(lower.id)
    # The twin is untouched on disk -- the defect was a second merge here.
    assert "Twin" not in agents.get_agent_system_prompt(upper.id)


async def test_retry_by_local_id_clears_and_checks_only_that_row(api) -> None:
    client, root, hub = api
    _agents, lower, upper = _seed_twins(root, hub)
    await client.post("/v1/desktop/hub/updates/check", json=mutation(kind="agent"))

    def fail_both(doc: dict[str, Any]) -> None:
        for item in doc["items"].values():
            item.update(attempts=3, error_class="hub-error", auto_retry=False)

    store = svc_module.HubSyncContext(
        config_dir=root, config_manager=ConfigManager(root), client_for_tenant=lambda _t: None
    ).store()
    store.mutate(fail_both)

    retried = await client.post(
        "/v1/desktop/hub/updates/retry", json=mutation(kind="agent", local_id=upper.id)
    )
    assert retried.status_code == 200 and _ids(retried) == [upper.id]
    items = store.load()["items"]
    assert items[f"agent:{upper.id}"]["attempts"] == 0
    # The case twin keeps its failure: a retry by id never clears a second row.
    assert items[f"agent:{lower.id}"]["attempts"] == 3
    assert items[f"agent:{lower.id}"]["error_class"] == "hub-error"


async def test_a_manual_retry_by_local_id_retires_only_that_rows_failure(api) -> None:
    """U10 for the desktop's per-row Retry: the id arm of ``clear_failure``.

    MANUAL mode is the point: there a retry is a dry run (``would-merge``) and the
    stale failure is retired by ``clear_failure`` -- with auto on the retry merges
    and that call never runs, which is why the auto-mode retry test above cannot
    see a ``clear_failure`` that drops ``local_id`` (it would then match no row).
    """

    client, root, hub = api
    _agents, lower, upper = _seed_twins(root, hub)
    ConfigManager(root).set_config_value("hub", {"auto_update": {"agents": False}})
    await client.post("/v1/desktop/hub/updates/check", json=mutation(kind="agent"))

    def fail_both(doc: dict[str, Any]) -> None:
        for item in doc["items"].values():
            item.update(error_class="hub-error", last_error="500 Server Error")

    store = svc_module.HubSyncContext(
        config_dir=root, config_manager=ConfigManager(root), client_for_tenant=lambda _t: None
    ).store()
    store.mutate(fail_both)

    retried = await client.post(
        "/v1/desktop/hub/updates/retry", json=mutation(kind="agent", local_id=lower.id)
    )
    assert retried.status_code == 200, retried.text
    (report,) = retried.json()["result"]["reports"]
    assert report["local_id"] == lower.id and report["outcome"] == "would-merge"
    items = store.load()["items"]
    assert items[f"agent:{lower.id}"]["error_class"] is None
    assert items[f"agent:{lower.id}"]["state"] == "available"
    # The case twin was not addressed, so its failure stands.
    assert items[f"agent:{upper.id}"]["error_class"] == "hub-error"


async def test_a_team_is_addressed_by_local_id_and_never_as_another_kind(api) -> None:
    client, root, hub = api
    _seed(root, hub)
    team = _pull_team(TeamRegistry(root), hub, root)
    hub.teams["ht1"] = {**hub.teams["ht1"], "instructions": "## Brief\nShip it.\n\n## More\nX."}

    checked = await client.post(
        "/v1/desktop/hub/updates/check", json=mutation(kind="team", local_id=team.id)
    )
    assert checked.status_code == 200 and _ids(checked) == [team.id]
    preview = await client.post(
        "/v1/desktop/hub/updates/apply",
        json=mutation(kind="team", local_id=team.id, dry_run=True),
    )
    assert preview.status_code == 200 and _ids(preview) == [team.id]
    # A team id sent as an agent is a plain 404, never a cross-kind hit.
    wrong = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", local_id=team.id)
    )
    assert wrong.status_code == 404


async def test_an_unknown_local_id_is_the_same_404_shape_as_an_unknown_name(api) -> None:
    client, root, hub = api
    _seed(root, hub)
    by_id = await client.post(
        "/v1/desktop/hub/updates/apply",
        json=mutation(kind="agent", local_id=str(uuid.uuid4())),
    )
    by_name = await client.post(
        "/v1/desktop/hub/updates/apply", json=mutation(kind="agent", name="ghost")
    )
    assert by_id.status_code == by_name.status_code == 404
    assert set(by_id.json()) == set(by_name.json())
    assert isinstance(by_id.json()["detail"], str) and isinstance(by_name.json()["detail"], str)


@pytest.mark.parametrize(
    ("route", "fields"),
    [
        # Both arms at once is ambiguous; neither is no address for apply/retry.
        ("apply", {"kind": "agent", "name": "coder", "local_id": "x"}),
        ("apply", {"kind": "agent"}),
        ("retry", {"kind": "agent", "name": "coder", "local_id": "x"}),
        ("retry", {"kind": "agent"}),
        ("check", {"kind": "agent", "name": "coder", "local_id": "x"}),
        # An id without a kind would be searched across kinds: refused instead.
        ("check", {"local_id": "x"}),
        # Bounded like `name`.
        ("check", {"kind": "agent", "local_id": ""}),
        ("apply", {"kind": "agent", "local_id": "   "}),
        ("retry", {"kind": "agent", "local_id": "\t"}),
        ("check", {"kind": "agent", "local_id": "x" * 129}),
        # The resolve arm looks its row up by name; an id there is refused.
        (
            "apply",
            {
                "kind": "agent",
                "local_id": "x",
                "resolve_source": True,
                "hub_source": "https://hub.test/v1",
                "visibility": "public",
            },
        ),
    ],
)
async def test_an_ambiguous_or_unbounded_address_is_refused(api, route, fields) -> None:
    client, *_ = api
    res = await client.post(f"/v1/desktop/hub/updates/{route}", json=mutation(**fields))
    assert res.status_code == 422, res.text


async def test_capabilities_advertise_hub_items_by_id() -> None:
    from local_operator.server.features import feature_flags

    assert feature_flags()["hub_items_by_id"] == 1
