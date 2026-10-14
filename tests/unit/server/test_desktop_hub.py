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
from local_operator.server.routes import capabilities, desktop_hub, desktop_sessions
from tests.unit.hub_sync.test_service import BASE, Hub, _pull_agent

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

    monkeypatch.setattr("local_operator.agents._fetch_hub_profile", boom)
    failed = await client.post("/v1/desktop/hub/updates/check", json=mutation())
    (item,) = failed.json()["result"]["status"]["items"]
    assert item["error_class"] == "hub-error" and item["state"] == "available"

    monkeypatch.setattr(
        "local_operator.agents._fetch_hub_profile", lambda _c, h, **_k: hub.agents[h]
    )
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
    # A LEGACY link (no confirmed identity) is listed under `unresolved`, not in
    # `links`, so a client offers resolve instead of acting on a guessed origin.
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

    link_id = await _resolved_link_id(client, root)

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
    link_id = await _resolved_link_id(client, root)
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


async def _resolved_link_id(client: Any, root: Path) -> str:
    """Resolve the pulled agent's legacy link to a confirmed identity, return its id."""
    resolved = await client.post(
        "/v1/desktop/hub/updates/apply",
        json=mutation(
            kind="agent",
            name="coder",
            resolve_source=True,
            hub_source="https://api.radienthq.com/v1",
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
