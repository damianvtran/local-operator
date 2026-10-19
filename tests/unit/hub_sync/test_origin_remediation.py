"""Production-caller and ancestry regressions from PR 2663 review round one."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync.check import ItemCheck, check_hub_teams
from local_operator.hub_sync.identity import link_origin
from tests.unit.hub_sync.test_link_origin import SOURCE, TENANT, _ctx, _team


@pytest.mark.parametrize("tenant", [None, TENANT])
def test_first_publication_records_accepted_ancestor(tmp_path, tenant):
    ctx = _ctx(tmp_path)
    origin = link_origin(SOURCE, "org" if tenant else "public", tenant)
    accepted = {
        "description": "Accepted canonical",
        "manager": "manager",
        "members": [],
        "instructions": "accepted",
        "project": "",
    }
    local = {**accepted, "description": "Locally authored"}
    assert svc.seed_receipt_after_publish(
        ctx.config_dir,
        "team",
        "local-1",
        accepted=accepted,
        local_fields=local,
        revision='"r1"',
        origin=origin,
        hub_id="hub-1",
    )
    record = prov.read_baseline(ctx.config_dir, "team", "local-1")
    assert record.recorded_by == "publish" and record.fields == accepted
    assert record.outbound.local_fields == local
    snapshot = svc.links_snapshot(ctx)
    row = snapshot["links"][0]
    assert snapshot["unresolved"] == [] and row["link_id"]
    assert row["tenant_id"] == tenant
    assert row["eligibility"]["can_auto_push"] is False
    assert row["policy"]["auto_push"] is False
    print(
        f"Q1 first publish: scope={row['visibility']} tenant={tenant} links=1 unresolved=0 "
        "baseline=publish canonical_ancestor=True auto_push=False "
        f"reason={row['eligibility']['reason']}"
    )


@pytest.mark.parametrize("conflict", ["hub", "scope", "source"])
def test_conflicting_publication_preserves_every_baseline_byte(tmp_path, conflict):
    ctx = _ctx(tmp_path)
    origin = link_origin(SOURCE, "org", TENANT)
    fields = {"description": "original"}
    assert svc.seed_receipt_after_publish(
        ctx.config_dir,
        "team",
        "local-1",
        accepted=fields,
        local_fields=fields,
        revision='"original"',
        origin=origin,
        hub_id="hub-1",
    )
    path = prov.baseline_path(ctx.config_dir, "team", "local-1")
    before = path.read_bytes()
    different = {
        "hub": origin,
        "scope": link_origin(SOURCE, "public"),
        "source": link_origin("https://other.test/v1", "org", TENANT),
    }[conflict]
    assert not svc.seed_receipt_after_publish(
        ctx.config_dir,
        "team",
        "local-1",
        accepted={"description": "other publication"},
        local_fields={"description": "other"},
        revision='"other"',
        origin=different,
        hub_id="hub-other" if conflict == "hub" else "hub-1",
    )
    assert path.read_bytes() == before
    print(f"R4 conflicting {conflict}: refused=True baseline_byte_identical=True")


def test_ambiguous_legacy_tenant_is_not_certified(tmp_path):
    ctx = _ctx(tmp_path)
    prov.record_team_pull(ctx.config_dir, _team("local-1"), {"id": "hub-1"}, tenant_id=TENANT)
    path = prov.baseline_path(ctx.config_dir, "team", "local-1")
    before = path.read_bytes()
    assert not svc.attach_served_origin(
        ctx,
        ItemCheck("team", "local-1", "crew", "hub-1", TENANT, "up-to-date", served_by=SOURCE),
        source=SOURCE,
    )
    assert path.read_bytes() == before
    assert svc.links_snapshot(ctx)["links"] == []
    print(
        "R2 legacy publisher tenant: unresolved=True "
        "baseline_byte_identical=True org_identity=False"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [None, TENANT])
async def test_desktop_team_scope_comes_from_caller(tmp_path, monkeypatch, declared):
    from local_operator.server.routes import agents as route
    from local_operator.teams import TeamRegistry

    ctx = _ctx(tmp_path)
    document = {
        "id": "10000000-0000-4000-8000-000000000001",
        "name": "scope-crew",
        "tenant_id": TENANT,
        "description": "d",
        "manager": "manager",
        "members": [],
        "instructions": "i",
        "project": "",
        "document_version": 1,
    }
    client = SimpleNamespace(base_url=SOURCE, get_team=lambda *a, **k: document)
    monkeypatch.setattr(
        route, "_org_radient_credentials", AsyncMock(return_value=(SecretStr("synthetic"), SOURCE))
    )
    monkeypatch.setattr(route, "RadientClient", lambda **kw: client)
    await route.pull_team_from_radient(
        team_id=document["id"],
        tenant_id=declared,
        config_manager=ctx.config_manager,
        provider_auth_store=None,
    )
    team = TeamRegistry(tmp_path).list_teams()[0]
    record = prov.read_baseline(tmp_path, "team", team.id)
    if declared:
        assert record.identity.visibility == "org" and record.tenant_id == declared
    else:
        assert record.identity is None and not record.scope_known
        assert not svc.attach_served_origin(
            ctx,
            ItemCheck(
                "team", team.id, team.name, document["id"], None, "up-to-date", served_by=SOURCE
            ),
            source=SOURCE,
        )
    print(
        f"R1 desktop team: declared={declared} "
        f"visibility={record.identity.visibility if record.identity else 'unresolved'}"
    )


@pytest.mark.asyncio
async def test_desktop_agent_download_uses_configured_base(tmp_path, monkeypatch):
    from local_operator.env import EnvConfig
    from local_operator.hub_sync.identity import LinkIdentity
    from local_operator.server.routes import agents as route

    ctx = _ctx(tmp_path)
    base = "https://configured-hub.test/v1"
    ctx.config_manager.set_config_value("radient_base_url", base)
    observed = []
    monkeypatch.setattr(
        route,
        "RadientClient",
        lambda **kw: observed.append(kw["base_url"]) or SimpleNamespace(**kw),
    )
    agent = SimpleNamespace(model_dump=lambda: {"id": "local-1"})
    registry = SimpleNamespace(
        download_agent_from_radient=lambda *a, **kw: SimpleNamespace(
            agent=agent, renamed_from=None, model_notice=None
        )
    )
    await route.download_agent_from_radient(
        agent_id="hub-1",
        agent_registry=registry,
        env_config=EnvConfig(),
        provider_auth_store=None,
        config_manager=ctx.config_manager,
    )
    assert observed == [base]
    pulled = LinkIdentity("agent", "local-1", observed[0], "public", None, "hub-1")
    resolved = LinkIdentity(
        "agent", "local-1", prov.canonical_hub_source(ctx.config_manager), "public", None, "hub-1"
    )
    assert pulled.link_id == resolved.link_id
    print(
        f"R3 desktop agent: route_base={observed[0]} "
        f"resolve_base={resolved.hub_source} same_link_id=True"
    )


def test_public_team_check_uses_optional_auth(tmp_path):
    team = _team("local-1")
    prov.record_team_pull(
        tmp_path, team, {"id": "hub-1"}, tenant_id="", origin=link_origin(SOURCE, "public")
    )
    registry = SimpleNamespace(
        config_dir=tmp_path, list_teams=lambda: [team], get_team=lambda _: team
    )
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from local_operator.clients.radient import RadientClient

    requests_seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests_seen.append((self.path, self.headers.get("Authorization") is not None))
            body = json.dumps(
                {
                    "result": {
                        "id": "hub-1",
                        "description": "upstream edit",
                        "manager": "manager",
                        "members": [],
                        "instructions": "i",
                        "project": "",
                    }
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # Real client, auth gate and loopback HTTP; no bearer and no mocked network.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RadientClient(api_key=None, base_url=f"http://127.0.0.1:{server.server_port}/v1")
        checks = check_hub_teams(registry, client_for_tenant=lambda _: client)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert checks[0].verdict == "available", checks[0]
    assert requests_seen == [("/v1/teams/hub-1", False)]
    print("Q2 anonymous team check: real_HTTP=200 authorization=False verdict=available")
