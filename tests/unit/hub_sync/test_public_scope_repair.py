"""The evidence-based repair of legacy public rows recorded under a tenant.

WHY THESE TESTS EXIST. Before the public pull stopped recording the served
document's ``tenant_id`` (the PUBLISHER's -- a public row carries it), a public
pull wrote that tenant into the baseline. Such a row has a tenant and no
identity; every repair path refuses it by design because a tenant cannot be
told apart from a real organization scope by GUESSING. The operator's
software-development and support-desk teams were stuck that way, showing "?" in
the desktop. ``service.repair_proven_public_scope`` replaces the guess with
proof: a 200 to a credential-free GET of the exact hub id is the public route's
own definition of public.

These tests pin, against a real ``RadientClient`` and a loopback HTTP hub (no
mocked network): the repair itself; every fail-closed path (404, 5xx, timeout,
transport error, redirect, mismatched id, an already-identified row) leaving
the baseline byte-identical; that the merge base is not laundered; that the
probe never carries a credential; and that a repaired row cannot publish.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from pydantic import SecretStr

from local_operator.clients.radient import RadientClient
from local_operator.config import ConfigManager
from local_operator.hub_sync import check as chk
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync import service as svc
from local_operator.hub_sync.check import ItemCheck
from local_operator.hub_sync.identity import LinkIdentity, link_origin

# The publisher tenant shape the live rows carried; synthetic local/hub ids.
PUBLISHER = "5b72d9f4-8ce7-4821-ac8a-f92c365ed5b3"
LOCAL = "b46e458e-0000-4000-8000-000000000001"
HUB = "01d2117f-0000-4000-8000-000000000001"

TEAM_DOC = {
    "id": HUB,
    "tenant_id": PUBLISHER,
    "name": "software-development",
    "description": "d",
    "manager": "manager",
    "members": [],
    "instructions": "i",
    "project": "",
}


class FakeHub:
    """A loopback hub whose answer per path is set by the test, recording headers."""

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, Any, dict[str, str]]] = {}
        self.seen: list[tuple[str, bool]] = []
        self.delay = 0.0
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                hub.seen.append((self.path, self.headers.get("Authorization") is not None))
                if hub.delay:
                    time.sleep(hub.delay)
                status, body, headers = hub.routes.get(self.path, (404, {"msg": "nf"}, {}))
                raw = json.dumps(body).encode()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Joined on close, so a handler still sleeping past the client's
        # timeout (the timeout case) ends with its test instead of leaking.
        self.server.daemon_threads = False
        self.base = f"http://127.0.0.1:{self.server.server_port}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def team(self, status: int = 200, document: Any = None, **headers: str) -> None:
        body = {"result": document if document is not None else TEAM_DOC}
        self.routes[f"/v1/teams/{HUB}"] = (status, body, headers)


@pytest.fixture
def hub() -> Iterator[FakeHub]:
    fake = FakeHub()
    fake.thread.start()
    try:
        yield fake
    finally:
        fake.server.shutdown()
        fake.server.server_close()
        fake.thread.join()


def _team() -> Any:
    return SimpleNamespace(
        id=LOCAL,
        name="software-development",
        description="d",
        manager="manager",
        members=[],
        instructions="i",
        project="",
    )


def _ctx(tmp_path: Path, base: str, *, for_tenant: Any = None) -> Any:
    cm = ConfigManager(tmp_path)
    cm.set_config_value("radient_base_url", base)
    team = _team()
    registry = SimpleNamespace(
        config_dir=cm.config_dir, list_teams=lambda: [team], get_team=lambda _id: team
    )
    if for_tenant is None:
        # THE PRODUCTION FACTORY, signed out: a key-less public client for
        # ``None`` and no organization client at all.
        for_tenant, _ = svc._clients(base, None)
    return svc.HubSyncContext(
        config_dir=cm.config_dir,
        config_manager=cm,
        client_for_tenant=for_tenant,
        team_registry=registry,
    )


def _legacy(ctx: Any, *, scope_known: bool = True) -> Path:
    """A baseline shaped like the live b46e458e row: pull, publisher tenant, no identity."""

    record = prov.record_team_pull(
        ctx.config_dir, _team(), TEAM_DOC, tenant_id=PUBLISHER, scope_known=scope_known
    )
    assert record is not None and record.identity is None and record.tenant_id == PUBLISHER
    path = prov.baseline_path(ctx.config_dir, "team", LOCAL)
    # Lineage, revision and a hand-set fingerprint/payload the repair must keep.
    raw = json.loads(path.read_text())
    raw.update(lineage="unknown", remote_revision='"rev-7"', fingerprint="sha256:kept")
    raw["fields"]["description"] = "merge-base text"
    if not scope_known:
        raw.pop("scope_known")  # the 10e73ee3 shape: the key is absent
    path.write_text(json.dumps(raw))
    return path


def test_anonymous_200_repairs_the_legacy_team_to_public(tmp_path, hub, caplog):
    hub.team()
    ctx = _ctx(tmp_path, hub.base)
    path = _legacy(ctx)
    before = json.loads(path.read_text())
    assert svc.links_snapshot(ctx)["links"] == []  # unresolved before

    caplog.set_level("INFO", logger="local_operator.hub_sync.service")
    out = svc.check_items(ctx, kinds=("team",))

    after = json.loads(path.read_text())
    source = prov.canonical_hub_source(ctx.config_manager)
    assert source is not None
    expected = LinkIdentity("team", LOCAL, source, "public", None, HUB)
    assert after["identity"] == expected.to_json()
    assert after["tenant_id"] is None and after["scope_known"] is True
    # A repair is not an integration: everything but scope/identity is kept.
    for key in ("fields", "fingerprint", "remote_revision", "lineage", "recorded_by", "outbound"):
        assert after[key] == before[key], key
    # Same identity a fresh public pull mints for this row.
    fresh = prov.link_identity("team", LOCAL, HUB, origin=link_origin(source, "public"))
    assert fresh is not None and after["identity"]["link_id"] == fresh.link_id
    # Probe + one re-read under the repaired scope, neither with a credential.
    assert hub.seen == [(f"/v1/teams/{HUB}", False), (f"/v1/teams/{HUB}", False)]
    assert out[0].tenant_id is None and out[0].verdict != "unavailable"
    row = svc.links_snapshot(ctx)["links"][0]
    assert row["visibility"] == "public" and row["tenant_id"] is None
    assert row["pull"]["state"] == "idle", row["pull"]
    lines = [r.getMessage() for r in caplog.records if "repaired legacy hub scope" in r.message]
    assert lines == [f"repaired legacy hub scope to public for team {LOCAL} (hub {HUB})"]
    assert PUBLISHER not in " ".join(r.getMessage() for r in caplog.records)


def test_absent_scope_known_key_is_repaired_too(tmp_path, hub):
    hub.team()
    ctx = _ctx(tmp_path, hub.base)
    path = _legacy(ctx, scope_known=False)
    svc.check_items(ctx, kinds=("team",))
    after = json.loads(path.read_text())
    assert after["identity"]["visibility"] == "public" and after["tenant_id"] is None


@pytest.mark.parametrize(
    "answer",
    ["404", "503", "mismatched-id", "redirect", "timeout", "refused"],
)
def test_unproven_answers_change_nothing(tmp_path, hub, monkeypatch, answer):
    base = hub.base
    if answer == "404":
        hub.team(404, {"msg": "team_not_found"})
    elif answer == "503":
        hub.team(503, {"msg": "down"})
    elif answer == "mismatched-id":
        hub.team(document={**TEAM_DOC, "id": "another-hub-id"})
    elif answer == "redirect":
        # Points at a path that WOULD answer 200: following it would "prove" it.
        hub.routes["/v1/elsewhere"] = (200, {"result": TEAM_DOC}, {})
        hub.team(302, {}, Location=f"{hub.base}/elsewhere")
    elif answer == "timeout":
        hub.team()
        hub.delay = 0.6
        monkeypatch.setattr(chk, "REQUEST_TIMEOUT_S", 0.2)
    else:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = f"http://127.0.0.1:{probe.getsockname()[1]}/v1"
    ctx = _ctx(tmp_path, base)
    path = _legacy(ctx)
    before = path.read_bytes()

    svc.check_items(ctx, kinds=("team",))

    assert path.read_bytes() == before
    assert svc.links_snapshot(ctx)["links"] == []
    assert "/v1/elsewhere" not in [p for p, _ in hub.seen]
    assert all(not auth for _, auth in hub.seen)


def test_identified_row_is_never_probed_or_touched(tmp_path, hub):
    hub.team()
    ctx = _ctx(tmp_path, hub.base)
    source = prov.canonical_hub_source(ctx.config_manager)
    prov.record_team_pull(
        ctx.config_dir,
        _team(),
        TEAM_DOC,
        tenant_id=PUBLISHER,
        origin=link_origin(source, "org", PUBLISHER),
    )
    path = prov.baseline_path(ctx.config_dir, "team", LOCAL)
    before = path.read_bytes()
    check = ItemCheck("team", LOCAL, "crew", HUB, PUBLISHER, "up-to-date", served_by=source)

    assert svc.repair_proven_public_scope(ctx, check, source=source) is None
    assert path.read_bytes() == before
    assert hub.seen == []  # an identified row costs no request


def test_probe_sends_no_credential_even_from_a_keyed_client(tmp_path, hub):
    """``with_credential=False`` is what keeps the bearer off, not the client choice."""

    hub.team()
    keyed = RadientClient(api_key=SecretStr("probe-must-not-send-this"), base_url=hub.base)
    ctx = _ctx(tmp_path, hub.base, for_tenant=lambda tenant: keyed if tenant is None else None)
    _legacy(ctx)
    svc.check_items(ctx, kinds=("team",))
    assert hub.seen and all(not auth for _, auth in hub.seen), hub.seen


def test_repaired_row_cannot_publish_without_destination_consent(tmp_path, hub):
    hub.team()
    ctx = _ctx(tmp_path, hub.base)
    _legacy(ctx)
    svc.check_items(ctx, kinds=("team",))
    row = svc.links_snapshot(ctx)["links"][0]
    assert row["policy"]["auto_push"] is False
    assert row["eligibility"]["can_auto_push"] is False
    values = ctx.config_manager.get_config().values
    assert row["link_id"] not in (values.get("hub", {}) or {}).get("public_egress_grants", {})
    assert row["push"]["state"] != "pushing"


def test_writer_refuses_a_rescope_in_any_other_shape(tmp_path):
    """The writer's carve-out is one-directional: never toward an organization."""

    ctx = _ctx(tmp_path, "https://hub.test/v1")
    _legacy(ctx)
    record = prov.read_baseline(ctx.config_dir, "team", LOCAL)
    assert record is not None
    org = prov.link_identity(
        "team", LOCAL, HUB, origin=link_origin("https://hub.test/v1", "org", "other-tenant")
    )
    with pytest.raises(prov.LinkError):
        prov.write_baseline(
            ctx.config_dir,
            prov.replace(record, tenant_id="other-tenant", identity=org),
            proven_public_rescope=True,
        )


def test_agent_repair_reads_the_envelope_and_refuses_redirects(tmp_path, hub):
    ctx = _ctx(tmp_path, hub.base)
    source = prov.canonical_hub_source(ctx.config_manager)
    prov.write_baseline(
        ctx.config_dir,
        prov.make_record(
            "agent",
            "agent-local",
            "agent-hub",
            PUBLISHER,
            {"instructions": "x", "description": ""},
            "pull",
        ),
    )
    path = prov.baseline_path(ctx.config_dir, "agent", "agent-local")
    check = ItemCheck("agent", "agent-local", "a", "agent-hub", PUBLISHER, "up-to-date")

    hub.routes["/v1/agents/agent-hub"] = (302, {}, {"Location": f"{hub.base}/agents/x"})
    hub.routes["/v1/agents/x"] = (200, {"result": {"id": "agent-hub"}}, {})
    before = path.read_bytes()
    assert svc.repair_proven_public_scope(ctx, check, source=source) is None
    assert path.read_bytes() == before

    hub.routes["/v1/agents/agent-hub"] = (200, {"result": {"id": "agent-hub"}}, {})
    repaired = svc.repair_proven_public_scope(ctx, check, source=source)
    assert repaired is not None and repaired.tenant_id is None
    record = prov.read_baseline(ctx.config_dir, "agent", "agent-local")
    assert record is not None and record.identity is not None
    assert record.identity.visibility == "public" and record.tenant_id is None
    assert ("/v1/agents/x", False) not in hub.seen
    assert all(not auth for _, auth in hub.seen)
