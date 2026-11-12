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
        # The Authorization VALUE per request (``seen`` keeps only presence),
        # so a test can tell netrc's ``Basic`` from the org bearer.
        self.auth: list[str | None] = []
        self.delay = 0.0
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                hub.seen.append((self.path, self.headers.get("Authorization") is not None))
                hub.auth.append(self.headers.get("Authorization"))
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


# -- review round 1 remediation ------------------------------------------------------------


def _netrc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``requests`` at a netrc with an entry for the loopback hub's host."""

    path = tmp_path / "netrc"
    path.write_text("machine 127.0.0.1 login probe password must-not-be-sent\n")
    path.chmod(0o600)
    monkeypatch.setenv("NETRC", str(path))


def test_probe_ignores_an_ambient_netrc_entry(tmp_path, hub, monkeypatch):
    """R1: "no credential, ever" includes the one ``requests`` reads from netrc.

    The NEGATIVE CONTROL comes first: a plain ``requests.get`` under the same
    ``NETRC`` does send Basic auth to this hub, so the rig can see the leak the
    probe must not have.
    """

    import requests

    _netrc(tmp_path, monkeypatch)
    hub.team()
    requests.get(f"{hub.base}/teams/{HUB}", timeout=5)
    assert hub.seen == [(f"/v1/teams/{HUB}", True)], "control: netrc must reach a plain GET"
    hub.seen.clear()

    ctx = _ctx(tmp_path, hub.base)
    path = _legacy(ctx)
    svc.check_items(ctx, kinds=("team",))
    assert json.loads(path.read_text())["identity"]["visibility"] == "public"
    # The FIRST GET is the probe -- the request whose 200 is read as proof.
    # The second is the ordinary public check re-reading the repaired row; a
    # public read never carries a credential either (Q2-netrc(b)).
    assert hub.seen == [(f"/v1/teams/{HUB}", False)] * 2, hub.seen

    # The agent arm of the same probe.
    hub.seen.clear()
    source = prov.canonical_hub_source(ctx.config_manager)
    prov.write_baseline(
        ctx.config_dir,
        prov.make_record(
            "agent", "agent-local", "agent-hub", PUBLISHER, {"instructions": "x"}, "pull"
        ),
    )
    hub.routes["/v1/agents/agent-hub"] = (200, {"result": {"id": "agent-hub"}}, {})
    check = ItemCheck("agent", "agent-local", "a", "agent-hub", PUBLISHER, "up-to-date")
    assert svc.repair_proven_public_scope(ctx, check, source=source) is not None
    assert hub.seen[0] == ("/v1/agents/agent-hub", False), hub.seen


def test_credential_free_refuses_a_bearer_request(tmp_path):
    client = RadientClient(api_key=None, base_url="http://127.0.0.1:9/v1")
    with pytest.raises(ValueError):
        client.get_team(HUB, credential_free=True)  # the default sends the bearer
    with pytest.raises(ValueError):
        client.get_agent(HUB, credential_free=True, **{"with_" + "credential": True})


def test_credential_free_agent_read_never_follows_a_redirect(tmp_path, hub, monkeypatch):
    """A followed redirect re-reads netrc for its target, so credential_free refuses it."""

    _netrc(tmp_path, monkeypatch)
    hub.routes["/v1/agents/agent-hub"] = (302, {}, {"Location": f"{hub.base}/agents/x"})
    hub.routes["/v1/agents/x"] = (200, {"result": {"id": "agent-hub"}}, {})
    client = RadientClient(api_key=None, base_url=hub.base)
    with pytest.raises(RuntimeError):
        client.get_agent("agent-hub", credential_free=True)
    assert hub.seen == [("/v1/agents/agent-hub", False)]


def _probes(ctx: Any) -> dict[str, Any]:
    return ctx.store().load().get(svc.PUBLIC_PROBE_STORE_KEY) or {}


def test_a_404_is_probed_at_most_once_a_day_and_a_later_200_still_repairs(tmp_path, hub):
    """R2: a genuine org row (404 to everyone outside it) stops costing a GET per check."""

    hub.team(404, {"msg": "team_not_found"})
    ctx = _ctx(tmp_path, hub.base)
    path = _legacy(ctx)
    before = path.read_bytes()
    probe = (f"/v1/teams/{HUB}", False)

    svc.check_items(ctx, kinds=("team",))
    svc.check_items(ctx, kinds=("team",))
    svc.check_items(ctx, kinds=("team",))
    assert hub.seen == [probe]  # one probe, not one per check
    assert path.read_bytes() == before
    source = prov.canonical_hub_source(ctx.config_manager)
    entry = _probes(ctx)[f"team:{HUB}"]
    assert entry["source"] == source and PUBLISHER not in json.dumps(entry)

    # The publisher makes the row public; once the backoff lapses, a check repairs it.
    hub.team()
    aged = (svc.datetime.now(svc.timezone.utc) - svc.PUBLIC_PROBE_BACKOFF).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    ctx.store().mutate(
        lambda doc: doc[svc.PUBLIC_PROBE_STORE_KEY][f"team:{HUB}"].update(not_public_at=aged)
    )
    svc.check_items(ctx, kinds=("team",))
    assert json.loads(path.read_text())["identity"]["visibility"] == "public"
    assert _probes(ctx) == {}  # the stale 404 stamp is cleared by the repair


def test_a_404_stamp_from_another_hub_source_does_not_suppress_the_probe(tmp_path, hub):
    hub.team()
    ctx = _ctx(tmp_path, hub.base)
    path = _legacy(ctx)
    ctx.store().mutate(
        lambda doc: doc.setdefault(svc.PUBLIC_PROBE_STORE_KEY, {}).update(
            {f"team:{HUB}": {"source": "https://other.test/v1", "not_public_at": prov.now_iso()}}
        )
    )
    svc.check_items(ctx, kinds=("team",))
    assert json.loads(path.read_text())["identity"]["visibility"] == "public"


@pytest.mark.parametrize("answer", ["503", "mismatched-id"])
def test_unproven_answers_are_not_remembered(tmp_path, hub, answer):
    """Only a 404 backs off: a passing outage must not delay the repair by a day."""

    if answer == "503":
        hub.team(503, {"msg": "down"})
    else:
        hub.team(document={**TEAM_DOC, "id": "another-hub-id"})
    ctx = _ctx(tmp_path, hub.base)
    _legacy(ctx)
    svc.check_items(ctx, kinds=("team",))
    svc.check_items(ctx, kinds=("team",))
    assert len(hub.seen) == 2 and _probes(ctx) == {}


def test_writer_refuses_a_rescope_with_no_current_record(tmp_path):
    """R3: a prune racing the repair must not have its deletion undone."""

    ctx = _ctx(tmp_path, "https://hub.test/v1")
    path = _legacy(ctx)
    record = prov.read_baseline(ctx.config_dir, "team", LOCAL)
    assert record is not None
    public = prov.link_identity(
        "team", LOCAL, HUB, origin=link_origin("https://hub.test/v1", "public", None)
    )
    repaired = prov.replace(record, tenant_id=None, scope_known=True, identity=public)
    path.unlink()  # the prune landed while the probe was in flight

    with pytest.raises(prov.LinkError) as refused:
        prov.write_baseline(ctx.config_dir, repaired, proven_public_rescope=True)
    assert refused.value.code == "origin-conflict"
    assert not path.exists()


def test_operator_ask_repaired_public_team_keeps_syncing(tmp_path, hub):
    """R4: the ask -- "make sure that public hub syncs work properly" -- end to end.

    repair -> a later check costs exactly one GET -> a new hub revision is
    ``available`` -> the unattended apply merges and writes it, on a link that
    is subscribed by the creation default (no stored policy).
    """

    from local_operator.teams import TeamRegistry

    ctx = _ctx(tmp_path, hub.base)
    teams = TeamRegistry(ctx.config_dir)
    doc = {**TEAM_DOC, "instructions": "## Brief\nShip it.", "version": "1.0.0"}
    hub.team(document=doc)
    team = teams.import_hub_team(doc).team
    prov.record_team_pull(
        ctx.config_dir, teams.get_team(team.id), doc, tenant_id=PUBLISHER, scope_known=True
    )
    ctx = svc.HubSyncContext(
        config_dir=ctx.config_dir,
        config_manager=ctx.config_manager,
        client_for_tenant=ctx.client_for_tenant,
        team_registry=teams,
    )

    svc.check_items(ctx, kinds=("team",))  # the repair
    row = svc.links_snapshot(ctx)["links"][0]
    assert row["visibility"] == "public" and row["tenant_id"] is None
    assert row["policy"]["subscribe"] is True and row["policy"]["auto_push"] is False

    hub.seen.clear()
    (later,) = svc.check_items(ctx, kinds=("team",))
    assert later.verdict == "up-to-date"
    assert hub.seen == [(f"/v1/teams/{HUB}", False)]  # exactly one GET, no credential

    hub.team(document={**doc, "instructions": "## Brief\nShip it carefully.", "version": "1.1.0"})
    (available,) = svc.check_items(ctx, kinds=("team",))
    assert available.verdict == "available"
    report = svc.apply_one(ctx, available, auto=True, allow_llm=False)
    assert report.outcome == "merged" and report.applied, report
    assert teams.get_team(team.id).instructions.strip() == "## Brief\nShip it carefully."
    record = prov.read_baseline(ctx.config_dir, "team", team.id)
    assert record is not None and record.identity is not None
    assert record.identity.visibility == "public" and record.tenant_id is None


# --- Q2-netrc(b): EVERY public hub-sync read is credential-free ----------------
#
# WHY. #3813 hardened only the repair PROBE. ``requests`` attaches an ambient
# ``~/.netrc`` entry for the hub's host as HTTP Basic auth to any request that
# passes no ``auth`` -- so the post-repair re-read and every ordinary update
# check of a PUBLIC team/agent link leaked that secret to the hub, and their
# answer could be a signed-in one recorded as the public one. These tests put a
# real ``~/.netrc`` in an isolated HOME (the file ``requests`` reads when NETRC
# is unset) and read the Authorization header the loopback hub actually
# received -- never whether a keyword was passed. Org links must keep their
# bearer: an org row is 404 to anyone who cannot prove membership.

ORG = "9c1e0d2a-0000-4000-8000-0000000000aa"
AGENT_LOCAL = "agent-local"
AGENT_HUB = "agent-hub"


def _home_netrc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An isolated HOME whose ``~/.netrc`` has an entry for the loopback hub."""

    home = tmp_path / "home"
    home.mkdir()
    netrc = home / ".netrc"
    netrc.write_text("machine 127.0.0.1 login probe password netrc-secret\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NETRC", raising=False)


def _netrc_control(hub: FakeHub) -> None:
    """NEGATIVE CONTROL: a plain GET under this HOME does send netrc's Basic auth."""

    import requests

    requests.get(f"{hub.base}/teams/{HUB}", timeout=5)
    assert hub.auth and str(hub.auth[0]).startswith("Basic "), hub.auth
    hub.seen.clear()
    hub.auth.clear()


def _agent_ctx(tmp_path: Path, hub: FakeHub, *, tenant: str | None, for_tenant: Any = None) -> Any:
    """A linked hub agent (``hub:`` marker + baseline) in ``tenant``'s scope."""

    ctx = _ctx(tmp_path, hub.base, for_tenant=for_tenant)
    agent = SimpleNamespace(id=AGENT_LOCAL, name="a", description="d", tags=[f"hub:{AGENT_HUB}"])
    registry = SimpleNamespace(
        config_dir=ctx.config_dir,
        list_agents=lambda: [agent],
        get_agent=lambda _id: agent,
        get_agent_system_prompt=lambda _id: "x",
    )
    prov.write_baseline(
        ctx.config_dir,
        prov.make_record(
            "agent",
            AGENT_LOCAL,
            AGENT_HUB,
            tenant,
            {"instructions": "x", "description": "d"},
            "pull",
        ),
    )
    hub.routes[f"/v1/agents/{AGENT_HUB}"] = (
        200,
        {"result": {"id": AGENT_HUB, "instructions": "x", "description": "d"}},
        {},
    )
    return SimpleNamespace(ctx=ctx, registry=registry)


def _org_clients(base: str) -> Any:
    """Signed in: a keyed org client for ORG, the key-less public client otherwise."""

    keyed = RadientClient(api_key=SecretStr("org-bearer"), base_url=base)
    public = RadientClient(api_key=None, base_url=base)
    return lambda tenant: keyed if tenant == ORG else public


def test_public_team_update_check_sends_no_netrc_credential(tmp_path, hub, monkeypatch):
    _home_netrc(tmp_path, monkeypatch)
    hub.team()
    _netrc_control(hub)
    ctx = _ctx(tmp_path, hub.base)
    record = prov.record_team_pull(
        ctx.config_dir, _team(), TEAM_DOC, tenant_id=None, scope_known=True
    )
    assert record is not None and record.tenant_id is None

    (check,) = chk.check_hub_teams(ctx.teams(), client_for_tenant=ctx.client_for_tenant)

    assert check.verdict == "up-to-date", check
    assert hub.seen == [(f"/v1/teams/{HUB}", False)] and hub.auth == [None], hub.auth


def test_public_agent_update_check_sends_no_netrc_credential(tmp_path, hub, monkeypatch):
    _home_netrc(tmp_path, monkeypatch)
    hub.team()
    _netrc_control(hub)
    linked = _agent_ctx(tmp_path, hub, tenant=None)

    (check,) = chk.check_hub_agents(linked.registry, client_for_tenant=linked.ctx.client_for_tenant)

    assert check.verdict == "up-to-date", check
    # One detail read, no credential -- and no zip fallback (it would be a second GET).
    assert hub.seen == [(f"/v1/agents/{AGENT_HUB}", False)] and hub.auth == [None], hub.auth


def test_post_repair_reread_sends_no_netrc_credential(tmp_path, hub, monkeypatch):
    """The re-read right after a scope repair is a public read too."""

    _home_netrc(tmp_path, monkeypatch)
    hub.team()
    _netrc_control(hub)
    ctx = _ctx(tmp_path, hub.base)
    path = _legacy(ctx)

    svc.check_items(ctx, kinds=("team",))

    assert json.loads(path.read_text())["identity"]["visibility"] == "public"
    # probe, then the re-read of the repaired row: both bare.
    assert hub.seen == [(f"/v1/teams/{HUB}", False)] * 2, hub.seen
    assert hub.auth == [None, None], hub.auth


def test_org_reads_still_send_their_bearer(tmp_path, hub, monkeypatch):
    """Org links keep their credential: the org bearer reaches the hub.

    Isolated HOME WITHOUT a netrc: ``requests`` lets a matching netrc entry
    REPLACE an explicit ``Authorization`` header, so with one present an org
    read sends netrc's Basic instead of the bearer. That is pre-existing org
    behaviour this change deliberately leaves alone; here the point is only
    that the public-read fix did not strip the org credential.
    """

    home = tmp_path / "bare-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NETRC", raising=False)
    hub.team(document={**TEAM_DOC, "tenant_id": ORG})
    ctx = _ctx(tmp_path, hub.base, for_tenant=_org_clients(hub.base))
    prov.record_team_pull(
        ctx.config_dir, _team(), {**TEAM_DOC, "tenant_id": ORG}, tenant_id=ORG, scope_known=True
    )
    chk.check_hub_teams(ctx.teams(), client_for_tenant=ctx.client_for_tenant)
    assert hub.seen == [(f"/v1/teams/{HUB}", True)], hub.seen
    assert hub.auth == ["Bearer org-bearer"], hub.auth

    hub.seen.clear()
    hub.auth.clear()
    linked = _agent_ctx(tmp_path, hub, tenant=ORG, for_tenant=_org_clients(hub.base))
    chk.check_hub_agents(linked.registry, client_for_tenant=linked.ctx.client_for_tenant)
    assert hub.seen[0] == (f"/v1/agents/{AGENT_HUB}", True), hub.seen
    assert hub.auth[0] == "Bearer org-bearer", hub.auth


# --- Round-1 R1/Q-m1: the agent ZIP FALLBACK and SKILL reads are public reads too --
#
# WHY. A public agent check falls back to the zip download whenever its detail
# read fails (5xx, timeout, a refused 3xx) or carries no ``instructions``. That
# download used to be ambient -- netrc's Basic attached, redirects followed with
# netrc re-read for the target -- so a degraded hub moved the leak instead of
# closing it, and a 200 there would record a signed-in answer as the public one.
# The public skill check had the same exposure. Every zip route below answers a
# 302 to a URL that WOULD answer 200: a followed redirect shows up in ``seen``.

AGENT_DETAIL = f"/v1/agents/{AGENT_HUB}"
AGENT_ZIP = f"/v1/agents/{AGENT_HUB}/download"


def _zip_redirects(hub: FakeHub) -> None:
    hub.routes[AGENT_ZIP] = (302, {}, {"Location": f"{hub.base}/elsewhere-zip"})
    hub.routes["/v1/elsewhere-zip"] = (200, {}, {})


@pytest.mark.parametrize(
    "detail",
    [
        pytest.param((500, {"msg": "boom"}, {}), id="detail-500"),
        pytest.param((302, {}, {"Location": "/v1/elsewhere-agent"}), id="detail-302"),
        pytest.param((200, {"result": {"id": AGENT_HUB}}, {}), id="detail-no-instructions"),
    ],
)
def test_public_agent_zip_fallback_sends_no_netrc_credential(tmp_path, hub, monkeypatch, detail):
    _home_netrc(tmp_path, monkeypatch)
    hub.team()
    _netrc_control(hub)
    linked = _agent_ctx(tmp_path, hub, tenant=None)
    hub.routes[AGENT_DETAIL] = detail
    hub.routes["/v1/elsewhere-agent"] = (
        200,
        {"result": {"id": AGENT_HUB, "instructions": "x"}},
        {},
    )
    _zip_redirects(hub)

    (check,) = chk.check_hub_agents(linked.registry, client_for_tenant=linked.ctx.client_for_tenant)

    # Detail read, then the zip fallback -- both bare, neither redirect followed.
    assert hub.seen == [(AGENT_DETAIL, False), (AGENT_ZIP, False)], hub.seen
    assert hub.auth == [None, None], hub.auth
    # A refused redirect is an ordinary failed check, never a crash.
    assert check.verdict == "unavailable", check


def test_post_repair_agent_reread_sends_no_netrc_credential(tmp_path, hub, monkeypatch):
    """The agent twin of the team re-read, through the zip fallback.

    The detail payload carries no ``instructions`` -- the shape that falls to
    the zip download -- so the re-read's second GET is the fallback itself.
    """

    import dataclasses

    _home_netrc(tmp_path, monkeypatch)
    hub.team()
    _netrc_control(hub)
    linked = _agent_ctx(tmp_path, hub, tenant=PUBLISHER)  # the legacy shape: tenant, no identity
    ctx = dataclasses.replace(linked.ctx, agent_registry=linked.registry)
    hub.routes[AGENT_DETAIL] = (200, {"result": {"id": AGENT_HUB}}, {})
    _zip_redirects(hub)
    check = ItemCheck("agent", AGENT_LOCAL, "a", AGENT_HUB, PUBLISHER, "up-to-date")

    again = svc.repair_proven_public_scope(
        ctx, check, source=prov.canonical_hub_source(ctx.config_manager)
    )

    assert again is not None and again.tenant_id is None, again
    # probe, re-read detail, re-read zip fallback: all bare, no redirect followed.
    assert hub.seen == [(AGENT_DETAIL, False), (AGENT_DETAIL, False), (AGENT_ZIP, False)], hub.seen
    assert hub.auth == [None, None, None], hub.auth


def test_public_skill_update_check_sends_no_netrc_credential(tmp_path, monkeypatch):
    """A PUBLIC skill link (tenant-free baseline, the production signed-out factory)."""

    from .test_skill_conditional_check import StubHub, _profile
    from .test_skill_storage import _input

    _home_netrc(tmp_path, monkeypatch)
    with StubHub(_input()[1]) as stub:
        import requests

        requests.get(f"{stub.base_url}/skills/sk-1", timeout=5)  # NEGATIVE CONTROL
        assert str(stub.requests[0].get("Authorization", "")).startswith("Basic "), stub.requests
        stub.requests.clear()
        _identity, root, _ctx_ = _profile(tmp_path, stub)
        public, _ = svc._clients(stub.base_url, None)

        (check,) = chk.check_hub_skills(config_dir=root, client_for_tenant=public)

    assert check.tenant_id is None and check.verdict == "up-to-date", check
    assert len(stub.requests) == 1 and "Authorization" not in stub.requests[0], stub.requests


def test_credential_free_download_and_skill_refuse_a_bearer_request():
    client = RadientClient(api_key=None, base_url="http://127.0.0.1:9/v1")
    bearer = {"with_" + "credential": True}
    with pytest.raises(ValueError):
        client.download_agent_from_marketplace(
            AGENT_HUB, Path("unused.zip"), credential_free=True, **bearer
        )
    with pytest.raises(ValueError):
        client.get_skill(AGENT_HUB, credential_free=True, **bearer)
