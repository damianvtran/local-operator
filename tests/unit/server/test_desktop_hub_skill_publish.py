"""The skills-publish route: preview receipt, confirm, refusals, reconciliation.

Driven against the REAL route stack (FastAPI + ASGI transport + the receipts and
error wrappers), the real ``hub_sync`` publication service, and the REAL
``ManagedSkillStorage`` on an isolated HOME/config root. Only the credentialed
client picker is faked, so the wire boundary is the one under test and every
assertion reads a PERSISTED effect (files, baseline, policy, inventory), never a
call count of the subject's own helpers.

The contract pinned here is the desktop's ``hub.skillPublish`` arm: the preview
object field-for-field, dry-run/confirm pairing, the reviewed-bytes freshness
checks, the hub-before-local ordering, and the real ``links_snapshot`` row the
successful answer carries.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.clients._http import APIError
from local_operator.config import ConfigManager, read_config_values
from local_operator.env import EnvConfig
from local_operator.hub_sync import policies as pol
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.identity import LinkIdentity
from local_operator.server import features as svc_features
from local_operator.server.routes import desktop_hub, desktop_sessions

pytestmark = pytest.mark.asyncio

DESKTOP_TOKEN = "desktop-test-token"
AUTH = {"Authorization": f"Bearer {DESKTOP_TOKEN}"}
SOURCE = "alpha"
ALIAS = "alpha"


def skill_files(name: str, description: str = "A small authored skill.", body: str = "Body.\n"):
    return [
        {
            "path": "SKILL.md",
            "content": (
                "---\n"
                f"name: {name}\n"
                f"description: {description}\n"
                "---\n\n"
                f"# {name}\n\n{body}"
            ),
        },
        {"path": "references/extra.md", "content": "More detail.\n"},
    ]


def write_authored_skill(base: Path, name: str, **fields: Any) -> Path:
    directory = base / name
    directory.mkdir(parents=True)
    for row in skill_files(name, **fields):
        path = directory / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(row["content"])
    return directory


def manifest(name: str, description: str = "A small authored skill.", body: str = "Body.\n"):
    return {
        "document_version": 1,
        "name": name,
        "description": description,
        "files": skill_files(name, description=description, body=body),
    }


class PublishClient:
    """The publish seam: records calls, returns one result or raises one error."""

    def __init__(self, base_url: str, *, api_key: str | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.calls: list[dict[str, Any]] = []
        self.result: Any = None
        self.error: BaseException | None = None

    def publish_skill_document(self, document: dict[str, Any], **kw: Any) -> Any:
        self.calls.append({"document": document, **kw})
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        name = str(document.get("name") or "")
        return {
            "skill": {"id": "sk-published", "name": name},
            "accepted_document": {**document, "id": "sk-published"},
            "revision": '"hub-rev-1"',
            "content_revision": 1,
            "operation_id": "op-published",
            "sync_contract": {"revision_cas": 1, "skills": 1},
        }


def publish_body(**fields: Any) -> dict[str, Any]:
    body = {
        "request_id": str(uuid.uuid4()),
        "dry_run": True,
        "source_skill": SOURCE,
        "local_alias": ALIAS,
        "visibility": "public",
        "tenant_id": None,
        "subscribe": True,
        "auto_push": False,
        "confirm_public_egress": False,
    }
    body.update(fields)
    return body


def confirm_body(preview: dict[str, Any], **fields: Any) -> dict[str, Any]:
    receipt = (preview.get("preview") or preview)["preview_receipt"]
    body = publish_body(
        dry_run=False,
        preview_receipt=receipt,
        confirm_publish=True,
        **fields,
    )
    return body


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    holder: dict[str, Any] = {}

    async def fake_clients(_cm: Any, _store: Any = None):
        def for_tenant(tenant: Any) -> Any:
            # None = the anonymous public client; any string (including "") = the
            # signed-in client, which is the one a publication may spend.
            return holder.get("client" if tenant is not None else "anon")

        return for_tenant, "ok"

    monkeypatch.setattr("local_operator.hub_sync.service.build_clients", fake_clients)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.env_config = EnvConfig()
    app.include_router(desktop_hub.router)
    app.include_router(desktop_sessions.router)
    app.dependency_overrides[desktop_hub.get_provider_auth_store] = lambda: None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=AUTH
    ) as client:
        yield client, root, holder
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def base_url(root: Path) -> str:
    from local_operator.providers.radient_credentials import configured_radient_base_url

    return configured_radient_base_url(ConfigManager(root))


def linked_identity(row: dict[str, Any]) -> LinkIdentity:
    return LinkIdentity(
        "skill",
        row["local_id"],
        row["hub_source"],
        row["visibility"],
        row["tenant_id"],
        row["hub_id"],
    )


async def preview(client: AsyncClient, **fields: Any) -> dict[str, Any]:
    response = await client.post("/v1/desktop/hub/skills/publish", json=publish_body(**fields))
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def test_a_dry_run_returns_the_pinned_preview_and_writes_no_link_state(api) -> None:
    client, root, holder = api
    source = write_authored_skill(root / "skills", SOURCE)
    holder["client"] = PublishClient(base_url(root))
    before = read_config_values(root)

    result = await preview(client)

    assert result["link"] is None and result["local_link_receipt"] is None
    assert result["publication"] is None
    row = result["preview"]
    assert row["kind"] == "skill" and row["direction"] == "push"
    assert row["publication"] == "create"
    assert row["source_skill"] == SOURCE and row["local_alias"] == ALIAS
    assert row["visibility"] == "public" and row["tenant_id"] is None
    assert row["hub_source"] == base_url(root)
    assert row["policy"] == {
        "subscribe": True,
        "auto_push": False,
        "confirm_public_egress": False,
    }
    assert row["proposal"] == {"kind": "skill", "fields": manifest(ALIAS)}
    # Fingerprints are real and distinct facts: the reviewed source bytes, the
    # alias-rendered draft (equal here) and the proposal wrapper.
    assert len(row["local_fingerprint"]) == 64
    assert row["draft_fingerprint"] == row["local_fingerprint"]
    assert len(row["proposal_fingerprint"]) == 64
    assert row["proposal_fingerprint"] not in ("", row["draft_fingerprint"])
    assert row["can_confirm"] is True and row["reason"] is None
    assert row["warnings"] == []
    # No hub call, no baseline, no policy, no inventory row, no config write.
    assert holder["client"].calls == []
    assert list(prov.iter_baselines(root)) == []
    assert read_config_values(root) == before
    assert (source / "SKILL.md").exists()
    snapshot = await client.get("/v1/desktop/hub/links", params={"kind": "skill"})
    assert snapshot.status_code == 200, snapshot.text
    assert snapshot.json()["result"]["links"] == []


@pytest.mark.parametrize(
    "fields",
    [
        {"visibility": "public", "tenant_id": "org-1"},
        {"visibility": "org", "tenant_id": None},
        {"dry_run": True, "preview_receipt": "x" * 16},
        {"dry_run": True, "confirm_publish": True},
        {"dry_run": False},
        {"dry_run": False, "preview_receipt": "x" * 16},
        {"dry_run": False, "confirm_publish": True},
    ],
)
async def test_the_reviewed_draft_pairing_is_refused_before_any_work(api, fields) -> None:
    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    holder["client"] = PublishClient(base_url(root))

    response = await client.post("/v1/desktop/hub/skills/publish", json=publish_body(**fields))
    assert response.status_code == 422, response.text
    assert holder["client"].calls == []
    assert list(prov.iter_baselines(root)) == []


async def test_confirm_publishes_then_materializes_the_real_inventory_row(api) -> None:
    client, root, holder = api
    source = write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root))
    holder["client"] = hub

    previewed = await preview(client)
    confirm = confirm_body(previewed)
    response = await client.post("/v1/desktop/hub/skills/publish", json=confirm)
    assert response.status_code == 200, response.text
    result = response.json()["result"]

    # ONE publication, with the exact reviewed manifest and the public target.
    (call,) = hub.calls
    assert call["document"] == manifest(ALIAS)
    assert call["visibility"] == "public" and call["tenant_id"] is None

    publication = result["publication"]
    assert publication["skill"] == {"id": "sk-published", "name": ALIAS}
    assert publication["accepted_document"] == manifest(ALIAS)
    assert publication["revision"] == '"hub-rev-1"'
    assert publication["content_revision"] == 1
    assert publication["operation_id"] == "op-published"
    assert publication["sync_contract"] == {"revision_cas": 1, "skills": 1}

    row = result["link"]
    receipt = result["local_link_receipt"]
    assert receipt["state"] == "complete"
    assert receipt["link_id"] == row["link_id"] and receipt["operation_id"]
    assert row["kind"] == "skill" and row["name"] == ALIAS
    assert row["hub_id"] == "sk-published"
    assert row["hub_source"] == base_url(root)
    assert row["baseline_state"] == "publish"

    # The answered row is a REAL inventory row: the same GET the sidebar reads
    # returns it, with the same policy revision.
    listed = await client.get("/v1/desktop/hub/links", params={"kind": "skill"})
    assert listed.status_code == 200, listed.text
    (inventory_row,) = listed.json()["result"]["links"]
    assert inventory_row == row

    # The authored folder was ADOPTED in place (same path, no byte rewrite).
    assert row["local_id"]
    assert (source / "SKILL.md").read_text() == manifest(ALIAS)["files"][0]["content"]
    # A published baseline records the hub's accepted canonical document, and
    # seeds the outbound receipt a later auto-push derives from.
    record = prov.read_baseline(root, "skill", row["local_id"])
    assert record is not None
    assert record.identity == linked_identity(row)
    assert record.recorded_by == "publish" and record.lineage == "clean"
    assert record.remote_revision == '"hub-rev-1"'
    assert record.local_alias == ALIAS
    assert dict(record.fields) == manifest(ALIAS)
    assert record.outbound is not None
    assert record.outbound.remote_revision == '"hub-rev-1"'
    assert record.outbound.accepted_fields == manifest(ALIAS)
    # Creation policy, materialized and readable through the same reader every
    # surface uses (the answer was reloaded, not a stale pre-write snapshot).
    policy = pol.read_policy(read_config_values(root) or {}, linked_identity(row))
    assert policy.policy.subscribe is True and policy.policy.auto_push is False
    assert row["policy"]["subscribe"] is True

    # The completed receipt is consumed with the reconcile (retention rule), so
    # the same REQUEST ID is the idempotent path: it replays the recorded answer
    # without a second publication.
    again = await client.post("/v1/desktop/hub/skills/publish", json=confirm)
    assert again.status_code == 200, again.text
    assert again.json()["result"].get("replayed") is True
    assert again.json()["result"]["link"]["link_id"] == row["link_id"]
    assert len(hub.calls) == 1

    # A NEW request id cannot start a second act from the consumed receipt.
    fresh = await client.post("/v1/desktop/hub/skills/publish", json=confirm_body(previewed))
    assert fresh.status_code == 409, fresh.text
    assert fresh.json()["detail"]["code"] == "preview_expired"
    assert len(hub.calls) == 1


async def test_an_org_publication_uses_the_signed_in_client_and_records_the_tenant(api) -> None:
    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root), api_key="org-key")
    holder["client"] = hub
    holder["anon"] = PublishClient(base_url(root))

    previewed = await preview(client, visibility="org", tenant_id="org-1")
    assert previewed["preview"]["visibility"] == "org"
    assert previewed["preview"]["tenant_id"] == "org-1"

    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, visibility="org", tenant_id="org-1"),
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["link"]["visibility"] == "org" and result["link"]["tenant_id"] == "org-1"
    assert len(hub.calls) == 1 and hub.calls[0]["tenant_id"] == "org-1"
    assert holder["anon"].calls == []
    record = prov.read_baseline(root, "skill", result["link"]["local_id"])
    assert record is not None and record.tenant_id == "org-1"


async def test_a_hub_refusal_leaves_local_skill_state_unchanged(api) -> None:
    client, root, holder = api
    # A copy-sourced skill: the destination does not exist before the confirm, so
    # "nothing was written" is observable as an absent managed directory too.
    write_authored_skill(Path(os.environ["HOME"]) / ".claude" / "skills", "beta")
    hub = PublishClient(base_url(root))
    hub.error = APIError("That name is taken.", status_code=422, code="name_taken")
    holder["client"] = hub
    before = read_config_values(root)

    previewed = await preview(client, source_skill="beta", local_alias="beta")
    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, source_skill="beta", local_alias="beta"),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "name_taken"
    assert len(hub.calls) == 1
    assert read_config_values(root) == before
    assert list(prov.iter_baselines(root)) == []
    assert not (root / "skills" / "beta").exists()
    listed = await client.get("/v1/desktop/hub/links", params={"kind": "skill"})
    assert listed.json()["result"]["links"] == []


async def test_an_expired_receipt_is_refused_without_a_hub_call(api) -> None:
    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    previewed = await preview(client)
    path = root / "hub" / "publication-previews" / f"{previewed['preview']['preview_receipt']}.json"
    record = json.loads(path.read_text())
    record["expires_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(record))

    response = await client.post("/v1/desktop/hub/skills/publish", json=confirm_body(previewed))
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "preview_expired"
    assert hub.calls == []
    assert list(prov.iter_baselines(root)) == []


async def test_an_edited_source_is_preview_stale_and_never_published(api) -> None:
    client, root, holder = api
    source = write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    previewed = await preview(client)
    (source / "references" / "extra.md").write_text("Edited after the review.\n")

    response = await client.post("/v1/desktop/hub/skills/publish", json=confirm_body(previewed))
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "preview_stale"
    assert hub.calls == []
    assert list(prov.iter_baselines(root)) == []


async def test_a_changed_draft_tuple_cannot_reuse_a_receipt(api) -> None:
    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    previewed = await preview(client, subscribe=True)

    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, subscribe=False),
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "preview_mismatch"
    assert hub.calls == []


async def test_a_renaming_copy_rewrites_the_managed_draft_and_keeps_the_source(api) -> None:
    client, root, holder = api
    home = Path(os.environ["HOME"])
    source = write_authored_skill(home / ".claude" / "skills", "beta")
    hub = PublishClient(base_url(root))
    hub.result = {
        "skill": {"id": "sk-gamma", "name": "gamma"},
        "accepted_document": manifest("gamma", body="Beat body.\n"),
        "revision": '"r-g"',
        "content_revision": 2,
        "operation_id": "op-g",
    }
    holder["client"] = hub

    previewed = await preview(client, source_skill="beta", local_alias="gamma")
    assert previewed["preview"]["warnings"] == [
        "This publication renames the skill from 'beta' to 'gamma'."
    ]
    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, source_skill="beta", local_alias="gamma"),
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]

    managed = root / "skills" / "gamma"
    assert (managed / "SKILL.md").exists()
    assert "name: gamma" in (managed / "SKILL.md").read_text()
    assert (managed / "references" / "extra.md").read_text() == "More detail.\n"
    # The authored source is untouched: publication copies, it never renames it.
    assert "name: beta" in (source / "SKILL.md").read_text()
    assert result["link"]["name"] == "gamma"
    assert hub.calls[0]["document"]["name"] == "gamma"
    token = previewed["preview"]["preview_receipt"]
    assert not (root / "hub" / "publication-previews" / f"{token}.json").exists()


async def test_a_local_failure_after_acceptance_is_pending_and_retry_reconciles(
    api, monkeypatch
) -> None:
    client, root, holder = api
    write_authored_skill(Path(os.environ["HOME"]) / ".claude" / "skills", "beta")
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    from local_operator.hub_sync import skill_storage

    real_apply = skill_storage.ManagedSkillStorage.apply

    attempts: list[int] = []

    def failing_once(self, *args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("synthetic local write failure")
        return real_apply(self, *args, **kwargs)

    monkeypatch.setattr(skill_storage.ManagedSkillStorage, "apply", failing_once)
    previewed = await preview(client, source_skill="beta", local_alias="beta")
    body = confirm_body(previewed, source_skill="beta", local_alias="beta")
    # THE CLIENT'S OWN RETRY SHAPE: the same preview, draft tuple and request id.
    # A recorded pending answer would replay forever and never reconcile, so the
    # route must leave this id unclaimed.
    body["request_id"] = "00000000-0000-4000-8000-000000000001"
    response = await client.post("/v1/desktop/hub/skills/publish", json=body)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["publication"]["skill"]["id"] == "sk-published"
    assert result["link"] is None
    assert result["local_link_receipt"]["state"] == "pending-reconciliation"
    assert list(prov.iter_baselines(root)) == []
    assert not (root / "skills" / "beta").exists()
    assert len(attempts) == 1

    # The SAME request id re-enters the service and resumes local
    # reconciliation without republishing.
    retry = await client.post("/v1/desktop/hub/skills/publish", json=body)
    assert retry.status_code == 200, retry.text
    assert retry.json()["result"].get("replayed") is not True
    retried = retry.json()["result"]
    assert retried["local_link_receipt"]["state"] == "complete"
    assert retried["link"]["hub_id"] == "sk-published"
    assert len(attempts) == 2
    assert len(hub.calls) == 1
    record = prov.read_baseline(root, "skill", retried["link"]["local_id"])
    assert record is not None and record.recorded_by == "publish"


def test_hub_skills_advertises_the_whole_adapter_and_every_promised_route_exists() -> None:
    """The key is a promise about the COMPLETE adapter, not about pull alone.

    Removing any one of the routes the capability-gate receipt names -- the
    desktop pull/publish pair, or either org skill proxy operation -- must fail
    here, which is what makes "advertise hub_skills" a reviewed act rather than a
    one-line claim.
    """

    flags = svc_features.feature_flags()
    assert flags.get("hub_skills") == 1

    routes = {
        (getattr(route, "path", ""), method)
        for route in desktop_hub.router.routes
        for method in getattr(route, "methods", set())
    }
    assert ("/v1/desktop/hub/skills/pull", "POST") in routes
    assert ("/v1/desktop/hub/skills/publish", "POST") in routes
    assert ("/v1/desktop/hub/links", "GET") in routes

    # The DISPATCHER, not the declared literal: a literal left behind after its
    # handler is removed would still satisfy an annotation check while the
    # advertised operation answers an unknown-operation refusal.
    from local_operator.server.routes.desktop_radient import RadientRequest, endpoint

    assert endpoint(RadientRequest(operation="org_skills.list", tenant_id="t-1")) == (
        "GET",
        "/tenants/t-1/skills",
    )
    assert endpoint(
        RadientRequest(operation="org_skill.get", tenant_id="t-1", skill_id="sk-1")
    ) == ("GET", "/tenants/t-1/skills/sk-1")


async def test_an_unconfirmable_preview_cannot_be_confirmed_by_its_receipt(api) -> None:
    """Disabling Confirm in the renderer is not the only gate.

    A caller replaying a receipt directly must hit the same local/capability/
    consent checks, BEFORE the hub is asked to create a row -- otherwise a
    preview that said ``can_confirm: false`` could still publish and then strand
    the link in pending reconciliation.
    """

    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root))
    holder["client"] = hub

    previewed = await preview(client, auto_push=True, confirm_public_egress=True)
    assert previewed["preview"]["can_confirm"] is False
    assert previewed["preview"]["reason"] == "unsupported-precondition"

    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, auto_push=True, confirm_public_egress=True),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "unsupported-precondition"
    assert hub.calls == []
    assert list(prov.iter_baselines(root)) == []


async def test_a_preview_names_an_occupied_alias_as_its_own_reason(api) -> None:
    """The PREVIEW is the surface a colliding user actually meets.

    An unowned personal skill holding the target alias answered
    ``origin-conflict``, which the desktop renderer maps to its merge-lineage
    sentence -- the wrong story for a name collision. The route must carry the
    new ``alias-occupied`` reason through unchanged (the renderer keys on it),
    with Confirm disabled, and Confirm's own re-check must refuse the same way
    before the hub is ever asked to create a row.
    """

    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    personal = write_authored_skill(
        root / "skills", "personal", description="My own copy, not the hub's."
    )
    before = (personal / "SKILL.md").read_bytes()
    hub = PublishClient(base_url(root))
    holder["client"] = hub

    result = await preview(client, local_alias="personal")

    assert result["link"] is None
    assert result["preview"]["can_confirm"] is False
    assert result["preview"]["reason"] == "alias-occupied"
    # No link state, no hub call, and the personal copy is byte-identical.
    assert list(prov.iter_baselines(root)) == []
    assert hub.calls == []
    assert (personal / "SKILL.md").read_bytes() == before

    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(result, local_alias="personal"),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "alias-occupied"
    assert hub.calls == []
    assert list(prov.iter_baselines(root)) == []
    assert (personal / "SKILL.md").read_bytes() == before


async def test_an_alias_owned_by_a_link_is_still_an_origin_conflict(api) -> None:
    """Only an UNOWNED copy is a name collision; a linked alias is an origin conflict.

    ``alias_owner`` means a hub LINK owns the name, present or absent on disk --
    reusing it is the lineage conflict the renderer already has language for,
    and it must not be folded into the new ``alias-occupied`` reason.
    """

    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    from dataclasses import replace

    from local_operator.providers.radient_credentials import configured_radient_base_url

    base = configured_radient_base_url(ConfigManager(root))
    identity = LinkIdentity("skill", "skl-taken", base, "public", None, "sk-taken")
    record = replace(
        prov.make_record("skill", "skl-taken", "sk-taken", None, manifest("taken"), "pull"),
        identity=identity,
        local_alias="taken",
        recorded_by="pull",
        lineage="clean",
    )
    prov.write_baseline(root, record)
    hub = PublishClient(base_url(root))
    holder["client"] = hub

    result = await preview(client, local_alias="taken")

    assert result["preview"]["can_confirm"] is False
    assert result["preview"]["reason"] == "origin-conflict"
    assert hub.calls == []


def attest_skill_capability(root: Path) -> None:
    """Credit the configured base with the skill capability as a real check does."""

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.check import ItemCheck
    from local_operator.providers.radient_credentials import configured_radient_base_url

    base = configured_radient_base_url(ConfigManager(root))
    ctx = svc.HubSyncContext(
        config_dir=root,
        config_manager=ConfigManager(root),
        client_for_tenant=lambda tenant: None,
    )
    svc.record_checks(
        ctx,
        [
            ItemCheck(
                kind="skill",
                local_id="attest",
                name="attest",
                hub_id="attest",
                tenant_id=None,
                verdict="up-to-date",
                remote_revision='"attest"',
                sync_contract={"revision_cas": 1, "skills": 1},
                served_by=base,
            )
        ],
    )


async def test_a_pending_outcome_survives_the_preview_ttl(api, monkeypatch) -> None:
    """The accepted row outlives the reviewed draft; only the draft expires.

    The hub id exists nowhere else until a baseline lands, so pruning or
    expiring the receipt would silently orphan the publication.
    """

    client, root, holder = api
    write_authored_skill(Path(os.environ["HOME"]) / ".claude" / "skills", "beta")
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    from local_operator.hub_sync import skill_storage

    real_apply = skill_storage.ManagedSkillStorage.apply
    failing = {"now": True}

    def maybe_failing(self, *args, **kwargs):
        if failing["now"]:
            raise OSError("synthetic local write failure")
        return real_apply(self, *args, **kwargs)

    monkeypatch.setattr(skill_storage.ManagedSkillStorage, "apply", maybe_failing)
    previewed = await preview(client, source_skill="beta", local_alias="beta")
    body = confirm_body(previewed, source_skill="beta", local_alias="beta")
    body["request_id"] = "00000000-0000-4000-8000-000000000002"
    first = await client.post("/v1/desktop/hub/skills/publish", json=body)
    assert first.json()["result"]["local_link_receipt"]["state"] == "pending-reconciliation"

    token = previewed["preview"]["preview_receipt"]
    path = root / "hub" / "publication-previews" / f"{token}.json"
    record = json.loads(path.read_text())
    record["expires_at"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(record))

    # A later preview prunes expired receipts; this one must survive because its
    # hub row is still unreconciled.
    write_authored_skill(Path(os.environ["HOME"]) / ".claude" / "skills", "delta")
    await preview(client, source_skill="delta", local_alias="delta")
    assert path.exists()

    failing["now"] = False
    retry = await client.post("/v1/desktop/hub/skills/publish", json=body)
    assert retry.status_code == 200, retry.text
    retried = retry.json()["result"]
    assert retried["local_link_receipt"]["state"] == "complete"
    assert retried["link"]["hub_id"] == "sk-published"
    assert len(hub.calls) == 1
    # Reconciled: the durable receipt is removed with the link now materialized.
    assert not path.exists()


async def test_a_linked_skill_cannot_be_republished_from_the_managed_root(api) -> None:
    """A pulled private-org skill must not leave through a public rename."""

    client, root, holder = api
    write_authored_skill(root / "skills", "gamma")
    from dataclasses import replace

    from local_operator.providers.radient_credentials import configured_radient_base_url

    base = configured_radient_base_url(ConfigManager(root))
    identity = LinkIdentity("skill", "skl-gamma", base, "org", "tenant-private", "sk-g")
    fields = manifest("gamma")
    record = replace(
        prov.make_record("skill", "skl-gamma", "sk-g", "tenant-private", fields, "pull"),
        identity=identity,
        local_alias="gamma",
        recorded_by="pull",
        lineage="clean",
    )
    prov.write_baseline(root, record)
    hub = PublishClient(base_url(root))
    holder["client"] = hub

    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=publish_body(source_skill="gamma", local_alias="gamma-public"),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "origin-conflict"
    assert hub.calls == []
    assert [row.local_alias for row in prov.iter_baselines(root)] == ["gamma"]


async def test_an_unprojectable_2xx_is_pending_and_reprojects_the_raw_result(api) -> None:
    """A 2xx the client cannot read still names a row; answer pending, then recover."""

    client, root, holder = api
    write_authored_skill(Path(os.environ["HOME"]) / ".claude" / "skills", "beta")
    hub = PublishClient(base_url(root))
    hub.result = {"accepted_document": manifest("beta")}
    holder["client"] = hub
    previewed = await preview(client, source_skill="beta", local_alias="beta")
    body = confirm_body(previewed, source_skill="beta", local_alias="beta")
    body["request_id"] = "00000000-0000-4000-8000-000000000003"

    first = await client.post("/v1/desktop/hub/skills/publish", json=body)
    assert first.status_code == 200, first.text
    pending = first.json()["result"]
    assert pending["publication"] is None
    assert pending["local_link_receipt"]["state"] == "pending-reconciliation"
    assert "could not be read" in pending["local_link_receipt"]["safe_message"]

    # The raw accepted result was persisted; a retry re-projects it instead of
    # paying for a second publication.
    token = previewed["preview"]["preview_receipt"]
    path = root / "hub" / "publication-previews" / f"{token}.json"
    record = json.loads(path.read_text())
    record["remote_result"]["skill"] = {"id": "sk-raw", "name": "beta"}
    record["remote_result"]["revision"] = '"r-raw"'
    path.write_text(json.dumps(record))

    retry = await client.post("/v1/desktop/hub/skills/publish", json=body)
    assert retry.status_code == 200, retry.text
    retried = retry.json()["result"]
    assert retried["local_link_receipt"]["state"] == "complete"
    assert retried["link"]["hub_id"] == "sk-raw"
    assert len(hub.calls) == 1


async def test_public_auto_push_without_consent_is_unsafe_publication(api) -> None:
    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    attest_skill_capability(root)

    previewed = await preview(client, auto_push=True, confirm_public_egress=False)
    assert previewed["preview"]["can_confirm"] is False
    assert previewed["preview"]["reason"] == "unsafe-publication"

    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, auto_push=True, confirm_public_egress=False),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "unsafe-publication"
    assert hub.calls == []
    assert list(prov.iter_baselines(root)) == []


async def test_publish_accepts_the_unversioned_renderer_base_setting(api, monkeypatch) -> None:
    """The publish surface sees the renderer's unversioned value via its config.

    `configured_radient_base_url` completes it to `/v1`; the preview receipt and
    the materialized link must both carry that versioned origin.
    """

    client, root, holder = api
    write_authored_skill(root / "skills", SOURCE)
    monkeypatch.setenv("RADIENT_API_BASE_URL", "https://api.radienthq.com")
    hub = PublishClient("https://api.radienthq.com/v1")
    holder["client"] = hub

    previewed = await preview(client)
    assert previewed["preview"]["hub_source"] == "https://api.radienthq.com/v1"
    assert previewed["preview"]["can_confirm"] is True

    response = await client.post("/v1/desktop/hub/skills/publish", json=confirm_body(previewed))
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["local_link_receipt"]["state"] == "complete"
    assert result["link"]["hub_source"] == "https://api.radienthq.com/v1"
