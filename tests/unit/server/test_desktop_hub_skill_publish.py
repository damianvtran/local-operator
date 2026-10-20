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
from local_operator.server.routes import desktop_hub, desktop_radient, desktop_sessions

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
    response = await client.post("/v1/desktop/hub/skills/publish", json=confirm_body(previewed))
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

    # Repeating the same confirm is idempotent: no second publication, same link.
    again = await client.post("/v1/desktop/hub/skills/publish", json=confirm_body(previewed))
    assert again.status_code == 200, again.text
    assert again.json()["result"]["link"]["link_id"] == row["link_id"]
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


async def test_a_local_failure_after_acceptance_is_pending_and_retry_reconciles(
    api, monkeypatch
) -> None:
    client, root, holder = api
    write_authored_skill(Path(os.environ["HOME"]) / ".claude" / "skills", "beta")
    hub = PublishClient(base_url(root))
    holder["client"] = hub
    from local_operator.hub_sync import skill_storage

    real_apply = skill_storage.ManagedSkillStorage.apply

    def failing_apply(self, *args, **kwargs):
        raise OSError("synthetic local write failure")

    monkeypatch.setattr(skill_storage.ManagedSkillStorage, "apply", failing_apply)
    previewed = await preview(client, source_skill="beta", local_alias="beta")
    response = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, source_skill="beta", local_alias="beta"),
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["publication"]["skill"]["id"] == "sk-published"
    assert result["link"] is None
    assert result["local_link_receipt"]["state"] == "pending-reconciliation"
    assert list(prov.iter_baselines(root)) == []
    assert not (root / "skills" / "beta").exists()

    # The SAME receipt resumes local reconciliation without republishing.
    monkeypatch.setattr(skill_storage.ManagedSkillStorage, "apply", real_apply)
    retry = await client.post(
        "/v1/desktop/hub/skills/publish",
        json=confirm_body(previewed, source_skill="beta", local_alias="beta"),
    )
    assert retry.status_code == 200, retry.text
    retried = retry.json()["result"]
    assert retried["local_link_receipt"]["state"] == "complete"
    assert retried["link"]["hub_id"] == "sk-published"
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

    from typing import get_args

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

    operations = get_args(desktop_radient.RadientRequest.model_fields["operation"].annotation)
    assert "org_skills.list" in operations
    assert "org_skill.get" in operations


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
