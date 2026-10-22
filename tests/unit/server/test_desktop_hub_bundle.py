"""The desktop hub skill routes carry the bundle carrier's fields.

The publish/pull services are pinned in ``tests/unit/hub_sync``; this file pins
the WIRE contract those routes expose: the pull body accepts (and forwards) the
bundle confirmation + dry run, the publish body accepts the explicit form, an
unknown field is still refused, and a real dry-run pull against the route stack
answers the bundle summary and publisher while writing nothing.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.hub_sync.skill_bundle import build_bundle
from local_operator.server.routes import desktop_hub, desktop_sessions

pytestmark = pytest.mark.asyncio

DESKTOP_TOKEN = "desktop-test-token"
AUTH = {"Authorization": f"Bearer {DESKTOP_TOKEN}"}
HUB_ID = "sk-bundle-1"
NAME = "pdf-forms"
PUBLISHER = {"account_id": "acct-9", "name": "Ada"}


def _document(built: Any) -> dict[str, Any]:
    return {
        "document_version": 1,
        "id": HUB_ID,
        "name": built.name,
        "description": built.description,
        "bundle": built.bundle.to_json(),
        "visibility": "public",
        "tenant_id": None,
        "revision": '"hub-rev-1"',
        "content_revision": 1,
        "published_by": dict(PUBLISHER),
        "sync_contract": {"revision_cas": 1, "skills": 1, "skill_bundles": 1},
    }


class BundleClient:
    def __init__(self, base: str, document: dict[str, Any], data: bytes) -> None:
        self.base_url = base
        self.api_key = None
        self.document = document
        self.data = data
        self.downloads = 0

    def get_skill(self, skill_id: str, **_kw: Any) -> dict[str, Any]:
        return dict(self.document)

    def download_skill_bundle(self, skill_id: str, *, dest: Any = None, **_kw: Any) -> Any:
        self.downloads += 1
        if dest is not None:
            dest.write(self.data)
        return SimpleNamespace(data=None, not_modified=False, etag=None, server_hash=None)


def _source(root: Path) -> Path:
    directory = root / NAME
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\n" f"name: {NAME}\n" "description: Fill and validate PDF forms.\n" "---\n\n# Body\n",
        encoding="utf-8",
    )
    (directory / "references.md").write_text("alpha\n", encoding="utf-8")
    return directory


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", DESKTOP_TOKEN)
    built = build_bundle(_source(tmp_path / "authored"))
    from local_operator.providers.radient_credentials import configured_radient_base_url

    base = configured_radient_base_url(ConfigManager(root))
    client = BundleClient(base, _document(built), built.data)
    holder: dict[str, Any] = {"client": client, "built": built, "base": base}

    async def fake_clients(_cm: Any, _store: Any = None):
        return (lambda _tenant: holder["client"]), "ok"

    monkeypatch.setattr("local_operator.hub_sync.service.build_clients", fake_clients)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.state.env_config = EnvConfig()
    app.include_router(desktop_hub.router)
    app.include_router(desktop_sessions.router)
    app.dependency_overrides[desktop_hub.get_provider_auth_store] = lambda: None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=AUTH
    ) as session:
        yield session, root, holder
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def pull_body(**fields: Any) -> dict[str, Any]:
    body = {
        "request_id": str(uuid.uuid4()),
        "hub_id": HUB_ID,
        "hub_source": "https://api.example.invalid",
        "visibility": "public",
        "tenant_id": None,
        "local_alias": NAME,
        "subscribe": True,
        "dry_run": True,
    }
    body.update(fields)
    return body


def test_the_pull_body_accepts_the_bundle_confirmation_and_dry_run() -> None:
    body = desktop_hub.SkillPull(
        **pull_body(
            form="bundle",
            confirm_publisher_id=PUBLISHER["account_id"],
            confirm_bundle_hash="0" * 64,
            confirm=True,
        )
    )

    assert body.form == "bundle"
    assert body.dry_run is True
    assert body.confirm is True
    assert body.confirm_publisher_id == PUBLISHER["account_id"]


def test_the_pull_body_still_forbids_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        desktop_hub.SkillPull(**pull_body(bundle_hash="0" * 64))


def test_the_publish_body_accepts_the_explicit_bundle_form() -> None:
    body = desktop_hub.SkillPublish(
        request_id=str(uuid.uuid4()),
        dry_run=True,
        source_skill=NAME,
        local_alias=NAME,
        visibility="public",
        tenant_id=None,
        subscribe=True,
        auto_push=False,
        confirm_public_egress=False,
        form="bundle",
    )

    assert body.form == "bundle"


async def test_a_dry_run_pull_answers_the_bundle_summary_and_writes_nothing(api) -> None:
    client, root, holder = api
    body = pull_body(hub_source=holder["base"])

    response = await client.post("/v1/desktop/hub/skills/pull", json=body)

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["dry_run"] is True
    assert result["bundle"]["bundle"]["hash"] == holder["built"].bundle.hash
    assert result["publisher"] == {"id": PUBLISHER["account_id"], "name": "Ada"}
    # Nothing was fetched beyond the detail read, and nothing was written.
    assert holder["client"].downloads == 0
    assert not (root / "hub" / "skills" / NAME).exists()
    assert not (root / "skills" / NAME).exists()
