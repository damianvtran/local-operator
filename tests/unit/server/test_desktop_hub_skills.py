"""The skills-pull route: scoped reads, managed install, refusals, replay.

Driven against the REAL route stack (FastAPI + the real ASGI transport + the
receipts/errors wrappers), the real ``hub_sync`` services, and the REAL
``ManagedSkillStorage`` on an isolated HOME/config root. Only the credentialed
client picker is faked -- with a client that records which credential class it
was asked for -- so the network boundary is the one under test, and every
assertion reads PERSISTED effects (files, baseline, policy, journal), never a
call count of the subject's own helpers.
"""

from __future__ import annotations

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
from local_operator.hub_sync.service import skill_link_local_id
from local_operator.hub_sync.skill_document import SkillDocument
from local_operator.hub_sync.skill_storage import ManagedSkillStorage
from local_operator.server import features as svc_features
from local_operator.server.routes import capabilities, desktop_hub, desktop_sessions

pytestmark = pytest.mark.asyncio

AUTH = {"Authorization": "Bearer test-token"}
BASE = "https://api.radienthq.com/v1"
ALIAS = "weather-buddy"
HUB_ID = "sk-1"


def manifest(name: str = ALIAS, body: str = "Say hi.\n") -> dict[str, Any]:
    # SKILL.md carries STRICT YAML frontmatter whose name/description match the
    # manifest fields -- construction refuses a document where the two disagree
    # (the same strictness every real manifest passes).
    return {
        "document_version": 1,
        "name": name,
        "description": "A small test skill for the pull route.",
        "files": [
            {
                "path": "SKILL.md",
                "content": (
                    "---\n"
                    f"name: {name}\n"
                    "description: A small test skill for the pull route.\n"
                    "---\n\n"
                    f"# {name}\n\n{body}"
                ),
            },
            {"path": "references/extra.md", "content": "More detail.\n"},
        ],
    }


def envelope(document: dict[str, Any], revision: str = '"hub-rev-1"') -> dict[str, Any]:
    """The DETAIL response a real server sends: manifest + metadata envelope."""

    payload = dict(document)
    payload.update(
        {
            "revision": revision,
            "content_revision": 3,
            "sync_contract": {"revision_cas": 1, "skills": 1},
            "id": HUB_ID,
            "visibility": "public",
            "tenant_id": None,
            "moderation": {"flagged": False},
        }
    )
    return payload


class SkillClient:
    """The fetch seam's client: records calls, serves one payload or one error."""

    def __init__(self, base_url: str, payload: Any, *, api_key: str | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get_skill(self, skill_id: str, **kw: Any) -> Any:
        self.calls.append({"skill_id": skill_id, **kw})
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


def pull_body(**fields: Any) -> dict[str, Any]:
    body = {
        "request_id": str(uuid.uuid4()),
        "hub_id": HUB_ID,
        "hub_source": BASE,
        "visibility": "public",
        "tenant_id": None,
        "local_alias": ALIAS,
    }
    body.update(fields)
    return body


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "test-token")
    holder: dict[str, Any] = {}

    async def fake_clients(_cm: Any, _store: Any = None):
        def for_tenant(tenant: Any) -> Any:
            # None = the anonymous public client; any tenant = the org client.
            return holder.get("client" if tenant is not None else "anon")

        return for_tenant, "ok"

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
        yield client, root, holder
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def nothing_installed(root: Path, alias: str = ALIAS) -> bool:
    return not (root / "skills" / alias).exists()


def linked_identity(row: dict[str, Any]) -> LinkIdentity:
    return LinkIdentity(
        "skill",
        row["local_id"],
        row["hub_source"],
        row["visibility"],
        row["tenant_id"],
        row["hub_id"],
    )


async def test_a_public_pull_installs_every_file_and_reads_back_the_contract(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 200, response.text
    result = response.json()["result"]

    # The directional report the UI's installed state reads (applied + link_id
    # + direction), never a bare merge report.
    (report,) = result["reports"]
    assert report["outcome"] == "merged" and report["applied"] is True
    assert report["direction"] == "pull" and report["kind"] == "skill"
    assert report["operation_id"]

    row = result["link"]
    assert row["link_id"] == report["link_id"] and row["kind"] == "skill"
    assert row["name"] == ALIAS and row["hub_source"] == BASE
    assert row["visibility"] == "public" and row["tenant_id"] is None
    assert row["hub_id"] == HUB_ID
    assert result["links"]["capability"] == "hub_links:1"
    assert any(link["link_id"] == row["link_id"] for link in result["links"]["links"])

    # Creation defaults, read back through the SAME reader every surface uses.
    policy = pol.read_policy(read_config_values(root) or {}, linked_identity(row))
    assert policy.policy.subscribe is True and policy.policy.auto_push is False

    # ALL manifest files on disk, through the managed root.
    skill_dir = root / "skills" / ALIAS
    skill_md = (skill_dir / "SKILL.md").read_text()
    assert skill_md.startswith("---\n") and f"name: {ALIAS}" in skill_md
    assert "Say hi." in skill_md
    assert (skill_dir / "references" / "extra.md").read_text() == "More detail.\n"

    # The canonical inbound baseline: the remote document at a live revision,
    # vouched as a clean pull of this origin.
    record = prov.read_baseline(root, "skill", row["local_id"])
    assert record is not None and record.identity == linked_identity(row)
    assert record.recorded_by == "pull" and record.lineage == "clean"
    assert record.remote_revision == '"hub-rev-1"'
    assert record.local_alias == ALIAS and record.hub_id == HUB_ID
    assert dict(record.fields)["files"] == manifest()["files"]

    # The partial-adapter milestone must NOT advertise `hub_skills` yet.
    assert "hub_skills" not in svc_features.feature_flags()
    assert "hub_links" in svc_features.feature_flags()

    # A public fetch is anonymous: the client carried no credential.
    (call,) = holder["anon"].calls
    assert call["with_credential"] is False


async def test_an_org_pull_uses_the_org_client_and_records_the_tenant(api) -> None:
    client, root, holder = api
    holder["client"] = SkillClient(BASE, envelope(manifest()), api_key="org-key")

    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(visibility="org", tenant_id="org-1"),
    )
    assert response.status_code == 200, response.text
    row = response.json()["result"]["link"]
    assert row["visibility"] == "org" and row["tenant_id"] == "org-1"
    record = prov.read_baseline(root, "skill", row["local_id"])
    assert record is not None and record.tenant_id == "org-1"

    # The org fetch spent the signed-in credential class, not the anonymous one.
    (call,) = holder["client"].calls
    assert call["with_credential"] is True


async def test_an_org_pull_without_a_credential_is_a_signin_refusal(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(visibility="org", tenant_id="org-1"),
    )
    assert response.status_code == 401
    assert (
        "sign" in response.json()["detail"].lower() or "login" in response.json()["detail"].lower()
    )
    assert nothing_installed(root)


async def test_missing_and_foreign_skills_are_one_plain_404(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, APIError("Not Found", status_code=404))

    first = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    foreign = await client.post("/v1/desktop/hub/skills/pull", json=pull_body(hub_id="sk-other"))
    assert first.status_code == 404 and foreign.status_code == 404
    # One identical body: the caller must not be able to tell "missing" from
    # "not yours" (the upstream answers 404 for both; this side never re-derives).
    assert first.json() == foreign.json()
    assert nothing_installed(root)


@pytest.mark.parametrize(
    "fields",
    [
        # Pairing: the one field where failing open would be a disclosure.
        {"visibility": "public", "tenant_id": "org-1"},
        {"visibility": "org", "tenant_id": None},
        # The alias must be a valid skill name.
        {"local_alias": "../escape"},
        {"local_alias": "Not A Name"},
        # Policy values refuse truthy strings by name, never coerce.
        {"subscribe": "yes"},
        {"auto_push": 1},
    ],
)
async def test_invalid_requests_refuse_by_name_and_write_nothing(api, fields) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body(**fields))
    assert response.status_code == 422, response.text
    assert nothing_installed(root)
    assert holder["anon"].calls == []


async def test_an_unapproved_or_mismatched_source_never_reaches_the_fetch(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    # A source that is not the ONE configured base: refused before any client
    # could be selected, so a stored credential can never be routed to it.
    response = await client.post(
        "/v1/desktop/hub/skills/pull", json=pull_body(hub_source="https://evil.invalid/v1")
    )
    assert response.status_code == 422, response.text
    assert nothing_installed(root)
    assert holder["anon"].calls == []

    # The client that answers must be the requested source; a rogue client
    # whose base differs is refused, never credited.
    holder["anon"] = SkillClient("https://elsewhere.invalid/v1", envelope(manifest()))
    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 422, response.text
    assert nothing_installed(root)


async def test_an_invalid_remote_manifest_refuses_without_writing(api) -> None:
    client, root, holder = api
    bad = manifest()
    bad["files"].append({"path": "references/../escape.md", "content": "nope"})
    holder["anon"] = SkillClient(BASE, envelope(bad))

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 422, response.text
    assert nothing_installed(root)
    assert not (root / "escape.md").exists()


async def test_an_authored_alias_is_never_overwritten(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    authored = root / "skills" / ALIAS
    authored.mkdir(parents=True)
    (authored / "SKILL.md").write_text(f"# {ALIAS}\n\nMy own authored copy.\n")

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 422, response.text
    assert (authored / "SKILL.md").read_text() == f"# {ALIAS}\n\nMy own authored copy.\n"
    assert (
        prov.read_baseline(
            root,
            "skill",
            skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID),
        )
        is None
    )


async def test_a_symlinked_alias_is_refused_and_its_target_untouched(api, tmp_path) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.md").write_text("keep\n")
    (root / "skills").mkdir(parents=True, exist_ok=True)
    os.symlink(outside, root / "skills" / ALIAS)

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 422, response.text
    assert (outside / "keep.md").read_text() == "keep\n"
    assert sorted(os.listdir(outside)) == ["keep.md"]


async def test_an_alias_owned_by_another_origin_is_refused(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    foreign = LinkIdentity(
        "skill", "other-id", "https://other.invalid/v1", "org", "org-2", "sk-other"
    )
    record = prov.make_record("skill", "other-id", "sk-other", "org-2", manifest(), "pull")
    prov.write_baseline(
        root,
        prov.replace(record, identity=foreign, local_alias=ALIAS, remote_revision='"r"'),
    )

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 422, response.text
    kept = prov.read_baseline(root, "skill", "other-id")
    assert kept is not None and kept.identity == foreign


async def test_a_crashed_install_recovers_and_the_retry_completes(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    # Simulate a crash mid-transaction for the SAME deterministic identity the
    # route will mint, then let the route run: the storage's recovery must
    # settle the crashed attempt and the install must still complete.
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)
    storage = ManagedSkillStorage(root)
    document = SkillDocument.from_json(manifest()).with_name(ALIAS)
    baseline = prov.replace(
        prov.make_record("skill", local_id, HUB_ID, None, manifest(), "pull"),
        identity=identity,
        local_alias=ALIAS,
        remote_revision='"hub-rev-1"',
        lineage="clean",
    )

    def crash(phase: str) -> None:
        if phase == "staged":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        storage.apply(
            identity,
            ALIAS,
            document,
            baseline,
            expected_fingerprint=None,
            checkpoint=crash,
        )

    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 200, response.text
    assert (root / "skills" / ALIAS / "SKILL.md").exists()
    assert storage.journal.pending(kind="skill", local_id=local_id) == []
    record = prov.read_baseline(root, "skill", local_id)
    assert record is not None and record.remote_revision == '"hub-rev-1"'


async def test_replays_and_a_second_pull_never_double_install(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    body = pull_body()

    first = await client.post("/v1/desktop/hub/skills/pull", json=body)
    assert first.status_code == 200
    assert len(holder["anon"].calls) == 1

    # The same request id replays the stored response: no second fetch, no
    # second install (the receipts discipline every mutation shares).
    replayed = await client.post("/v1/desktop/hub/skills/pull", json=body)
    assert replayed.status_code == 200
    # The receipts layer replays the STORED response and marks the result as a
    # replay; every other byte is identical and no second fetch was made.
    replayed_result = dict(replayed.json()["result"])
    assert replayed_result.pop("replayed", None) is True
    assert replayed_result == first.json()["result"]
    assert len(holder["anon"].calls) == 1

    # A NEW request for the same remote identity is the idempotent arm: it
    # confirms the link and reports `unchanged`, and still never writes twice.
    second = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert second.status_code == 200, second.text
    result = second.json()["result"]
    assert result["reports"][0]["outcome"] == "unchanged"
    assert result["reports"][0]["applied"] is False
    assert result["link"]["link_id"] == first.json()["result"]["link"]["link_id"]
    assert "Say hi." in (root / "skills" / ALIAS / "SKILL.md").read_text()
