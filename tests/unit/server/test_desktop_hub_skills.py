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

import errno
import os
import subprocess
import sys
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


def org_envelope(
    document: dict[str, Any], tenant_id: str = "org-1", revision: str = '"hub-rev-1"'
) -> dict[str, Any]:
    """The DETAIL response for an ORG row: same shape, scoped metadata."""

    payload = envelope(document, revision)
    payload.update({"visibility": "org", "tenant_id": tenant_id})
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

    # The complete-adapter milestone advertises `hub_skills` beside the links key;
    # the route pin below fails if the routes it promises are removed.
    assert svc_features.feature_flags().get("hub_skills") == 1
    assert "hub_links" in svc_features.feature_flags()

    # A public fetch is anonymous: the client carried no credential.
    (call,) = holder["anon"].calls
    assert call["with_credential"] is False


async def test_an_unversioned_renderer_source_is_completed_not_refused(api) -> None:
    """The UI sends VITE_RADIENT_SERVER_BASE_URL verbatim, without `/v1`.

    The route must complete it with the same rule the configured base uses and
    answer the work, while the minted link carries the versioned origin.
    """

    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(hub_source="https://api.radienthq.com"),
    )
    assert response.status_code == 200, response.text
    row = response.json()["result"]["link"]
    assert row["hub_source"] == BASE
    record = prov.read_baseline(root, "skill", row["local_id"])
    assert record is not None and record.identity is not None
    assert record.identity.hub_source == BASE

    # A DIFFERENT origin is still refused after completion.
    holder["anon"].calls.clear()
    refused = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(hub_source="https://elsewhere.invalid"),
    )
    assert refused.status_code == 422, refused.text
    assert holder["anon"].calls == []


async def test_a_malformed_hub_source_is_the_normal_origin_conflict(api) -> None:
    """A bracketed host fails in ``urlsplit``; the route must not leak that.

    Completion runs before approval, so a raw ``ValueError`` used to reach the
    route's generic handler and answer 409 with the parser's own sentence. The
    refusal is the same nondisclosing 422 every unapproved origin gets, and no
    client is built.
    """

    client, _root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(hub_source="http://[::1"),
    )
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "origin-conflict"
    assert "Invalid IPv6 URL" not in detail["message"]
    assert holder["anon"].calls == []


async def test_an_org_pull_uses_the_org_client_and_records_the_tenant(api) -> None:
    client, root, holder = api
    holder["client"] = SkillClient(BASE, org_envelope(manifest()), api_key="org-key")

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


# -- remediation round 1 (R1/Q2, R2/Q1, Q3, Q4): identity at the boundary, ------
# -- atomic initial policy, the cross-process lease, and crash resumption -----


def attest_skill_capability(root: Path) -> None:
    """Credit BASE with the skill capability exactly the way a real check does."""

    from local_operator.hub_sync import service as svc
    from local_operator.hub_sync.check import ItemCheck

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
                served_by=BASE,
            )
        ],
    )


async def test_a_scope_mismatched_row_is_a_nondisclosing_404(api) -> None:
    client, root, holder = api
    # The canonical missing-row refusal, to compare bodies against.
    holder["anon"] = SkillClient(
        BASE, APIError("not found", status_code=404, code="hub-item-missing")
    )
    reference = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert reference.status_code == 404

    # An ORG request answered by a PUBLIC row: the relabel is refused.
    holder["client"] = SkillClient(BASE, envelope(manifest()), api_key="k")
    public_for_org = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(visibility="org", tenant_id="org-1"),
    )
    assert public_for_org.status_code == 404
    assert public_for_org.json() == reference.json()

    # An ORG row from ANOTHER tenant answered this tenant's request.
    holder["client"] = SkillClient(
        BASE, org_envelope(manifest(), tenant_id="other-org"), api_key="k"
    )
    foreign_tenant = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(visibility="org", tenant_id="org-1"),
    )
    assert foreign_tenant.status_code == 404
    assert foreign_tenant.json() == reference.json()

    # A DIFFERENT identity answered the requested hub id.
    wrong = envelope(manifest())
    wrong["id"] = "not-requested"
    holder["anon"] = SkillClient(BASE, wrong)
    wrong_id = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert wrong_id.status_code == 404
    assert wrong_id.json() == reference.json()

    # None of the three refusals wrote anything.
    assert nothing_installed(root)
    assert list(prov.iter_baselines(root)) == []


async def test_a_refused_install_leaves_config_untouched_and_retry_defaults_on(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    authored = root / "skills" / ALIAS
    authored.mkdir(parents=True)
    (authored / "SKILL.md").write_text("# authored\n\nNot ours to overwrite.\n")
    before = read_config_values(root)

    first = await client.post("/v1/desktop/hub/skills/pull", json=pull_body(subscribe=False))
    assert first.status_code == 422, first.text
    # R2/Q1: the refusal changed NOTHING -- no orphan policy for a link that
    # was never installed.
    assert read_config_values(root) == before
    assert (authored / "SKILL.md").read_text().endswith("Not ours to overwrite.\n")

    # A retry for the same remote identity at a FREE alias with no overrides
    # gets the creation default ON -- never the dead request's OFF.
    retry = await client.post(
        "/v1/desktop/hub/skills/pull", json=pull_body(local_alias="free-alias")
    )
    assert retry.status_code == 200, retry.text
    row = retry.json()["result"]["link"]
    policy = pol.read_policy(read_config_values(root) or {}, linked_identity(row))
    assert policy.policy.subscribe is True and policy.policy.auto_push is False


async def test_an_orphan_policy_from_a_crashed_attempt_never_beats_the_request(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    manager = ConfigManager(root)

    # Simulate a crashed earlier attempt: an explicit entry exists, no baseline.
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)
    current = pol.read_policy(read_config_values(root) or {}, identity)
    pol.update_policy(manager, identity, expected_revision=current.revision, subscribe=False)

    # With no override the request's creation default wins, not the orphan's OFF.
    response = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert response.status_code == 200, response.text
    row = response.json()["result"]["link"]
    policy = pol.read_policy(read_config_values(root) or {}, linked_identity(row))
    assert policy.policy.subscribe is True and policy.policy.auto_push is False

    # And when the next request DOES ask for OFF, the request wins over an
    # orphan that claimed ON.
    second = envelope(manifest(name="second-buddy"), '"hub-rev-2"')
    second["id"] = "sk-2"
    holder["anon"] = SkillClient(BASE, second)
    local_id2 = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id="sk-2")
    identity2 = LinkIdentity("skill", local_id2, BASE, "public", None, "sk-2")
    current2 = pol.read_policy(read_config_values(root) or {}, identity2)
    pol.update_policy(manager, identity2, expected_revision=current2.revision, subscribe=True)
    off = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(hub_id="sk-2", local_alias="second-buddy", subscribe=False),
    )
    assert off.status_code == 200, off.text
    row2 = off.json()["result"]["link"]
    policy2 = pol.read_policy(read_config_values(root) or {}, linked_identity(row2))
    assert policy2.policy.subscribe is False and policy2.policy.auto_push is False


async def test_an_ineligible_auto_push_refuses_before_any_write(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    before = read_config_values(root)

    # No capability attestation for BASE: the auto-push request is refused with
    # its own named reason and NOTHING is written, not even for a moment.
    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert response.status_code == 422, response.text
    assert read_config_values(root) == before
    assert list(prov.iter_baselines(root)) == []
    assert nothing_installed(root)


async def test_a_staged_grant_is_not_rolled_back_over_a_newer_write(api, monkeypatch) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    attest_skill_capability(root)
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)

    real_apply = ManagedSkillStorage.apply
    raced: list[str] = []

    def racing_apply(self, identity_arg, alias_arg, document, baseline, **kw):
        # ANOTHER WRITER lands between the staged auto-push write and the
        # storage's policy CAS: the transaction must refuse, and the revert
        # must NOT roll the newer value back.
        values = read_config_values(root) or {}
        current = pol.read_policy(values, identity_arg)
        snapshot = pol.update_policy(
            ConfigManager(root), identity_arg, expected_revision=current.revision, subscribe=False
        )
        raced.append(snapshot.revision)
        return real_apply(self, identity_arg, alias_arg, document, baseline, **kw)

    monkeypatch.setattr(ManagedSkillStorage, "apply", racing_apply)
    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert response.status_code == 409, response.text
    stored = pol.read_policy(read_config_values(root) or {}, identity)
    assert stored.revision == raced[0]
    assert list(prov.iter_baselines(root)) == []


async def test_a_held_runner_lease_refuses_the_install(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    from local_operator.hub_sync import store as st

    lease = st.RunnerLease(root)
    assert lease.acquire(0.0)
    try:
        held = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
        assert held.status_code == 409, held.text
        assert nothing_installed(root)
        assert list(prov.iter_baselines(root)) == []
    finally:
        lease.release()

    after = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert after.status_code == 200, after.text


async def test_a_lease_held_by_another_process_also_refuses(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    script = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from local_operator.hub_sync import store as st\n"
        "lease = st.RunnerLease(Path(sys.argv[1]))\n"
        "assert lease.acquire(0.0)\n"
        "print('READY', flush=True)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(root)],
        stdout=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        held = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
        assert held.status_code == 409, held.text
        assert nothing_installed(root)
    finally:
        proc.kill()
        proc.wait(timeout=10)


async def test_a_finalize_crash_is_resumed_by_the_next_install(api, monkeypatch) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    from local_operator.hub_sync import skill_storage

    real = skill_storage.write_baseline
    calls: list[int] = []

    def failing_once(config_dir, record):
        if not calls:
            calls.append(1)
            raise RuntimeError("simulated finalize crash")
        return real(config_dir, record)

    monkeypatch.setattr(skill_storage, "write_baseline", failing_once)
    first = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert first.status_code != 200, first.text
    # Content committed, baseline NOT finalized: honestly not claimed complete.
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    assert (root / "skills" / ALIAS / "SKILL.md").exists()
    assert prov.read_baseline(root, "skill", local_id) is None

    monkeypatch.setattr(skill_storage, "write_baseline", real)
    retry = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    assert retry.status_code == 200, retry.text
    result = retry.json()["result"]
    # Q4: the first retry RECOVERS the crashed transaction and settles as the
    # idempotent completed-install result -- consistent files, baseline, policy.
    assert result["reports"][0]["applied"] is False
    assert result["reports"][0]["outcome"] == "unchanged"
    record = prov.read_baseline(root, "skill", local_id)
    assert record is not None and record.remote_revision == '"hub-rev-1"'
    assert record.lineage == "clean"
    assert len(list(prov.iter_baselines(root))) == 1
    values = read_config_values(root) or {}
    assert pol.stored_link_policy(values, result["link"]["link_id"]) is not None
    assert (root / "skills" / ALIAS / "references" / "extra.md").read_text() == "More detail.\n"


async def test_a_receipt_lost_after_the_install_still_reconciles(api, monkeypatch) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))

    from local_operator.server.utils import desktop_receipts

    real_finish = desktop_receipts.DesktopReceipts._finish
    calls: list[int] = []

    def failing_once(self, key, result):
        if not calls:
            calls.append(1)
            raise RuntimeError("simulated crash before the receipt was recorded")
        return real_finish(self, key, result)

    monkeypatch.setattr(desktop_receipts.DesktopReceipts, "_finish", failing_once)
    body = pull_body()
    first = await client.post("/v1/desktop/hub/skills/pull", json=body)
    assert first.status_code != 200, first.text
    # The install COMPLETED before the receipt was lost.
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    assert prov.read_baseline(root, "skill", local_id) is not None

    monkeypatch.setattr(desktop_receipts.DesktopReceipts, "_finish", real_finish)
    retry = await client.post("/v1/desktop/hub/skills/pull", json=body)
    assert retry.status_code == 200, retry.text
    result = retry.json()["result"]
    assert result["reports"][0]["applied"] is False
    assert len(list(prov.iter_baselines(root))) == 1

    # Once the retry records the outcome, the id replays like any other.
    third = await client.post("/v1/desktop/hub/skills/pull", json=body)
    assert third.status_code == 200
    assert third.json()["result"].get("replayed") is True


async def test_a_too_large_local_rendering_never_leaks_a_staged_grant(api) -> None:
    client, root, holder = api
    # The REMOTE skill's SKILL.md sits exactly AT the 200 KiB file bound and its
    # remote name is one character. A 64-character local alias re-renders that
    # scalar and pushes the SAME file past the bound, so the local proposal
    # itself must refuse -- BEFORE any policy or grant is staged (R2/Q1b).
    remote = manifest(name="a", body="")
    md_index = next(i for i, f in enumerate(remote["files"]) if f["path"] == "SKILL.md")
    current_bytes = len(remote["files"][md_index]["content"].encode())
    remote["files"][md_index]["content"] += "x" * (200 * 1024 - current_bytes)
    assert len(remote["files"][md_index]["content"].encode()) == 200 * 1024
    holder["anon"] = SkillClient(BASE, envelope(remote))
    attest_skill_capability(root)
    before = read_config_values(root)
    alias64 = "b" * 64

    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True, local_alias=alias64),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "skill-validation"

    # NOTHING was staged: no policy entry, no public grant, no baseline, no file.
    assert read_config_values(root) == before
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)
    values = read_config_values(root) or {}
    assert pol.stored_link_policy(values, identity.link_id) is None
    assert pol.read_policy(values, identity).public_egress_grant is None
    assert list(prov.iter_baselines(root)) == []
    assert nothing_installed(root, alias64)


async def test_a_storage_enospc_pre_commit_never_leaks_a_staged_grant(api, monkeypatch) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    attest_skill_capability(root)
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)

    real_stage = ManagedSkillStorage._stage
    calls: list[int] = []

    def failing_stage(self, *args, **kwargs):
        if not calls:
            calls.append(1)
            raise OSError(errno.ENOSPC, "synthetic precommit filesystem full")
        return real_stage(self, *args, **kwargs)

    monkeypatch.setattr(ManagedSkillStorage, "_stage", failing_stage)
    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert response.status_code == 507, response.text
    assert response.json()["detail"]["code"] == "store_out_of_space"

    # Q1c: the staged entry and grant are cleaned up on the unhandled exit.
    # The staged write itself materializes config defaults, so the contract is
    # asserted on the ATTEMPTED LINK (not whole-file bytes): no stored policy,
    # no grant, no baseline, no content, and the request id stays usable.
    values = read_config_values(root) or {}
    assert pol.stored_link_policy(values, identity.link_id) is None
    assert pol.read_policy(values, identity).public_egress_grant is None
    assert list(prov.iter_baselines(root)) == []
    assert nothing_installed(root)

    # A healthy retry installs cleanly from the same state.
    monkeypatch.setattr(ManagedSkillStorage, "_stage", real_stage)
    retry = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert retry.status_code == 200, retry.text


async def test_an_unexpected_pre_commit_failure_never_leaks_a_staged_grant(
    api, monkeypatch
) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    attest_skill_capability(root)
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)

    real_stage = ManagedSkillStorage._stage
    calls: list[int] = []

    def failing_stage(self, *args, **kwargs):
        if not calls:
            calls.append(1)
            raise RuntimeError("synthetic unexpected pre-commit failure")
        return real_stage(self, *args, **kwargs)

    monkeypatch.setattr(ManagedSkillStorage, "_stage", failing_stage)
    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert response.status_code >= 500, response.text
    # An UNEXPECTED exception type must not orphan the staged entry either: the
    # guard is the exit, not the class.
    values = read_config_values(root) or {}
    assert pol.stored_link_policy(values, identity.link_id) is None
    assert pol.read_policy(values, identity).public_egress_grant is None
    assert list(prov.iter_baselines(root)) == []
    assert nothing_installed(root)


async def test_a_content_committed_failure_keeps_the_staged_grant(api, monkeypatch) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    attest_skill_capability(root)

    from local_operator.hub_sync import skill_storage

    real = skill_storage.write_baseline
    calls: list[int] = []

    def failing_once(config_dir, record):
        if not calls:
            calls.append(1)
            raise RuntimeError("simulated finalize crash")
        return real(config_dir, record)

    monkeypatch.setattr(skill_storage, "write_baseline", failing_once)
    first = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert first.status_code != 200, first.text
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)
    values = read_config_values(root) or {}
    # The content LANDED; the staged grant must survive for the recovering
    # journal -- rolling it back here would silently strip the recovered
    # install of the policy the user asked for.
    stored = pol.stored_link_policy(values, identity.link_id)
    assert stored is not None and stored.auto_push is True
    assert pol.read_policy(values, identity).public_egress_grant is not None
    assert prov.read_baseline(root, "skill", local_id) is None

    monkeypatch.setattr(skill_storage, "write_baseline", real)
    retry = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert retry.status_code == 200, retry.text
    record = prov.read_baseline(root, "skill", local_id)
    assert record is not None and record.lineage == "clean"
    kept = pol.stored_link_policy(read_config_values(root) or {}, identity.link_id)
    assert kept is not None and kept.auto_push is True


async def test_a_matching_authored_folder_is_never_adopted_and_never_leaks(api) -> None:
    client, root, holder = api
    # An unknown authored folder whose files are BYTE-IDENTICAL to the manifest
    # this install renders. Content equality is not ownership: the storage
    # refuses the takeover and NOTHING may be adopted -- no baseline, no
    # policy, no grant -- because the refused attempt never owned that folder.
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    attest_skill_capability(root)
    rendered = SkillDocument.from_json(manifest()).with_name(ALIAS)
    authored = root / "skills" / ALIAS
    authored.mkdir(parents=True)
    for file in rendered.files:
        path = authored / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file.content)

    # Sanity: the authored tree IS byte-identical to what the install renders.
    assert SkillDocument.from_directory(authored).fingerprint == rendered.fingerprint

    response = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "origin-conflict"
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)
    values = read_config_values(root) or {}
    assert pol.stored_link_policy(values, identity.link_id) is None
    assert pol.read_policy(values, identity).public_egress_grant is None
    assert list(prov.iter_baselines(root)) == []
    # The authored tree is intact and untouched.
    assert SkillDocument.from_directory(authored).fingerprint == rendered.fingerprint


async def test_a_crash_in_the_rename_window_keeps_the_grant_and_recovers(api, monkeypatch) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    attest_skill_capability(root)

    from local_operator.hub_sync import skill_storage

    real = skill_storage.TransactionJournal.transition
    calls: list[int] = []

    def failing(self, record, phase, **kw):
        if phase == "content-committed" and not calls:
            calls.append(1)
            raise RuntimeError("simulated crash in the rename window")
        return real(self, record, phase, **kw)

    monkeypatch.setattr(skill_storage.TransactionJournal, "transition", failing)
    first = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert first.status_code != 200, first.text
    local_id = skill_link_local_id(source=BASE, visibility="public", tenant_id=None, hub_id=HUB_ID)
    identity = LinkIdentity("skill", local_id, BASE, "public", None, HUB_ID)
    values = read_config_values(root) or {}
    # The new tree IS in place and the journal is still ours (phase
    # 'prepared'): the rename boundary was crossed, so the staged grant must
    # survive for recovery even though the phase alone is coarse.
    stored = pol.stored_link_policy(values, identity.link_id)
    assert stored is not None and stored.auto_push is True
    assert (root / "skills" / ALIAS / "SKILL.md").exists()
    assert prov.read_baseline(root, "skill", local_id) is None

    monkeypatch.setattr(skill_storage.TransactionJournal, "transition", real)
    retry = await client.post(
        "/v1/desktop/hub/skills/pull",
        json=pull_body(auto_push=True, confirm_public_egress=True),
    )
    assert retry.status_code == 200, retry.text
    record = prov.read_baseline(root, "skill", local_id)
    assert record is not None and record.lineage == "clean"
    kept = pol.stored_link_policy(read_config_values(root) or {}, identity.link_id)
    assert kept is not None and kept.auto_push is True


# -- subscribe / unpull -------------------------------------------------------------


def subscribe_body(**fields: Any) -> dict[str, Any]:
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


async def test_a_subscribe_links_without_installing_and_replays(api) -> None:
    """The link exists with no bytes; the same request returns the same link."""

    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    body = subscribe_body()

    response = await client.post("/v1/desktop/hub/skills/subscribe", json=body)
    assert response.status_code == 200, response.text
    row = response.json()["result"]["link"]

    assert row["kind"] == "skill" and row["name"] == ALIAS
    assert row["content_state"] == "absent" and row["local_path"] is None
    assert row["policy"]["subscribe"] is True and row["policy"]["auto_push"] is False
    assert row["eligibility"]["can_auto_push"] is False
    assert row["auto_will_apply"] is False
    assert nothing_installed(root)

    record = prov.read_baseline(root, "skill", row["local_id"])
    assert record is not None and record.content_state == "absent"
    assert record.local_alias == ALIAS and record.identity is not None
    assert record.recorded_by == "subscribe"
    assert record.fields == manifest()  # the remote text this link names
    assert len(holder["anon"].calls) == 1

    # The same request id replays the stored response: no second fetch, no
    # second baseline (the receipts discipline every mutation shares).
    replayed = await client.post("/v1/desktop/hub/skills/subscribe", json=body)
    replayed_result = dict(replayed.json()["result"])
    assert replayed_result.pop("replayed", None) is True
    assert replayed_result == response.json()["result"]
    assert len(holder["anon"].calls) == 1

    # The capability-gated inventory carries the state a client renders.
    listed = await client.get("/v1/desktop/hub/links", params={"kind": "skill"})
    (link_row,) = listed.json()["result"]["links"]
    assert link_row["link_id"] == row["link_id"]
    assert link_row["content_state"] == "absent" and link_row["local_path"] is None

    # Replay, not a duplicate: a fresh request id returns the SAME link.
    again = await client.post(
        "/v1/desktop/hub/skills/subscribe", json={**body, "request_id": str(uuid.uuid4())}
    )
    assert again.status_code == 200, again.text
    assert again.json()["result"]["link"]["link_id"] == row["link_id"]
    assert len([r for r in prov.iter_baselines(root) if r.kind == "skill"]) == 1


async def test_the_unpull_route_removes_the_copy_and_keeps_the_link(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, envelope(manifest()))
    pulled = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    link = pulled.json()["result"]["link"]
    assert (root / "skills" / ALIAS).exists()

    body = {"request_id": str(uuid.uuid4()), "link_id": link["link_id"]}
    response = await client.post("/v1/desktop/hub/skills/unpull", json=body)
    assert response.status_code == 200, response.text
    row = response.json()["result"]["link"]

    assert row["link_id"] == link["link_id"] and row["content_state"] == "absent"
    assert row["local_path"] is None
    assert not (root / "skills" / ALIAS).exists()
    record = prov.read_baseline(root, "skill", link["local_id"])
    assert record is not None and record.identity == linked_identity(link)
    assert record.content_state == "absent"

    # Unpull is a state transition, so repeating it succeeds with the same row.
    again = await client.post(
        "/v1/desktop/hub/skills/unpull", json={**body, "request_id": str(uuid.uuid4())}
    )
    assert again.status_code == 200, again.text
    assert again.json()["result"]["link"]["content_state"] == "absent"


async def test_a_public_subscribe_of_an_org_skill_is_one_plain_404(api) -> None:
    client, root, holder = api
    holder["anon"] = SkillClient(BASE, org_envelope(manifest()))

    foreign = await client.post("/v1/desktop/hub/skills/subscribe", json=subscribe_body())
    holder["anon"] = SkillClient(BASE, APIError("Not Found", status_code=404))
    missing = await client.post("/v1/desktop/hub/skills/subscribe", json=subscribe_body())

    # The PULL arm refuses the same row identically: one body for "missing" and
    # "not yours", on both the install and the subscribe path.
    holder["anon"] = SkillClient(BASE, org_envelope(manifest()))
    pulled_foreign = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())
    holder["anon"] = SkillClient(BASE, APIError("Not Found", status_code=404))
    pulled_missing = await client.post("/v1/desktop/hub/skills/pull", json=pull_body())

    assert foreign.status_code == 404 and missing.status_code == 404
    assert foreign.json() == missing.json()
    assert pulled_foreign.status_code == 404
    assert pulled_foreign.json() == pulled_missing.json() == missing.json()
    assert nothing_installed(root)
    assert not [r for r in prov.iter_baselines(root) if r.kind == "skill"]
