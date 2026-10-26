"""``POST /v1/desktop/hub/updates/push``: preview-then-confirm republish of a linked skill.

Driven against the REAL route stack (FastAPI + ASGI transport + receipts/error
wrappers), the real ``hub_sync`` service, the real outbox and the REAL managed skill
storage on an isolated HOME/config root. Only the two network seams are faked: the
credential-free check read (``get_skill``) and the push transport, so "the hub saw
nothing" is an assertion about requests that would have left the machine, and every
state assertion reads PERSISTED effects (receipt files, baseline, inventory).

The contract is the desktop's ``hub.push`` arm: the preview object field for field
(``HubPushPreview``), dry-run/confirm pairing, and the typed codes the renderer keys
on (``publication-preview-changed``, ``remote-revision-changed``).
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

from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.hub_sync import provenance as prov
from local_operator.hub_sync.push import PushRefused
from local_operator.server.routes import desktop_hub, desktop_sessions
from tests.unit.hub_sync.test_push import FakeTransport
from tests.unit.server.test_desktop_hub_skills import (
    ALIAS,
    BASE,
    SkillClient,
    envelope,
    manifest,
    pull_body,
)

pytestmark = pytest.mark.asyncio

AUTH = {"Authorization": "Bearer desktop-test-token"}
REVISION = '"hub-rev-1"'


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "desktop-test-token")
    holder: dict[str, Any] = {}

    async def fake_clients(_cm: Any, _store: Any = None):
        def for_tenant(tenant: Any) -> Any:
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
        yield client, root, holder, monkeypatch
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


def push_body(**fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "link_id": "",
        "direction": "push",
        "dry_run": True,
    }
    body.update(fields)
    return body


def confirm_body(link_id: str, receipt: str, **fields: Any) -> dict[str, Any]:
    return push_body(
        link_id=link_id, dry_run=False, preview_receipt=receipt, confirm_publish=True, **fields
    )


async def link_skill(rig, *, alias: str = ALIAS, hub_id: str = "sk-1") -> tuple[str, SkillClient]:
    """Pull a public skill (a real link + baseline) and arm a push transport fake."""

    client, root, holder, monkeypatch = rig
    anon = SkillClient(BASE, {**envelope(manifest(alias)), "id": hub_id})
    holder["anon"] = anon
    pulled = await client.post(
        "/v1/desktop/hub/skills/pull", json=pull_body(local_alias=alias, hub_id=hub_id)
    )
    assert pulled.status_code == 200, pulled.text
    return pulled.json()["result"]["link"]["link_id"], anon


def edit_locally(root: Path, alias: str = ALIAS, body: str = "Say hello, loudly.\n") -> None:
    document = manifest(alias, body=body)
    skill_md = document["files"][0]["content"]
    (root / "skills" / alias / "SKILL.md").write_text(skill_md)


def arm_transport(rig, link_alias: str = ALIAS, **kw: Any) -> FakeTransport:
    """A recording hub: a republish returns the edited manifest as accepted."""

    _, root, _, monkeypatch = rig
    edited = manifest(link_alias, body="Say hello, loudly.\n")
    transport = FakeTransport(revision=REVISION, accepted=edited, **kw)
    monkeypatch.setattr(
        "local_operator.hub_sync.service.push_transport", lambda _ctx, _rec: (transport, None)
    )
    return transport


async def preview(client: AsyncClient, link_id: str) -> dict[str, Any]:
    response = await client.post(
        "/v1/desktop/hub/updates/push", json=push_body(link_id=link_id, dry_run=True)
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


def spent(root: Path) -> list[str]:
    directory = root / "hub" / "push-previews"
    return sorted(p.name for p in directory.glob("*.json")) if directory.exists() else []


async def test_preview_returns_the_pinned_proposal_and_sends_nothing(api) -> None:
    client, root, _, _ = api
    link_id, anon = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    reads_before = len(anon.calls)

    result = await preview(client, link_id)

    pv = result["preview"]
    # HubPushPreview, field for field.
    assert set(pv) == {
        "preview_receipt",
        "link_id",
        "direction",
        "hub_source",
        "visibility",
        "tenant_id",
        "local_fingerprint",
        "expected_policy_revision",
        "expected_remote_revision",
        "proposal_fingerprint",
        "expires_at",
        "proposal",
        "warnings",
        "can_confirm",
        "reason",
    }
    assert pv["link_id"] == link_id and pv["direction"] == "push"
    assert pv["hub_source"] == BASE and pv["visibility"] == "public" and pv["tenant_id"] is None
    assert pv["expected_remote_revision"] == REVISION
    assert pv["can_confirm"] is True and pv["reason"] is None
    assert pv["proposal"]["kind"] == "skill"
    assert "loudly" in json.dumps(pv["proposal"]["fields"])
    assert result["reports"] == []
    # The link row the UI binds against carries the SAME policy/remote revisions.
    (row,) = [r for r in result["links"]["links"] if r["link_id"] == link_id]
    assert row["policy"]["revision"] == pv["expected_policy_revision"]
    assert row["remote_revision"] == pv["expected_remote_revision"]
    # NOTHING left the machine: no hub read, no preview call, no commit.
    assert transport.preview_calls == [] and transport.commit_calls == []
    assert len(anon.calls) == reads_before
    # The only write is the receipt.
    assert spent(root) == [pv["preview_receipt"] + ".json"]


async def test_confirm_pushes_exactly_once_and_replay_does_not_resend(api) -> None:
    client, root, _, _ = api
    link_id, anon = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    receipt = (await preview(client, link_id))["preview"]["preview_receipt"]

    body = confirm_body(link_id, receipt)
    first = await client.post("/v1/desktop/hub/updates/push", json=body)
    assert first.status_code == 200, first.text
    (report,) = first.json()["result"]["reports"]
    assert report["outcome"] == "pushed" and report["applied"] is True
    assert report["direction"] == "push" and report["link_id"] == link_id
    assert len(transport.commit_calls) == 1
    # The edited bytes are what was sent.
    assert "loudly" in json.dumps(transport.commit_calls[0]["document"])
    # The outbound receipt advanced and the preview receipt was spent.
    record = prov.read_baseline(root, "skill", report["local_id"])
    assert record is not None and record.outbound is not None
    assert spent(root) == []

    # Same request id: the recorded answer is replayed, nothing is re-sent.
    again = await client.post("/v1/desktop/hub/updates/push", json=body)
    assert again.status_code == 200 and again.json()["result"]["replayed"] is True
    # A NEW request id with the SPENT receipt finds no receipt: a typed refusal.
    other = await client.post("/v1/desktop/hub/updates/push", json=confirm_body(link_id, receipt))
    assert other.status_code == 409, other.text
    assert other.json()["detail"]["code"] == "publication-preview-changed"
    assert len(transport.commit_calls) == 1


async def test_a_confirm_without_a_receipt_or_a_dry_run_with_one_is_refused(api) -> None:
    client, root, _, _ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)

    bare = await client.post(
        "/v1/desktop/hub/updates/push", json=push_body(link_id=link_id, dry_run=False)
    )
    assert bare.status_code == 422
    smuggled = await client.post(
        "/v1/desktop/hub/updates/push",
        json=push_body(link_id=link_id, dry_run=True, preview_receipt="x" * 32),
    )
    assert smuggled.status_code == 422
    unconfirmed = await client.post(
        "/v1/desktop/hub/updates/push",
        json=push_body(link_id=link_id, dry_run=False, preview_receipt="x" * 32),
    )
    assert unconfirmed.status_code == 422
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_unknown_or_path_shaped_receipts_are_refused_before_any_send(api) -> None:
    client, root, _, _ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    for token in ("A" * 40, "../../etc/passwd", "short"):
        response = await client.post(
            "/v1/desktop/hub/updates/push", json=confirm_body(link_id, token)
        )
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == "publication-preview-changed"
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_an_expired_receipt_is_refused_and_nothing_is_sent(api) -> None:
    client, root, _, _ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    receipt = (await preview(client, link_id))["preview"]["preview_receipt"]
    path = root / "hub" / "push-previews" / f"{receipt}.json"
    record = json.loads(path.read_text())
    record["expires_at"] = "2000-01-01T00:00:00Z"
    path.write_text(json.dumps(record))

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, receipt)
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "publication-preview-changed"
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_a_moved_hub_revision_is_refused_with_the_typed_code_and_no_send(api) -> None:
    client, root, _, _ = api
    link_id, anon = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    receipt = (await preview(client, link_id))["preview"]["preview_receipt"]
    # Somebody else published: the hub now serves a different revision.
    anon.payload = envelope(manifest(ALIAS, body="Theirs.\n"), revision='"hub-rev-2"')

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, receipt)
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "remote-revision-changed"
    assert transport.commit_calls == [] and transport.preview_calls == []
    # The receipt is NOT spent by a refusal, but a re-preview is needed: it is stale
    # against the moved hub and keeps refusing rather than resending.
    again = await client.post("/v1/desktop/hub/updates/push", json=confirm_body(link_id, receipt))
    assert again.status_code == 409 and transport.commit_calls == []


async def test_a_changed_policy_revision_is_refused_and_nothing_is_sent(api) -> None:
    client, root, _, _ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    pv = (await preview(client, link_id))["preview"]
    patched = await client.patch(
        f"/v1/desktop/hub/links/{link_id}",
        json={
            "request_id": str(uuid.uuid4()),
            "expected_policy_revision": pv["expected_policy_revision"],
            "subscribe": False,
        },
    )
    assert patched.status_code == 200, patched.text

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, pv["preview_receipt"])
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "policy-revision-changed"
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_a_local_edit_after_the_review_is_refused_and_nothing_is_sent(api) -> None:
    client, root, _, _ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api)
    edit_locally(root)
    receipt = (await preview(client, link_id))["preview"]["preview_receipt"]
    edit_locally(root, body="Unreviewed bytes.\n")

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, receipt)
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "publication-preview-changed"
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_a_receipt_minted_for_one_link_cannot_publish_another(api) -> None:
    """Scope binding: B's confirm carrying A's receipt is refused, A's skill untouched."""

    client, root, holder, _ = api
    link_a, _ = await link_skill(api, alias="alpha", hub_id="sk-a")
    link_b, _ = await link_skill(api, alias="beta", hub_id="sk-b")
    transport = arm_transport(api, "alpha")
    edit_locally(root, "alpha")
    edit_locally(root, "beta")
    receipt_a = (await preview(client, link_a))["preview"]["preview_receipt"]

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_b, receipt_a)
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "publication-preview-changed"
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_a_private_skill_cannot_be_pushed_by_a_caller_outside_its_scope(api) -> None:
    """The negative case: the hub refuses membership, the route answers a typed 403.

    The credential selector is the caller's identity here: the hub answers
    ``not_a_member`` to the commit, which must surface as a typed refusal (not a
    success, not an untyped 500), leave the preview receipt unspent, and record no
    accepted outbound state.
    """

    client, root, _, _ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api, fail_commit=PushRefused("not_a_member", "not a member"))
    edit_locally(root)
    receipt = (await preview(client, link_id))["preview"]["preview_receipt"]

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, receipt)
    )
    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "not_a_member"
    assert len(transport.commit_calls) == 1  # attempted exactly once, and refused
    record = prov.read_baseline(root, "skill", prov.iter_baselines(root)[0].local_id)
    assert record is not None and record.outbound is not None
    # Not advanced to the (refused) edit: the accepted pair is still the pulled tree.
    assert "loudly" not in json.dumps(dict(record.outbound.accepted_fields))
    assert len(spent(root)) == 1


async def test_a_signed_out_caller_cannot_preview_a_confirmable_push_or_confirm(api) -> None:
    client, root, holder, _ = api
    link_id, _ = await link_skill(api)
    edit_locally(root)
    holder.pop("client", None)  # the real push_transport: no signed-in client

    pv = (await preview(client, link_id))["preview"]
    assert pv["can_confirm"] is False and pv["reason"] == "no-credential"

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, pv["preview_receipt"])
    )
    # The receipt says can_confirm=false, so the confirm is refused by NAME before
    # any request is made, whatever the renderer's disabled button did. A missing
    # credential is the 401 whose remedy is signing in.
    assert response.status_code == 401, response.text


async def test_an_unknown_link_is_a_plain_404(api) -> None:
    client, *_ = api
    response = await client.post(
        "/v1/desktop/hub/updates/push", json=push_body(link_id="f" * 64, dry_run=True)
    )
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "link-not-found"


async def test_an_unchanged_skill_previews_as_not_confirmable(api) -> None:
    client, *_ = api
    link_id, _ = await link_skill(api)
    transport = arm_transport(api)

    pv = (await preview(client, link_id))["preview"]

    assert pv["can_confirm"] is False and pv["reason"] == "no-changes"
    assert transport.commit_calls == [] and transport.preview_calls == []


async def test_the_push_route_is_registered_beside_its_hub_siblings() -> None:
    routes = {
        (getattr(route, "path", ""), method)
        for route in desktop_hub.router.routes
        for method in getattr(route, "methods", set())
    }
    assert ("/v1/desktop/hub/updates/push", "POST") in routes
    assert ("/v1/desktop/hub/updates/apply", "POST") in routes


class SignedInClient(SkillClient):
    """A signed-in client: the transport builder reads its bearer headers."""

    api_key = "k"

    def bearer_headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer k"}


async def test_a_client_for_another_origin_cannot_publish_this_link(api) -> None:
    """The REAL transport builder: a credential never follows a link to a foreign base.

    ``push_transport`` verifies the answering client's base against the link's
    recorded ``hub_source`` before any credential could leave, so a signed-in client
    pointed at a different hub is refused (``origin-conflict``) with zero requests.
    """

    client, root, holder, _ = api
    link_id, _ = await link_skill(api)
    edit_locally(root)
    foreign = SignedInClient("https://other-hub.example/v1", envelope(manifest()))
    holder["client"] = foreign
    pv = (await preview(client, link_id))["preview"]
    assert pv["can_confirm"] is False and pv["reason"] == "origin-conflict"

    response = await client.post(
        "/v1/desktop/hub/updates/push", json=confirm_body(link_id, pv["preview_receipt"])
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "origin-conflict"
    assert foreign.calls == []
