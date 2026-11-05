"""Physical-origin races through the real built tool and persisted owner.

Only credentials and HTTP are scripted. No owner/admission fence is patched;
MockTransport records dispatch, not a real network/real-hub acceptance claim.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.harness.types import ToolContext
from local_operator.providers.auth_store import AuthStore
from local_operator.team_bridge import client as client_module
from local_operator.team_bridge import host, tool
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from tests.unit.team_bridge.test_agent_attachments import AID1, AID2, _png_bytes
from tests.unit.team_bridge.test_agent_files import CAP
from tests.unit.team_bridge.test_client import FakeRow, FakeStore

ORIGIN = "https://hub.example/v1"
OTHER = "https://not-approved.invalid/v1"
ACCOUNT = "acc_origin_a"
TENANT = "tnt_origin_a"
ENDPOINT = "bep_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
CONVERSATION = "bcv_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["status", "mixed_status", "cleanup", "mixed_cleanup", "mixed_partial"]
)
@pytest.mark.parametrize("window", ["freshness", "refresh"])
@pytest.mark.parametrize("destination", ["same-origin", "changed-origin", "newly-disallowed"])
async def test_normal_tool_pins_every_status_and_cleanup_attempt(
    tmp_path, monkeypatch, operation, window, destination
):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))
    ConfigManager(cfg).set_config_value("team_bridge", {"enabled": True, "session": "qa-origin"})
    native = AuthStore(db_path=cfg / "auth.db", config_dir=cfg)
    credential = native.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "access": "owned-local-token",
            "refresh": "unused",
            "expires": int((time.time() + 3600) * 1000),
            "account_id": ACCOUNT,
        },
    )
    native.close()
    is_status_operation = operation.endswith("status")
    state = {
        "active": False,
        "armed": False,
        "challenged": False,
        "switched": False,
        "allowed": True,
    }
    wire = []
    holder = {}

    class Freshness(FakeStore):
        async def ensure_oauth_fresh_or_raise(self, credential_id, *, force=False):
            data = await super().ensure_oauth_fresh_or_raise(credential_id, force=force)
            await asyncio.sleep(0)  # the real dispatch boundary must survive this wait
            if state["armed"] and ((window == "refresh") == force):
                state["armed"] = False
                state["switched"] = True
                holder["client"]._base_url = OTHER if destination == "changed-origin" else ORIGIN
                state["allowed"] = destination != "newly-disallowed"
            return data

    credentials = Freshness(
        rows=[
            FakeRow(
                credential.id,
                data={"access": "owned-wire-token", "account_id": ACCOUNT},
            )
        ],
        refresh={"access": "owned-refreshed-token"},
    )
    monkeypatch.setattr(
        client_module,
        "org_oauth_destination_allowed",
        lambda base: base == ORIGIN and state["allowed"],
    )
    image_uploads = 0

    def answer(result, status=200):
        return httpx.Response(status, json={"msg": "ok", "result": result})

    def revoke_and_arm():
        # Cleanup is NOT a new send. A full authority/admission guard here would
        # strand the original staged rows after an ordinary permission switch.
        ConfigManager(cfg).set_config_value("team_bridge", {"enabled": False})
        holder["runtime"].tenant_id = "tnt_after_switch"
        state["armed"] = True

    def handler(request):
        nonlocal image_uploads
        wire.append(
            (
                request.method,
                request.url.host,
                request.url.path,
                dict(request.url.params),
                "authorization" in request.headers,
                state["allowed"],
            )
        )
        is_status = request.url.path.endswith("/status")
        is_delete = request.method == "DELETE"
        target = state["active"] and (is_status if is_status_operation else is_delete)
        if target and window == "refresh" and not state["challenged"]:
            state["challenged"] = True
            if not is_status_operation:
                state["armed"] = True
            return httpx.Response(401, json={"code": "refresh_required"})
        if is_status:
            return answer(
                {
                    "api_version": "1",
                    "tenants": [
                        {
                            "tenant_id": TENANT,
                            "eligible": True,
                            "endpoint": {"id": ENDPOINT, "state": "active"},
                        }
                    ],
                    **CAP,
                }
            )
        if request.method == "POST" and request.url.path.endswith("/attachments"):
            if request.headers.get("x-bridge-attachment-mode") == "file":
                return answer({"file": {"id": AID2}}, 201)
            image_uploads += 1
            if operation == "mixed_partial" and image_uploads == 2:
                revoke_and_arm()
                return httpx.Response(403, json={"code": "agent_not_permitted"})
            return answer({"attachment": {"id": AID1}}, 201)
        if request.method == "POST" and request.url.path.endswith("/messages"):
            if not is_status_operation:
                revoke_and_arm()
                return httpx.Response(403, json={"code": "agent_not_permitted"})
            return answer({"id": "bcmsg_origin_control", "seq": 1}, 201)
        if is_delete:
            return httpx.Response(204)
        raise AssertionError("unexpected owned fixture request")

    client = client_module.TeamBridgeClient(
        cfg, base_url=ORIGIN, store=credentials, transport=httpx.MockTransport(handler)
    )
    holder["client"] = client
    owner = OwnerSnapshot(
        credential_id=credential.id,
        account_id=ACCOUNT,
        tenant_id=TENANT,
        endpoint_id=ENDPOINT,
        canonical_base_url=ORIGIN,
        enrollment_generation=1,
    ).validated()
    store = Store(cfg)
    store.commit_enrollment(
        owner,
        OwnerReceipt(consent_version="bridge_consent_v1", consent_ref=ENDPOINT),
        cos_session_id="qa-origin",
        require_no_owner=True,
    )
    assert store.active_owner() == owner
    assert store.current_generation_is_live(owner)
    assert store.owner_consent_receipt(owner)
    runtime = tool.BridgeRuntime(
        client=client,
        config_dir=cfg,
        tenant_id=TENANT,
        owner=owner,
        session_id="qa-origin",
    )
    holder["runtime"] = runtime
    assert host.outbound_tool_fence(runtime) is None
    assert await host.admit_bridge_runtime(runtime) == "admitted"
    context = ToolContext(cwd=str(tmp_path), bridge_runtime=runtime)
    built = tool.build_bridge_tool(context)
    assert built is not None
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    file = payload_dir / "owned.txt"
    file.write_bytes(b"owned generic-file origin control")
    args = {
        "op": "channel_post",
        "channel_id": CONVERSATION,
        "body": "",
        "files": [str(file)],
    }
    if operation.startswith("mixed"):
        images = []
        for index in range(2 if operation == "mixed_partial" else 1):
            path = payload_dir / f"image-{index}.png"
            path.write_bytes(_png_bytes(size=(16 + index, 16 + index)))
            images.append(str(path))
        args["attachments"] = images
    wire.clear()
    state["active"] = True
    if is_status_operation:
        state["armed"] = True
    result = await built.execute("owned-origin-call", args, None, None, context)
    outcome = result.details or {}
    assert state["switched"], "freshness/refresh hook did not run"
    assert all(entry[1] == "hub.example" for entry in wire), "changed-origin dispatch escaped"
    assert all(entry[5] for entry in wire), "newly disallowed dispatch escaped"
    assert all(entry[4] for entry in wire)
    if destination != "same-origin":
        assert outcome.get("ok") is not True
        if window == "freshness":
            assert not any(
                (r[0] == "DELETE" if not is_status_operation else r[2].endswith("/status"))
                for r in wire
            )
    elif is_status_operation:
        assert outcome.get("ok") is True, outcome
    else:
        deletes = [r for r in wire if r[0] == "DELETE"]
        assert len(deletes) == (2 if operation == "mixed_cleanup" else 1) + (window == "refresh")
        assert all(r[3] == {"tenant_id": TENANT, "actor_endpoint_id": ENDPOINT} for r in deletes)
        assert outcome.get("ok") is not True
    if window == "refresh":
        assert state["challenged"]
        assert any(force for _, force in credentials.refresh_calls)
