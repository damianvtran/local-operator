"""Send-only agent file semantics; recorders are NOT real hub/Mongo evidence."""

from __future__ import annotations

import dataclasses
import json

import pytest

from local_operator.team_bridge import host
from local_operator.team_bridge.client import (
    BridgeRefusal,
    agent_files_capability,
    files_capability,
)
from local_operator.team_bridge.tool import BridgeToolParams, execute_bridge
from tests.unit.team_bridge.test_agent_attachments import (
    AID1,
    AID2,
    CAP_ON,
    FakeHub,
    _assert_no_bytes,
    _dm,
    _png_bytes,
    _post,
    _runtime,
    _write,
)

CAP = {
    "files": {
        "version": 1,
        "enabled": True,
        "agent_upload": True,
        "download_only": True,
        "classes": [
            {"class": "text", "max_bytes": 2 * 1024 * 1024},
            {"class": "container", "max_bytes": 25 * 1024 * 1024},
        ],
        "max_items": 10,
        "max_images": 4,
    },
    **CAP_ON,
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(host, "owner_fence_refusal", lambda runtime: None)
    # This recorder represents an approved hub; real-client origin races live
    # in test_agent_file_origins with actual native ownership/admission.
    monkeypatch.setattr(
        "local_operator.team_bridge.client.org_oauth_destination_allowed",
        lambda base: base == "https://hub.example",
    )
    monkeypatch.setattr(host, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host, "admission_refusal", lambda runtime, *, is_read=False: None)


class Hub(FakeHub):
    base_url = "https://hub.example"

    async def status(self, **kwargs):
        return await super().status()

    async def upload_attachment(self, conversation_id, data, **kwargs):
        guard = kwargs.get("pre_http_guard")
        if guard:
            guard()
        result = await super().upload_attachment(conversation_id, data, **kwargs)
        return {"file": result["attachment"]} if kwargs.get("mode") == "file" else result


@pytest.mark.parametrize(
    "change",
    [
        {"agent_upload": False},
        {"agent_upload": None},
        {"agent_upload": 1},
        {"enabled": False},
        {"version": True},
        {"version": 1.0},
        {"version": 2},
        {"download_only": False},
        {"classes": []},
        {"classes": "text"},
    ],
)
def test_strict_agent_file_grant(change):
    payload = {"files": {**CAP["files"], **change}}
    assert agent_files_capability(payload) is None
    assert agent_files_capability(CAP) is not None
    assert files_capability({"files": {**CAP["files"], "agent_upload": False}}) is not None
    assert agent_files_capability(CAP_ON) is None


def test_files_are_discoverable_and_actor_is_not_model_selected():
    properties = BridgeToolParams.model_json_schema()["properties"]
    assert "files" in properties
    assert "agent_upload=true" in properties["files"]["description"]
    assert "actor_endpoint_id" not in properties


@pytest.mark.asyncio
@pytest.mark.parametrize("args", [_dm(body=""), _post(body="")])
async def test_exact_files_only_ordered_actor_scoped_send(tmp_path, args):
    payloads = [b"exact text\r\n", b"PK\x03\x04\x00raw-container\xff"]
    paths = [_write(tmp_path, f"file-{i}.bin", value) for i, value in enumerate(payloads)]
    hub = Hub(status=CAP)
    outcome = await execute_bridge(_runtime(hub), {**args, "files": paths})
    assert outcome["ok"], outcome
    assert hub.uploads == payloads
    post = hub.named("conversation_send")[0]
    assert post["files"] == [AID1, AID2]
    assert post["actor_endpoint_id"] == "bep_1"
    assert post["expected_account_id"] == "acc_owner"
    assert post["text"] == ""
    assert not hub.named("delete_attachment")
    assert outcome["file_ids"] == [AID1, AID2]
    _assert_no_bytes(outcome, *payloads)
    assert all(path not in json.dumps(outcome) for path in paths)


@pytest.mark.asyncio
async def test_mixed_images_and_files_have_separate_lists(tmp_path):
    image = _write(tmp_path, "image.png", _png_bytes())
    content = b"file canary exact\r\n"
    path = _write(tmp_path, "note.txt", content)
    hub = Hub(status=CAP)
    out = await execute_bridge(
        _runtime(hub), _post(files=[path], attachments=[image], thread_root_id="bcmsg_root0001")
    )
    assert out["ok"], out
    post = hub.named("conversation_send")[0]
    assert post["attachment_ids"] == [AID1]
    assert post["files"] == [AID2]
    assert post["thread_root_id"] == "bcmsg_root0001"
    assert hub.uploads[1] == content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [{}, CAP_ON, {"files": {**CAP["files"], "agent_upload": False}}]
)
async def test_unavailable_never_probes_with_bytes_or_opens_dm(tmp_path, payload):
    hub = Hub(status=payload)
    out = await execute_bridge(_runtime(hub), _dm(files=[str(tmp_path / "does-not-exist")]))
    assert out["code"] == "files_unavailable"
    assert not hub.named("upload_attachment")
    assert not hub.named("create_conversation")
    assert not hub.named("conversation_send")


@pytest.mark.asyncio
async def test_failed_status_is_dark(tmp_path):
    hub = Hub(status_failure=TimeoutError())
    out = await execute_bridge(_runtime(hub), _post(files=[str(tmp_path / "missing")]))
    assert out["code"] == "files_unavailable"
    assert not hub.uploads


@pytest.mark.asyncio
async def test_duplicate_and_secret_file_fail_without_upload(tmp_path):
    path = _write(tmp_path, "note.txt", b"one exact file")
    hub = Hub(status=CAP)
    out = await execute_bridge(_runtime(hub), _post(files=[path, path]))
    assert out["code"] == "attachment_invalid"
    secret = _write(tmp_path, ".env", b"not-a-real-secret-canary")
    out = await execute_bridge(_runtime(hub), _post(files=[secret]))
    assert not out["ok"]
    assert not hub.uploads
    assert "not-a-real-secret-canary" not in json.dumps(out)


@pytest.mark.asyncio
async def test_partial_stage_and_definitive_send_cleanup_keep_actor(tmp_path):
    paths = [_write(tmp_path, f"file{i}.txt", f"file{i}".encode()) for i in range(2)]
    for hub in [
        Hub(status=CAP, fail_upload_at=1),
        Hub(status=CAP, send_failure=BridgeRefusal("denied", "refused", http_status=403)),
    ]:
        out = await execute_bridge(_runtime(hub), _post(files=paths))
        assert not out["ok"]
        deletes = hub.named("delete_attachment")
        assert deletes
        assert all(
            d["actor_endpoint_id"] == "bep_1" and d["expected_account_id"] == "acc_owner"
            for d in deletes
        )


@pytest.mark.asyncio
async def test_unknown_commit_replays_exact_ref_and_ids_without_cleanup(tmp_path):
    path = _write(tmp_path, "note.txt", b"uncertain file")
    hub = Hub(status=CAP, send_failure=TimeoutError())
    out = await execute_bridge(_runtime(hub), _post(files=[path]))
    assert out["code"] == "delivery_uncertain", out
    attempts = hub.named("conversation_send")
    assert len(attempts) == 2
    assert attempts[0]["client_ref"] == attempts[1]["client_ref"]
    assert attempts[0]["files"] == attempts[1]["files"] == [AID1]
    assert len(hub.uploads) == 1
    assert not hub.named("delete_attachment")


@pytest.mark.asyncio
async def test_owner_change_during_capability_read_cannot_stage(tmp_path):
    path = _write(tmp_path, "note.txt", b"owner fenced")
    hub = Hub(status=CAP)
    runtime = _runtime(hub)
    original = hub.status

    async def switched(**kwargs):
        result = await original(**kwargs)
        runtime.owner = dataclasses.replace(runtime.owner, account_id="acc_other")
        return result

    hub.status = switched
    out = await execute_bridge(runtime, _dm(files=[path]))
    assert out["code"] == "owner_changed", out
    assert not hub.uploads
    assert not hub.named("create_conversation")


@pytest.mark.asyncio
async def test_lost_response_then_same_ref_success_never_restages(tmp_path):
    path = _write(tmp_path, "note.txt", b"same ref recovery")
    hub = Hub(status=CAP, send_failure=TimeoutError())
    original = hub.conversation_send

    async def once(conversation_id, **kwargs):
        try:
            return await original(conversation_id, **kwargs)
        finally:
            hub.send_failure = None

    hub.conversation_send = once
    out = await execute_bridge(_runtime(hub), _post(files=[path]))
    assert out["ok"], out
    attempts = hub.named("conversation_send")
    assert len(attempts) == 2
    assert attempts[0]["client_ref"] == attempts[1]["client_ref"]
    assert attempts[0]["files"] == attempts[1]["files"]
    assert len(hub.uploads) == 1
    assert not hub.named("delete_attachment")


@pytest.mark.asyncio
async def test_combined_count_and_pair_lane_never_upload(tmp_path):
    path = _write(tmp_path, "note.txt", b"bounded count")
    image = _write(tmp_path, "image.png", _png_bytes())
    hub = Hub(status=CAP)
    out = await execute_bridge(_runtime(hub), _post(files=[path] * 10, attachments=[image]))
    assert out["code"] == "attachment_invalid"
    out = await execute_bridge(
        _runtime(hub), {"op": "send", "text": "not a file lane", "files": [path]}
    )
    assert not out["ok"]
    assert not hub.uploads


@pytest.mark.asyncio
async def test_generic_parent_symlink_swap_cannot_read_config_bytes(tmp_path, monkeypatch):
    import os

    from local_operator.team_bridge import agent_attachments as aa

    outside = tmp_path / "outside"
    outside.mkdir()
    path = _write(outside, "payload.bin", b"allowed")
    secret = tmp_path / "cfg" / "private"
    secret.mkdir()
    (secret / "payload.bin").write_bytes(b"never-upload-this-config-canary")
    original_open = os.open
    swapped = False

    def swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "/" and not swapped:
            swapped = True
            outside.rename(tmp_path / "old-outside")
            outside.symlink_to(secret, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(aa.os, "open", swap)
    monkeypatch.setattr(aa.os, "supports_dir_fd", os.supports_dir_fd | {swap})
    hub = Hub(status=CAP)
    out = await execute_bridge(_runtime(hub), _post(files=[path]))
    assert swapped, "negative control must reach anchored open, not fail setup"
    assert not out["ok"]
    assert not hub.uploads
    assert "never-upload-this-config-canary" not in json.dumps(out)


@pytest.mark.asyncio
async def test_real_tool_to_client_wire_preserves_exact_file_and_actor(tmp_path, monkeypatch):
    import httpx

    from local_operator.team_bridge import client as client_module
    from tests.unit.team_bridge.test_client import _make_client

    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda base: True)
    requests = []
    raw = b"exact-wire-file\r\n\x00\xff"
    path = _write(tmp_path, "wire.bin", raw)

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"result": CAP})
        if request.url.path.endswith("/attachments"):
            return httpx.Response(201, json={"result": {"file": {"id": AID1}}})
        assert request.url.path.endswith("/messages")
        return httpx.Response(201, json={"result": {"id": "bcmsg_wire0001", "seq": 1}})

    client = _make_client(handler)
    # The shared helper builds a FakeHub fixture; install the real wire client
    # on its typed runtime instead of passing an unrelated type to that helper.
    runtime = _runtime(Hub())
    runtime.client = client
    runtime.owner = dataclasses.replace(
        runtime.owner, account_id="acc_1", canonical_base_url=client.base_url
    )
    out = await execute_bridge(runtime, _post(files=[path]))
    assert out["ok"], out
    stage = [r for r in requests if r.url.path.endswith("/attachments")][0]
    post = [r for r in requests if r.url.path.endswith("/messages")][0]
    assert stage.content == raw
    assert stage.url.params["actor_endpoint_id"] == "bep_1"
    assert stage.url.params["tenant_id"] == runtime.tenant_id
    assert stage.headers["x-bridge-attachment-mode"] == "file"
    assert stage.headers["x-bridge-files-version"] == "1"
    assert json.loads(post.content)["files"] == [AID1]
    assert json.loads(post.content)["actor_endpoint_id"] == "bep_1"
    assert "attachments" not in json.loads(post.content)
    _assert_no_bytes(out, raw)


@pytest.mark.asyncio
async def test_cancelled_physical_file_send_retains_possibly_bound_rows(tmp_path):
    import asyncio

    path = _write(tmp_path, "note.txt", b"cancelled physical send")
    hub = Hub(status=CAP, send_failure=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await execute_bridge(_runtime(hub), _post(files=[path]))
    assert len(hub.named("conversation_send")) == 1
    assert not hub.named("delete_attachment")


@pytest.mark.asyncio
async def test_tenant_switch_after_first_stage_cleans_only_original_tuple(tmp_path):
    paths = [_write(tmp_path, f"tenant{i}.txt", f"tenant{i}".encode()) for i in range(2)]
    hub = Hub(status=CAP)
    runtime = _runtime(hub)
    original = hub.upload_attachment

    async def switched(conversation_id, data, **kwargs):
        result = await original(conversation_id, data, **kwargs)
        runtime.tenant_id = "ten_other"
        return result

    hub.upload_attachment = switched
    out = await execute_bridge(runtime, _post(files=paths))
    assert not out["ok"]
    assert len(hub.uploads) == 1
    assert not hub.named("conversation_send")
    deletes = hub.named("delete_attachment")
    assert deletes
    assert all(d["tenant_id"] == "ten_1" and d["actor_endpoint_id"] == "bep_1" for d in deletes)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["origin", "tenant", "endpoint"])
async def test_initial_owner_tuple_mismatch_refuses_before_status(tmp_path, changed):
    path = _write(tmp_path, "note.txt", b"must remain local")
    hub = Hub(status=CAP)
    runtime = _runtime(hub)
    if changed == "origin":
        hub.base_url = "https://other.example"
    elif changed == "tenant":
        runtime.tenant_id = "ten_other"
    else:
        runtime.owner = dataclasses.replace(runtime.owner, endpoint_id="")
    out = await execute_bridge(runtime, _post(files=[path]))
    assert not out["ok"]
    assert not hub.named("status")
    assert not hub.uploads
    assert not hub.named("conversation_send")
