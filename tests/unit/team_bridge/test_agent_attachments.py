"""Agent image attachments on ``bridge`` ``dm`` / ``channel_post`` (slice S1).

Pins the contract in ``local_operator/team_bridge/agent_attachments.py``:

* the option exists only when the hub advertises ``attachments.agent_upload``
  (``True`` literally), and with it absent a text-only send is the SAME client
  call as before;
* the flow is validate -> stage as the agent -> send with ``attachment_ids``,
  and a send that does not land deletes the staged ids;
* NOTHING but ``batt_`` ids and metadata reaches the tool result: the no-bytes
  tests search the whole outcome for the image's base64 and sha256;
* an inline ``data:image/...;base64,`` run in message text is refused with an
  error that points at ``attachments`` (the root cause of the operator's
  multi-hundred-KiB base64 audit row).

The hub is a recorder, not a live agent-server: this is TOOL and CLIENT-WIRE
semantics (the server side is agent-server PR #283). Nothing here proves a real
hub accepts the upload.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from PIL import Image

from local_operator.team_bridge import agent_attachments as aa
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import (
    BridgeRefusal,
    TeamBridgeClient,
    attachments_capability,
)
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.tool import (
    BridgeRuntime,
    BridgeToolParams,
    execute_bridge,
)
from tests.unit.team_bridge.test_client import _make_client, _ok

TENANT = "ten_1"
CID = "bcv_abcd1234"
DM_ID = "bcv_dm123456"
AID1 = "batt_0f8fad5b-d9cb-469f-a165-70867728950e"
AID2 = "batt_7c9e6679-7425-40de-944b-e07fc1f90ae7"
PEER = "acc_peer"
PEER_ENDPOINT = "bep_peer"

CAP_ON = {
    "attachments": {
        "enabled": True,
        "agent_upload": True,
        "mime_types": ["image/png", "image/jpeg"],
        "max_count": 4,
        "max_bytes": 5 * 1024 * 1024,
    }
}
CAP_HUMAN_ONLY = {
    "attachments": {"enabled": True, "mime_types": ["image/png", "image/jpeg"], "max_count": 4}
}


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The reply guard writes its budget under the runtime's config dir (".").
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _png_bytes(
    color: tuple[int, int, int] = (200, 30, 30), size: tuple[int, int] = (64, 48)
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


class FakeHub:
    """Records every call the tool makes; scripts the hub's answers."""

    def __init__(
        self,
        *,
        status: Any = CAP_ON,
        status_failure: BaseException | None = None,
        send_failure: BaseException | None = None,
        fail_upload_at: int | None = None,
    ) -> None:
        self.status_payload = status
        self.status_failure = status_failure
        self.send_failure = send_failure
        self.fail_upload_at = fail_upload_at
        self.calls: list[dict[str, Any]] = []
        self.uploads: list[bytes] = []
        self._ids = [AID1, AID2, "batt_11111111-2222-3333-4444-555555555555"]

    def named(self, method: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method]

    async def status(self) -> Any:
        self.calls.append({"method": "status"})
        if self.status_failure is not None:
            raise self.status_failure
        return self.status_payload

    async def directory(self, tenant_id: str, *, expected_account_id: str | None = None) -> Any:
        self.calls.append({"method": "directory"})
        return {
            "members": [
                {
                    "account_id": PEER,
                    "display_label": "Robin",
                    "role": "member",
                    "bridge_enabled": True,
                    "bridge_state": "enabled",
                    "agent": {"endpoint_id": PEER_ENDPOINT, "name": "Aida", "state": "active"},
                }
            ]
        }

    async def create_conversation(self, body: Any, **kwargs: Any) -> Any:
        self.calls.append({"method": "create_conversation"})
        return {"conversation": {"id": DM_ID, "kind": "dm", "members": []}}

    async def upload_attachment(self, conversation_id: str, data: bytes, **kwargs: Any) -> Any:
        index = len(self.uploads)
        self.calls.append(
            {"method": "upload_attachment", "conversation_id": conversation_id, **kwargs}
        )
        if self.fail_upload_at is not None and index == self.fail_upload_at:
            raise BridgeRefusal("agent_not_permitted", "refused", http_status=403)
        self.uploads.append(data)
        return {"attachment": {"id": self._ids[index], "state": "staged"}}

    async def delete_attachment(
        self, conversation_id: str, attachment_id: str, **kwargs: Any
    ) -> Any:
        self.calls.append(
            {
                "method": "delete_attachment",
                "conversation_id": conversation_id,
                "attachment_id": attachment_id,
                **kwargs,
            }
        )
        return {}

    async def conversation_send(self, conversation_id: str, **kwargs: Any) -> Any:
        guard = kwargs.pop("pre_http_guard", None)
        if guard is not None:
            guard()
        self.calls.append(
            {"method": "conversation_send", "conversation_id": conversation_id, **kwargs}
        )
        if self.send_failure is not None:
            raise self.send_failure
        return {"id": "bcmsg_post0001", "seq": 7}


def _runtime(hub: FakeHub) -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, hub),
        config_dir=".",
        tenant_id=TENANT,
        owner=OwnerSnapshot(
            credential_id=1,
            account_id="acc_owner",
            tenant_id=TENANT,
            endpoint_id="bep_1",
            canonical_base_url="https://hub.example",
            enrollment_generation=1,
        ),
        session_id="sess_1",
    )


def _post(**extra: Any) -> dict[str, Any]:
    return {"op": "channel_post", "channel_id": CID, "body": "see attached", **extra}


def _dm(**extra: Any) -> dict[str, Any]:
    return {"op": "dm", "endpoint_id": PEER_ENDPOINT, "body": "see attached", **extra}


def _assert_no_bytes(outcome: Any, *images: bytes) -> None:
    """The whole outcome carries no base64, data URI or digest of any image."""

    blob = json.dumps(outcome, default=str)
    assert "data:image" not in blob and "base64" not in blob
    for data in images:
        assert base64.b64encode(data).decode("ascii") not in blob
        assert hashlib.sha256(data).hexdigest() not in blob


# -- capability + client wire -------------------------------------------------


def test_agent_upload_is_carried_only_when_literally_true() -> None:
    assert attachments_capability(CAP_ON) == {
        "enabled": True,
        "agent_upload": True,
        "mime_types": ["image/png", "image/jpeg"],
        "max_count": 4,
        "max_bytes": 5 * 1024 * 1024,
    }
    for bad in (False, None, "true", 1, 0, []):
        block = dict(CAP_ON["attachments"], agent_upload=bad)
        assert "agent_upload" not in (attachments_capability({"attachments": block}) or {})
    # a hub that never heard of the key: the normalized block is unchanged.
    assert attachments_capability(CAP_HUMAN_ONLY) == {
        "enabled": True,
        "mime_types": ["image/png", "image/jpeg"],
        "max_count": 4,
    }


@pytest.mark.asyncio
async def test_upload_and_delete_name_the_actor_only_when_given() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        return _ok({"attachment": {"id": AID1, "state": "staged"}}, status=201)

    client = _make_client(handler)
    data = _png_bytes()
    await client.upload_attachment(CID, data, tenant_id="tnt_1")
    await client.upload_attachment(CID, data, tenant_id="tnt_1", actor_endpoint_id="bep_1")
    await client.delete_attachment(CID, AID1, tenant_id="tnt_1", actor_endpoint_id="bep_1")
    await client.delete_attachment(CID, AID1)
    human, agent, agent_delete, human_delete = seen
    assert dict(human.url.params) == {"tenant_id": "tnt_1"}  # unchanged human wire
    assert dict(agent.url.params) == {"tenant_id": "tnt_1", "actor_endpoint_id": "bep_1"}
    assert agent.content == data and agent.headers["content-type"] == "application/octet-stream"
    assert "x-bridge-attachment-mode" not in {k.lower() for k in agent.headers}  # image mode only
    assert dict(agent_delete.url.params) == {"tenant_id": "tnt_1", "actor_endpoint_id": "bep_1"}
    assert dict(human_delete.url.params) == {}


# -- params -------------------------------------------------------------------


def test_attachments_param_is_not_published_in_the_tool_schema() -> None:
    # Default-off: a hub-gated option must not move every session's schema.
    assert "attachments" not in BridgeToolParams.model_json_schema()["properties"]
    assert BridgeToolParams(op="dm", attachments=["/a.png"]).attachments == ["/a.png"]
    assert BridgeToolParams(op="dm").attachments == []


@pytest.mark.asyncio
async def test_more_than_four_or_non_list_attachments_fail_validation(tmp_path: Path) -> None:
    hub = FakeHub()
    runtime = _runtime(hub)
    five = [_write(tmp_path, f"{i}.png", _png_bytes((i, 0, 0))) for i in range(5)]
    for bad in (five, "/a.png", [1]):
        out = await execute_bridge(runtime, _post(attachments=bad))
        assert out["ok"] is False and out["code"] == "invalid_request"
    assert hub.calls == []


@pytest.mark.asyncio
async def test_attachments_on_other_ops_are_refused_before_any_call() -> None:
    hub = FakeHub()
    out = await execute_bridge(
        _runtime(hub), {"op": "channel_history", "channel_id": CID, "attachments": ["/a.png"]}
    )
    assert out["ok"] is False and out["code"] == "attachment_invalid"
    assert hub.calls == []


# -- capability off => identical behaviour ------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CAP_HUMAN_ONLY, {}, {"attachments": {"enabled": False}}])
async def test_capability_off_refuses_attachments_and_sends_nothing(
    tmp_path: Path, status: Any
) -> None:
    hub = FakeHub(status=status)
    path = _write(tmp_path, "a.png", _png_bytes())
    for args in (_post(attachments=[path]), _dm(attachments=[path])):
        out = await execute_bridge(_runtime(hub), args)
        assert out["ok"] is False and out["code"] == "attachments_unavailable"
    assert hub.named("upload_attachment") == []
    assert hub.named("conversation_send") == []
    assert hub.named("create_conversation") == []  # a refused DM leaves no conversation behind


@pytest.mark.asyncio
async def test_unreadable_status_reads_as_capability_off(tmp_path: Path) -> None:
    hub = FakeHub(status_failure=RuntimeError("down"))
    out = await execute_bridge(
        _runtime(hub), _post(attachments=[_write(tmp_path, "a.png", _png_bytes())])
    )
    assert out["code"] == "attachments_unavailable"
    assert hub.named("upload_attachment") == []


@pytest.mark.asyncio
async def test_text_only_send_is_the_pre_attachment_call_and_asks_no_status() -> None:
    hub = FakeHub(status=CAP_HUMAN_ONLY)
    out = await execute_bridge(_runtime(hub), _post())
    assert out["ok"] is True and "attachments" not in out
    (send,) = hub.named("conversation_send")
    assert "attachment_ids" not in send
    assert hub.named("status") == [] and hub.named("upload_attachment") == []
    assert out["content"].startswith(f"posted to {CID}: message bcmsg_post0001")


# -- the happy paths ----------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_post_stages_as_the_agent_then_sends_ids_only(tmp_path: Path) -> None:
    hub = FakeHub()
    original = _png_bytes()
    out = await execute_bridge(
        _runtime(hub), _post(attachments=[_write(tmp_path, "a.png", original)])
    )
    assert out["ok"] is True, out
    (upload,) = hub.named("upload_attachment")
    assert upload["conversation_id"] == CID
    assert upload["actor_endpoint_id"] == "bep_1" and upload["tenant_id"] == TENANT
    (send,) = hub.named("conversation_send")
    assert send["attachment_ids"] == [AID1] and send["actor_endpoint_id"] == "bep_1"
    assert [c["method"] for c in hub.calls].index("upload_attachment") < [
        c["method"] for c in hub.calls
    ].index("conversation_send")
    assert out["attachments"] == [
        {"id": AID1, "mime": "image/png", "size": len(hub.uploads[0]), "width": 64, "height": 48}
    ]
    assert AID1 in out["content"]
    _assert_no_bytes(out, original, hub.uploads[0])


@pytest.mark.asyncio
async def test_dm_stages_into_the_opened_dm_and_keeps_order(tmp_path: Path) -> None:
    hub = FakeHub()
    first, second = _png_bytes((9, 9, 9)), _png_bytes((1, 2, 3), (32, 32))
    out = await execute_bridge(
        _runtime(hub),
        _dm(attachments=[_write(tmp_path, "1.png", first), _write(tmp_path, "2.png", second)]),
    )
    assert out["ok"] is True, out
    methods = [c["method"] for c in hub.calls]
    assert methods.index("create_conversation") < methods.index("upload_attachment")
    assert {c["conversation_id"] for c in hub.named("upload_attachment")} == {DM_ID}
    (send,) = hub.named("conversation_send")
    assert send["attachment_ids"] == [AID1, AID2]
    assert [a["id"] for a in out["attachments"]] == [AID1, AID2]
    _assert_no_bytes(out, first, second, *hub.uploads)


@pytest.mark.asyncio
async def test_jpeg_is_accepted_and_sniffed_by_bytes_not_extension(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), (0, 80, 160)).save(buffer, format="JPEG")
    hub = FakeHub()
    out = await execute_bridge(
        _runtime(hub), _post(attachments=[_write(tmp_path, "photo.dat", buffer.getvalue())])
    )
    assert out["ok"] is True and out["attachments"][0]["mime"] == "image/jpeg"


# -- refusals -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrong_type_missing_relative_directory_and_duplicate_are_refused(
    tmp_path: Path,
) -> None:
    hub = FakeHub()
    gif = io.BytesIO()
    Image.new("RGB", (8, 8)).save(gif, format="GIF")
    good = _write(tmp_path, "ok.png", _png_bytes())
    cases = {
        "attachment_type_unsupported": [
            _write(tmp_path, "notes.png", b"hello this is text"),  # lies about its extension
            _write(tmp_path, "anim.gif", gif.getvalue()),
        ],
        "attachment_unreadable": [str(tmp_path / "missing.png")],
        "attachment_invalid": ["relative/a.png", str(tmp_path), "   "],
    }
    for code, paths in cases.items():
        for path in paths:
            out = await execute_bridge(_runtime(hub), _post(attachments=[path]))
            assert out["ok"] is False and out["code"] == code, (path, out)
    dup = await execute_bridge(_runtime(hub), _post(attachments=[good, good]))
    assert dup["code"] == "attachment_invalid"
    assert hub.named("upload_attachment") == [] and hub.named("conversation_send") == []


@pytest.mark.asyncio
async def test_oversize_source_is_refused_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aa, "_SOURCE_MAX_BYTES", 1000)
    hub = FakeHub()
    path = _write(tmp_path, "big.png", _png_bytes((5, 5, 5), (400, 400)))
    assert Path(path).stat().st_size > 0
    out = await execute_bridge(_runtime(hub), _post(attachments=[path]))
    assert out["code"] == "attachment_too_large"
    assert hub.named("upload_attachment") == []


@pytest.mark.asyncio
async def test_hub_size_limit_is_honoured(tmp_path: Path) -> None:
    status = {"attachments": dict(CAP_ON["attachments"], max_bytes=10)}
    hub = FakeHub(status=status)
    out = await execute_bridge(
        _runtime(hub), _post(attachments=[_write(tmp_path, "a.png", _png_bytes())])
    )
    assert out["code"] == "attachment_too_large" and hub.named("upload_attachment") == []


@pytest.mark.asyncio
async def test_refusals_do_not_echo_the_full_path(tmp_path: Path) -> None:
    hub = FakeHub()
    out = await execute_bridge(_runtime(hub), _post(attachments=[str(tmp_path / "missing.png")]))
    assert "missing.png" in out["error"] and str(tmp_path) not in out["error"]


# -- cleanup ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_send_deletes_the_staged_ids_as_the_agent(tmp_path: Path) -> None:
    hub = FakeHub(send_failure=BridgeRefusal("attachment_not_found", "gone", http_status=404))
    out = await execute_bridge(
        _runtime(hub),
        _post(
            attachments=[
                _write(tmp_path, "1.png", _png_bytes()),
                _write(tmp_path, "2.png", _png_bytes((0, 0, 9))),
            ]
        ),
    )
    assert out["ok"] is False and out["code"] == "post_failed"
    deleted = hub.named("delete_attachment")
    assert [d["attachment_id"] for d in deleted] == [AID1, AID2]
    assert all(d["actor_endpoint_id"] == "bep_1" and d["conversation_id"] == CID for d in deleted)


@pytest.mark.asyncio
async def test_failed_dm_send_deletes_the_staged_ids(tmp_path: Path) -> None:
    hub = FakeHub(send_failure=RuntimeError("boom"))
    out = await execute_bridge(
        _runtime(hub), _dm(attachments=[_write(tmp_path, "1.png", _png_bytes())])
    )
    assert out["code"] == "post_failed"
    assert [d["attachment_id"] for d in hub.named("delete_attachment")] == [AID1]


@pytest.mark.asyncio
async def test_partial_upload_failure_discards_what_was_staged_and_sends_nothing(
    tmp_path: Path,
) -> None:
    hub = FakeHub(fail_upload_at=1)
    out = await execute_bridge(
        _runtime(hub),
        _post(
            attachments=[
                _write(tmp_path, "1.png", _png_bytes()),
                _write(tmp_path, "2.png", _png_bytes((0, 0, 9))),
            ]
        ),
    )
    assert out["ok"] is False and out["code"] == "attachment_failed"
    assert [d["attachment_id"] for d in hub.named("delete_attachment")] == [AID1]
    assert hub.named("conversation_send") == []


@pytest.mark.asyncio
async def test_cleanup_failure_never_replaces_the_send_failure(tmp_path: Path) -> None:
    hub = FakeHub(send_failure=RuntimeError("boom"))

    async def broken_delete(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("hub down")

    hub.delete_attachment = broken_delete  # type: ignore[method-assign]
    out = await execute_bridge(
        _runtime(hub), _post(attachments=[_write(tmp_path, "1.png", _png_bytes())])
    )
    assert out["code"] == "post_failed"


@pytest.mark.asyncio
async def test_malformed_hub_attachment_id_is_not_sent(tmp_path: Path) -> None:
    hub = FakeHub()
    hub._ids[0] = "../../etc/passwd"
    out = await execute_bridge(
        _runtime(hub), _post(attachments=[_write(tmp_path, "1.png", _png_bytes())])
    )
    assert out["code"] == "attachment_failed" and hub.named("conversation_send") == []


# -- inline data-URI guard ----------------------------------------------------


def _data_uri(chars: int, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64," + "A" * chars


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [CAP_ON, CAP_HUMAN_ONLY])
async def test_long_inline_image_data_is_refused_with_the_attachments_hint(status: Any) -> None:
    hub = FakeHub(status=status)
    payload = _data_uri(2000)
    for args in (
        _post(body=f"[image; sha256 abc] {payload}"),
        _dm(body=payload),
        {"op": "dm", "endpoint_id": PEER_ENDPOINT, "text": payload},
    ):
        out = await execute_bridge(_runtime(hub), args)
        assert out["ok"] is False and out["code"] == "inline_image_data"
        assert "`attachments`" in out["error"]
        assert "AAAA" not in out["error"]  # the run is never echoed
    assert hub.named("conversation_send") == [] and hub.named("create_conversation") == []


@pytest.mark.asyncio
async def test_wrapped_inline_image_data_is_still_caught() -> None:
    wrapped = "data:image/png;base64," + "\n".join(["A" * 76] * 20)
    out = await execute_bridge(_runtime(FakeHub()), _post(body=wrapped))
    assert out["code"] == "inline_image_data"


@pytest.mark.asyncio
async def test_short_or_non_image_data_uris_are_ordinary_text() -> None:
    hub = FakeHub()
    for body in (_data_uri(200), "data:text/plain;base64," + "A" * 5000, "an ordinary message"):
        out = await execute_bridge(_runtime(hub), _post(body=body))
        assert out["ok"] is True, body[:40]


def test_guard_threshold_is_about_one_kib() -> None:
    assert aa.refuse_inline_image_data(_data_uri(aa.INLINE_IMAGE_DATA_MIN_CHARS - 1)) is None
    assert aa.refuse_inline_image_data(_data_uri(aa.INLINE_IMAGE_DATA_MIN_CHARS)) is not None
