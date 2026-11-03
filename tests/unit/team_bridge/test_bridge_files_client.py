"""Team Bridge generic files: the client half of agent-server S3a/S3b.

Human file upload (``upload_attachment(mode="file")``), the ``files`` send key
and the ``status.files`` capability reader. The hub is ``httpx.MockTransport``
recording every request, so each acceptance claim is asserted on the bytes
that would have left the machine: the text-only send stays byte-identical, a
wrong account and an oversized file never produce a request, and the
negotiation headers are exact.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    BridgeRefusal,
    encode_file_name,
    files_capability,
)
from tests.unit.team_bridge.test_client import _make_client, _ok

CID = "bcv_abcdefgh"
AID = "batt_0f8fad5b-d9cb-469f-a165-70867728950e"
AID2 = "batt_7c9e6679-7425-40de-944b-e07fc1f90ae7"
NAME = "Q4 rëport+v2 100%/final 📄.pdf"
NAME_WIRE = "Q4%20r%C3%ABport%2Bv2%20100%25%2Ffinal%20%F0%9F%93%84.pdf"
# Only the async tests: a module-level ``pytestmark`` would warn on the sync ones.
asyncio = pytest.mark.asyncio
FILE_HEADERS = ("x-bridge-attachment-mode", "x-bridge-files-version", "x-bridge-file-name")


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _recorder() -> tuple[list[httpx.Request], Any]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/attachments"):
            if request.headers.get("x-bridge-attachment-mode") == "file":
                return _ok({"file": {"id": AID, "class": "pdf", "state": "staged"}}, status=201)
            return _ok({"attachment": {"id": AID, "state": "staged"}}, status=201)
        return _ok({"id": "bcmsg_abcdef01"})

    return seen, handler


# -- send ---------------------------------------------------------------------


@asyncio
async def test_text_only_send_wire_is_byte_identical() -> None:
    """``files=None`` and ``files=[]`` produce exactly the pre-files bytes."""

    seen, handler = _recorder()
    client = _make_client(handler)
    await client.conversation_send(CID, client_ref="r1", text="hi")
    await client.conversation_send(CID, client_ref="r1", text="hi", files=None)
    await client.conversation_send(CID, client_ref="r1", text="hi", files=[])
    assert [r.content for r in seen] == [b'{"client_ref":"r1","body":"hi"}'] * 3
    # The image-only send is equally untouched by the new parameter.
    await client.conversation_send(CID, client_ref="r2", text="", attachment_ids=[AID])
    assert json.loads(seen[-1].content) == {"client_ref": "r2", "body": "", "attachment_ids": [AID]}


@asyncio
async def test_send_carries_files_in_order_beside_images() -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    await client.conversation_send(CID, client_ref="r", text="", files=[AID2, AID])
    assert json.loads(seen[-1].content) == {"client_ref": "r", "body": "", "files": [AID2, AID]}
    await client.conversation_send(
        CID, client_ref="r", text="x", attachment_ids=[AID], files=[AID2]
    )
    body = json.loads(seen[-1].content)
    assert body["attachment_ids"] == [AID] and body["files"] == [AID2]


@pytest.mark.parametrize(
    ("images", "files"),
    [
        (None, [AID, AID]),
        ([AID], [AID]),
        (None, ["../../status"]),
        ([f"batt_{i:08d}-x" for i in range(4)], [f"batt_{i:08d}-y" for i in range(7)]),
        (None, [f"batt_{i:08d}-y" for i in range(11)]),
    ],
    ids=["dup", "both-lists", "grammar", "combined-eleven", "eleven"],
)
@asyncio
async def test_send_refuses_bad_file_lists_before_io(images, files) -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    with pytest.raises(ValueError):
        await client.conversation_send(
            CID, client_ref="r", text="x", attachment_ids=images, files=files
        )
    assert seen == []


# -- upload -------------------------------------------------------------------


@asyncio
async def test_file_upload_sends_the_exact_negotiation_headers() -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    out = await client.upload_attachment(
        CID, b"%PDF-1.7", tenant_id="tnt_1", mode="file", name=NAME
    )
    assert out == {"file": {"id": AID, "class": "pdf", "state": "staged"}}
    request = seen[-1]
    assert request.method == "POST"
    assert request.url.path == f"/v1/messaging/conversations/{CID}/attachments"
    assert dict(request.url.params) == {"tenant_id": "tnt_1"}
    assert request.content == b"%PDF-1.7"
    assert request.headers["content-type"] == "application/octet-stream"
    assert request.headers["x-bridge-attachment-mode"] == "file"
    assert request.headers["x-bridge-files-version"] == "1"
    assert request.headers["x-bridge-file-name"] == NAME_WIRE
    # The name never reaches the URL.
    assert "port" not in str(request.url)


@asyncio
async def test_file_upload_without_a_name_omits_the_name_header() -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    for name in (None, ""):
        await client.upload_attachment(CID, b"x", mode="file", name=name)
        assert "x-bridge-file-name" not in seen[-1].headers
        assert seen[-1].headers["x-bridge-files-version"] == "1"


@asyncio
async def test_image_upload_wire_is_unchanged_and_refuses_a_name() -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    await client.upload_attachment(CID, b"\x89PNG\r\n\x1a\n")
    assert not set(FILE_HEADERS) & {k.lower() for k in seen[-1].headers}
    with pytest.raises(ValueError):
        await client.upload_attachment(CID, b"x", name="a.png")
    with pytest.raises(ValueError):
        await client.upload_attachment(CID, b"x", mode="video")  # type: ignore[arg-type]
    assert len(seen) == 1


@asyncio
async def test_26_mib_file_is_refused_before_any_request() -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as refused:
        await client.upload_attachment(CID, b"\0" * (26 * 1024 * 1024), mode="file", name="big.zip")
    assert refused.value.code == "attachment_too_large"
    assert refused.value.http_status == 413
    with pytest.raises(BridgeRefusal):
        await client.upload_attachment(CID, b"\0" * (tb_client.FILE_MAX_BYTES + 1), mode="file")
    with pytest.raises(BridgeRefusal) as empty:
        await client.upload_attachment(CID, b"", mode="file")
    assert empty.value.code == "attachment_invalid"
    assert seen == []
    # The boundary itself is allowed: 25 MiB exactly reaches the hub.
    await client.upload_attachment(CID, b"\0" * tb_client.FILE_MAX_BYTES, mode="file")
    assert len(seen) == 1 and len(seen[0].content) == tb_client.FILE_MAX_BYTES


@asyncio
async def test_wrong_account_file_upload_is_refused_before_any_bytes() -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as refused:
        await client.upload_attachment(
            CID, b"%PDF", mode="file", name=NAME, expected_account_id="acc_other"
        )
    assert refused.value.code == tb_client.ACCOUNT_MISMATCH_CODE
    assert seen == []
    # Positive control: the store's own account goes through.
    await client.upload_attachment(
        CID, b"%PDF", mode="file", name=NAME, expected_account_id="acc_1"
    )
    assert len(seen) == 1


@asyncio
async def test_hub_file_refusal_maps_to_its_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(415, json={"error": "off", "code": "file_class_disabled"})

    client = _make_client(handler)
    with pytest.raises(BridgeRefusal) as refused:
        await client.upload_attachment(CID, b"x", mode="file")
    assert refused.value.code == "file_class_disabled" and refused.value.http_status == 415


def test_file_name_encoding_matches_go_path_unescape() -> None:
    assert encode_file_name(NAME) == NAME_WIRE
    assert encode_file_name("a+b") == "a%2Bb"  # never a space on the hub
    assert encode_file_name("x\r\nSet-Cookie: y") == "x%0D%0ASet-Cookie%3A%20y"
    with pytest.raises(BridgeRefusal) as refused:
        encode_file_name("é" * 200)  # 6 encoded bytes each: 1200 > 1024
    assert refused.value.code == "attachment_invalid"


@asyncio
async def test_display_name_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    seen, handler = _recorder()
    client = _make_client(handler)
    with caplog.at_level(logging.DEBUG):
        await client.upload_attachment(CID, b"%PDF", mode="file", name=NAME)
        with pytest.raises(BridgeRefusal):
            await client.upload_attachment(CID, b"%PDF", mode="file", name="é" * 200)
    assert seen[0].headers["x-bridge-file-name"] == NAME_WIRE  # control: it was sent
    for needle in (NAME, NAME_WIRE, "rëport", "é" * 10):
        assert needle not in caplog.text


# -- capability ---------------------------------------------------------------

HUB_BLOCK = {
    "version": 1,
    "enabled": True,
    "download_only": True,
    "max_items": 10,
    "max_images": 4,
    "classes": [
        {"class": "pdf", "max_bytes": 20971520},
        {"class": "text", "max_bytes": 2097152},
        {"class": "container", "max_bytes": 26214400},
        {"class": "other", "max_bytes": 26214400},
    ],
}


def test_files_capability_reads_the_hub_block() -> None:
    assert files_capability({"files": HUB_BLOCK}) == HUB_BLOCK


@pytest.mark.parametrize(
    "block",
    [
        None,
        "on",
        {**HUB_BLOCK, "enabled": False},
        {**HUB_BLOCK, "version": 2},
        {**HUB_BLOCK, "version": True},
        {**HUB_BLOCK, "version": "1"},
        {**HUB_BLOCK, "download_only": False},
        {**HUB_BLOCK, "classes": []},
        {**HUB_BLOCK, "classes": [{"class": "exe", "max_bytes": 1}]},
        {**HUB_BLOCK, "classes": [{"class": "pdf", "max_bytes": 26214401}]},
    ],
    ids=[
        "absent",
        "scalar",
        "disabled",
        "v2",
        "bool-version",
        "str-version",
        "not-download-only",
        "no-classes",
        "unknown-class",
        "cap-over-transport",
    ],
)
def test_files_capability_off_unless_usable_v1(block) -> None:
    payload = {} if block is None else {"files": block}
    assert files_capability(payload) is None


def test_files_capability_drops_junk_rows_and_fields() -> None:
    block = {
        **HUB_BLOCK,
        "max_items": True,
        "secret": "x",
        "classes": [
            {"class": "pdf", "max_bytes": 1},
            {"class": "pdf", "max_bytes": 2},
            {"class": "other", "max_bytes": 0},
            "junk",
        ],
    }
    assert files_capability({"files": block}) == {
        "version": 1,
        "enabled": True,
        "download_only": True,
        "classes": [{"class": "pdf", "max_bytes": 1}],
        "max_images": 4,
    }


def test_status_files_block_is_normalized_or_dropped() -> None:
    from local_operator.server.routes.desktop_team_bridge import (
        _status_with_attachments,
    )

    raw = {"api_version": "1", "files": {**HUB_BLOCK, "extra": "x"}}
    out = _status_with_attachments(raw, tb_client.attachments_capability, files_capability)
    assert out["files"] == HUB_BLOCK
    assert raw["files"]["extra"] == "x"  # cached payload untouched
    # Files OFF (the production answer today): the key is gone, not forwarded.
    off = _status_with_attachments(
        {"api_version": "1", "files": {**HUB_BLOCK, "enabled": False}},
        tb_client.attachments_capability,
        files_capability,
    )
    assert off == {"api_version": "1"}
    v2 = _status_with_attachments(
        {"files": {**HUB_BLOCK, "version": 2}}, tb_client.attachments_capability, files_capability
    )
    assert "files" not in v2
    # Pre-files hub: nothing to normalize, payload returned as is.
    plain = {"api_version": "1"}
    assert (
        _status_with_attachments(plain, tb_client.attachments_capability, files_capability) is plain
    )
