"""A refused agent-file stage keeps the hub's closed code in the tool result.

WHY (incident 2026-11-19, Sana ``bep_bfc1dacf…``): two DM file sends both
answered "the file upload did not confirm; nothing was sent" and nothing else.
Whatever the hub said — a 415 ``file_class_disabled`` for a ``.tar``, a 409
``attachment_stage_conflict`` for bytes already staged, a 403
``agent_not_permitted`` — was dropped at the ``tool.py`` file arm, while the
image arm beside it already carried :func:`tool._failure_text`. The DM arm also
had no ``AttachmentRefused`` branch, so the batch's own ``owner_changed`` /
duplicate refusals collapsed into the same sentence.

These tests drive the REAL :class:`TeamBridgeClient` over an in-process
``httpx.MockTransport`` hub (the same rig as the wire test in
``test_agent_files``): real status read, real ``_refusal`` decoding, real
cleanup DELETEs, with every request counted. Only process identity/admission
fences are patched (``test_agent_files``' autouse fixture); no secret guard.
The hub's free-text ``error`` sentence carries a sentinel that must NEVER
reach the result, together with the file name, path and digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import httpx
import pytest

from local_operator.team_bridge import client as client_module
from local_operator.team_bridge.tool import execute_bridge
from tests.unit.team_bridge.test_agent_attachments import (
    AID1,
    CID,
    PEER,
    PEER_ENDPOINT,
    _dm,
    _post,
    _runtime,
    _write,
)
from tests.unit.team_bridge.test_agent_files import CAP, Hub, isolated  # noqa: F401
from tests.unit.team_bridge.test_client import _make_client

SENTINEL = "SENTINEL-provider-body /Users/someone/secret/archive.tar"
GENERIC = "the file upload did not confirm; nothing was sent"


class WireHub:
    """Counts every request; answers the Nth file stage with a scripted reply."""

    def __init__(self, stage_answers):
        self.requests: list[httpx.Request] = []
        self.stage_answers = list(stage_answers)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/status"):
            return httpx.Response(200, json={"result": CAP})
        if path.endswith("/messaging/directory"):
            # The frozen desktop directory shape, one peer with an agent: the
            # DM resolves its target here before the conversation is opened.
            member = {
                "account_id": PEER,
                "display_label": "Robin",
                "role": "member",
                "bridge_enabled": True,
                "bridge_state": "enabled",
                "agent": {"endpoint_id": PEER_ENDPOINT, "name": "Aida", "state": "active"},
            }
            return httpx.Response(200, json={"result": {"members": [member]}})
        if request.method == "POST" and path.endswith("/messaging/conversations"):
            return httpx.Response(201, json={"result": {"conversation": {"id": CID}}})
        if request.method == "POST" and path.endswith("/attachments"):
            answer = self.stage_answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if request.method == "DELETE" and "/attachments/" in path:
            return httpx.Response(204)
        if path.endswith("/messages"):
            return httpx.Response(201, json={"result": {"id": "bcmsg_wire0001", "seq": 1}})
        raise AssertionError(f"unexpected request {request.method} {path}")

    def count(self, method: str, suffix: str) -> int:
        return sum(1 for r in self.requests if r.method == method and suffix in r.url.path)


def _refused(status: int, code: str, details=None) -> httpx.Response:
    body = {"code": code, "error": SENTINEL, "details": details or {}, "raw": SENTINEL}
    return httpx.Response(status, json=body)


def _wire_runtime(monkeypatch, hub: WireHub):
    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda base: True)
    client = _make_client(hub.handler)
    runtime = _runtime(Hub())
    runtime.client = client
    runtime.owner = dataclasses.replace(
        runtime.owner, account_id="acc_1", canonical_base_url=client.base_url
    )
    return runtime


def _args(op: str, paths):
    return _dm(files=paths) if op == "dm" else _post(files=paths)


def _assert_no_leak(out, *, paths, data):
    blob = json.dumps(out, default=str)
    assert "SENTINEL" not in blob
    for path in paths:
        assert path not in blob
        assert path.rsplit("/", 1)[-1] not in blob
    for raw in data:
        assert hashlib.sha256(raw).hexdigest() not in blob


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["dm", "post"])
@pytest.mark.parametrize(
    ("status", "code", "details", "expected"),
    [
        # agent-server sends these three with ``details: nil``
        # (``apperrors.NewBridgeError(code, nil)``); 403 is also what a wrong
        # tenant or a non-member endpoint receives.
        (415, "file_class_disabled", None, "[refusal code=file_class_disabled]"),
        (409, "attachment_stage_conflict", None, "[refusal code=attachment_stage_conflict]"),
        (403, "agent_not_permitted", None, "[refusal code=agent_not_permitted]"),
        # The validation shape (``details.field``/``rule``) travels too; the
        # pair here is illustrative, the shape is the hub's 400 contract.
        (
            400,
            "invalid_request",
            {"field": "x_bridge_files_version", "rule": "unsupported", "path": SENTINEL},
            "[refusal code=invalid_request field=x_bridge_files_version rule=unsupported]",
        ),
    ],
)
async def test_hub_refusal_code_survives_without_body_or_post(
    tmp_path, monkeypatch, op, status, code, details, expected
):
    data = [b"first staged file", b"second refused file"]
    paths = [_write(tmp_path, f"bundle{i}.tar", raw) for i, raw in enumerate(data)]
    # The first file stages, the second is refused: cleanup must delete exactly
    # the first id, with the pinned actor, and no message may be posted.
    hub = WireHub(
        [
            httpx.Response(201, json={"result": {"file": {"id": AID1}}}),
            _refused(status, code, details),
        ]
    )
    runtime = _wire_runtime(monkeypatch, hub)
    out = await execute_bridge(runtime, _args(op, paths))
    assert not out["ok"]
    assert out["code"] == "attachment_failed"
    assert out["error"] == f"{GENERIC} {expected}"
    assert hub.count("POST", "/messages") == 0
    deletes = [r for r in hub.requests if r.method == "DELETE"]
    # Only the staged id is ever deleted. It is deleted TWICE today
    # (``FileBatch.stage``'s own cleanup, then the tool arm's) — pre-existing,
    # same id and tuple, and out of scope for this text-only fix.
    assert {r.url.path.rsplit("/", 1)[-1] for r in deletes} == {AID1}
    for delete in deletes:
        assert delete.url.params["actor_endpoint_id"] == "bep_1"
        assert delete.url.params["tenant_id"] == runtime.tenant_id
    _assert_no_leak(out, paths=paths, data=data)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["dm", "post"])
async def test_transport_timeout_stays_unconfirmed_without_footer(tmp_path, monkeypatch, op):
    raw = b"timeout file"
    path = _write(tmp_path, "slow.txt", raw)
    hub = WireHub([httpx.ReadTimeout("read timed out", request=None)])
    out = await execute_bridge(_wire_runtime(monkeypatch, hub), _args(op, [path]))
    assert not out["ok"]
    assert out["code"] == "attachment_failed"
    # No hub answer exists to quote: the sentence is byte-identical to before.
    assert out["error"] == GENERIC
    assert hub.count("POST", "/messages") == 0
    _assert_no_leak(out, paths=[path], data=[raw])


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["dm", "post"])
async def test_malformed_receipt_stays_unconfirmed(tmp_path, monkeypatch, op):
    raw = b"malformed receipt file"
    path = _write(tmp_path, "receipt.txt", raw)
    hub = WireHub([httpx.Response(201, json={"result": {"file": {"id": SENTINEL}}})])
    out = await execute_bridge(_wire_runtime(monkeypatch, hub), _args(op, [path]))
    assert not out["ok"]
    assert out["code"] == "attachment_failed"
    assert out["error"] == "the file upload did not confirm"
    assert hub.count("POST", "/messages") == 0
    assert hub.count("DELETE", "/attachments/") == 0
    _assert_no_leak(out, paths=[path], data=[raw])


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["dm", "post"])
async def test_owner_change_mid_stage_names_owner_changed(tmp_path, monkeypatch, op):
    """Wrong-tenant negative: the local guard refuses; cleanup keeps the old tuple."""

    data = [b"tenant one", b"tenant two"]
    paths = [_write(tmp_path, f"tenant{i}.txt", raw) for i, raw in enumerate(data)]
    hub = WireHub(
        [
            httpx.Response(201, json={"result": {"file": {"id": AID1}}}),
            AssertionError("a second stage must never be attempted"),
        ]
    )
    runtime = _wire_runtime(monkeypatch, hub)
    original = hub.handler

    def switching(request):
        response = original(request)
        if request.method == "POST" and request.url.path.endswith("/attachments"):
            runtime.tenant_id = "ten_other"
        return response

    hub.handler = switching  # type: ignore[method-assign]
    runtime.client = _make_client(switching)
    out = await execute_bridge(runtime, _args(op, paths))
    assert not out["ok"]
    assert out["code"] == "owner_changed"
    assert hub.count("POST", "/attachments") == 1
    assert hub.count("POST", "/messages") == 0
    deletes = [r for r in hub.requests if r.method == "DELETE"]
    # Cleanup is the one write after the change, and only under the ORIGINAL
    # tuple: never the new tenant.
    assert all(d.url.params["tenant_id"] == "ten_1" for d in deletes)
    assert all(d.url.params["actor_endpoint_id"] == "bep_1" for d in deletes)
    _assert_no_leak(out, paths=paths, data=data)


def test_dm_target_constant_is_an_endpoint():
    # Guards the DM rows above from silently becoming channel rows if the
    # shared helper changes its target shape.
    assert _dm()["endpoint_id"] == PEER_ENDPOINT
