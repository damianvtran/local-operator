"""MockTransport seam tests; no renderer/main trust or live hub is established."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as client_module
from local_operator.team_bridge import file_history_action as history
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from local_operator.team_bridge.file_action_orchestrator import (
    HumanFileActionOrchestrator,
    TrustedMainView,
)
from local_operator.team_bridge.file_action_registry import (
    ActionUnavailable,
    InertFileActionRegistry,
)

pytestmark = pytest.mark.asyncio


@dataclass
class Row:
    id: int = 1
    provider: str = "radient"
    credential_type: str = "oauth"
    disabled_cause: str | None = None
    updated_at: int = 42
    data: dict[str, Any] = field(
        default_factory=lambda: {
            "access": "synthetic-token",
            "account_id": "acc_1",
            "expires": 4102444800000,
        }
    )


class Store:
    db_path = Path("synthetic-auth.db")

    def __init__(self):
        self.row = Row()
        self.incarnation = "a" * 32
        self.refreshes = 0

    def native_lifecycle_snapshot(self, _path, _provider):
        return [(1, 42, self.incarnation, None, "oauth", "acc_1", None, None, None)]

    def get_credential(self, credential_id):
        return self.row if credential_id == 1 else None

    async def ensure_oauth_fresh_or_raise(self, *args, **kwargs):
        self.refreshes += 1
        raise AssertionError("unexpected refresh")


def scope(registry):
    return history.HistoryScope(
        1,
        "acc_1",
        "a" * 32,
        "tnt_alpha",
        "window_1",
        4,
        "bcv_aaaaaaaa",
        "bcmsg_aaaaaaaa",
        registry.current_epoch(),
        1001,
    )


def file(file_id="batt_aaaaaaaa", **changes):
    return {
        "id": file_id,
        "mime": "application/octet-stream",
        "download_only": True,
        "state": "bound",
        "class": "other",
        "size": 0,
        "expires_at": "2099-01-01T00:00:00Z",
        **changes,
    }


def page(files=None):
    return {
        "msg": "ok",
        "result": {
            "messages": [
                {
                    "id": "bcmsg_aaaaaaaa",
                    "seq": 1001,
                    "conversation_id": "bcv_aaaaaaaa",
                    "files": files if files is not None else [file()],
                }
            ]
        },
    }


def streamed_json(value):
    # MockTransport's json= eagerly consumes the body; real HTTP streams do not.
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        stream=httpx.ByteStream(json.dumps(value).encode()),
    )


def client(monkeypatch, store, handler):
    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda _: True)
    return TeamBridgeClient(
        base_url="https://synthetic.example/v1", store=store, transport=httpx.MockTransport(handler)
    )


async def test_exact_human_history_commissions_only_response_files(monkeypatch):
    store, registry, requests = Store(), InertFileActionRegistry(), []

    def handler(request):
        requests.append(request)
        return streamed_json(page())

    s = scope(registry)
    handle = await history.commission_human_history(
        client(monkeypatch, store, handler), registry, s, lambda: s
    )
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/messaging/conversations/bcv_aaaaaaaa/messages"
    assert dict(requests[0].url.params) == {
        "tenant_id": "tnt_alpha",
        "limit": "1",
        "before_seq": "1002",
    }
    op = registry.start(handle, "batt_aaaaaaaa")
    assert registry.claim(op).binding.account_id == "acc_1"
    with pytest.raises(ActionUnavailable):
        registry.start(handle, "batt_bbbbbbbb")
    assert store.refreshes == 0


async def test_far_older_message_uses_one_authenticated_targeted_read(monkeypatch):
    store, registry, queries = Store(), InertFileActionRegistry(), []
    s = scope(registry)

    # Model the hub's newest-first exclusive cursor with 900 newer roots.
    target = page()["result"]["messages"][0]
    newer = [
        {
            "id": f"bcmsg_{seq:08d}",
            "seq": seq,
            "conversation_id": "bcv_aaaaaaaa",
            "files": [file("batt_bbbbbbbb")],
        }
        for seq in range(1002, 1902)
    ]

    def handler(request):
        queries.append(dict(request.url.params))
        before = int(request.url.params["before_seq"])
        messages = [m for m in [*reversed(newer), target] if m["seq"] < before]
        return streamed_json({"result": {"messages": messages[:1]}})

    handle = await history.commission_human_history(
        client(monkeypatch, store, handler), registry, s, lambda: s
    )
    assert queries == [{"tenant_id": "tnt_alpha", "limit": "1", "before_seq": "1002"}]
    registry.claim(registry.start(handle, "batt_aaaaaaaa"))
    with pytest.raises(ActionUnavailable):
        registry.start(handle, "batt_bbbbbbbb")
    assert store.refreshes == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "bcmsg_bbbbbbbb"),
        ("conversation_id", "bcv_bbbbbbbb"),
        ("seq", 1000),
        ("seq", True),
        ("seq", "1001"),
    ],
)
async def test_targeted_history_rejects_wrong_response_identity(monkeypatch, field, value):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = scope(registry)
    result = page([file("batt_bbbbbbbb")])
    result["result"]["messages"][0][field] = value
    c = client(monkeypatch, store, lambda r: (requests.append(r) or streamed_json(result)))
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert len(requests) == 1 and registry._contexts == {} and store.refreshes == 0


@pytest.mark.parametrize("seq", [None, True, 0, -1, "1001", (1 << 53) - 1, 1 << 63])
async def test_invalid_sequence_hint_never_dispatches(monkeypatch, seq):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = replace(scope(registry), message_seq=seq)
    c = client(monkeypatch, store, lambda r: (requests.append(r) or streamed_json(page())))
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert not requests and registry._contexts == {} and store.refreshes == 0


@pytest.mark.parametrize("rotation", ["credential", "epoch", "view"])
async def test_rotation_during_targeted_history_refuses_commission(monkeypatch, rotation):
    from local_operator.team_bridge import file_action_registry as registry_module

    monkeypatch.setattr(registry_module, "_MAX_WINDOW_TOMBSTONES", 1)
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = scope(registry)
    current = [s]
    if rotation == "epoch":
        registry.invalidate_view("window_seed", 1)

    def handler(request):
        requests.append(request)
        if rotation == "credential":
            store.incarnation = "b" * 32
        elif rotation == "epoch":
            registry.invalidate_view("window_rotate", 1)
        else:
            current[0] = replace(s, view_generation=s.view_generation + 1)
        return streamed_json(page())

    with pytest.raises((BridgeRefusal, ActionUnavailable)):
        await history.commission_human_history(
            client(monkeypatch, store, handler), registry, s, lambda: current[0]
        )
    assert len(requests) == 1 and registry._contexts == {} and store.refreshes == 0


def thread_page():
    reply = page()["result"]["messages"][0]
    reply["thread_root_id"] = "bcmsg_rootaaaa"
    return {
        "result": {
            "root": {"id": "bcmsg_rootaaaa", "conversation_id": "bcv_aaaaaaaa", "seq": 1},
            "replies": [reply],
        }
    }


async def test_distant_thread_reply_is_selected_with_inclusive_cursor(monkeypatch):
    store, registry, queries = Store(), InertFileActionRegistry(), []
    s = replace(scope(registry), thread_root_id="bcmsg_rootaaaa")
    target: dict[str, Any] = thread_page()["result"]["replies"][0]
    older: list[dict[str, Any]] = [
        dict(target, id=f"bcmsg_{seq:08d}", seq=seq, files=[file("batt_bbbbbbbb")])
        for seq in range(1, 901)
    ]

    def handler(request):
        queries.append((request.url.path, dict(request.url.params)))
        after = int(request.url.params["after_seq"])
        result = thread_page()
        result["result"]["replies"] = [m for m in [*older, target] if m["seq"] >= after][:1]
        return streamed_json(result)

    handle = await history.commission_human_history(
        client(monkeypatch, store, handler), registry, s, lambda: s
    )
    assert queries == [
        (
            "/v1/messaging/conversations/bcv_aaaaaaaa/messages/bcmsg_rootaaaa/replies",
            {"tenant_id": "tnt_alpha", "limit": "1", "after_seq": "1001"},
        )
    ]
    registry.claim(registry.start(handle, "batt_aaaaaaaa"))
    with pytest.raises(ActionUnavailable):
        registry.start(handle, "batt_bbbbbbbb")
    binding = registry.inspect_context(handle)
    assert binding.thread_root_id == "bcmsg_rootaaaa"
    view = TrustedMainView(
        "window_1", 4, "tnt_alpha", "bcv_aaaaaaaa", "bcmsg_aaaaaaaa", 1001, "bcmsg_rootaaaa"
    )
    assert HumanFileActionOrchestrator._same_view(binding, view)
    assert not HumanFileActionOrchestrator._same_view(
        binding, replace(view, thread_root_id="bcmsg_otheraaa")
    )
    assert store.refreshes == 0


@pytest.mark.parametrize(
    "part,field,value",
    [
        ("root", "id", "bcmsg_otheraaa"),
        ("root", "conversation_id", "bcv_bbbbbbbb"),
        ("root", "thread_root_id", "bcmsg_otheraaa"),
        ("reply", "id", "bcmsg_bbbbbbbb"),
        ("reply", "conversation_id", "bcv_bbbbbbbb"),
        ("reply", "thread_root_id", "bcmsg_otheraaa"),
        ("reply", "seq", 1000),
        ("reply", "seq", True),
    ],
)
async def test_thread_reply_identity_is_independently_checked(monkeypatch, part, field, value):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = replace(scope(registry), thread_root_id="bcmsg_rootaaaa")
    result = thread_page()
    item = result["result"]["root"] if part == "root" else result["result"]["replies"][0]
    item[field] = value
    c = client(monkeypatch, store, lambda r: (requests.append(r) or streamed_json(result)))
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert len(requests) == 1 and registry._contexts == {} and store.refreshes == 0


@pytest.mark.parametrize("shape", ["multiple", "oversized"])
async def test_thread_reply_response_limits_fail_closed(monkeypatch, shape):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = replace(scope(registry), thread_root_id="bcmsg_rootaaaa")
    result = thread_page()
    if shape == "multiple":
        result["result"]["replies"].append(result["result"]["replies"][0])
    else:
        result["result"]["root"]["padding"] = "x" * history._HISTORY_MAX_BYTES
    c = client(monkeypatch, store, lambda r: (requests.append(r) or streamed_json(result)))
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert len(requests) == 1 and registry._contexts == {} and store.refreshes == 0


@pytest.mark.parametrize("root", ["../../bad", "bcmsg_aaaaaaaa", 1])
async def test_invalid_trusted_thread_root_never_dispatches(monkeypatch, root):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = replace(scope(registry), thread_root_id=root)
    c = client(monkeypatch, store, lambda r: (requests.append(r) or streamed_json(thread_page())))
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert not requests and registry._contexts == {} and store.refreshes == 0


@pytest.mark.parametrize("rotation", ["view", "credential", "epoch"])
async def test_thread_rotation_invalidates_current_scope(monkeypatch, rotation):
    from local_operator.team_bridge import file_action_registry as registry_module

    monkeypatch.setattr(registry_module, "_MAX_WINDOW_TOMBSTONES", 1)
    store, registry, requests = Store(), InertFileActionRegistry(), []
    if rotation == "epoch":
        registry.invalidate_view("window_seed", 1)
    s = replace(scope(registry), thread_root_id="bcmsg_rootaaaa")
    current = [s]

    def handler(request):
        requests.append(request)
        if rotation == "view":
            current[0] = replace(s, thread_root_id="bcmsg_otheraaa")
        elif rotation == "credential":
            store.incarnation = "b" * 32
        else:
            registry.invalidate_view("window_rotate", 1)
        return streamed_json(thread_page())

    with pytest.raises((BridgeRefusal, ActionUnavailable)):
        await history.commission_human_history(
            client(monkeypatch, store, handler), registry, s, lambda: current[0]
        )
    assert len(requests) == 1 and registry._contexts == {} and store.refreshes == 0


async def test_stale_before_request_and_401_never_refresh(monkeypatch):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    s = scope(registry)
    c = client(
        monkeypatch, store, lambda request: (requests.append(request) or httpx.Response(401))
    )
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(
            c, registry, s, lambda: replace(s, view_generation=5)
        )
    assert requests == [] and store.refreshes == 0
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert len(requests) == 1 and store.refreshes == 0


@pytest.mark.parametrize(
    "alter",
    [
        "wrong_conversation",
        "other_message",
        "other_message_file",
        "duplicate_message",
        "duplicate_file",
        "bad_mime",
        "agent_file",
        "unknown_state",
        "expired",
    ],
)
async def test_malformed_or_cross_message_history_never_commissions(monkeypatch, alter):
    store, registry = Store(), InertFileActionRegistry()
    obj = page()
    message = obj["result"]["messages"][0]
    if alter == "wrong_conversation":
        message["conversation_id"] = "bcv_bbbbbbbb"
    if alter == "other_message":
        message["id"] = "bcmsg_bbbbbbbb"
    if alter == "other_message_file":
        obj["result"]["messages"].append(
            {"id": "bcmsg_bbbbbbbb", "conversation_id": "bcv_aaaaaaaa", "files": [file()]}
        )
        message["files"] = []
    if alter == "duplicate_message":
        obj["result"]["messages"].append(dict(message))
    if alter == "duplicate_file":
        message["files"].append(file())
    if alter == "bad_mime":
        message["files"][0]["mime"] = "image/png"
    if alter == "agent_file":
        message["files"][0]["download_only"] = False
    if alter == "unknown_state":
        message["files"][0]["state"] = "staged"
    if alter == "expired":
        message["files"][0]["state"] = "expired"
    s = scope(registry)
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(
            client(monkeypatch, store, lambda _: streamed_json(obj)),
            registry,
            s,
            lambda: s,
        )
    assert registry._contexts == {}


async def test_lifecycle_and_view_revocation_after_response_refuses_commit(monkeypatch):
    for revoke in ("login", "tenant", "generation", "account", "window"):
        store, registry = Store(), InertFileActionRegistry()
        s = scope(registry)
        current = [s]

        def handler(_):
            if revoke == "login":
                store.incarnation = "b" * 32
            if revoke == "tenant":
                current[0] = replace(s, tenant_id="tnt_other")
            if revoke == "generation":
                current[0] = replace(s, view_generation=5)
            if revoke == "account":
                current[0] = replace(s, account_id="acc_2")
            if revoke == "window":
                current[0] = replace(s, window_id="window_2")
            return streamed_json(page())

        with pytest.raises(BridgeRefusal):
            await history.commission_human_history(
                client(monkeypatch, store, handler), registry, s, lambda: current[0]
            )
        assert registry._contexts == {} and store.refreshes == 0


async def test_no_agent_addressing_api_or_invalid_json(monkeypatch):
    store, registry = Store(), InertFileActionRegistry()
    s = scope(registry)
    c = client(
        monkeypatch,
        store,
        lambda _: httpx.Response(
            200,
            stream=httpx.ByteStream(b'{"result":{"messages":[],"messages":[]}}'),
            headers={"content-type": "application/json"},
        ),
    )
    for key in ("endpoint_id", "actor_endpoint_id", "purpose"):
        with pytest.raises(TypeError):
            await history.commission_human_history(c, registry, s, lambda: s, **{key: "agent"})
    with pytest.raises(BridgeRefusal):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert registry._contexts == {}


@pytest.mark.parametrize(
    "change", ["expired", "missing_expiry", "missing_access", "dead", "wrong_account"]
)
async def test_stale_or_wrong_credential_never_dispatches_or_refreshes(monkeypatch, change):
    store, registry, requests = Store(), InertFileActionRegistry(), []
    if change == "expired":
        store.row.data["expires"] = 1
    if change == "missing_expiry":
        store.row.data.pop("expires")
    if change == "missing_access":
        store.row.data.pop("access")
    if change == "dead":
        store.row.data["grant_dead_at"] = 1
    if change == "wrong_account":
        store.row.data["account_id"] = "acc_2"
    s = scope(registry)
    c = client(
        monkeypatch,
        store,
        lambda request: (requests.append(request) or streamed_json(page())),
    )
    with pytest.raises((BridgeRefusal, ValueError)):
        await history.commission_human_history(c, registry, s, lambda: s)
    assert requests == [] and store.refreshes == 0 and registry._contexts == {}


async def test_invalidation_between_final_guard_and_insert_blocks_resurrection(monkeypatch):
    # The interposed insert waits until an independent thread completed view
    # invalidation. Even an obsolete reader that still echoes the old scope
    # cannot resurrect it; this tests the shared registry watermark, not IPC.
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    store, registry = Store(), InertFileActionRegistry()
    s = scope(registry)
    ready, invalidated = Event(), Event()
    original = registry.commission
    guards = []

    def reader():
        guards.append("checked")
        return s

    def paused_insert(binding, *, ttl_seconds, expected_epoch):
        assert guards
        ready.set()
        assert invalidated.wait(3), "invalidation did not finish"
        return original(binding, ttl_seconds=ttl_seconds, expected_epoch=expected_epoch)

    def invalidate():
        assert ready.wait(3), "insert was not reached"
        registry.invalidate_view(s.window_id, s.view_generation)
        invalidated.set()

    monkeypatch.setattr(registry, "commission", paused_insert)
    c = client(monkeypatch, store, lambda _: streamed_json(page()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(invalidate)
        with pytest.raises(ActionUnavailable):
            await history.commission_human_history(c, registry, s, reader)
        worker.result(timeout=3)
    assert registry._contexts == {}
    monkeypatch.setattr(registry, "commission", original)
    newer = replace(s, view_generation=s.view_generation + 1)
    handle = await history.commission_human_history(c, registry, newer, lambda: newer)
    op = registry.start(handle, "batt_aaaaaaaa")
    registry.claim(op)
    registry.revoke(handle)
    with pytest.raises(ActionUnavailable):
        registry.current(op)


async def test_rotation_after_history_guard_refuses_old_epoch_and_accepts_fresh_scope(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from local_operator.team_bridge import file_action_registry as registry_module

    monkeypatch.setattr(registry_module, "_MAX_WINDOW_TOMBSTONES", 1)
    store, registry = Store(), InertFileActionRegistry()
    s = scope(registry)
    registry.invalidate_view("window_seed", 1)
    ready, rotated = Event(), Event()
    original = registry.commission

    def paused_insert(candidate, *, ttl_seconds, expected_epoch):
        ready.set()
        assert rotated.wait(3), "epoch rotation did not finish"
        return original(candidate, ttl_seconds=ttl_seconds, expected_epoch=expected_epoch)

    def rotate():
        assert ready.wait(3), "insert not reached"
        registry.invalidate_view("window_rotate", 1)
        rotated.set()

    monkeypatch.setattr(registry, "commission", paused_insert)
    c = client(monkeypatch, store, lambda _: streamed_json(page()))
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(rotate)
        with pytest.raises(ActionUnavailable):
            await history.commission_human_history(c, registry, s, lambda: s)
        worker.result(timeout=3)
    assert registry.current_epoch() != s.registry_epoch
    assert registry._contexts == {}
    # A trusted owner must reauthorize before constructing this new scope;
    # this synthetic fixture cannot establish actual owner provenance.
    monkeypatch.setattr(registry, "commission", original)
    fresh = scope(registry)
    context = await history.commission_human_history(c, registry, fresh, lambda: fresh)
    operation = registry.start(context, "batt_aaaaaaaa")
    registry.claim(operation)
    registry.revoke(context)
    with pytest.raises(ActionUnavailable):
        registry.current(operation)
