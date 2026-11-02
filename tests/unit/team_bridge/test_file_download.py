"""Unwired human-file helper against a synthetic HTTP transport; no server or desktop route."""

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as client_module
from local_operator.team_bridge import file_download as files
from local_operator.team_bridge.client import (
    BridgeRefusal,
    BridgeSignInRequired,
    TeamBridgeClient,
)

pytestmark = pytest.mark.asyncio
LOGIN_A = "a" * 32
LOGIN_B = "b" * 32


@dataclass
class Row:
    id: int = 1
    provider: str = "radient"
    credential_type: str = "oauth"
    data: dict[str, Any] = field(
        default_factory=lambda: {
            "access": "synthetic-token",
            "account_id": "acc_1",
            "expires": 4_102_444_800_000,
        }
    )
    disabled_cause: str | None = None
    updated_at: int = 10


class Store:
    def __init__(self) -> None:
        self.row = Row()
        self.incarnation: str | None = LOGIN_A
        self.refreshes = 0
        self.db_path = Path("/synthetic/unused-auth.db")
        self.snapshot_available = True
        self.mesh_origin: str | None = None

    def list_credentials(self, *args: Any, **kwargs: Any) -> list[Row]:
        return [self.row]

    def get_credential(self, credential_id: int) -> Row | None:
        return self.row if credential_id == self.row.id else None

    def native_lifecycle_snapshot(
        self, db_path: Path, provider: str
    ) -> list[tuple[Any, ...]] | None:
        assert db_path == self.db_path and provider == "radient"
        if not self.snapshot_available:
            return None
        return [
            (
                self.row.id,
                self.row.updated_at,
                self.incarnation,
                self.row.disabled_cause,
                self.row.credential_type,
                self.row.data.get("account_id"),
                None,
                None,
                self.mesh_origin,
            )
        ]

    async def ensure_oauth_fresh_or_raise(
        self, credential_id: int, *, force: bool = False
    ) -> dict[str, Any]:
        self.refreshes += int(force)
        return dict(self.row.data)


def make_client(monkeypatch: pytest.MonkeyPatch, handler: Any, store: Store) -> TeamBridgeClient:
    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda _: True)
    return TeamBridgeClient(
        base_url="https://synthetic.example/v1", store=store, transport=httpx.MockTransport(handler)
    )


def make_fence(**changes: Any) -> files.FileActionFence:
    return replace(
        files.FileActionFence(
            "acc_1", 1, LOGIN_A, "tnt_test", 1, "bcv_12345678", "bcmsg_1", "batt_12345678"
        ),
        **changes,
    )


async def invoke(client: TeamBridgeClient, **changes: Any) -> bytes:
    fence = make_fence()
    args = {
        "conversation_id": "bcv_12345678",
        "message_id": "bcmsg_1",
        "attachment_id": "batt_12345678",
        "tenant_id": "tnt_test",
        "expected_account_id": "acc_1",
        "action_fence": fence,
        "current_action_fence": lambda: fence,
    }
    args.update(changes)
    return await files.download_human_file(client, **args)


class StaticStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.closed = False

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        self.closed = True


async def test_human_request_is_explicit_tenant_only_and_accepts_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=StaticStream(b"")
        )

    assert await invoke(make_client(monkeypatch, handler, Store())) == b""
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert (
        seen[0].url.path
        == "/v1/messaging/conversations/bcv_12345678/messages/bcmsg_1/attachments/batt_12345678"
    )
    assert list(seen[0].url.params.multi_items()) == [("tenant_id", "tnt_test")]
    assert seen[0].headers["accept-encoding"] == "identity"
    assert seen[0].headers["accept"] == "application/octet-stream, application/json"


@pytest.mark.parametrize(
    "changes",
    [
        {"conversation_id": "bcv_foo/../x"},
        {"message_id": "bcmsg_1?endpoint_id=x"},
        {"attachment_id": "batt_a/../x"},
        {"tenant_id": "tnt_a&purpose=context"},
        {"expected_account_id": ""},
    ],
)
async def test_invalid_input_never_dispatches(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, str]
) -> None:
    seen = []
    client = make_client(monkeypatch, lambda request: seen.append(request), Store())
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client, **changes)
    assert exc.value.code == "invalid_request"
    assert seen == []


async def test_agent_addressing_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    client = make_client(monkeypatch, lambda request: seen.append(request), Store())
    for forbidden_field in ("endpoint_id", "actor_endpoint_id", "purpose"):
        with pytest.raises(TypeError):
            await invoke(client, **{forbidden_field: "synthetic"})
    assert seen == []


@pytest.mark.parametrize(
    "status, headers, content, code",
    [
        (200, {"content-type": "image/png"}, b"x", "bridge_invalid_response"),
        (
            200,
            {"content-type": "application/octet-stream; charset=utf-8"},
            b"x",
            "bridge_invalid_response",
        ),
        (
            200,
            {"content-type": "application/octet-stream", "content-encoding": "gzip"},
            b"not-gzip",
            "bridge_invalid_response",
        ),
        (302, {"location": "https://not-allowed.example/"}, b"", "bridge_invalid_response"),
        (
            404,
            {"content-type": "application/json"},
            b"sensitive upstream message",
            "attachment_not_found",
        ),
        (
            403,
            {"content-type": "application/json"},
            b"sensitive upstream message",
            "attachment_not_found",
        ),
    ],
)
async def test_invalid_responses_are_structured_and_never_echo_content(
    monkeypatch: pytest.MonkeyPatch, status: int, headers: dict[str, str], content: bytes, code: str
) -> None:
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(status, headers=headers, stream=StaticStream(content)),
        Store(),
    )
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == code
    assert "sensitive" not in str(exc.value)
    assert "synthetic-token" not in str(exc.value)


class Stream(httpx.AsyncByteStream):
    def __init__(self, store: Store, *, switch: bool = False) -> None:
        self.store, self.switch, self.closed = store, switch, False

    async def __aiter__(self):
        yield b"12345"
        if self.switch:
            self.store.incarnation = LOGIN_B
        yield b"67890"

    async def aclose(self) -> None:
        self.closed = True


async def test_stream_cap_and_close_before_return(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files, "FILE_DOWNLOAD_MAX_BYTES", 8)
    store = Store()
    stream = Stream(store)
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=stream
        ),
        store,
    )
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == "attachment_too_large"
    assert stream.closed


async def test_account_switch_during_body_refuses_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()
    stream = Stream(store, switch=True)
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=stream
        ),
        store,
    )
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == "account_mismatch"
    assert stream.closed


async def test_switch_before_dispatch_and_401_never_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    seen = []
    client = make_client(
        monkeypatch,
        lambda request: (
            seen.append(request)
            or httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                stream=StaticStream(b"ok"),
            )
        ),
        store,
    )
    with pytest.raises(BridgeRefusal) as exc:
        other = make_fence(account_id="acc_2")
        await invoke(
            client,
            expected_account_id="acc_2",
            action_fence=other,
            current_action_fence=lambda: other,
        )
    assert exc.value.code == "account_mismatch"
    assert not seen
    store = Store()
    calls = []

    def unauthorized(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(401, stream=StaticStream(b"sensitive"))

    client = make_client(monkeypatch, unauthorized, store)
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == "file_reauthentication_required"
    assert len(calls) == 1 and store.refreshes == 0
    assert "sensitive" not in str(exc.value)


async def test_login_change_after_401_prevents_refresh_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = Store()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        store.incarnation = LOGIN_B
        return httpx.Response(401, content=b"synthetic")

    client = make_client(monkeypatch, handler, store)
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == "account_mismatch"
    assert len(calls) == 1
    assert store.refreshes == 0


@pytest.mark.parametrize(
    "current",
    [
        make_fence(login_incarnation=LOGIN_B),
        make_fence(tenant_id="tnt_other"),
        make_fence(generation=2),
    ],
)
async def test_stale_action_never_dispatches_or_refreshes(
    monkeypatch: pytest.MonkeyPatch, current: files.FileActionFence
) -> None:
    store, seen = Store(), []
    client = make_client(monkeypatch, lambda request: seen.append(request), store)
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client, current_action_fence=lambda: current)
    assert exc.value.code == "account_mismatch"
    assert seen == [] and store.refreshes == 0


async def test_action_switch_during_stream_refuses_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()
    current = [make_fence()]

    class SwitchStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"part"
            current[0] = make_fence(tenant_id="tnt_other", generation=2)
            yield b"rest"

        async def aclose(self) -> None:
            self.closed = True

    stream = SwitchStream()
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=stream
        ),
        store,
    )
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client, current_action_fence=lambda: current[0])
    assert exc.value.code == "account_mismatch" and stream.closed


async def test_many_tiny_chunks_and_cancellation_close(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()

    class TinyStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            for _ in range(10000):
                yield b"x"

        async def aclose(self) -> None:
            self.closed = True

    stream = TinyStream()
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=stream
        ),
        store,
    )
    assert await invoke(client) == b"x" * 10000
    assert stream.closed
    started = asyncio.Event()

    class WaitingStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"never"

        async def aclose(self) -> None:
            self.closed = True

    waiting = WaitingStream()
    client = make_client(
        monkeypatch,
        lambda _: httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, stream=waiting
        ),
        store,
    )
    task = asyncio.create_task(invoke(client))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert waiting.closed


@pytest.mark.parametrize(
    "change",
    [
        "expired",
        "near_expiry",
        "missing_expiry",
        "bad_expiry",
        "missing_access",
        "dead",
        "unknown_dead",
    ],
)
async def test_unusable_bearer_never_refreshes_or_dispatches(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    store, calls = Store(), []
    if change == "expired":
        store.row.data["expires"] = 1
    if change == "near_expiry":
        store.row.data["expires"] = int(files.time.time() * 1000) + 1000
    if change == "missing_expiry":
        store.row.data.pop("expires")
    if change == "bad_expiry":
        store.row.data["expires"] = True
    if change == "missing_access":
        store.row.data.pop("access")
    if change == "dead":
        store.row.data["grant_dead_at"] = 1
    if change == "unknown_dead":
        store.row.data["grant_dead_at"] = "unknown"
    client = make_client(monkeypatch, lambda request: calls.append(request), store)
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == "file_auth_refresh_required"
    assert calls == [] and store.refreshes == 0


@pytest.mark.parametrize(
    "change",
    ["unknown_snapshot", "unknown_incarnation", "disabled", "mesh", "same_account_relogin"],
)
async def test_unknown_or_revoked_native_authority_never_dispatches(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    store, calls = Store(), []
    if change == "unknown_snapshot":
        store.snapshot_available = False
    if change == "unknown_incarnation":
        store.incarnation = None
    if change == "disabled":
        store.row.disabled_cause = "synthetic-disabled"
    if change == "mesh":
        store.mesh_origin = "object"
    if change == "same_account_relogin":
        store.incarnation = LOGIN_B
    client = make_client(monkeypatch, lambda request: calls.append(request), store)
    with pytest.raises((BridgeRefusal, BridgeSignInRequired)):
        await invoke(client)
    assert calls == [] and store.refreshes == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("conversation_id", "bcv_87654321"),
        ("message_id", "bcmsg_2"),
        ("attachment_id", "batt_87654321"),
        ("credential_id", 2),
    ],
)
async def test_context_resource_and_credential_are_bound(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    store, calls = Store(), []
    client = make_client(monkeypatch, lambda request: calls.append(request), store)
    fence = make_fence(**{field: value})
    with pytest.raises((BridgeRefusal, BridgeSignInRequired)):
        await invoke(client, action_fence=fence, current_action_fence=lambda: fence)
    assert calls == [] and store.refreshes == 0


async def test_save_dialog_completion_requires_fresh_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    store = Store()
    client = make_client(monkeypatch, lambda _: httpx.Response(200), store)
    fence = make_fence()
    assert files.verify_file_action_current(client, fence, lambda: fence) is None
    # The future caller must perform this second check after its asynchronous
    # dialog and before publishing a file; the standalone helper does not save.
    with pytest.raises(BridgeRefusal) as exc:
        files.verify_file_action_current(client, fence, lambda: replace(fence, generation=2))
    assert exc.value.code == "account_mismatch"


async def test_login_changes_while_bearer_row_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    class SwitchingStore(Store):
        def get_credential(self, credential_id: int) -> Row | None:
            row = super().get_credential(credential_id)
            self.incarnation = LOGIN_B
            return row

    store, calls = SwitchingStore(), []
    client = make_client(monkeypatch, lambda request: calls.append(request), store)
    with pytest.raises(BridgeRefusal) as exc:
        await invoke(client)
    assert exc.value.code == "account_mismatch"
    assert calls == [] and store.refreshes == 0
