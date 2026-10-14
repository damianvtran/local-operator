"""The UI host's session-side client, availability answers, and failure copy.

Two kinds of test here, deliberately separated:

* the TRANSPORT ones route the client's own `httpx.AsyncClient` through a mock
  transport, mirroring `tests/unit/browser_bridge/test_backend.py` — same shape,
  because it is the same transport;
* the REACHABILITY ones stand up a REAL loopback HTTP listener, because the
  `/health` acquittal is the one place this host's rule differs from the
  bridge's (it must check the pid that answered, not just HTTP 200) and a mock
  would assert only that the mock was called.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.browser_bridge.backend import (
    BridgeError,
    BridgeUnreachable,
    format_error,
)
from local_operator.browser_bridge.protocol import PROTO_VERSION, ErrorCode
from local_operator.ui_browser import backend, state


def publish(tmp_path: Path, **updates: Any) -> state.UiHostState:
    values: dict[str, Any] = {
        "pid": os.getpid(),
        "port": 52133,
        "session_key": secrets.token_hex(24),
        "proto": PROTO_VERSION,
        "app_version": "0.21.0",
    }
    values.update(updates)
    current = state.UiHostState.model_validate(values)
    state.publish(current, tmp_path)
    return current


def _client_with_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: object
) -> backend.UiHostClient:
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)  # type: ignore[arg-type]
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    return backend.UiHostClient(tmp_path)


@pytest.mark.asyncio
async def test_round_trip_and_the_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        current = state.read(tmp_path)
        assert current is not None
        assert request.headers["X-Bridge-Key"] == current.session_key
        body = json.loads(request.content)
        if body["method"] == "status":
            return httpx.Response(200, json={"id": body["id"], "ok": True, "result": {"tabs": 2}})
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "ok": False,
                "error": {
                    "code": "proto_mismatch",
                    "message": "proto 2 vs 1",
                    "data": {"proto": 2},
                },
            },
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    assert await client.call("status", {}) == {"tabs": 2}
    with pytest.raises(BridgeError) as excinfo:
        await client.call("open", {"url": "https://example.com"})
    assert excinfo.value.code == ErrorCode.PROTO_MISMATCH


@pytest.mark.asyncio
async def test_unreachable_copy_names_the_app_and_never_installs_a_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The UI host's transport failures must not read as the daemon's.

    \"Run 'lop browser install'\" tells the user of a running desktop app to set up
    a daemon that is not what they are driving, and — if they obey — changes
    nothing about the app they quit.
    """
    publish(tmp_path)
    real_async_client = httpx.AsyncClient

    def fail(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    def fake_async_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(fail)  # type: ignore[arg-type]
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    client = backend.UiHostClient(tmp_path)
    with pytest.raises(BridgeUnreachable) as excinfo:
        await client.call("read", {})
    message = str(excinfo.value)
    assert "no longer answering at 127.0.0.1:52133" in message
    assert "re-open it and retry" in message
    assert "lop browser install" not in message
    assert "browser bridge" not in message


@pytest.mark.asyncio
async def test_missing_record_does_not_infer_the_app_is_closed(tmp_path: Path) -> None:
    with pytest.raises(BridgeUnreachable) as excinfo:
        await backend.UiHostClient(tmp_path).call("status", {})
    message = str(excinfo.value)
    assert "unavailable" in message and "incomplete browser startup" in message
    assert "does not prove the app is closed" in message
    assert "lop browser install" not in message


@pytest.mark.asyncio
async def test_rejected_key_names_the_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publish(tmp_path)
    client = _client_with_transport(
        tmp_path, monkeypatch, lambda _request: httpx.Response(401, json={})
    )
    with pytest.raises(BridgeUnreachable, match="restart the app"):
        await client.call("status", {})


@pytest.mark.asyncio
async def test_a_malformed_envelope_is_not_silently_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(tmp_path)
    client = _client_with_transport(
        tmp_path, monkeypatch, lambda _request: httpx.Response(200, json={"nonsense": True})
    )
    with pytest.raises(BridgeUnreachable, match="invalid response"):
        await client.call("status", {})


@pytest.mark.asyncio
async def test_a_timeout_after_connect_names_the_app_not_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(tmp_path)

    def hang(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read deadline")

    client = _client_with_transport(tmp_path, monkeypatch, hang)
    with pytest.raises(BridgeUnreachable) as excinfo:
        await client.call("goto", {"url": "https://example.com"})
    message = str(excinfo.value)
    assert "accepted 'goto' but did not answer" in message
    assert "unreachable" not in message
    assert "check the app's browser tab" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["401", "409", "connect", "health"])
async def test_modern_ui_retries_only_changed_record_predelivery_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    first = publish(tmp_path, instance_id="instance-a")
    counts = {"post": 0, "get": 0}
    generation = "instance-a"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generation
        if request.method == "GET":
            counts["get"] += 1
            if refusal == "health" and counts["get"] == 1:
                generation = "instance-b"
                publish(tmp_path, instance_id=generation)
            return httpx.Response(
                200, json={"host": "ui", "pid": first.pid, "instance_id": generation}
            )
        counts["post"] += 1
        assert request.headers["X-UI-Browser-Instance"] == generation
        if counts["post"] == 1 and refusal != "health":
            generation = "instance-b"
            publish(tmp_path, instance_id=generation)
            if refusal == "connect":
                raise httpx.ConnectError("connection refused")
            if refusal == "409":
                return httpx.Response(409, json={"detail": "host instance changed"})
            return httpx.Response(401, json={})
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"id": body["id"], "ok": True, "result": {"generation": generation}}
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    assert await client.call("open", {"url": "https://example.test"}) == {
        "generation": "instance-b"
    }
    assert counts == {"get": 2, "post": 1 if refusal == "health" else 2}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ReadError]
)
async def test_modern_ui_never_replays_ambiguous_action_even_after_record_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[httpx.RequestError]
) -> None:
    current = publish(tmp_path, instance_id="instance-a")
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(
                200, json={"host": "ui", "pid": current.pid, "instance_id": "instance-a"}
            )
        posts += 1
        publish(tmp_path, instance_id="instance-b")
        raise failure("after possible side effect")

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable):
        await client.call("open", {"url": "https://example.test"})
    assert posts == 1


@pytest.mark.asyncio
async def test_call_once_refuses_when_the_shared_deadline_is_already_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CR2: only the REMAINING allowance is passed into each stage.

    The one monotonic deadline is shared across health, the initial delivery and
    a pre-delivery rediscovery. `call_once` therefore refuses, without sending,
    when the deadline it is handed is already in the past — which is what stops
    a retry from receiving a fresh full budget (the defect CR2 names). Pinned
    deterministically on the injected deadline rather than on wall-clock sleeps.
    """
    publish(tmp_path, instance_id="instance-a")
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        posts += 1
        pytest.fail("a spent deadline must refuse before dispatch")

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    current = client._read()
    assert current is not None
    # A deadline in the past: remaining allowance is spent.
    past = asyncio.get_running_loop().time() - 1.0
    with pytest.raises((TimeoutError, BridgeUnreachable)):
        await client.call_once(current, "open", {"url": "https://x.test"}, deadline=past)
    assert posts == 0, "a spent shared deadline must refuse before any dispatch"


@pytest.mark.asyncio
async def test_modern_ui_changed_instance_never_sends_command_to_wrong_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = publish(tmp_path, instance_id="expected-instance")
    posts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(
                200, json={"host": "ui", "pid": current.pid, "instance_id": "wrong-instance"}
            )
        posts += 1
        pytest.fail("command dispatched despite health identity mismatch")

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable, match="no command was sent"):
        await client.call("open", {"url": "https://example.test"})
    assert posts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("beyond", [False, True])
async def test_public_retry_is_clamped_to_the_shared_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, beyond: bool
) -> None:
    """CR2 at the PUBLIC entry, deterministically.

    A pre-delivery refusal must re-enter the transport with the REMAINDER of one
    deadline, never a renewed full budget; and when the allowance is spent the
    rediscovery must dispatch NOTHING.

    HOW IT IS DETERMINISTIC: this test does not sleep and does not mutate the real
    event loop. It installs a proxy in place of `asyncio` in BOTH browser modules
    whose `get_running_loop().time()` is the REAL time plus a controlled offset,
    while `asyncio.timeout` delegates to the REAL `asyncio.timeout` (a generous
    real outer budget that never fires). The offset therefore moves only the
    module's own deadline ARITHMETIC — the thing under test — with no wall-clock
    race.

      * `beyond=False`: after the first 401 the record is rotated AND the offset
        is advanced to leave a small remainder; the replacement attempt is
        clamped to that remainder (its health and POST httpx timeouts are both
        captured, and both are strictly smaller than the first attempt's).
      * `beyond=True`: the offset is advanced PAST the deadline; the replacement
        preflight computes a non-positive allowance and refuses — ZERO second
        health/POST.
    """
    from local_operator.browser_bridge import backend as bridge_backend
    from local_operator.ui_browser import backend as ui_backend

    BUDGET = 10.0
    state_box = {"offset": 0.0}

    class _Clock:
        def time(self) -> float:
            return asyncio.get_running_loop().time() + state_box["offset"]

        def __getattr__(self, name: str) -> Any:  # any other loop API is real
            return getattr(asyncio.get_running_loop(), name)

    class _AsyncioProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

        def get_running_loop(self) -> Any:
            return _Clock()

    proxy = _AsyncioProxy()
    monkeypatch.setattr(ui_backend, "asyncio", proxy)
    monkeypatch.setattr(bridge_backend, "asyncio", proxy)

    publish(tmp_path, instance_id="instance-a")
    gets = 0
    posts = 0
    timeouts: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal gets, posts
        if request.method == "GET":
            gets += 1
            cur = state.read(tmp_path)
            assert cur is not None
            return httpx.Response(
                200, json={"host": "ui", "pid": cur.pid, "instance_id": cur.instance_id}
            )
        posts += 1
        # Rotate the record on the FIRST delivery so the pre-delivery rediscovery
        # is selected, and move the controlled clock:
        if posts == 1:
            publish(tmp_path, instance_id="instance-b")
            state_box["offset"] = BUDGET + 1.0 if beyond else BUDGET - 1.0
            return httpx.Response(401, json={})
        body = json.loads(request.content)
        return httpx.Response(
            200, json={"id": body["id"], "ok": True, "result": {"replaced": True}}
        )

    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: Any) -> httpx.AsyncClient:
        timeouts.append(kwargs.get("timeout"))
        kwargs["transport"] = httpx.MockTransport(handler)  # type: ignore[arg-type]
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    client = backend.UiHostClient(tmp_path)
    client.timeout_for = lambda method, params: BUDGET  # type: ignore[assignment]

    if beyond:
        with pytest.raises(BridgeUnreachable):
            await client.call("open", {"url": "https://example.test"})
        assert (gets, posts) == (1, 1), "a spent budget forbids a second health or POST"
        assert len(timeouts) == 2
        return
    assert await client.call("open", {"url": "https://example.test"}) == {"replaced": True}
    # Remainder case: attempt1 health + POST, then attempt2 health + POST.
    assert (gets, posts) == (2, 2), "a changed-record refusal redispatches exactly once"
    assert len(timeouts) == 4, "two attempts x (health, post)"
    assert timeouts[2] > 0 and timeouts[3] > 0
    assert timeouts[2] < timeouts[0] and timeouts[3] < timeouts[1], (
        "the retry is clamped to the REMAINDER, never a renewed full budget"
    )


@pytest.mark.asyncio
async def test_modern_ui_delayed_timeout_after_delivery_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An action that may have been DELIVERED, then a record rotation, then a
    delay past the budget: exactly ONE dispatch, no replay.

    Distinct from the immediate-transport-failure test above: here the POST is
    accepted (a 200 arrives) but the response is DELAYED past the outer budget,
    and the record rotates before the retry branch could run. The outer
    `asyncio.timeout` must fire first and land as `BridgeUnreachable` with the
    transport never sending a second request — the pre-delivery retry branch is
    reachability to only a `HostPreDeliveryFailure`, which a post-send stall can
    never produce.
    """
    publish(tmp_path, instance_id="instance-a")
    posts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.method == "GET":
            return httpx.Response(
                200, json={"host": "ui", "pid": os.getpid(), "instance_id": "instance-a"}
            )
        posts += 1
        # A second incarnation is advertised while our request is in flight.
        publish(tmp_path, instance_id="instance-b")
        await asyncio.sleep(2.0)  # outlives the tiny budget below
        return httpx.Response(200, json={"ok": True})

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    client.timeout_for = lambda method, params: 0.1  # type: ignore[assignment]
    with pytest.raises(BridgeUnreachable):
        await client.call("open", {"url": "https://example.test"})
    assert posts == 1, "a possibly-delivered action is never replayed after a record rotation"


# --- availability and the /health acquittal, against a real listener ----------


class _Health(BaseHTTPRequestHandler):
    #: Read through the CLASS on every request, so a test can change what the
    #: listener claims without restarting it (the pid/host assertions below
    #: depend on exactly that).
    body: dict[str, Any] = {}

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
        payload = json.dumps(_Health.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence the stdlib access log so the suite stays readable."""


def _listener(body: dict[str, Any]) -> tuple[HTTPServer, int]:
    _Health.body = body
    server = HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "advertised_instance, reachable", [("instance-a", True), ("instance-b", False)]
)
async def test_fresh_modern_record_requires_real_health_generation_match(
    tmp_path: Path, advertised_instance: str, reachable: bool
) -> None:
    server, port = _listener({"host": "ui", "pid": os.getpid(), "instance_id": "instance-a"})
    try:
        publish(tmp_path, port=port, instance_id=advertised_instance)
        assert await backend.ui_browser_reachable(tmp_path) is reachable
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_real_ui_transport_rotates_once_before_delivery_without_duplicate_side_effect(
    tmp_path: Path,
) -> None:
    """Real HTTP/key/header path; only the page dispatcher is synthetic."""
    facts: dict[str, Any] = {"instance": "instance-a", "gets": 0, "posts": 0, "effects": 0}

    class RpcHandler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            facts["gets"] += 1
            self.reply(200, {"host": "ui", "pid": os.getpid(), "instance_id": facts["instance"]})

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            facts["posts"] += 1
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if facts["posts"] == 1:
                facts["instance"] = "instance-b"
                publish(tmp_path, port=server.server_address[1], instance_id="instance-b")
                self.reply(401, {})
                return
            assert self.headers["X-UI-Browser-Instance"] == "instance-b"
            record = state.read(tmp_path)
            assert record is not None
            assert self.headers["X-Bridge-Key"] == record.session_key
            facts["effects"] += 1
            body = json.loads(raw)
            self.reply(200, {"id": body["id"], "ok": True, "result": {"created": True}})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), RpcHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        publish(tmp_path, port=server.server_address[1], instance_id="instance-a")
        result = await backend.UiHostClient(tmp_path).call("open", {"url": "https://example.test"})
        assert result == {"created": True}
        assert facts == {"instance": "instance-b", "gets": 2, "posts": 2, "effects": 1}
        print(
            "real UI transport: health=2, POST=2, first=401/pre-delivery, "
            "second=200/created, side_effects=1"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.asyncio
async def test_reachability_matrix(tmp_path: Path) -> None:
    # ABSENT: nothing to acquit, and no probe is attempted (no listener exists).
    assert await backend.ui_browser_reachable(tmp_path) is False
    assert backend.ui_browser_available(tmp_path) is False
    assert backend.ui_browser_advertisable(tmp_path) is False

    server, port = _listener({"host": "ui", "proto": PROTO_VERSION, "pid": os.getpid()})
    try:
        # FRESH is a definite yes and costs no probe: point the record at a dead
        # socket and it must still answer True.
        publish(tmp_path, port=1)
        assert await backend.ui_browser_reachable(tmp_path) is True

        # STALE + a listener that reports our own pid: acquitted by the probe.
        stale = state.read(tmp_path)
        assert stale is not None
        stale.port = port
        stale.heartbeat_at = 0.0
        state.state_path(tmp_path).write_text(stale.model_dump_json())
        assert state.liveness(tmp_path)[0] is state.Liveness.STALE
        assert await backend.ui_browser_reachable(tmp_path) is True

        # STALE + a listener answering for a DIFFERENT process: a recycled port
        # must not be acquitted as this host.
        stale.pid = os.getpid()
        state.state_path(tmp_path).write_text(stale.model_dump_json())
        _Health.body = {"host": "ui", "proto": PROTO_VERSION, "pid": os.getpid() + 1}
        assert await backend.ui_browser_reachable(tmp_path) is False

        # STALE + a listener that is not this host at all (e.g. the bridge's
        # daemon on a recycled port): the `host` field is what separates them.
        _Health.body = {"host": "extension", "proto": PROTO_VERSION, "pid": os.getpid()}
        assert await backend.ui_browser_reachable(tmp_path) is False

        # STALE + a listener answering for us but at a skewed proto: still
        # reachable, because the typed proto_mismatch is the honest answer.
        _Health.body = {"host": "ui", "proto": PROTO_VERSION + 1, "pid": os.getpid()}
        assert await backend.ui_browser_reachable(tmp_path) is True
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_client_reads_the_ui_namespace_and_not_the_bridge_one(tmp_path: Path) -> None:
    """The two hosts' records are separate namespaces, and the client knows it.

    A valid bridge record in the SAME root must not be readable as the UI host:
    otherwise a session would treat a running daemon's port and key as the
    desktop app's and dial the wrong process with the wrong credential.
    """
    from local_operator.browser_bridge import state as bridge_state

    bridge_state.publish(
        bridge_state.BridgeState(
            pid=os.getpid(),
            port=4099,
            session_key=secrets.token_hex(24),
            proto=PROTO_VERSION,
            extension_connected=True,
        ),
        tmp_path,
    )
    assert backend.ui_browser_available(tmp_path) is False
    client = backend.UiHostClient(tmp_path)
    assert client.host == "ui"
    with pytest.raises(BridgeUnreachable, match="unavailable"):
        await client.call("status", {})


# --- the UI host's voice, and the extension's, side by side -------------------


def _error(code: ErrorCode, data: dict[str, Any] | None = None) -> BridgeError:
    return BridgeError(code, "detail", data or {})


@pytest.mark.parametrize(
    ("code", "data", "must", "forbidden"),
    [
        (
            ErrorCode.EXTENSION_DISCONNECTED,
            {},
            ("no browser tab attached", "open a browser tab in the app"),
            ("chrome://extensions", "lop browser install"),
        ),
        (
            ErrorCode.EXTENSION_UNRESPONSIVE,
            {},
            ("restart the desktop app",),
            ("chrome://extensions", "browser extension"),
        ),
        (
            ErrorCode.PROTO_MISMATCH,
            {},
            ("different bridge protocol", "desktop app or Local Operator"),
            ("browser extension",),
        ),
        (
            ErrorCode.PROTO_MISMATCH,
            {"proto": 2},
            ("speaks bridge protocol 2", "this Local Operator speaks 1"),
            ("browser extension",),
        ),
        (
            ErrorCode.ORIGIN_DENIED,
            {"origin": "https://example.com"},
            ("desktop app's browser tab", "Do not retry"),
            ("extension popup",),
        ),
        (
            ErrorCode.ORIGIN_PROMPT_PENDING,
            {},
            ("Local Operator desktop app is waiting",),
            ("extension popup",),
        ),
        (
            ErrorCode.ORIGIN_NOT_ALLOWED,
            {"origin": "https://example.com", "url": "https://example.com/page"},
            ("desktop app's browser tab", "action='request_access'"),
            ("extension popup",),
        ),
        (
            ErrorCode.INTERNAL,
            {"stalled": "read"},
            ("restart the Local Operator desktop app",),
            ("chrome://extensions",),
        ),
    ],
    ids=[
        "no-tab",
        "unresponsive",
        "proto-mismatch-static",
        "proto-mismatch-numbered",
        "origin-denied",
        "prompt-pending",
        "origin-not-allowed",
        "stalled",
    ],
)
def test_ui_host_copy_never_sends_the_user_to_the_extension(
    code: ErrorCode, data: dict[str, Any], must: tuple[str, ...], forbidden: tuple[str, ...]
) -> None:
    """Every code whose REMEDY is a process must name the app, not the extension.

    Naming chrome://extensions to the user of a desktop app (or telling them to
    run `lop browser install`) sends them somewhere that cannot fix anything —
    which is why this table is per host rather than one set of sentences.
    """
    rendered = format_error(_error(code, data), action="open", host="ui")
    for fragment in must:
        assert fragment in rendered, rendered
    for fragment in forbidden:
        assert fragment not in rendered, rendered


def test_shared_codes_keep_one_spelling_for_every_host() -> None:
    """Codes whose copy names no process must not fork, or the two drift."""
    for code, data in (
        (ErrorCode.TAB_CLOSED, {}),
        (ErrorCode.NAV_FAILED, {}),
        (ErrorCode.ELEMENT_NOT_FOUND, {}),
        (ErrorCode.TAB_LIMIT, {}),
        (ErrorCode.TAB_AMBIGUOUS, {}),
        (ErrorCode.BUSY, {}),
        (ErrorCode.DEBUGGER_CONFLICT, {}),
    ):
        assert format_error(_error(code, data), action="read", host="ui") == format_error(
            _error(code, data), action="read", host="extension"
        ), code


def test_the_extension_host_copy_is_unchanged_by_this_seam() -> None:
    """Spot checks on the sentences the extension's users already read.

    The UI variants were added as an OVERRIDE table, so the extension's table must
    render byte-identically to before — a UI feature that rewords the extension's
    errors would be a regression nobody asked for.
    """
    assert format_error(
        _error(ErrorCode.EXTENSION_DISCONNECTED), action="open", host="extension"
    ) == (
        "browser extension not connected: the bridge daemon is running but no browser is "
        "attached. Ask the user to open their browser (the extension reconnects "
        "automatically), or check the extension is enabled."
    )
    assert (
        format_error(
            _error(ErrorCode.PROTO_MISMATCH, {"proto": 2}), action="open", host="extension"
        )
        == "browser bridge protocol mismatch: update Local Operator and the browser "
        "extension, then restart the bridge daemon."
    )
    assert (
        format_error(_error(ErrorCode.ORIGIN_DENIED, {"origin": "https://x.test"}), action="open")
        == "navigation to https://x.test was denied by the user (or the permission prompt "
        "went unanswered). Do not retry the same origin; ask the user to allow it from the "
        "extension popup if it is needed."
    )
