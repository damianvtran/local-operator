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
async def test_missing_record_says_open_the_app(tmp_path: Path) -> None:
    with pytest.raises(BridgeUnreachable) as excinfo:
        await backend.UiHostClient(tmp_path).call("status", {})
    message = str(excinfo.value)
    assert "not running" in message and "Open the desktop app" in message
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
    with pytest.raises(BridgeUnreachable, match="not running"):
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
