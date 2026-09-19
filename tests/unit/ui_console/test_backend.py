"""The console's session-side client: transport, copy, budgets and reachability.

Two kinds of test, separated the way the browser host's client tests separate
them:

* the TRANSPORT ones route the client's own `httpx.AsyncClient` through a mock
  transport — same shape as `tests/unit/ui_browser/test_backend.py`, because it is
  the same transport with two seams overridden;
* the REACHABILITY ones stand up a REAL loopback listener, because the acquittal
  rule (a STALE record is condemned only if `/health` refuses to answer as the pid
  the file named) is one a mock can only assert was called.

What is NOT re-tested here: the RPC envelope, the error taxonomy and the timeout
arithmetic are the shared transport's, and they have their own tests. What this
module adds is where the console must DIFFER — its own copy, and its own budget
that never waits on a human prompt.
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
    HostClient,
)
from local_operator.browser_bridge.protocol import PROTO_VERSION, ErrorCode
from local_operator.ui_console import backend, state


def publish(tmp_path: Path, **updates: Any) -> state.ConsoleHostState:
    values: dict[str, Any] = {
        "pid": os.getpid(),
        "port": 52133,
        "session_key": secrets.token_hex(24),
        "proto": PROTO_VERSION,
        "app_version": "0.26.0",
        "console": True,
    }
    values.update(updates)
    current = state.ConsoleHostState.model_validate(values)
    state.publish(current, tmp_path)
    return current


def _client_with_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler: Any
) -> backend.ConsoleHostClient:
    real_async_client = httpx.AsyncClient

    def fake_async_client(**kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", fake_async_client)
    return backend.ConsoleHostClient(tmp_path)


@pytest.mark.asyncio
async def test_a_round_trip_sends_the_key_and_the_console_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = publish(tmp_path)
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers["X-Bridge-Key"]
        body = json.loads(request.content)
        seen["body"] = body
        return httpx.Response(
            200,
            json={"id": body["id"], "ok": True, "result": {"surfaces": []}},
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    result = await client.call("console_list", {"session_id": "s-1"})
    assert result == {"surfaces": []}
    assert seen["key"] == current.session_key
    assert seen["body"]["method"] == "console_list"
    assert seen["body"]["params"] == {"session_id": "s-1"}


@pytest.mark.asyncio
async def test_a_typed_refusal_comes_back_as_its_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "ok": False,
                "error": {
                    "code": "secure_input_active",
                    "message": "the user holds the secure span",
                    "data": {"surface": "con:1:a"},
                },
            },
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeError) as caught:
        await client.call("console_read", {"surface": "con:1:a", "mode": "viewport"})
    assert caught.value.code is ErrorCode.SECURE_INPUT_ACTIVE
    assert caught.value.data == {"surface": "con:1:a"}


@pytest.mark.asyncio
async def test_a_wrong_key_is_reported_with_the_console_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy is the console's, not the app's browser host's.

    The two share the process, the record and the conversation partner, so the
    only thing that makes the console's sentence different is that it names the
    CONSOLE — and a reader told to go and open a browser tab has been sent to look
    in the wrong place, which is worse than being told nothing.
    """
    publish(tmp_path)
    client = _client_with_transport(tmp_path, monkeypatch, lambda request: httpx.Response(401))
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_list", {})
    assert "rejected the state-file key" in str(caught.value)
    assert "browser" not in str(caught.value)


@pytest.mark.asyncio
async def test_an_unreadable_body_is_not_reported_as_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publish(tmp_path)
    client = _client_with_transport(
        tmp_path, monkeypatch, lambda request: httpx.Response(200, text="<html>nope</html>")
    )
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_list", {})
    assert "invalid response" in str(caught.value)


@pytest.mark.asyncio
async def test_a_read_timeout_says_accepted_but_unanswered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distinction the shared transport already draws, kept for the console.

    A WEDGED app accepted the command and never answered; an unreachable one was
    never reached. Collapsing them sends a session to restart a healthy app, which
    is the misdiagnosis the browser path's own copy exists to prevent.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    publish(tmp_path)
    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_list", {})
    message = str(caught.value)
    assert "did not answer within" in message
    assert "unresponsive" in message


@pytest.mark.asyncio
async def test_a_refused_connection_is_unreachable_not_unresponsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    publish(tmp_path)
    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_list", {})
    assert "no longer answering" in str(caught.value)


@pytest.mark.asyncio
async def test_no_record_at_all_is_the_absence_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = backend.ConsoleHostClient(tmp_path)
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_list", {})
    message = str(caught.value)
    assert "not running" in message
    assert "cannot outlive it" in message


def test_the_console_budget_is_its_own_not_the_browsers() -> None:
    """No console method waits on a human, so none may inherit the prompt window.

    The browser's arithmetic is `base + ORIGIN_PROMPT_WINDOW_S + margin` because
    its host can sit on a site-approval popup. An unknown console method on the
    shared default would be budgeted the same way — 190 seconds — and a wedged app
    would then hang a tool call for three minutes instead of answering honestly.
    """
    console = backend.ConsoleHostClient()
    browser = HostClient()
    assert console.host == "ui"
    assert console.timeout_for("console_list", {}) == backend.CONSOLE_TIMEOUTS["console_list"] + 5
    assert console.timeout_for("console_invented", {}) == max(backend.CONSOLE_TIMEOUTS.values()) + 5
    assert browser.timeout_for("console_list", {}) > console.timeout_for("console_list", {})
    assert max(backend.CONSOLE_TIMEOUTS.values()) + 5 < 190


def test_every_console_method_has_a_budget_and_a_name() -> None:
    """A method without a budget silently takes the unknown-method fallback, and a
    name outside the frozen vocabulary is a call the app will refuse."""
    assert set(backend.CONSOLE_METHODS) == set(backend.CONSOLE_TIMEOUTS)
    assert len(backend.CONSOLE_METHODS) == 10


def test_the_discovery_helpers_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """They run while constructing every session and at diagnostic sites."""
    monkeypatch.setattr(state, "available", _boom)
    monkeypatch.setattr(state, "advertisable", _boom)
    monkeypatch.setattr(state, "liveness", _boom)
    assert backend.ui_console_available() is False
    assert backend.ui_console_advertisable() is False
    status, current = backend.ui_console_liveness()
    assert status is state.Liveness.ABSENT and current is None


def _boom(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("unreadable record")


# --- the real listener: the one rule a mock cannot assert -------------------


class _Health(BaseHTTPRequestHandler):
    payload: dict[str, Any] = {}

    def do_GET(self) -> None:  # noqa: N802 - http.server's spelling
        body = json.dumps(type(self).payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base name
        # Overridden only to keep a test peer quiet on stderr. The parameter is
        # named `format` because the base class names it that way and pyright
        # checks override compatibility; nothing here formats anything.
        return


def _serve(payload: dict[str, Any]) -> tuple[HTTPServer, int]:
    handler = type("_Handler", (_Health,), {"payload": payload})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _publish_stale(tmp_path: Path, **updates: Any) -> state.ConsoleHostState:
    """A record whose heartbeat is genuinely old.

    `publish` refreshes `heartbeat_at` by contract (it is the writer a live host
    models), so the stale case is produced by rewriting the JSON — the same move
    the browser host's state tests make, and for the same reason: weakening the
    publisher to test the reader is how a publisher stops being trustworthy.
    """
    current = publish(tmp_path, **updates)
    current.heartbeat_at = 0.0
    state.state_path(tmp_path).write_text(current.model_dump_json())
    return current


@pytest.mark.asyncio
async def test_a_stale_but_alive_app_is_acquitted_by_one_probe(tmp_path: Path) -> None:
    server, port = _serve({"host": "ui", "pid": os.getpid(), "console": True})
    try:
        _publish_stale(tmp_path, port=port)
        assert state.liveness(tmp_path)[0] is state.Liveness.STALE
        assert await backend.ui_console_reachable(tmp_path) is True
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_a_recycled_port_answering_as_another_pid_is_not_this_host(
    tmp_path: Path,
) -> None:
    """The pid check is the UI host's stand-in for the bridge's device check: a
    stale record whose port another process has taken must not be acquitted."""
    server, port = _serve({"host": "ui", "pid": os.getpid() + 1})
    try:
        _publish_stale(tmp_path, port=port)
        assert await backend.ui_console_reachable(tmp_path) is False
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_a_console_less_app_is_never_probed_at_all(tmp_path: Path) -> None:
    """Asking a host that has already said "no console" would only confirm the
    refusal, and the answer is already in the record."""
    server, port = _serve({"host": "ui", "pid": os.getpid(), "console": False})
    try:
        _publish_stale(tmp_path, port=port, console=False)
        assert await backend.ui_console_reachable(tmp_path) is False
        assert backend.ui_console_advertisable(tmp_path) is False
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_an_absent_record_is_not_a_probe_candidate(tmp_path: Path) -> None:
    assert await backend.ui_console_reachable(tmp_path) is False


@pytest.mark.asyncio
async def test_a_code_this_version_does_not_model_is_a_newer_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q2: an unknown-but-VALID refusal is version skew, not a broken app.

    `ErrorDetail.code` is typed on the shared enum, so a code a newer app added
    fails `Response.model_validate` exactly where a torn body does — and the
    generic sentence ("returned an invalid response (HTTP 200)") sends the reader
    to look at a broken app. Design §15 makes this direction the ORDINARY one: a
    peer that does not know a code never emits it, so a code this side cannot
    parse is one it has not learned yet. The raw body is read for the one field
    that tells the two cases apart.
    """
    publish(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "ok": False,
                "error": {
                    "code": "some_future_code",
                    "message": "a host sentence the model must not be shown",
                    "data": {"anything": 1},
                },
            },
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_status", {"surface": "con:1:a"})
    message = str(caught.value)
    assert "some_future_code" in message, message
    assert "does not model" in message, message
    assert "invalid response" not in message, message
    assert "newer" in message, message
    assert "a host sentence the model must not be shown" not in message, message


@pytest.mark.asyncio
async def test_a_known_code_that_fails_validation_keeps_the_generic_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook names a code only when the code is genuinely unmodelled.

    A body that is broken in ANY other way (here: an `error` with no `message`)
    fails validation too, and calling that "a code this version does not model"
    would be a false statement of its own — `surface_unavailable` is very much
    modelled.
    """
    publish(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"id": body["id"], "ok": False, "error": {"code": "surface_unavailable"}},
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_status", {"surface": "con:1:a"})
    message = str(caught.value)
    assert "invalid response" in message, message
    assert "does not model" not in message, message


@pytest.mark.asyncio
async def test_an_unknown_code_from_a_peer_that_is_not_ours_is_still_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a well-formed `ok: false` envelope is read for its code.

    An `ok: true` body that fails validation has no refusal to name, and reading a
    code out of one would invent a refusal that never happened.
    """
    publish(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # Fails validation on an UNEXPECTED TOP-LEVEL FIELD, with a `code`-looking
        # value sitting outside `error` where the hook must not look for one.
        return httpx.Response(
            200,
            json={
                "id": body["id"],
                "ok": True,
                "result": {"surfaces": []},
                "code": "some_future_code",
            },
        )

    client = _client_with_transport(tmp_path, monkeypatch, handler)
    with pytest.raises(BridgeUnreachable) as caught:
        await client.call("console_list", {})
    assert "does not model" not in str(caught.value)


def test_the_proto_skew_copy_names_both_numbers_and_never_the_host() -> None:
    """Q1: §15 asks for the UI variant, which reads `data["proto"]`.

    `proto_mismatch` was the one code interpolating the host's free-text message,
    and losing the two revision numbers is losing the only actionable part —
    "update the app or Local Operator" without knowing which side is behind is a
    shrug. The host's prose must not reach the transcript either (the module's own
    contract, and what every other code in the table honours).
    """
    error = BridgeError(
        ErrorCode.PROTO_MISMATCH,
        "a host sentence the model must not be shown",
        {"proto": 7},
    )
    text = backend.console_error_text(error)
    assert "protocol 7" in text, text
    assert f"speaks {PROTO_VERSION}" in text, text
    assert "a host sentence the model must not be shown" not in text, text

    # No revision reported: one number rather than none, and still no host prose.
    bare = backend.console_error_text(BridgeError(ErrorCode.PROTO_MISMATCH, "host prose", {}))
    assert "did not report" in bare, bare
    assert str(PROTO_VERSION) in bare, bare
    assert "host prose" not in bare, bare


def test_the_grid_refusal_reads_the_nested_clamp_and_the_fixed_reason() -> None:
    """C-2: the clamp is nested, and a fixed grid is not a clamp at all.

    Two conditions share `invalid_grid`. Reading only a flat `cols`/`rows` pair
    printed nothing for the nested clamp (the app's spelling) and printed the
    surface's OWN grid as "it applied NxM instead" for a fixed surface — a false
    statement about what the app did.
    """
    nested = backend.console_error_text(
        BridgeError(
            ErrorCode.INVALID_GRID,
            "9000x9000 is outside the supported grid; the clamp would be 400x200",
            {"requested": {"cols": 9000, "rows": 9000}, "clamp": {"cols": 400, "rows": 200}},
        )
    )
    assert "It applied 400x200 instead." in nested, nested

    fixed = backend.console_error_text(
        BridgeError(
            ErrorCode.INVALID_GRID,
            "this surface was created with a fixed grid",
            {"reason": "fixed", "cols": 100, "rows": 30},
        )
    )
    assert "fixed grid" in fixed, fixed
    assert "applied" not in fixed, fixed

    # A flat pair is the same two numbers for a host that spells it flat: accepted
    # defensively, never read as `None`.
    flat = backend.console_error_text(
        BridgeError(ErrorCode.INVALID_GRID, "out of range", {"cols": 40, "rows": 10})
    )
    assert "It applied 40x10 instead." in flat, flat


def test_the_queue_refusal_reads_the_accepted_count_not_the_payload_size() -> None:
    """C-2: `accepted` is the bytes the host TOOK; `bytes` is the refused size.

    Reading `bytes` as an accepted count states the opposite of what happened, and
    it is the reason the key table is published rather than left to either half.
    """
    text = backend.console_error_text(
        BridgeError(
            ErrorCode.INPUT_QUEUE_FULL,
            "that payload is 9000 bytes and this surface accepts 4096 at a time",
            {"accepted": 0, "limit": 4096, "bytes": 9000},
        )
    )
    assert "0 bytes were accepted" in text, text
    assert "9000" not in text, text

    # A host that reports no count gets the honest sentence rather than a guess.
    bare = backend.console_error_text(
        BridgeError(ErrorCode.INPUT_QUEUE_FULL, "queue full", {"bytes": 9000})
    )
    assert "the queue filled" in bare, bare
    assert "9000" not in bare, bare


def test_the_record_is_read_through_the_console_model(tmp_path: Path) -> None:
    """`ConsoleHostClient` must read THIS namespace's model, or the capability bit
    is dropped on the floor and every call looks like a console-less host."""
    publish(tmp_path, console_surfaces=3)
    current = backend.ConsoleHostClient(tmp_path)._read()
    assert isinstance(current, state.ConsoleHostState)
    assert current.console is True and current.console_surfaces == 3
