"""The selection recount over a REAL socket, as uvicorn/h11 parse it (QA Q1, #3103).

Why a socket and not ``TestClient``: the failure was in the HTTP parser, which
an in-process ASGI client never runs. h11 refuses a request head buffered past
16 KiB without its terminating blank line (``DEFAULT_MAX_INCOMPLETE_EVENT_SIZE``)
with ``400`` before the route is called, so a GET recount carrying 500 ask-shaped
``excluded_ids`` (~50 KB of URL) failed whenever the kernel delivered it in more
than one read. The body form (``PATCH op:"preview"``) is not bound by that limit.

The server is the production router on an in-process ``uvicorn.Server`` with the
same ``Config`` defaults ``lop serve`` uses (no h11 override), bound to an
ephemeral 127.0.0.1 port. Requests are written with ``sendall`` in 8 KiB chunks
and a short gap, so the parser sees the request split across reads.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI

from local_operator.server.routes.desktop_notifications import router
from tests.unit.notifications.test_bulk import seed
from tests.unit.notifications.test_bulk_asks import queue
from tests.unit.notifications.test_inbox import SYNTHETIC_TOKEN

URL = "/v1/desktop/notifications"
CHUNK = 8 * 1024
#: The largest valid ask ref: ``ask:<12 hex>:<64 id chars>``, 81 bytes. 499 of
#: these plus one notify id is the 500-id worst case the UI may send.
ASK_IDS = [f"ask:{index:012x}:" + "a" * 64 for index in range(499)]


@pytest.fixture
def server(tmp_path: Path, monkeypatch) -> Iterator[int]:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    app = FastAPI()
    app.include_router(router)
    app.state.config_manager = SimpleNamespace(config_dir=tmp_path)
    started = threading.Event()

    class Server(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            started.set()

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    # http="h11" pins the parser that refused the request: httptools is not a
    # dependency, and pinning keeps the test meaning the same if it ever is.
    instance = Server(uvicorn.Config(app, lifespan="off", log_level="critical", http="h11"))
    thread = threading.Thread(target=instance.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    assert started.wait(10), "loopback HTTP did not start"
    try:
        yield listener.getsockname()[1]
    finally:
        instance.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "loopback HTTP did not stop"


def _head(method: str, target: str, port: int, extra: str = "") -> bytes:
    return (
        f"{method} {target} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Authorization: Bearer {SYNTHETIC_TOKEN}\r\n"
        f"{extra}"
        "Connection: close\r\n\r\n"
    ).encode()


def _exchange(port: int, payload: bytes, *, stop_at: int | None = None) -> tuple[int, bytes]:
    """Write ``payload`` in 8 KiB chunks, then read the whole response.

    ``stop_at`` writes only that many bytes before reading: the server must then
    answer from a partial request, which is how the negative control proves
    the parser limit without depending on how the kernel coalesces writes.
    """
    sent = payload if stop_at is None else payload[:stop_at]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for start in range(0, len(sent), CHUNK):
            conn.sendall(sent[start : start + CHUNK])
            time.sleep(0.02)
        response = b""
        while chunk := conn.recv(65536):
            response += chunk
    status_line, _, rest = response.partition(b"\r\n")
    return int(status_line.split()[1]), rest.partition(b"\r\n\r\n")[2]


def _preview(port: int, query: str = "") -> dict[str, Any]:
    status, body = _exchange(port, _head("GET", URL + "?preview=true" + query, port))
    assert status == 200, body
    return json.loads(body)["result"]["selection"]


def _body_recount(port: int, token: str, excluded: list[str]) -> tuple[int, bytes]:
    body = json.dumps({"op": "preview", "selection_token": token, "excluded_ids": excluded})
    assert len(body) > 40_000  # well past h11's 16 KiB head limit, as a body
    extra = f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
    return _exchange(port, _head("PATCH", URL, port, extra) + body.encode())


def test_500_ask_shaped_exclusions_recount_in_a_chunked_body(server, tmp_path):
    ids = seed(tmp_path, 3)
    selection = _preview(server)
    assert selection["eligible_count"] == 3
    excluded = [ids[0], *ASK_IDS]
    assert len(excluded) == 500
    status, answer = _body_recount(server, selection["token"], excluded)
    assert status == 200, answer
    refreshed = json.loads(answer)["result"]["selection"]
    assert refreshed["eligible_count"] == 2
    for key in ("token", "through_seq", "expires_at"):
        assert refreshed[key] == selection[key]


def test_v2_ask_selection_recounts_500_real_and_padded_ask_refs_in_a_body(server, tmp_path):
    """#3105's v2 (``include_ask_reads``) selection through the same body form.

    V2 is the selection whose exclusions are mostly ask refs, so it is the one
    that most needs the body form. Real captured asks are excluded alongside
    maximum-length padding refs; only the real ones may change the ask count.
    """
    refs = queue(tmp_path, 3)
    seed(tmp_path, 2)
    selection = _preview(server, "&include_ask_reads=true&state=all")
    assert selection["selection_version"] == 2
    assert selection["eligible_ask_count"] == 3
    excluded = [refs[0], refs[1], *ASK_IDS[:498]]
    assert len(excluded) == 500
    status, answer = _body_recount(server, selection["token"], excluded)
    assert status == 200, answer
    refreshed = json.loads(answer)["result"]["selection"]
    assert refreshed["eligible_ask_count"] == 1 and refreshed["eligible_update_count"] == 2
    for key in ("token", "through_seq", "expires_at"):
        assert refreshed[key] == selection[key]


def test_negative_control_the_same_exclusions_in_a_url_are_refused(server, tmp_path):
    """Proves the instrument: the GET form of the same recount dies in the parser.

    Only the first 24 KiB of the head is written, with no terminating blank
    line, so h11 must refuse it however the kernel groups the chunks. If this
    ever answers 200, the parser limit moved and the test above no longer
    discriminates the body form from the URL form.
    """
    seed(tmp_path)
    selection = _preview(server)
    query = "&".join(f"excluded_ids={item}" for item in ASK_IDS)
    target = f"{URL}?preview=true&selection_token={selection['token']}&{query}"
    request = _head("GET", target, server)
    assert len(request) > 40_000
    status, _ = _exchange(server, request, stop_at=3 * CHUNK)
    assert status == 400
