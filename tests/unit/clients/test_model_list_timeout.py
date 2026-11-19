"""The provider ``list_models()`` calls give up on an upstream that never answers.

``requests`` has no default timeout. Before ``MODEL_LIST_TIMEOUT`` an accepted-but-
silent connection held ``list_models()`` until the OS dropped the socket (minutes),
and ``/v1/models`` with it. These tests point each client at a real local socket that
accepts and never replies, so the bound is proved at the socket, not by inspecting a
mock's kwargs. The module-level constant is shrunk per test so the run stays short;
the production values and their measurement live on ``MODEL_LIST_TIMEOUT`` itself.
"""

import socket
import threading
import time
from typing import Iterator

import pytest
from pydantic import SecretStr

from local_operator.clients import _http, openrouter, radient
from local_operator.clients.openrouter import OpenRouterClient
from local_operator.clients.radient import RadientClient

SHORT = (1.0, 1.0)


@pytest.fixture
def silent_upstream() -> Iterator[str]:
    """A listening socket that accepts connections and never writes a byte."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    held: list[socket.socket] = []

    def accept() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            held.append(conn)

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}/v1"
    finally:
        server.close()
        for conn in held:
            conn.close()


def test_production_bound_is_finite_on_both_phases() -> None:
    connect, read = _http.MODEL_LIST_TIMEOUT
    assert 0 < connect < float("inf")
    assert 0 < read < float("inf")


def test_radient_list_models_times_out_on_silent_upstream(silent_upstream, monkeypatch) -> None:
    monkeypatch.setattr(radient, "MODEL_LIST_TIMEOUT", SHORT)
    client = RadientClient(api_key=SecretStr("synthetic"), base_url=silent_upstream)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="(?i)timed out"):
        client.list_models()
    # Loose ceiling: the point is "returns at all", with generous slack for a
    # loaded CI host; the unbounded call would sit here for minutes.
    assert time.monotonic() - started < 30


def test_openrouter_list_models_times_out_on_silent_upstream(silent_upstream, monkeypatch) -> None:
    monkeypatch.setattr(openrouter, "MODEL_LIST_TIMEOUT", SHORT)
    client = OpenRouterClient(api_key=SecretStr("synthetic"), base_url=silent_upstream)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="(?i)timed out"):
        client.list_models()
    assert time.monotonic() - started < 30
