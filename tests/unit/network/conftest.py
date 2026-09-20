"""Fixtures and helpers for the mesh core's unit tests.

Every test builds its OWN config root: the modules take a ``root`` argument
precisely so a test never has to monkeypatch ``HOME``, and the one place that
matters is identity — a test that minted a key into the operator's real store
would be a defect wearing a green tick.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A private config root for one test."""
    return tmp_path


@pytest.fixture()
def socketpair() -> Any:
    """A connected pair of TCP sockets on loopback — real sockets, not a stub.

    The handshake's failure modes are about what goes on the wire (a silent close,
    a pipelined frame, a truncated read), so a fake stream would test the fake
    rather than the protocol.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname()[:2], timeout=5)
    accepted, _addr = server.accept()
    try:
        yield client, accepted
    finally:
        for sock in (client, accepted, server):
            try:
                sock.close()
            except OSError:
                pass
