"""Relay admission fence: a handoff pauses NEW peers, keeps existing links.

Focused on the fence contract the ready-successor handoff relies on, so it drives
a real serve-shaped relay on loopback (no live services, and the port is 0 so the
kernel picks).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from local_operator.network import identity, relay
from tests.unit.network.test_relay_e2e import serve_shaped_relay


def _relay(root: Path, monkeypatch: pytest.MonkeyPatch) -> relay.RelayServer:
    ident = identity.mint(root, name="fence-device")
    return serve_shaped_relay(root, monkeypatch, identity=ident)


def test_begin_reload_refuses_new_peers_but_leaves_the_listener_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _relay(tmp_path, monkeypatch)
    server.start()  # binds the listener AND runs the real accept loop
    try:
        host, port = server.settings.listen_address, server.settings.port
        server.begin_reload()

        # The listener still accepts at the KERNEL level (bind is untouched) — the
        # fence is admission, not shutdown — but the relay closes the arrival.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(3.0)
        try:
            probe.connect((host, port))
            # A closed-but-accepted connection reads EOF rather than a welcome frame.
            assert probe.recv(1) == b"", "a fenced relay must not answer a new peer"
        finally:
            probe.close()
        # Nothing became a link while fenced.
        assert server.links == {}

        server.end_reload()
        assert server.reload_blocker() is None
    finally:
        server.stop()


def test_reload_blocker_reports_an_inflight_handshake_not_the_semaphore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _relay(tmp_path, monkeypatch)
    assert server.reload_blocker() is None
    # Simulate one handshake mid-flight through the real counter the accept loop
    # and the handshake thread both use.
    with server._handshakes_lock:
        server._handshakes_active += 1
    try:
        assert server.reload_blocker() is not None
    finally:
        with server._handshakes_lock:
            server._handshakes_active -= 1
    assert server.reload_blocker() is None


def test_admission_defaults_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _relay(tmp_path, monkeypatch)
    assert server._admission.is_set(), "a fresh relay must admit peers"
