"""``lop serve``'s listener: bound here, announced here, handed to uvicorn.

These tests exist because of one requirement that cannot be satisfied after the
fact: the daemon's rendezvous record must carry the port ACTUALLY BOUND, and
``--port 0`` (what the UI's own child asks for) means only the process that
bound the socket can know it. So the CLI binds, announces the resolved
address, and hands the open socket to uvicorn — and these tests pin all three,
without starting a real server.
"""

from __future__ import annotations

import socket

import pytest

from local_operator.cli import _bind_serve_socket, serve_command
from local_operator.server import registry as serve_registry


@pytest.fixture
def captured_server(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``uvicorn.Server.run`` with a capture, so nothing serves.

    The capture also keeps the process from hanging: a real ``serve_command``
    runs a server until it is signalled, which is exactly what an automated
    test must not do.
    """
    import uvicorn

    captured: dict[str, object] = {}

    def fake_run(server: object, sockets: list[socket.socket] | None = None) -> None:
        captured["sockets"] = sockets
        captured["config"] = server.config  # type: ignore[attr-defined]
        captured["host"] = server.config.host  # type: ignore[attr-defined]
        captured["port"] = server.config.port  # type: ignore[attr-defined]

    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    return captured


def test_port_zero_announces_the_port_it_actually_bound(
    captured_server: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of ``--port 0``: the record gets the RESOLVED port.

    Asserted at every hop the value travels — the socket the kernel gave us,
    the ``Config`` uvicorn is handed, the environment the app reads at startup,
    and the record built from it — because a break at any one of them is a
    record that names a port nobody is listening on.
    """
    assert serve_command("127.0.0.1", 0, False) == 0

    sockets = captured_server["sockets"]
    assert isinstance(sockets, list) and len(sockets) == 1
    listener = sockets[0]
    try:
        bound = listener.getsockname()[1]
        assert bound > 0, "an ephemeral port was requested and one was granted"
        assert captured_server["port"] == bound, "uvicorn is told the bound port"
        assert serve_registry.advertised_address() == ("127.0.0.1", bound)
        record = serve_registry.build_record(instance_id="instance")
        assert (record.host, record.port) == ("127.0.0.1", bound)
    finally:
        listener.close()

    assert f"Starting server at http://127.0.0.1:{bound}" in capsys.readouterr().out


def test_a_fixed_port_is_bound_and_announced_unchanged(
    captured_server: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Port ``1111`` keeps working, and keeps being the default elsewhere.

    The socket is bound before uvicorn is handed it, so an occupied port is
    reported by US, with the address named.
    """
    probe = _bind_serve_socket("127.0.0.1", 0)
    port = probe.getsockname()[1]
    probe.close()

    assert serve_command("127.0.0.1", port, False) == 0
    sockets = captured_server["sockets"]
    assert isinstance(sockets, list)
    try:
        assert sockets[0].getsockname()[1] == port
        assert serve_registry.advertised_address() == ("127.0.0.1", port)
    finally:
        sockets[0].close()
    assert f"Starting server at http://127.0.0.1:{port}" in capsys.readouterr().out


def test_an_occupied_port_is_refused_with_the_address_named(
    captured_server: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The common failure on this host is a daemon already on 1111.

    uvicorn's own path logs ``[Errno 48] Address already in use`` and exits
    non-zero; the address is what makes it actionable, and the exit code is the
    same so a supervisor sees no difference.
    """
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    port = holder.getsockname()[1]
    try:
        assert serve_command("127.0.0.1", port, False) == 1
    finally:
        holder.close()

    # A refused bind must never have reached uvicorn (``captured_server`` would
    # otherwise have recorded the attempt).
    assert captured_server == {}
    assert f"cannot bind http://127.0.0.1:{port}" in capsys.readouterr().err


def test_reload_resolves_an_ephemeral_port_without_keeping_a_listener(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dev-only path, and why it is allowed to differ.

    uvicorn reloads only an app given as an import string, and the reloader
    re-imports it in a child, so the socket cannot be handed over and the port
    is resolved with a bind/close probe instead. The record still names the
    port the child is told to bind — the residual race is the one the design
    accepts, and the identity check (``/health``'s ``instance_id``) is what
    admits a candidate, never the record's port.
    """
    import uvicorn

    captured: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(kwargs, app=app))

    assert serve_command("127.0.0.1", 0, True) == 0

    port = captured["port"]
    assert isinstance(port, int) and port > 0
    assert captured["app"] == "local_operator.server.app:app"
    assert serve_registry.advertised_address() == ("127.0.0.1", port)
    assert f"Starting server at http://127.0.0.1:{port}" in capsys.readouterr().out
