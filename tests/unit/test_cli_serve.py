"""``lop serve``'s listener: bound here, announced here, handed to uvicorn.

These tests exist because of one requirement that cannot be satisfied after the
fact: the daemon's rendezvous record must carry the port ACTUALLY BOUND, and
``--port 0`` (what the UI's own child asks for) means only the process that
bound the socket can know it. So the CLI binds, announces the resolved
address, and hands the open socket to uvicorn — and these tests pin all three,
without starting a real server.

The announce has TWO shapes and the split is the point: the address goes on the
app OBJECT for the daemon served in-process (no environment, so nothing this
daemon spawns can inherit it), and through the environment only for the
``--reload`` child, which cannot be reached any other way and where the value
names the process that wrote it.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from local_operator import cli
from local_operator.cli import _bind_serve_socket, serve_command
from local_operator.server import registry as serve_registry
from local_operator.server.app import app as asgi_app


@pytest.fixture(autouse=True)
def clean_app_announcement() -> Iterator[None]:
    """Give the module-level app the state it had before the test.

    ``serve_command`` announces on the app OBJECT (the environment is the
    ``--reload`` channel only, and ``tests/conftest.py`` scrubs that), and these
    tests never run the lifespan that would publish from it — so without this,
    the next test in the worker would see an app still announced on a dead
    ephemeral port.

    Snapshotting through Starlette's mapping interface (``__iter__`` +
    ``__getitem__``) deliberately: ``vars(app.state)`` returns ``{'_state': …}``,
    whose value is the SAME dict the app keeps using, so a snapshot taken that
    way restores nothing.
    """
    saved = {key: asgi_app.state[key] for key in asgi_app.state}
    state = asgi_app.state
    try:
        yield
    finally:
        for key in list(state):
            del state[key]
        for key, value in saved.items():
            state[key] = value


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
    the ``Config`` uvicorn is handed, the app object the lifespan reads at
    startup, and the record built from it — because a break at any one of them
    is a record that names a port nobody is listening on.

    The environment is asserted to be UNTOUCHED on this path, which is the
    other half of the contract: a daemon that wrote its address into its own
    environment would hand it to every process it spawns.
    """
    assert serve_command("127.0.0.1", 0, False) == 0

    sockets = captured_server["sockets"]
    assert isinstance(sockets, list) and len(sockets) == 1
    listener = sockets[0]
    try:
        bound = listener.getsockname()[1]
        assert bound > 0, "an ephemeral port was requested and one was granted"
        assert captured_server["port"] == bound, "uvicorn is told the bound port"
        assert serve_registry.SERVE_ANNOUNCE_ENV not in os.environ, (
            "the served-in-process path announces on the app object, never in "
            "the environment: nothing this daemon spawns may inherit its address"
        )
        announced = serve_registry.advertised_address(asgi_app)
        assert announced == ("127.0.0.1", bound), "the app carries the address it serves"
        record = serve_registry.build_record(instance_id="instance", announced=announced)
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
        assert serve_registry.advertised_address(asgi_app) == ("127.0.0.1", port)
    finally:
        sockets[0].close()
    assert f"Starting server at http://127.0.0.1:{port}" in capsys.readouterr().out


def test_an_occupied_port_is_refused_with_the_address_named(
    captured_server: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The common failure on this host is a daemon already on 1111.

    The refusal names the address, which is what makes it actionable. The exit
    code is this path's own (1), NOT uvicorn's ``STARTUP_FAILURE`` (3) — the
    message is printed by us and nothing branches on the number.

    The holder is a LISTENER, on a plain socket with no ``SO_REUSEADDR``, and it
    stays open across the whole call. That is not incidental — it is what is
    PORTABLE, measured on both platforms this runs on. On the Linux CI runner
    ``SO_REUSEADDR`` lets a second socket bind an address a NON-listening holder
    only bound (that is the shape that failed shard 3 when this test used a bare
    bound socket); on this macOS host (Darwin 25.6.0, arm64) every permutation of
    that holder was refused. A live listener is refused by a second bind on
    BOTH, whatever ``SO_REUSEADDR`` says (only ``SO_REUSEPORT`` would defeat
    it), which is also how a real daemon holds its port — so this exercises the
    actual collision rather than a platform quirk.
    """
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
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
    announced = os.environ[serve_registry.SERVE_ANNOUNCE_ENV].split(" ")
    assert announced == [str(os.getpid()), "127.0.0.1", str(port)], (
        "the reload child is told the resolved port through the environment, and "
        "the announcement names THIS process so only a child of ours honours it"
    )
    assert f"Starting server at http://127.0.0.1:{port}" in capsys.readouterr().out


def test_a_reload_probe_that_cannot_bind_is_refused_like_the_listener(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``--reload`` probe reports a bind failure the way the listener does.

    An unbindable host (``--host 10.1.2.3``: not an address on this machine, and
    the reachable input the reviewer named) used to escape ``serve_command`` as
    an uncaught traceback on this branch while the non-reload branch printed a
    named refusal — the same operator mistake, two different reports. uvicorn
    must never be reached in that case.
    """

    def refuse(host: str, port: int) -> socket.socket:
        raise OSError(49, "Can't assign requested address")

    with patch.object(cli, "_bind_serve_socket", refuse), patch("uvicorn.run") as mock_run:
        assert serve_command("10.1.2.3", 0, True) == 1

    mock_run.assert_not_called()
    assert "cannot bind http://10.1.2.3:0" in capsys.readouterr().err
