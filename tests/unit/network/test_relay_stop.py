"""``RelayServer.stop()`` QUIESCES — the post-condition, and the race that broke it.

WHY THIS FILE EXISTS. ``stop()`` set ``_stop``, snapshotted ``links`` ONCE, waited 50 ms
and closed what it had seen — and nothing on the way into the table read ``_stop``. Two
arrivals therefore fell through the snapshot and were never closed:

* a connection the kernel completed while the accept loop was BLOCKED in ``accept()``,
  which was handed to a handshake thread after ``_stop`` had been set;
* a handshake already in flight, which finished inside that 50 ms window and registered
  a link the snapshot never saw.

Either one leaves a relay that ANSWERS a peer it has stopped for. That is the property
an incident control rests on — "make every peer disconnect" is not a control at all if
the relay keeps serving an established peer — and the pre-fix behaviour was measured
rather than inferred: across 40 natural runs the gap between a peer's registration and
``stop()``'s snapshot ran from -1.66 ms to +13.94 ms (clustered at 0-3 ms), and a 20 ms
shift of only thread schedules reproduced a relay answering an op after ``stop()``.

THE CELLS ARE ORDERING, NOT TIMING. Neither one sleeps and waits to see what happens:
each PARKS the racy step on a ``threading.Event`` until the test has called ``stop()``,
so the interleaving under test is the interleaving that runs. That is the whole
difference between this file and the 40-run reproduction — a schedule that has to be
hit again on the next machine is not a test, which is why the defect survived.

The post-condition both cells assert, in one sentence: **after ``stop()`` returns, a
peer that dials or handshakes late holds no live link, and the relay does not answer
it.**
"""

from __future__ import annotations

import socket
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import relay, store, wire
from local_operator.network.handshake import Credential, Handshake
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


def _handshake_that_waits_for_the_test(
    server: relay.RelayServer,
    sock: socket.socket,
    *,
    network_id: str,
    epoch: int,
    credential: Any,
    took_the_challenge: threading.Event,
    release: threading.Event,
    outcome: list[str],
) -> None:
    """One member handshake, driven the way ``RelayServer.dial`` drives it, PARKED.

    It stops between ``read_challenge`` and ``send_auth``, and that stop IS the
    mechanism: the challenge coming back is proof the listener is inside this handshake
    (past the hello, with a thread and a socket of its own via ``accept()``), and it is
    proof the listener has registered NOTHING yet, because a handshake registers only
    after its welcome. So the test knows the handshake is in flight when it calls
    ``stop()`` without racing it, and the peer finishes only when the test says so.

    ``outcome`` carries what the peer OBSERVED, because that observation is the
    assertion: a stopped relay must leave this peer with a closed socket, and ``acked``
    — reading a reply off a link established after ``stop()`` returned — is the
    pre-fix reading this file exists for.
    """
    handshake = Handshake.new(
        role="dialer",
        identity=server.identity,
        network_id=network_id,
        epoch=epoch,
        instance_id=server.instance_id,
        session_protocol=net_cli._session_protocol(),  # noqa: SLF001 — the CLI's own value
        mode="member",
        capabilities=list(wire.LINK_CAPABILITIES),
        build={},
    )
    handshake.send_hello(sock)
    reader = wire.FrameReader(sock)
    handshake.read_challenge(reader, wire.deadline_in(30.0))
    took_the_challenge.set()
    if not release.wait(30.0):  # pragma: no cover - the test always releases it
        raise AssertionError("the test never released the parked handshake")
    try:
        handshake.send_auth(sock, credential)
        handshake.read_welcome(reader, wire.deadline_in(30.0))
    except (OSError, TimeoutError, wire.LinkCryptoError) as exc:
        outcome.append(f"refused:{type(exc).__name__}")
        return
    # Reached only while the relay still answers a link made after it stopped.
    handshake.establish()
    codec = handshake.codec()
    sock.sendall(codec.seal({"op": "ping", "req": 4242, "locality": "remote"}))
    reply = codec.open(reader.read_record_payload(wire.deadline_in(30.0)))
    outcome.append("acked" if reply.get("op") == "ack" else f"answered:{reply.get('op')}")


def test_a_handshake_in_flight_when_stop_runs_leaves_no_live_link(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half of the race the 40-run reproduction caught: a link born after ``stop``.

    A member peer completes its hello and takes its challenge, and then WAITS — parked
    one frame short of the point that registers it. ``stop()`` runs to completion while
    it waits, and only then does the peer send its auth frame and use whatever link it
    gets. On the pre-fix tree it gets one, and the relay answers an op over it; the
    snapshot ``stop()`` took had already gone by.
    """
    # ``getfixturevalue`` by name, as ``test_credentials_real_link`` does it: a parameter
    # of the same name would shadow the import that makes the fixture reachable, and so
    # would a local of that name.
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _host, _port = _pair(pair_devices, monkeypatch)
    # THE PRE-CONDITION IS WAITED FOR, NOT ASSUMED. The ceremony leaves a link on A
    # that the joiner's own socket close has to drain, and a cell that started with it
    # live could not tell a link made by the parked handshake from one made by the
    # ceremony — so this asserts the event, not a sleep, and fails loudly rather than
    # silently testing nothing.
    assert net_fixtures.wait_for(
        lambda: not any(link.alive for link in server_a.links.values())
    ), f"the pairing link never drained: {list(server_a.links)}"

    state = store.require_secrets(record.network_id, server_b.root)
    credential = Credential(
        "epoch",
        record.epoch,
        wire.epoch_key(state.secret, record.network_id, record.epoch),
    )
    sock = socket.create_connection((host, port), timeout=30)
    took_the_challenge = threading.Event()
    release = threading.Event()
    outcome: list[str] = []
    peer = threading.Thread(
        target=_handshake_that_waits_for_the_test,
        args=(server_b, sock),
        kwargs={
            "network_id": record.network_id,
            "epoch": record.epoch,
            "credential": credential,
            "took_the_challenge": took_the_challenge,
            "release": release,
            "outcome": outcome,
        },
        name="parked-peer",
        daemon=True,
    )
    peer.start()
    try:
        assert took_the_challenge.wait(30.0), "the listener never answered the hello"
        # THE STATE THE RACE NEEDS, read rather than assumed: the peer is provably
        # inside its handshake (it holds a challenge) and provably unregistered, and
        # the relay holds no live link for the arrival at ``stop()`` to be about.
        assert not [
            link for link in server_a.links.values() if link.alive
        ], "a link was already live when stop() ran, so this cell proves nothing"

        server_a.stop()
        release.set()
        peer.join(30.0)
        assert not peer.is_alive(), "the parked peer never finished"

        # THE POST-CONDITION, in the two readings the classification took, reported
        # together so one run shows the whole state: what the PEER got, and what the
        # relay still HOLDS. Either half alone can pass by accident on a different
        # schedule — ``acked`` is the reading that says the peer reached an op.
        refused = bool(outcome) and outcome[0].startswith("refused:")
        live = [link.link_id for link in server_a.links.values() if link.alive]
        assert refused and not live, (
            "after stop() returned, this peer read "
            f"{outcome or ['nothing']} and the relay held {live or 'no'} live link(s)"
        )
    finally:
        release.set()
        peer.join(10)
        sock.close()


@pytest.mark.parametrize(
    ("socket_attr", "loop_attr", "handler_attr", "loop_name", "handler_name"),
    [
        pytest.param(
            "_listener",
            "_accept_loop",
            "_handshake_inbound",
            "listener-under-test",
            "a handshake thread",
            id="peer-listener",
        ),
        pytest.param(
            "_control",
            "_control_loop",
            "_control_connection",
            "control-under-test",
            "a control connection thread",
            id="loopback-control",
        ),
    ],
)
def test_a_connection_accepted_after_stop_is_closed_not_served(
    socket_attr: str,
    loop_attr: str,
    handler_attr: str,
    loop_name: str,
    handler_name: str,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other arrival, on BOTH accept loops: ``accept()`` returning after ``_stop``.

    Each loop's ``_stop`` check runs at the TOP of the loop, so a thread parked inside
    ``accept()`` is already past it — a connection the kernel completed during the
    shutdown was served by a relay that had stopped. The peer listener is the arriving
    peer; the loopback control surface is the same defect in the same shape, and both
    are checked here rather than one being left to the reader's confidence.

    What the gate stages is only WHEN ``accept()`` may return. The listener, the
    connection and the socket the loop really holds are all real, because the claim
    under test is about what the loop does with a socket it has.
    """
    server = serve_shaped_relay(root, monkeypatch)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()[:2]
    accepted = threading.Event()
    release = threading.Event()

    class _GatedListener:
        """A real listener whose single ``accept()`` waits for the test to let go."""

        def accept(self) -> tuple[socket.socket, Any]:
            sock, addr = listener.accept()
            accepted.set()
            release.wait(30.0)
            return sock, addr

        def close(self) -> None:
            listener.close()

    # noqa: SLF001 — the loop's own accept, and the loop under test on an unstarted relay
    setattr(server, socket_attr, cast(Any, _GatedListener()))
    served: list[str] = []
    monkeypatch.setattr(server, handler_attr, lambda *args: served.append(str(args[0])))
    loop = threading.Thread(
        target=getattr(server, loop_attr),  # noqa: SLF001 — the loop under test
        name=loop_name,
        daemon=True,
    )
    loop.start()
    client = socket.create_connection((host, port), timeout=30)
    try:
        assert accepted.wait(30.0), "the accept loop never took the connection"

        server.stop()
        release.set()
        loop.join(30.0)
        assert not loop.is_alive(), "the accept loop outlived stop()"

        assert served == [], f"a stopped relay handed an accepted connection to {handler_name}"
        client.settimeout(30.0)
        assert client.recv(64) == b"", "the connection was left open instead of closed"
    finally:
        release.set()
        loop.join(10)
        client.close()
        listener.close()
