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

AND THE BARRIER HAS TWO TERMS, which is why the panic cells live here rather than in a
file of their own: ``_register_or_refuse`` refuses a link for an UNTRUSTED network for
the same reason, and by the same lock, as one that arrives after a stop. The panic
paths are the two writers of that state, and a handshake already past the challenge
when one lands is this same race with a WIDER window — the whole auth → welcome →
register span rather than the 50 ms settle. What makes it worse than the stop case is
what is behind it: an admin's panic rotates the epoch, so an admitted survivor is
refused by the epoch check, but a NON-ADMIN panic rotates nothing, so a survivor that
gets in is authorised by the capability check and *served* — and the fan-out that would
have told that peer does not dial. Both cells below park the handshake across a real
panic, and the local one asserts ``rotated is False`` so the weaker half is the one
under test.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Iterator
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
        handshake.establish()
        codec = handshake.codec()
        sock.sendall(codec.seal({"op": "ping", "req": 4242, "locality": "remote"}))
        reply = codec.open(reader.read_record_payload(wire.deadline_in(30.0)))
    except Exception as exc:  # noqa: BLE001 — the KIND of failure is what is asserted
        # THE WHOLE POST-AUTH SEQUENCE IS INSIDE THE ``try``, and that is what makes
        # this observable: a relay that answers the handshake and THEN refuses the
        # link fails the peer at the read AFTER the welcome, so a ``try`` that stopped
        # at ``read_welcome`` would let the thread die silently and file an empty
        # reading as "nothing happened".
        kind = "refused" if isinstance(exc, _REFUSAL_EXCEPTIONS) else "error"
        outcome.append(f"{kind}:{type(exc).__name__}")
        return
    outcome.append("acked" if reply.get("op") == "ack" else f"answered:{reply.get('op')}")


#: What a REFUSED peer can raise: the socket ending under it, the deadline passing, a
#: frame that no longer decrypts, or the relay's own refusal arriving as an exception
#: rather than as a status (``MeshRefusal``). A failure outside this set is recorded as
#: ``error:`` rather than ``refused:``, so a cell cannot pass on a crash.
_REFUSAL_EXCEPTIONS = (OSError, TimeoutError, wire.LinkCryptoError, relay.MeshRefusal)


class _ParkedPeer:
    """A member handshake driven by a test, held one frame short of registration.

    The park is the mechanism and not a sleep: the challenge coming back proves the
    listener is inside this handshake (past the hello, with a thread and a socket of its
    own via ``accept()``) and that it has registered NOTHING yet, because a listener
    registers only after its welcome. A test can therefore know the handshake is in
    flight at the moment it stops a relay or panics a network, without racing either,
    and the peer finishes only when the test says so.
    """

    def __init__(
        self,
        dialer: relay.RelayServer,
        sock: socket.socket,
        *,
        network_id: str,
        epoch: int,
        credential: Any,
    ) -> None:
        self._sock = sock
        self._took_challenge = threading.Event()
        self._release = threading.Event()
        #: What the peer OBSERVED, because that observation is the assertion: a refused
        #: peer reads a closed socket, and ``acked`` — a reply read off a link
        #: established after the relay said no — is the pre-fix reading these cells
        #: exist for.
        self.outcome: list[str] = []
        self._thread = threading.Thread(
            target=_handshake_that_waits_for_the_test,
            args=(dialer, sock),
            kwargs={
                "network_id": network_id,
                "epoch": epoch,
                "credential": credential,
                "took_the_challenge": self._took_challenge,
                "release": self._release,
                "outcome": self.outcome,
            },
            name="parked-peer",
            daemon=True,
        )

    def start(self) -> _ParkedPeer:
        self._thread.start()
        return self

    def parked(self, timeout: float = 30.0) -> bool:
        """True once the listener answered the hello — proof the handshake is in flight."""
        return self._took_challenge.wait(timeout)

    def release(self, timeout: float = 30.0) -> None:
        """Let the peer finish, and return only once its side is done."""
        self._release.set()
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "the parked peer never finished"

    def close(self) -> None:
        self._release.set()
        self._thread.join(10.0)
        self._sock.close()


@contextlib.contextmanager
def _parked_member_handshake(
    dialer: relay.RelayServer,
    *,
    host: str,
    port: int,
    network_id: str,
    epoch: int,
    credential: Any,
) -> Iterator[_ParkedPeer]:
    """A connected socket, with :class:`_ParkedPeer` on it, for the body of a ``with``."""
    sock = socket.create_connection((host, port), timeout=30)
    peer = _ParkedPeer(
        dialer, sock, network_id=network_id, epoch=epoch, credential=credential
    ).start()
    try:
        yield peer
    finally:
        peer.close()


def _member_credential(root: Path, network_id: str, epoch: int) -> Credential:
    """The credential a member dials with: that device's OWN copy of the epoch secret."""
    state = store.require_secrets(network_id, root)
    return Credential("epoch", epoch, wire.epoch_key(state.secret, network_id, epoch))


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
    with _parked_member_handshake(
        server_b,
        host=host,
        port=port,
        network_id=record.network_id,
        epoch=record.epoch,
        credential=credential,
    ) as peer:
        assert peer.parked(), "the listener never answered the hello"
        # THE STATE THE RACE NEEDS, read rather than assumed: the peer is provably
        # inside its handshake (it holds a challenge) and provably unregistered, and
        # the relay holds no live link for the arrival at ``stop()`` to be about.
        assert not [
            link for link in server_a.links.values() if link.alive
        ], "a link was already live when stop() ran, so this cell proves nothing"

        server_a.stop()
        peer.release()

        # THE POST-CONDITION, in the two readings the classification took, reported
        # together so one run shows the whole state: what the PEER got, and what the
        # relay still HOLDS. Either half alone can pass by accident on a different
        # schedule — ``acked`` is the reading that says the peer reached an op.
        refused = bool(peer.outcome) and peer.outcome[0].startswith("refused:")
        live = [link.link_id for link in server_a.links.values() if link.alive]
        assert refused and not live, (
            "after stop() returned, this peer read "
            f"{peer.outcome or ['nothing']} and the relay held {live or 'no'} live link(s)"
        )


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


def test_a_handshake_in_flight_when_a_non_admin_panic_runs_is_refused(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panic's window, where nothing else refuses the survivor.

    An admin's panic rotates the epoch, so a link that registered after the alarm is
    refused by the epoch check on its next frame. A NON-ADMIN panic rotates nothing —
    the design gives it the trust half only — so a survivor that gets admitted passes
    the capability check and is SERVED, and the fan-out that would have told that peer
    does not dial. ``assert panicked["rotated"] is False`` pins which half this cell is
    about; without it the cell would also pass on a relay that only works because the
    epoch moved.

    The park is the same rig as the stop cell and the panic is the relay's own
    ``net_panic_local`` control op, so the state write under test is the product's.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _pair_host, _pair_port = _pair(pair_devices, monkeypatch, role="drive")
    # B is the device that panics, so B is the device the survivor meets: it has to be
    # listening, and its own table has to be empty for the park to mean anything.
    b_host, b_port = server_b.bind()
    server_b.start()
    try:
        assert net_fixtures.wait_for(
            lambda: not any(link.alive for link in server_b.links.values())
        ), f"the pairing link never drained: {list(server_b.links)}"
        with _parked_member_handshake(
            server_a,
            host=b_host,
            port=b_port,
            network_id=record.network_id,
            epoch=record.epoch,
            credential=_member_credential(server_a.root, record.network_id, record.epoch),
        ) as peer:
            assert peer.parked(), "the listener never answered the hello"

            panicked = server_b._ctl_panic(  # noqa: SLF001 — the CLI's own control op
                {"network": record.network_id, "reason": "test: the panic's window"}
            )
            assert panicked["rotated"] is False, panicked

            peer.release()
            refused = bool(peer.outcome) and peer.outcome[0].startswith("refused:")
            live = [link.link_id for link in server_b.links.values() if link.alive]
            assert refused and not live, (
                "the panic left a link the relay answered on: the mid-handshake peer "
                f"read {peer.outcome or ['nothing']} and the relay held "
                f"{live or 'no'} live link(s)"
            )
    finally:
        server_b.stop()


def test_a_handshake_in_flight_when_a_peer_panic_arrives_is_refused(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other writer of that state: a panic RECEIVED over a real member link.

    Same park, same hole, a different handler. ``_op_panic`` runs on the link's own
    reader thread, so its trust write and the table it closes have to be ordered by the
    lock the admission reads; otherwise the parked handshake is admitted while the
    handler is busy closing what it snapshotted, and the survivor is served.

    The panic travels as a real ``net_panic`` frame on a real link that was authorised
    the way the wire authorises one (``OP_CAPABILITY`` gives a ``drive`` member
    ``list``), because the frame being a genuine peer frame is part of what the cell is
    about — and a NON-admin frame is the one that rotates nothing, which is what leaves
    the survivor's ops authorised instead of refused by the epoch.
    """
    pair_devices: Devices = request.getfixturevalue("devices")
    server_a, server_b, host, port = pair_devices
    record, _pair_host, _pair_port = _pair(pair_devices, monkeypatch, role="drive")
    server_b.start()
    assert net_fixtures.wait_for(
        lambda: not any(link.alive for link in server_a.links.values())
    ), f"the pairing link never drained: {list(server_a.links)}"

    alarm, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert alarm is not None, reason
    try:
        with _parked_member_handshake(
            server_b,
            host=host,
            port=port,
            network_id=record.network_id,
            epoch=record.epoch,
            credential=_member_credential(server_b.root, record.network_id, record.epoch),
        ) as peer:
            assert peer.parked(), "the listener never answered the hello"

            frame = relay.panic(
                store.load(record.network_id, server_b.root),
                store.require_secrets(record.network_id, server_b.root),
                by=server_b.identity.device_id,
                is_admin=False,
                reason="test: a peer's panic",
                persist=False,
            )
            frame["req"] = 424242
            alarm.send(frame)
            # WAITED ON THE STATE, not on a sleep: the panic is applied or it is not.
            assert net_fixtures.wait_for(
                lambda: store.load(record.network_id, server_a.root).trust != "active"
            ), "the peer's panic never landed on this device"

            peer.release()
            refused = bool(peer.outcome) and peer.outcome[0].startswith("refused:")
            live = [link.link_id for link in server_a.links.values() if link.alive]
            assert refused and not live, (
                "this device went untrusted and still answered a link made after it: "
                f"the mid-handshake peer read {peer.outcome or ['nothing']} and the "
                f"relay held {live or 'no'} live link(s)"
            )
    finally:
        alarm.close("test")
