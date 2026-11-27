"""The WebSocket carrier: the same sealed records, riding ``ws://`` instead of TCP.

ENG-4225 Slice 1 (L1a). Two layers are pinned here, and they are pinned
separately on purpose:

* the ADAPTER — ``wstransport.WebSocketPeerSocket`` presents one WebSocket as a
  byte stream, so ``wire.FrameReader`` and ``wire.LinkCrypto`` drive it with no
  change at all. The stream tests below are about BYTES: partial reads, a message
  larger than one ``recv``, a timeout, and the two ways a link ends.
* the RELAY — a real ``RelayServer`` accepts a real peer over a real loopback
  WebSocket listener and answers a real op. That is what proves the handshake,
  the membership check and the record codec never notice the carrier.

Nothing here stubs the socket: a fake stream would test the fake.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from secrets import token_bytes
from typing import Any, Iterator, cast

import pytest

from local_operator.network import relay, store, types, wire
from local_operator.network import wstransport as ws
from tests.unit.network.conftest import wait_for

REPO = Path(__file__).resolve().parents[3]
NETWORK_NAME = "ws-net"


# ---------------------------------------------------------------------------
# Fixtures: two devices already in one network (the pairing ceremony is not this slice)
# ---------------------------------------------------------------------------


def _relay(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(listen_address="127.0.0.1", port=0)
    )


def _write_network(
    server: relay.RelayServer,
    *,
    network_id: str,
    secret: str,
    members: list[tuple[str, str, str, list[str]]],
) -> None:
    """Persist ONE device's view of a network both relays already belong to.

    The pairing ceremony is deliberately absent: this slice changes the CARRIER,
    and driving the SAS human step would test the pairing path instead. Both roots
    therefore hold the same membership and the same epoch secret — the exact state
    two paired devices are in — and every frame below is a real sealed record.
    """
    record = types.NetworkRecord(
        network_id=network_id,
        name=NETWORK_NAME,
        created_by=members[0][0],
        self_device_id=server.identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": server.settings.port, "advertised": []},
    )
    for device_id, public_key, name, endpoints in members:
        relay.admit(
            record,
            device_id=device_id,
            public_key=public_key,
            name=name,
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_by=members[0][0],
            added_via="self",
            endpoints=endpoints,
            root=server.root,
            persist=False,
        )
    store.save(record, server.root)
    store.save_secrets(
        types.SecretState(network_id=network_id, epoch=1, secret=secret), server.root
    )


@pytest.fixture()
def loopback_relays(
    tmp_path: Path,
) -> Iterator[tuple[relay.RelayServer, relay.RelayServer, str]]:
    """Two started relays in one network, the second accepting on a loopback WebSocket.

    Yields ``(dialer, listener, url)``. ``listener`` serves the WebSocket port and
    the ``dialer``'s record gives it that URL as the listener's ONLY candidate — so
    any link that comes up came over the carrier under test.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    server_a = _relay(tmp_path / "a")
    server_b = _relay(tmp_path / "b")
    server_a.start()
    server_b.start()
    host, port = server_a.serve_ws()
    url = f"ws://{host}:{port}"
    network_id = store.new_network_id()
    secret = wire.b64u(token_bytes(32))
    _write_network(
        server_a,
        network_id=network_id,
        secret=secret,
        members=[
            (server_a.identity.device_id, server_a.identity.public_key, "a", []),
            (server_b.identity.device_id, server_b.identity.public_key, "b", []),
        ],
    )
    _write_network(
        server_b,
        network_id=network_id,
        secret=secret,
        members=[
            (server_a.identity.device_id, server_a.identity.public_key, "a", [url]),
            (server_b.identity.device_id, server_b.identity.public_key, "b", []),
        ],
    )
    try:
        yield server_b, server_a, url
    finally:
        server_b.stop()
        server_a.stop()


class _EchoLink:
    """A loopback peer that speaks the record protocol and nothing else."""

    def __init__(self, keys: wire.LinkKeys, *, echo_bytes: int = 0) -> None:
        self.keys = keys
        self.echo_bytes = echo_bytes
        self.received: list[dict[str, Any]] = []
        self.failure: BaseException | None = None
        self.finished = threading.Event()

    def serve(self, sock: Any) -> None:
        codec = wire.LinkCrypto(self.keys, role="listener")
        reader = wire.FrameReader(sock)
        try:
            while True:
                frame = codec.open(reader.read_record_payload(wire.deadline_in(10)))
                self.received.append(frame)
                if frame.get("op") == "bye":
                    # A SMALL close reply, deliberately. A ``sendall`` of a bulky
                    # blob here would block once the socket buffers fill — the
                    # client has stopped reading by then — and the rig would
                    # deadlock on a peer that is only closing.
                    sock.sendall(codec.seal({"bye": True}))
                    return
                if self.echo_bytes:
                    sock.sendall(codec.seal({"blob": wire.b64u(os.urandom(self.echo_bytes))}))
                else:
                    sock.sendall(codec.seal({"echo": frame}))
        except BaseException as exc:  # noqa: BLE001 — reported, never swallowed
            self.failure = exc
        finally:
            sock.close()
            self.finished.set()


def _start_echo_server(
    *, echo_bytes: int = 0
) -> tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys]:
    """A WebSocket listener that runs :class:`_EchoLink` on every arrival."""
    keys = wire.link_keys(os.urandom(32), b"t" * 32, b"l" * 16)
    link = _EchoLink(keys, echo_bytes=echo_bytes)

    def on_accept(sock: Any, _addr: Any) -> None:
        threading.Thread(target=link.serve, args=(sock,), daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=relay.DEFAULT_MAX_HANDSHAKES)
    host, port = server.start()
    assert host == "127.0.0.1", host
    return server, link, keys


@pytest.fixture()
def echo_server() -> Iterator[tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys]]:
    server, link, keys = _start_echo_server()
    try:
        yield server, link, keys
    finally:
        server.stop()


def _dial(server: ws.WebSocketAcceptServer) -> ws.WebSocketPeerSocket:
    return ws.dial_ws(f"ws://127.0.0.1:{server.address[1]}", timeout=5.0)


def _reader(sock: ws.WebSocketPeerSocket) -> wire.FrameReader:
    """``wire.FrameReader`` over the adapter.

    ``FrameReader.__init__`` is annotated for a ``socket.socket`` and this module
    never touches ``wire``; the adapter IS socket-shaped (``recv``/``settimeout``),
    so the cast is the one place that seam is stated rather than sprinkled.
    """
    return wire.FrameReader(cast(Any, sock))


def _read_at_most(sock: ws.WebSocketPeerSocket, count: int, *, chunk: int) -> bytes:
    """``count`` bytes in ``chunk``-sized reads — and a hard stop at end of stream.

    The stop is the point: a read loop that treats ``b""`` as "no bytes YET" spins
    forever when the peer is gone, which is a deadlock in the rig rather than a
    finding about the carrier.
    """
    data = b""
    while len(data) < count:
        piece = sock.recv(min(chunk, count - len(data)))
        assert piece, f"end of stream after {len(data)} of {count} bytes"
        data += piece
    return data


# ---------------------------------------------------------------------------
# Address parsing: what is, and is not, a mesh WebSocket endpoint
# ---------------------------------------------------------------------------


def test_a_wss_url_parses_to_host_port_and_path() -> None:
    parsed = ws.parse_ws_endpoint("wss://run-abc-lop.example/tunnel/peer")
    assert (parsed.scheme, parsed.host, parsed.port) == ("wss", "run-abc-lop.example", 443)
    assert parsed.path == "/tunnel/peer"
    assert parsed.uri == "wss://run-abc-lop.example:443/tunnel/peer"


def test_a_bare_loopback_ws_url_keeps_its_explicit_port() -> None:
    parsed = ws.parse_ws_endpoint("ws://127.0.0.1:9101")
    assert (parsed.host, parsed.port, parsed.path) == ("127.0.0.1", 9101, "")


@pytest.mark.parametrize(
    "address",
    [
        "http://127.0.0.1:9000",
        "file:///etc/passwd",
        "tcp://127.0.0.1:9000",
        "wss://user:pass@host.example",
        "ws://127.0.0.1:notaport",
        "ws://",
        "127.0.0.1:9000",
        "ws://host.example:0",
        "ws://host.example#fragment",
    ],
)
def test_an_address_that_is_not_a_mesh_websocket_endpoint_is_refused(address: str) -> None:
    with pytest.raises(ws.WebSocketEndpointError):
        ws.parse_ws_endpoint(address)


def test_a_claimed_scheme_picks_the_carrier_even_when_the_url_is_malformed() -> None:
    """``carries_ws_scheme`` is the dial path's question, and it must not lie in the
    unsafe direction: a malformed ``ws://`` that answered "not a WebSocket" would be
    re-read as ``host:port`` and dialled as a host named ``ws://user``."""
    assert ws.carries_ws_scheme("ws://user:pass@host:9000") is True
    assert ws.carries_ws_scheme("WSS://host.example") is True
    assert ws.carries_ws_scheme("127.0.0.1:9000") is False
    assert ws.carries_ws_scheme("http://host.example") is False


def test_a_plaintext_dial_to_a_public_host_is_refused() -> None:
    """``ws://`` carries no TLS, so it is a loopback-only hop — and the refusal is an
    ``OSError`` so the dial seam reports it like any other failed connect."""
    with pytest.raises(ws.WebSocketDialError) as refused:
        ws.dial_ws("ws://192.0.2.10:9000", timeout=0.5)
    assert "loopback" in str(refused.value)


# ---------------------------------------------------------------------------
# The stream contract wire.FrameReader depends on
# ---------------------------------------------------------------------------


def test_sealed_records_round_trip_over_a_websocket_socket(
    echo_server: tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys],
) -> None:
    server, link, keys = echo_server
    sock = _dial(server)
    try:
        codec = wire.LinkCrypto(keys, role="dialer")
        reader = _reader(sock)
        sock.sendall(codec.seal({"op": "hello", "req": 1}))
        assert codec.open(reader.read_record_payload(wire.deadline_in(5))) == {
            "echo": {"op": "hello", "req": 1}
        }
        sock.sendall(codec.seal({"op": "bye"}))
        assert wait_for(link.finished.is_set), "the echo peer never finished its loop"
        assert link.failure is None, f"the echo peer failed: {link.failure!r}"
        assert [frame["op"] for frame in link.received] == ["hello", "bye"]
    finally:
        sock.close()


def test_a_small_recv_reassembles_one_large_message() -> None:
    """A WebSocket delivers MESSAGES; ``wire`` reads a STREAM of 64 KiB chunks.

    The adapter's buffer is what reconciles the two, so this reads with a 4 KiB
    ``recv`` and proves the bytes come back in order and complete — the failure it
    catches is a message silently re-cut, or a tail dropped between reads.
    """
    server, link, keys = _start_echo_server(echo_bytes=200_000)
    try:
        sock = _dial(server)
        try:
            codec = wire.LinkCrypto(keys, role="dialer")
            sock.sendall(codec.seal({"op": "blob"}))
            prefix = _read_at_most(sock, wire.LOCAL_PREFIX_LEN, chunk=4)
            length = int.from_bytes(prefix, "big")
            assert length > 199_000, f"expected the bulky reply, got {length} bytes"
            body = _read_at_most(sock, length, chunk=4096)
            # ``open`` verifies the AEAD tag, so a re-cut body cannot pass.
            assert isinstance(codec.open(body)["blob"], str)
            sock.sendall(codec.seal({"op": "bye"}))
            assert wait_for(link.finished.is_set), "the echo peer never finished"
            assert link.failure is None, f"the echo peer failed: {link.failure!r}"
        finally:
            sock.close()
    finally:
        server.stop()


def test_a_three_hundred_kilobyte_record_round_trips_through_frame_reader(
    echo_server: tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys],
) -> None:
    """Over one ``recv`` chunk (64 KiB), so ``FrameReader`` reassembles it across
    many reads — the case where a message-orientated carrier could lose a tail."""
    server, _link, keys = echo_server
    sock = _dial(server)
    try:
        codec = wire.LinkCrypto(keys, role="dialer")
        reader = _reader(sock)
        body = os.urandom(300_000)
        sock.sendall(codec.seal({"op": "blob", "data": wire.b64u(body)}))
        reply = codec.open(reader.read_record_payload(wire.deadline_in(10)))
        assert reply["echo"]["data"] == wire.b64u(body)
        sock.sendall(codec.seal({"op": "bye"}))
    finally:
        sock.close()


def test_a_silent_peer_raises_timeout_and_leaves_the_link_usable(
    echo_server: tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys],
) -> None:
    """A timeout is the caller's own clock expiring, not the link dying — exactly the
    distinction ``FrameReader._fill`` re-raises as ``TimeoutError``."""
    server, _link, keys = echo_server
    sock = _dial(server)
    try:
        sock.settimeout(0.2)
        with pytest.raises(TimeoutError):
            sock.recv(1024)
        codec = wire.LinkCrypto(keys, role="dialer")
        reader = _reader(sock)
        sock.sendall(codec.seal({"op": "hello"}))
        assert codec.open(reader.read_record_payload(wire.deadline_in(5))) == {
            "echo": {"op": "hello"}
        }
        sock.sendall(codec.seal({"op": "bye"}))
    finally:
        sock.close()


def test_a_peer_that_closes_reads_as_end_of_stream_then_makes_wire_raise(
    echo_server: tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys],
) -> None:
    server, link, keys = echo_server
    sock = _dial(server)
    try:
        codec = wire.LinkCrypto(keys, role="dialer")
        reader = _reader(sock)
        sock.sendall(codec.seal({"op": "bye"}))  # the echo peer closes on this
        assert codec.open(reader.read_record_payload(wire.deadline_in(5))) == {"bye": True}
        assert wait_for(link.finished.is_set), "the echo peer never closed"
        sock.settimeout(5.0)
        assert sock.recv(64) == b"", "a closed WebSocket must read as end-of-stream"
        # And ``wire``'s own sentence for that event, not a second vocabulary.
        with pytest.raises(ConnectionError):
            _reader(sock).read_record_payload(wire.deadline_in(5))
    finally:
        sock.close()


def test_a_truncated_record_is_a_connection_error_not_a_hang() -> None:
    """The peer announces 4096 bytes, sends three, and drops: the reader must fail
    rather than wait out its deadline on a record that can never arrive."""
    done = threading.Event()

    def on_accept(sock: Any, _addr: Any) -> None:
        def run() -> None:
            try:
                sock.sendall((4096).to_bytes(wire.LOCAL_PREFIX_LEN, "big") + b"abc")
                time.sleep(0.1)
            finally:
                sock.close()
                done.set()

        threading.Thread(target=run, daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=relay.DEFAULT_MAX_HANDSHAKES)
    server.start()
    try:
        sock = _dial(server)
        try:
            with pytest.raises(ConnectionError):
                _reader(sock).read_record_payload(wire.deadline_in(5))
        finally:
            sock.close()
        assert wait_for(done.is_set), "the truncating peer never finished"
    finally:
        server.stop()


def test_an_over_cap_message_closes_the_link_instead_of_buffering() -> None:
    """The library's read bound is ``wire``'s own record ceiling: a peer that sends
    more must lose the link, not make this process allocate the message."""
    sent = threading.Event()

    def on_accept(sock: Any, _addr: Any) -> None:
        def run() -> None:
            try:
                sock.sendall(b"\x00" * (ws.MAX_MESSAGE_BYTES + 1024))
            finally:
                sent.set()

        threading.Thread(target=run, daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=relay.DEFAULT_MAX_HANDSHAKES)
    server.start()
    try:
        sock = _dial(server)
        try:
            sock.settimeout(10.0)
            assert sock.recv(65536) == b"", "an over-cap message must read as end-of-stream"
        finally:
            sock.close()
        assert wait_for(sent.is_set), "the oversize sender never finished"
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# The relay itself, over the carrier
# ---------------------------------------------------------------------------


def test_two_relays_link_and_answer_an_op_over_a_loopback_websocket(
    loopback_relays: tuple[relay.RelayServer, relay.RelayServer, str],
) -> None:
    """The whole point of L1a: a sealed, mutually-authenticated link and a real
    authorisation path, with every byte inside WebSocket binary frames."""
    dialer, listener, url = loopback_relays
    member = listener.identity.device_id
    link, reason, attempts = dialer._ensure_link_with_probe(  # noqa: SLF001 — the dial path
        member, probe_timeout_s=15.0
    )
    assert link is not None, f"the handshake over {url} failed: {reason} {attempts}"
    try:
        chosen = [row for row in attempts if row.connected]
        assert chosen, f"no candidate connected: {attempts}"
        assert chosen[0].endpoint == url
        assert chosen[0].detail == relay.DETAIL_OK
        reply = link.request({"op": "net_catalog", "req": 41, "locality": "remote"})
        assert reply is not None, "no reply to net_catalog over the WebSocket carrier"
        assert reply["op"] == "ack" and reply["req"] == 41
        assert reply["detail"]["device"]["device_id"] == member
        assert isinstance(reply["detail"]["sessions"], list)
        assert link.stray_replies == 0, "the link answered something nobody asked"
    finally:
        link.close("test")


def test_the_dialer_reports_an_unreachable_websocket_as_a_failed_connect() -> None:
    """A ws candidate no listener answers must read as a connect failure — the
    refusal vocabulary listings and ``doctor`` already speak."""
    dead = _free_port()
    probe = relay.probe_candidates(
        [f"ws://127.0.0.1:{dead}"],
        deadline=time.monotonic() + 10.0,
        connect_cap=5.0,
        wait_all=True,
    )
    assert probe.sock is None, "a closed loopback port must not answer"
    details = [row.detail for row in probe.attempts]
    assert details == ["connect_failed:WebSocketDialError"], details


def test_a_malformed_websocket_candidate_is_a_bad_endpoint() -> None:
    probe = relay.probe_candidates(
        ["ws://user:pass@127.0.0.1:9"],
        deadline=time.monotonic() + 5.0,
        connect_cap=2.0,
        wait_all=True,
    )
    assert probe.sock is None
    assert [row.detail for row in probe.attempts] == [relay.DETAIL_BAD_ENDPOINT]


def test_the_raw_tcp_carrier_is_untouched(
    loopback_relays: tuple[relay.RelayServer, relay.RelayServer, str],
) -> None:
    """Almost every existing mesh link rides ``host:port``; a candidate in that form
    must still be dialled by the socket path, unchanged."""
    _dialer, listener, _url = loopback_relays
    listener_listener = listener._listener  # noqa: SLF001 — the bound TCP listener
    assert listener_listener is not None
    host, port = listener_listener.getsockname()[:2]
    probe = relay.probe_candidates(
        [f"{host}:{port}"], deadline=time.monotonic() + 10.0, connect_cap=5.0
    )
    try:
        assert probe.sock is not None, f"the raw TCP candidate failed: {probe.reason}"
        assert probe.attempts[0].detail == relay.DETAIL_OK
    finally:
        if probe.sock is not None:
            probe.sock.close()


def test_a_websocket_candidate_naming_this_host_is_not_dialled(
    loopback_relays: tuple[relay.RelayServer, relay.RelayServer, str],
) -> None:
    """The self-race guard must read a ws URL as the host:port it names, and it must
    compare it against the port the WS listener BOUND — not against the TCP one, or
    a stored ``ws://127.0.0.1:<our own ws port>`` is dialled and the relay handshakes
    with itself.

    Exercised through the server's own call shape (``_redial_candidates``), because a
    filter that is only correct when a test passes the right port is not the filter
    the relay runs.
    """
    _dialer, listener, url = loopback_relays
    ws_server = listener._ws_server  # noqa: SLF001 — the bound WS listener
    assert ws_server is not None
    ws_port = ws_server.address[1]
    assert ws_port != listener.settings.port, "the rig must bind the two listeners apart"
    assert (
        relay._reaches_own_listener(  # noqa: SLF001
            url, listener.settings.port, "127.0.0.1", ws_port=ws_port
        )
        is True
    )
    assert (
        relay._reaches_own_listener(  # noqa: SLF001
            url, listener.settings.port, "0.0.0.0", ws_port=ws_port
        )
        is True
    )
    # WITHOUT THE WS PORT the filter cannot see this relay's WebSocket listener at
    # all: that blindness IS the defect, so it stays pinned.
    assert (
        relay._reaches_own_listener(url, listener.settings.port, "127.0.0.1")  # noqa: SLF001
        is False
    )
    # A third port is a different listener, whatever the spelling.
    assert (
        relay._reaches_own_listener(  # noqa: SLF001
            url, listener.settings.port, "127.0.0.1", ws_port=ws_port + 1
        )
        is False
    )
    assert (
        relay._reaches_own_listener(  # noqa: SLF001
            "ws://10.0.0.5:9000", listener.settings.port, "127.0.0.1", ws_port=ws_port
        )
        is False
    )
    record = store.load(store.list_networks(listener.root)[0].network_id, listener.root)
    member = record.member(listener.identity.device_id)
    assert member is not None
    member.endpoints = [url]
    assert listener._redial_candidates(member) == []  # noqa: SLF001 — the real call
    assert (
        relay._dialable_candidates(  # noqa: SLF001
            member, listener.settings.port, "127.0.0.1", ws_port=ws_port
        )
        == []
    )


def test_a_socket_option_is_answered_not_faked(
    echo_server: tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys],
) -> None:
    """The adapter has no ``setsockopt``, and the one caller that asks for a socket
    option must read "not set" rather than be told it was applied.

    ``set_pair_keepalive`` is that caller: it catches every option it cannot set
    and returns whether ``SO_KEEPALIVE`` took. A no-op ``setsockopt`` here would
    have it report True for an option no socket received.
    """
    from local_operator.network.handshake import set_pair_keepalive

    server, _link, _keys = echo_server
    sock = _dial(server)
    try:
        assert not hasattr(sock, "setsockopt")
        assert set_pair_keepalive(sock) is False
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# The listener's own limits, and the import rule
# ---------------------------------------------------------------------------


def test_the_listener_refuses_a_non_loopback_bind(tmp_path: Path) -> None:
    (tmp_path / "r").mkdir()
    server = _relay(tmp_path / "r")
    with pytest.raises(ValueError, match="loopback"):
        server.serve_ws(host="0.0.0.0", port=0)


def test_the_listener_can_only_be_started_once(tmp_path: Path) -> None:
    (tmp_path / "r").mkdir()
    server = _relay(tmp_path / "r")
    server.serve_ws()
    try:
        with pytest.raises(RuntimeError, match="once"):
            server.serve_ws()
    finally:
        server.stop()


def test_the_startup_path_does_not_import_the_websocket_stack() -> None:
    """``websockets`` is imported inside the functions that use it: ``network/cli``
    sits on the CLI startup path, so a module-level import would tax every
    ``lop --version``."""
    probe = (
        "import importlib, json, sys;"
        " importlib.import_module('local_operator.network.relay');"
        " print(json.dumps(sorted(sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO)
    )
    assert proc.returncode == 0, f"importing the relay failed:\n{proc.stderr[-3000:]}"
    modules = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    offenders = sorted(name for name in modules if name.split(".", 1)[0] == "websockets")
    assert not offenders, f"websockets is back on the startup path; saw {offenders[:5]}"


def _free_port() -> int:
    """A loopback port nothing is listening on (bound, read, then released)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Round-1 remediation: the carrier's own limits, and what the library does not
# promise across the declared version floor
# ---------------------------------------------------------------------------


class _Redirector:
    """A loopback HTTP server that answers any handshake with ``302 Location: <target>``.

    The bypass it models is the one a check on the ADDRESS cannot see: the endpoint
    dialled is loopback and stays loopback in the URL, while the upgrade the client is
    told to follow lands somewhere else.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self.port = int(self._listener.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, name="ws-redirector", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _addr = self._listener.accept()
            except OSError:
                return  # the listener was closed: this server is done
            with connection:
                connection.settimeout(5.0)
                try:
                    connection.recv(65536)
                    connection.sendall(
                        b"HTTP/1.1 302 Found\r\n"
                        b"Location: " + self.target.encode() + b"\r\n"
                        b"Content-Length: 0\r\n"
                        b"Connection: close\r\n\r\n"
                    )
                except OSError:
                    pass

    def close(self) -> None:
        try:
            self._listener.close()
        except OSError:
            pass


def _stub_connection(method_calls: list[tuple[str, Any]]) -> Any:
    """A connection-shaped object with only the attributes this adapter uses.

    ``wait_closed`` is deliberately ABSENT: ``websockets`` grows it in 17.2 and this
    package declares ``websockets>=15.0.1``, where the attribute does not exist.
    """

    class _Stub:
        remote_address = ("127.0.0.1", 41000)
        local_address = ("127.0.0.1", 41001)

        def recv(self, timeout: Any = None) -> bytes:
            method_calls.append(("recv", timeout))
            return b""

        def send(self, data: bytes) -> None:
            method_calls.append(("send", data))

        def close_socket(self) -> None:
            method_calls.append(("close_socket", None))

    return _Stub()


def test_wait_closed_answers_its_own_event_not_an_optional_library_method() -> None:
    """B1: ``self._conn.wait_closed`` does not exist before ``websockets`` 17.2.

    The declared floor is 15.0.1, so the old implementation was an ``AttributeError``
    on a supported install — raised in the accept handler, the one thread whose
    survival keeps a live link open. The stub has no such attribute at all, which is
    exactly the install the adapter has to work on.
    """
    calls: list[tuple[str, Any]] = []
    connection = _stub_connection(calls)
    assert not hasattr(connection, "wait_closed"), "the stub must model the 15.x API"
    sock = ws.WebSocketPeerSocket(connection)
    assert sock.wait_closed(0.01) is False, "a live adapter must not report itself closed"
    released = threading.Event()

    def wait() -> None:
        sock.wait_closed()  # unbounded, as the accept handler calls it
        released.set()

    threading.Thread(target=wait, daemon=True).start()
    assert not released.is_set(), "nothing has closed the adapter yet"
    sock.close()
    assert released.wait(5.0), "close() must release a waiter parked in wait_closed()"
    assert sock.wait_closed(0.01) is True
    assert calls == [("close_socket", None)], calls


def test_a_loopback_redirect_cannot_move_a_plaintext_dial_off_loopback() -> None:
    """M1: a ``302`` from a loopback server must not become a plaintext dial elsewhere.

    The address check sees ``ws://127.0.0.1:<port>`` — loopback, allowed. The redirect
    target is TEST-NET-1, which is neither this machine nor reachable, so the only
    correct outcome is a refusal: either the library declines to re-dial a socket it
    was handed (17.x) or it does not follow redirects at all (15.x). What must never
    happen is a connection to the target.
    """
    target = f"ws://192.0.2.10:{_free_port()}"
    redirector = _Redirector(target)
    try:
        with pytest.raises(ws.WebSocketDialError) as refused:
            ws.dial_ws(f"ws://127.0.0.1:{redirector.port}", timeout=5.0)
    finally:
        redirector.close()
    text = str(refused.value).lower()
    assert "redirect" in text or "302" in text, f"refused for the wrong reason: {refused.value}"


def test_a_plaintext_hop_whose_connected_peer_is_not_loopback_is_refused() -> None:
    """The address check and the peer check answer different questions, and this is
    the one only the connected socket can answer."""

    def peer(ip: str) -> Any:
        """A connected-socket stand-in: ``getpeername`` is all the check asks."""

        class _Peer:
            def getpeername(self) -> tuple[str, int]:
                return (ip, 9100)

        return cast(Any, _Peer())

    endpoint = ws.parse_ws_endpoint("ws://localhost:9100")
    with pytest.raises(ws.WebSocketDialError, match="not a loopback"):
        ws._refuse_plaintext_off_loopback(endpoint, peer("10.0.0.5"))  # noqa: SLF001
    # Loopback in both families, and a wss endpoint is not this function's business:
    # its trust comes from the verified TLS context, whatever the peer IP is.
    ws._refuse_plaintext_off_loopback(endpoint, peer("127.0.0.1"))  # noqa: SLF001
    ws._refuse_plaintext_off_loopback(endpoint, peer("::1"))  # noqa: SLF001
    ws._refuse_plaintext_off_loopback(  # noqa: SLF001
        ws.parse_ws_endpoint("wss://localhost:9100"), peer("10.0.0.5")
    )


def test_a_plaintext_dial_to_localhost_is_still_admitted() -> None:
    """The new pre-upgrade peer check must not refuse the one plaintext hop the mesh
    actually uses: a loopback listener reached by a loopback NAME resolves to a
    loopback peer, whoever resolves it."""
    server, _link, keys = _start_echo_server()
    try:
        sock = ws.dial_ws(f"ws://localhost:{server.address[1]}", timeout=5.0)
        try:
            codec = wire.LinkCrypto(keys, role="dialer")
            sock.sendall(codec.seal({"op": "hello"}))
            reply = codec.open(_reader(sock).read_record_payload(wire.deadline_in(5)))
            assert reply == {"echo": {"op": "hello"}}
        finally:
            sock.close()
    finally:
        server.stop()


def _one_slot_server(
    *, max_connections: int = 1
) -> tuple[ws.WebSocketAcceptServer, _EchoLink, wire.LinkKeys]:
    """A listener with an explicit, small pre-upgrade cap and an echo peer behind it."""
    keys = wire.link_keys(os.urandom(32), b"t" * 32, b"l" * 16)
    link = _EchoLink(keys)

    def on_accept(sock: Any, _addr: Any) -> None:
        threading.Thread(target=link.serve, args=(sock,), daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=max_connections)
    host, _port = server.start()
    assert host == "127.0.0.1", host
    return server, link, keys


def test_a_connection_beyond_the_pre_upgrade_cap_is_refused_with_503() -> None:
    """M2: the listener holds ``max_connections`` and answers the next upgrade with
    HTTP 503 BEFORE the handler exists — a stranger must not buy threads faster than
    ``_admit_inbound`` can refuse them. The slot is handed back when the connection
    ends, so the cap is a bound and not a leak."""
    server, link, keys = _one_slot_server(max_connections=1)
    try:
        held = _dial(server)
        try:
            codec = wire.LinkCrypto(keys, role="dialer")
            held.sendall(codec.seal({"op": "hello"}))
            reply = codec.open(_reader(held).read_record_payload(wire.deadline_in(5)))
            assert reply == {"echo": {"op": "hello"}}
            with pytest.raises(ws.WebSocketDialError, match="503"):
                ws.dial_ws(f"ws://127.0.0.1:{server.address[1]}", timeout=5.0)
        finally:
            held.close()
        assert wait_for(link.finished.is_set), "the held link never ended"

        def the_slot_came_back() -> bool:
            try:
                recovered = _dial(server)
            except ws.WebSocketDialError:
                return False
            recovered.close()
            return True

        assert wait_for(the_slot_came_back), "the freed slot was never handed back"
    finally:
        server.stop()


def test_a_plain_http_probe_does_not_consume_a_connection_slot() -> None:
    """A refused or non-WebSocket handshake never reaches the handler, so its slot is
    released in ``process_response``. Without that every health probe would shrink the
    cap by one until the listener refused the peers it exists for."""
    server, _link, _keys = _one_slot_server(max_connections=1)
    try:
        for attempt in range(3):
            probe = socket.create_connection(("127.0.0.1", server.address[1]), timeout=5.0)
            with probe:
                probe.settimeout(5.0)
                probe.sendall(b"GET /health HTTP/1.1\r\nHost: probe\r\n\r\n")
                answer = probe.recv(64)
            assert answer.startswith(b"HTTP/1.1 4"), (attempt, answer)
        sock = _dial(server)
        sock.close()
    finally:
        server.stop()


def _frame_at_the_deliverable_ceiling() -> dict[str, Any]:
    """A frame whose sealed record is EXACTLY ``ws.MAX_MESSAGE_BYTES``.

    ``wire``'s reader bounds the LENGTH PREFIX — the ciphertext, GCM tag included —
    by ``MAX_RECORD_BYTES``, so the largest plaintext it will accept back is that
    ceiling minus the tag. The sealed record for it is one byte short of the library
    bound this carrier declares, which is the boundary a library default (1 MiB)
    would have refused and a bound of the plaintext ceiling would have miscounted.
    """
    plaintext = wire.MAX_RECORD_BYTES - ws.AEAD_TAG_BYTES
    padding = plaintext - len(wire.encode_peer_payload({"op": "blob", "data": ""}))
    frame = {"op": "blob", "data": "a" * padding}
    assert len(wire.encode_peer_payload(frame)) == plaintext
    return frame


def test_the_largest_record_the_wire_accepts_round_trips_over_this_carrier() -> None:
    """MINOR 1, MEASURED: what the carrier must not do is refuse a record the wire
    itself delivers.

    The message bound is ``wire``'s own: prefix plus ciphertext, tag included. This
    pins the boundary from both sides — the largest deliverable record round-trips
    byte-identically to a raw TCP socket, and its sealed length is exactly the bound
    the library is given — so the carrier is neither tighter than TCP (a legal record
    refused) nor wider than the wire (bytes buffered that the reader drops).
    """
    assert ws.MAX_MESSAGE_BYTES == wire.MAX_RECORD_BYTES + wire.LOCAL_PREFIX_LEN
    keys = wire.link_keys(os.urandom(32), b"t" * 32, b"l" * 16)
    frame = _frame_at_the_deliverable_ceiling()
    codec = wire.LinkCrypto(keys, role="dialer")
    sealed = codec.seal(frame)
    assert len(sealed) == ws.MAX_MESSAGE_BYTES, (
        "the rig must sit exactly at the carrier's bound, "
        f"got {len(sealed)} of {ws.MAX_MESSAGE_BYTES}"
    )

    # THE SAME BYTES OVER A RAW SOCKET: the record is the wire's, not the carrier's.
    left, right = socket.socketpair()
    # A socket's buffers are far smaller than this record, so the write has to be
    # concurrent with the read: ``sendall`` would block forever on a full buffer.
    writer = threading.Thread(target=left.sendall, args=(sealed,), daemon=True)
    writer.start()
    try:
        over_tcp = wire.LinkCrypto(keys, role="listener").open(
            wire.FrameReader(right).read_record_payload(wire.deadline_in(30))
        )
    finally:
        writer.join(timeout=30)
        left.close()
        right.close()
    assert over_tcp == frame

    received: list[dict[str, Any]] = []
    done = threading.Event()

    def on_accept(sock: Any, _addr: Any) -> None:
        def serve() -> None:
            try:
                received.append(
                    wire.LinkCrypto(keys, role="listener").open(
                        wire.FrameReader(sock).read_record_payload(wire.deadline_in(30))
                    )
                )
                sock.sendall(wire.LinkCrypto(keys, role="listener").seal({"ok": True}))
            finally:
                sock.close()
                done.set()

        threading.Thread(target=serve, daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=relay.DEFAULT_MAX_HANDSHAKES)
    server.start()
    try:
        sock = _dial(server)
        try:
            sock.sendall(sealed)
            reply = codec.open(_reader(sock).read_record_payload(wire.deadline_in(30)))
            assert reply == {"ok": True}
        finally:
            sock.close()
        assert wait_for(done.is_set), "the ceiling-sized record never arrived"
        assert received and received[0] == frame
    finally:
        server.stop()


def test_a_text_frame_is_refused_rather_than_reinterpreted() -> None:
    """Every byte a mesh link sends is binary. A serialiser that receives a text frame
    must lose the link, never decode it as if it were a record."""
    failure: list[BaseException] = []
    done = threading.Event()

    def on_accept(sock: Any, _addr: Any) -> None:
        def serve() -> None:
            try:
                sock.recv(64)
            except BaseException as exc:  # noqa: BLE001 — recorded, then asserted
                failure.append(exc)
            finally:
                sock.close()
                done.set()

        threading.Thread(target=serve, daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=relay.DEFAULT_MAX_HANDSHAKES)
    server.start()
    try:
        from websockets.sync.client import connect

        with connect(f"ws://127.0.0.1:{server.address[1]}", open_timeout=5) as raw:
            raw.send("this is not a mesh record")
        assert wait_for(done.is_set), "the text frame never reached the adapter"
        assert failure and isinstance(failure[0], ConnectionError), failure
        assert "text frame" in str(failure[0])
    finally:
        server.stop()


def test_an_empty_binary_frame_is_not_read_as_end_of_stream() -> None:
    """A zero-length message is legal and carries no bytes. Answering ``b""`` for one
    would report a live link as gone, and ``FrameReader`` would fail a record that is
    still on its way."""
    done = threading.Event()

    def on_accept(sock: Any, _addr: Any) -> None:
        def serve() -> None:
            try:
                sock.sendall(b"")
                sock.sendall(b"payload")
            finally:
                sock.close()
                done.set()

        threading.Thread(target=serve, daemon=True).start()

    server = ws.WebSocketAcceptServer(on_accept, max_connections=relay.DEFAULT_MAX_HANDSHAKES)
    server.start()
    try:
        sock = _dial(server)
        try:
            sock.settimeout(5.0)
            assert sock.recv(64) == b"payload"
        finally:
            sock.close()
        assert wait_for(done.is_set), "the empty-frame peer never finished"
    finally:
        server.stop()


def test_stop_closes_an_open_link_before_the_library_join() -> None:
    """MINOR 2: a handler parks in ``wait_closed`` for the life of its link, and the
    library's shutdown JOINS those handlers. ``stop`` therefore closes the links this
    server opened first, or teardown waits out a shutdown that cannot finish."""
    server, _link, keys = _start_echo_server()
    sock = _dial(server)
    try:
        codec = wire.LinkCrypto(keys, role="dialer")
        sock.sendall(codec.seal({"op": "hello"}))
        assert codec.open(_reader(sock).read_record_payload(wire.deadline_in(5)))["echo"] == {
            "op": "hello"
        }
        started = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started
        assert elapsed < ws.ACCEPT_THREAD_JOIN_S, f"stop took {elapsed:.1f}s"
        sock.settimeout(5.0)
        assert sock.recv(64) == b"", "the stop must close the link it opened"
    finally:
        sock.close()
        server.stop()
