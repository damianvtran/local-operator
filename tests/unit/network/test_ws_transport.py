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

    server = ws.WebSocketAcceptServer(on_accept)
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
    assert ws.is_ws_endpoint(address) is False


def test_a_claimed_scheme_picks_the_carrier_even_when_the_url_is_malformed() -> None:
    """``carries_ws_scheme`` is the dial path's question, and it must not lie in the
    unsafe direction: a malformed ``ws://`` that answered "not a WebSocket" would be
    re-read as ``host:port`` and dialled as a host named ``ws://user``."""
    assert ws.carries_ws_scheme("ws://user:pass@host:9000") is True
    assert ws.carries_ws_scheme("WSS://host.example") is True
    assert ws.is_ws_endpoint("ws://user:pass@host:9000") is False
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

    server = ws.WebSocketAcceptServer(on_accept)
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

    server = ws.WebSocketAcceptServer(on_accept)
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
    """The self-race guard must read a ws URL as the host:port it names, or a
    poisoned row makes a relay handshake with itself."""
    _dialer, listener, url = loopback_relays
    ws_server = listener._ws_server  # noqa: SLF001 — the bound WS listener
    assert ws_server is not None
    ws_port = ws_server.address[1]
    assert relay._reaches_own_listener(url, ws_port, "127.0.0.1") is True  # noqa: SLF001
    assert relay._reaches_own_listener(url, ws_port, "0.0.0.0") is True  # noqa: SLF001
    # A different port is a different listener, whatever the spelling.
    assert (
        relay._reaches_own_listener(url, listener.settings.port, "127.0.0.1") is False
    )  # noqa: SLF001
    assert (
        relay._reaches_own_listener("ws://10.0.0.5:9000", ws_port, "127.0.0.1") is False
    )  # noqa: SLF001
    record = store.load(store.list_networks(listener.root)[0].network_id, listener.root)
    member = record.member(listener.identity.device_id)
    assert member is not None
    member.endpoints = [url]
    assert relay._dialable_candidates(member, ws_port, "127.0.0.1") == []  # noqa: SLF001


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
