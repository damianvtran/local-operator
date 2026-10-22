"""A lost guest is self-diagnosing and its transport failure is classified precisely.

Two gaps, one per section (see ``lop_osworld_v2_adapter.guest_loss`` for the
incident that motivated both):

* ``execute`` flattened a CONNECT-phase failure (TCP never established, so the
  batch certainly did not apply) and a POST-SEND failure (the guest may already
  have typed the text) into one "outcome is unknown". The matrix below pins the
  verdict for each exception chain ``requests`` can produce, built from REAL
  sockets wherever a socket can produce it, and pins that an ambiguous batch is
  NEVER replayed by counting the requests that actually reach a listening server.
* Nothing diagnostic survived a lost guest, because the instance was terminated
  first. The console tests pin what is persisted, what is scrubbed, and -- the
  property that matters most on a failure path -- that capturing can neither
  raise into the caller nor delay the original error or the termination.

No cloud calls: EC2 is a botocore ``Stubber`` or a hand-written fake, and every
socket is on loopback.
"""

from __future__ import annotations

import base64
import errno
import http.client
import json
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import boto3
import pytest
import requests
import urllib3.exceptions as u3
from botocore.stub import Stubber
from lop_osworld_v2_adapter import guest_loss
from lop_osworld_v2_adapter.observation import NATIVE_SCREEN, write_png_rgb
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.aws import AwsProvider, GuestExecutionError

from tests.unit.evaluation.adapters.osworld.test_aws_provider import (
    _CONTROLLER,
    CREDS,
    EPISODE,
    INSTANCE,
    REGION,
    _BlindEnv,
    _cache_root,
    _FakeEnv,
    _instance,
    _Stubs,
)

# An autouse fixture is not inherited across modules; re-register by name (the
# same discipline ``test_guest_deadline`` documents).
from tests.unit.evaluation.adapters.osworld.test_aws_provider import (  # noqa: F401  isort:skip
    _hermetic_aws,
)

# ---------------------------------------------------------------------------
# Real exception chains
# ---------------------------------------------------------------------------


def _post(url: str, timeout: float) -> Exception:
    """The exception the PRODUCTION guest transport raises for ``url``.

    Uses ``build_clients(...).http_post_json`` itself, not a stand-in, so the
    chain under test is the one ``execute`` really receives.
    """

    transport = aws_mod.build_clients(CREDS, REGION).http_post_json
    try:
        transport(url, {"command": ["x"], "shell": False, "timeout": 1}, timeout)
    except Exception as error:  # noqa: BLE001 - the exception IS the fixture
        return error
    raise AssertionError("the transport did not fail")


_GUEST = "http://203.0.113.9:5000"


def _guest_request(path: str = "/execute") -> requests.PreparedRequest:
    """The request ``requests`` attaches to every connect error it raises."""

    return requests.Request("POST", _GUEST + path, json={}).prepare()


def _closed_port() -> int:
    """A loopback port nothing listens on: connect gets an immediate RST."""

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _OneShotServer:
    """A loopback server that misbehaves in exactly one named way, once.

    ``mode``: ``sleep`` (accept, read the request, then say nothing until the
    client's read timeout fires), ``close`` (accept, read, close with no response bytes -- the
    ``RemoteDisconnected`` shape), ``reset`` (accept, read, RST), ``500``.
    ``received`` records the request body, which is how a test proves the bytes
    DID arrive for a post-send failure.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.received: list[bytes] = []
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            connection, _ = self._server.accept()
        except OSError:
            return
        connection.settimeout(3)
        try:
            data = self._read_one_request(connection)
            self.received.append(data)
            if self.mode == "sleep":
                # Say nothing until the CLIENT gives up (its read timeout
                # closes the socket, which ends this ``recv``): event-driven,
                # so the test costs the client's deadline and not a fixed
                # sleep. The caller's deadline MUST comfortably exceed
                # connect+send on a loaded CI runner, or the client abandons
                # mid-write and the close below turns this into a RESET
                # instead of the read timeout the case means to exercise --
                # that is exactly what a 50ms deadline did on shard 3 of run
                # 38745006530 (ConnectionResetError, not ReadTimeout).
                connection.recv(1)
            elif self.mode == "reset":
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            elif self.mode == "500":
                connection.sendall(b"HTTP/1.1 500 X\r\nContent-Length: 0\r\n\r\n")
        except OSError:
            pass
        finally:
            connection.close()
            self._server.close()

    @staticmethod
    def _read_one_request(connection: socket.socket) -> bytes:
        """Read ONE HTTP request: its headers, then its declared body.

        A single ``recv`` is not a request. TCP may deliver only the request
        HEADERS in the first segment and the body in a later one, so reading
        once and then acting on the mode leaves unread bytes in the buffer --
        the close below then resets the connection, and the ``sleep`` arm sees
        an RST *before* its read deadline instead of the read timeout it exists
        to produce (review round 5 C3; deterministic under a header/body split,
        not merely a load artifact). Draining ``Content-Length`` first makes the
        fixture behave the same however the kernel segments the request.

        ``Content-Length`` is the only framing these tests send (``requests``
        never uses chunked encoding for a JSON body of this size); an absent or
        unparsable header simply ends the read, so a malformed probe cannot
        hang the helper.
        """

        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = connection.recv(65536)
            if not chunk:
                return buffer
            buffer += chunk
        head, _, body = buffer.partition(b"\r\n\r\n")
        declared = 0
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                declared = int(value.strip() or b"0") if value.strip().isdigit() else 0
        while len(body) < declared:
            chunk = connection.recv(65536)
            if not chunk:
                break
            body += chunk
        return head + b"\r\n\r\n" + body

    def close(self) -> None:
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=5)


def test_the_fixture_drains_a_body_that_arrives_after_its_headers() -> None:
    """C3: valid TCP framing may split one request across segments.

    The helper's first ``recv`` can legitimately return only the request
    HEADERS. If its mode behaviour then runs, it consumes the first body byte
    and closes with unread bytes in the buffer, so the client gets an RST
    instead of the mode's intended outcome -- and, for the ``sleep`` arm, an
    RST *before* its read deadline. The fixture must drain the declared
    ``Content-Length`` body first. Asserted from the SERVER side so the claim
    does not depend on client timing.
    """

    server = _OneShotServer("sleep")
    try:
        body = b'{"command": ["x"], "shell": false, "timeout": 1}'
        headers = (
            b"POST /execute HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: keep-alive\r\n\r\n"
        )
        connection = socket.create_connection(("127.0.0.1", server.port))
        connection.settimeout(3)
        try:
            connection.sendall(headers)
            # A real client may send the body in a LATER segment; this is valid
            # framing, not a fault. (A raw socket is used because ``requests``
            # gives no seam to split one logical request this way.)
            time.sleep(0.05)
            connection.sendall(body)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if body in b"".join(server.received):
                    break
                time.sleep(0.02)
        finally:
            connection.close()
        recorded = b"".join(server.received)
        assert body in recorded, f"fixture did not drain the body: {len(recorded)} bytes"
        assert b"POST /execute" in recorded
    finally:
        server.close()


def test_a_refused_connection_is_not_sent_and_names_its_kind() -> None:
    """The 043 shape: ``[Errno 61] Connection refused`` on ``/execute``."""

    port = _closed_port()
    error = _post(f"http://127.0.0.1:{port}/execute", 2.0)
    # Pin the premise: this IS the chain the incident journal recorded.
    assert isinstance(error, requests.ConnectionError)
    assert "NewConnectionError" in repr(error.args[0])

    verdict = guest_loss.classify_transport_failure(error, origin=f"http://127.0.0.1:{port}")

    assert verdict == guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "refused")


@pytest.mark.parametrize(
    ("mode", "client_timeout", "expected_type"),
    [
        ("sleep", 0.5, requests.ReadTimeout),
        ("close", 2.0, requests.ConnectionError),  # RemoteDisconnected
        ("reset", 2.0, requests.ConnectionError),  # ConnectionResetError
        ("500", 2.0, requests.HTTPError),
    ],
)
def test_every_failure_after_the_request_was_written_is_ambiguous(
    mode: str, client_timeout: float, expected_type: type[Exception]
) -> None:
    """Read timeout, dropped connection, reset and 5xx: the guest may have acted.

    Each is produced by a real socket, and ``received`` proves the request BYTES
    reached the server -- which is precisely why none of them may be called
    ``not_sent`` or replayed.
    """

    server = _OneShotServer(mode)
    try:
        error = _post(f"http://127.0.0.1:{server.port}/execute", client_timeout)
    finally:
        server.close()

    assert isinstance(error, expected_type), repr(error)
    assert server.received and b"POST /execute" in server.received[0]
    assert guest_loss.classify_transport_failure(
        error, origin=f"http://127.0.0.1:{server.port}"
    ) == guest_loss.TransportVerdict(guest_loss.DELIVERY_AMBIGUOUS)


def test_a_connect_timeout_is_not_sent() -> None:
    """A SYN that never completes sent no request. Shaped as requests/urllib3 build it."""

    reason = u3.ConnectTimeoutError(
        None, "Connection to 203.0.113.9 timed out. (connect timeout=3)"
    )
    error = requests.ConnectTimeout(
        u3.MaxRetryError(None, "/execute", reason),  # type: ignore[arg-type]
        request=_guest_request(),
    )

    assert guest_loss.classify_transport_failure(
        error, origin=_GUEST
    ) == guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "timeout")


def test_a_name_resolution_failure_is_not_sent() -> None:
    gai = socket.gaierror(8, "nodename nor servname")
    reason = u3.NameResolutionError("guest.invalid", None, gai)  # type: ignore[arg-type]
    error = requests.ConnectionError(
        u3.MaxRetryError(None, "/execute", reason),  # type: ignore[arg-type]
        request=_guest_request(),
    )

    assert guest_loss.classify_transport_failure(
        error, origin=_GUEST
    ) == guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "dns")


@pytest.mark.parametrize(
    "error",
    [
        # A bare builtin carries no proof about WHERE in the exchange it failed.
        ConnectionError("connection refused"),
        ConnectionRefusedError(61, "Connection refused"),
        TimeoutError("timed out"),
        RuntimeError("anything else"),
        # urllib3's post-handshake faults, bare and wrapped.
        requests.ConnectionError(
            u3.ProtocolError("Connection aborted.", ConnectionResetError(54, "reset"))
        ),
        requests.ConnectionError(
            u3.ProtocolError("Connection aborted.", http.client.RemoteDisconnected("closed"))
        ),
        requests.exceptions.ChunkedEncodingError("broken body"),
        requests.exceptions.ContentDecodingError("bad gzip"),
        BrokenPipeError(32, "Broken pipe"),
    ],
)
def test_anything_not_proven_connect_phase_is_ambiguous(error: BaseException) -> None:
    assert (
        guest_loss.classify_transport_failure(error, origin=_GUEST).delivery
        == guest_loss.DELIVERY_AMBIGUOUS
    )


def test_a_connect_failure_that_also_shows_post_send_evidence_stays_ambiguous() -> None:
    """Mixed evidence resolves to the SAFE side.

    ``__context__`` chains are not trustworthy about ordering, so a chain that
    contains both a refused connect and a post-send fault must not be read as
    "nothing was sent".
    """

    refused = _post(f"http://127.0.0.1:{_closed_port()}/execute", 2.0)
    tainted = RuntimeError("outer")
    tainted.__cause__ = refused
    refused.__context__ = u3.ProtocolError("Connection aborted.", ConnectionResetError(54, "reset"))

    assert (
        guest_loss.classify_transport_failure(tainted, origin=_GUEST).delivery
        == guest_loss.DELIVERY_AMBIGUOUS
    )


def test_a_cyclic_exception_chain_terminates() -> None:
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__context__, b.__context__ = b, a

    assert (
        guest_loss.classify_transport_failure(a, origin=_GUEST).delivery
        == guest_loss.DELIVERY_AMBIGUOUS
    )


# ---------------------------------------------------------------------------
# Review round 1, C1: a redirect hides a DELIVERED request behind a connect error
# ---------------------------------------------------------------------------


class _RedirectServer:
    """A loopback "guest" that READS the whole POST, then answers a redirect.

    ``posts`` counts the requests this first server received: the fact the
    classifier must not lose when the redirect TARGET is what refuses.
    """

    def __init__(self, target: str, code: int = 307) -> None:
        posts: list[bytes] = []
        self.posts = posts

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                posts.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(code)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_port
        self.origin = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def test_a_redirect_followed_into_a_refusal_is_not_called_not_sent() -> None:
    """The reviewer's repro, with the redirect followed (``requests`` default).

    The first server received the full command; the exception that escapes
    names only the refused TARGET. The classifier must not read that as "never
    connected" -- it is told which origin it meant to talk to and refuses.
    """

    dead = _closed_port()
    server = _RedirectServer(f"http://127.0.0.1:{dead}/execute")
    try:
        try:
            requests.post(server.origin + "/execute", json={"command": ["x"]}, timeout=2)
        except requests.ConnectionError as error:
            raised = error
        else:  # pragma: no cover - the premise of the test
            raise AssertionError("the redirect was not followed into a refusal")
    finally:
        server.close()

    assert len(server.posts) == 1 and b'"command"' in server.posts[0]  # DELIVERED
    # Premise: the exception really does look like a plain refusal.
    assert "NewConnectionError" in repr(raised.args[0])
    assert guest_loss.classify_transport_failure(
        raised, origin=server.origin
    ) == guest_loss.TransportVerdict(guest_loss.DELIVERY_AMBIGUOUS)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_the_production_guest_post_does_not_follow_a_redirect(code: int) -> None:
    """Contract: a guest command POST is sent once and a 3xx is an error, never a hop."""

    dead = _closed_port()
    server = _RedirectServer(f"http://127.0.0.1:{dead}/execute", code)
    try:
        error = _post(server.origin + "/execute", 2.0)
    finally:
        server.close()

    assert len(server.posts) == 1
    assert isinstance(error, requests.HTTPError), repr(error)
    assert error.response is not None and error.response.status_code == code
    assert guest_loss.classify_transport_failure(
        error, origin=server.origin
    ) == guest_loss.TransportVerdict(guest_loss.DELIVERY_AMBIGUOUS)


@pytest.mark.asyncio
async def test_execute_against_a_redirecting_guest_is_ambiguous_and_posts_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end through ``AwsProvider.execute``: one delivered POST, verdict ambiguous."""

    dead = _closed_port()
    server = _RedirectServer(f"http://127.0.0.1:{dead}/execute")
    monkeypatch.setattr(aws_mod, "GUEST_PORT", server.port)
    try:
        with _Stubs() as stubs:
            stubs.clients.ec2 = _NoConsoleEc2(output="console")
            provider = _provider_over(stubs, tmp_path, post=_CountingTransport())
            with pytest.raises(GuestExecutionError) as raised:
                await provider.execute(["x()"])
    finally:
        server.close()

    assert len(server.posts) == 1
    assert raised.value.delivery == guest_loss.DELIVERY_AMBIGUOUS
    assert "was not sent" not in str(raised.value)
    assert "delivery=ambiguous" in str(raised.value)


@pytest.mark.parametrize("evidence", ["response", "history"])
@pytest.mark.parametrize("nested", [False, True])
def test_a_connect_failure_with_response_evidence_is_ambiguous(evidence: str, nested: bool) -> None:
    reason = u3.NewConnectionError(None, "refused")  # type: ignore[arg-type]
    refused = requests.ConnectionError(
        u3.MaxRetryError(None, "/execute", reason),  # type: ignore[arg-type]
        request=_guest_request(),
    )
    # A negative control ensures this guard, rather than the origin check,
    # changes the verdict. Response truthiness is deliberately FALSE (503).
    assert guest_loss.classify_transport_failure(refused, origin=_GUEST).delivery == "not_sent"
    response = requests.Response()
    response.status_code = 503
    carrier = reason if nested else refused
    setattr(carrier, evidence, response if evidence == "response" else [response])

    assert (
        guest_loss.classify_transport_failure(refused, origin=_GUEST).delivery
        == guest_loss.DELIVERY_AMBIGUOUS
    )


def test_a_connect_failure_with_no_request_evidence_is_ambiguous() -> None:
    """Fail closed: not_sent needs POSITIVE evidence of which request failed."""

    reason = u3.NewConnectionError(None, "refused")  # type: ignore[arg-type]
    bare = requests.ConnectionError(
        u3.MaxRetryError(None, "/execute", reason)  # type: ignore[arg-type]
    )

    assert (
        guest_loss.classify_transport_failure(bare, origin=_GUEST).delivery
        == guest_loss.DELIVERY_AMBIGUOUS
    )


@pytest.mark.parametrize(
    "failing_url",
    [
        "http://203.0.113.9:50001/execute",  # a string prefix of the origin is not the origin
        "http://203.0.113.10:5000/execute",
        "https://203.0.113.9:5000/execute",
    ],
)
def test_a_connect_failure_on_another_origin_is_ambiguous(failing_url: str) -> None:
    reason = u3.NewConnectionError(None, "refused")  # type: ignore[arg-type]
    request = requests.Request("POST", failing_url, json={}).prepare()
    error = requests.ConnectionError(
        u3.MaxRetryError(None, "/execute", reason),  # type: ignore[arg-type]
        request=request,
    )

    assert (
        guest_loss.classify_transport_failure(error, origin=_GUEST).delivery
        == guest_loss.DELIVERY_AMBIGUOUS
    )


# ---------------------------------------------------------------------------
# execute: receipt fields, no replay, no leak
# ---------------------------------------------------------------------------


def _provider_over(stubs: _Stubs, tmp_path: Path, *, post: Any = None) -> AwsProvider:
    """An AwsProvider wired to a fake env and (optionally) the REAL HTTP transport."""

    if post is not None:
        stubs.clients.http_post_json = post
    provider = AwsProvider(
        CREDS,
        region=REGION,
        lease_ref="lop-ttl-x",
        clients=stubs.clients,
        sleep=stubs.sleep,
        # The stub clock, so the retry/recovery BUDGETS are exercised without
        # spending real seconds.
        monotonic=stubs.monotonic,
    )
    env = _FakeEnv()
    env.pkgs_prefix = "{command}"
    provider._env = env
    provider._public_ip = "127.0.0.1"
    provider._instance_id = INSTANCE
    provider._cache_root = _cache_root(tmp_path)
    return provider


class _CountingTransport:
    """The production ``http_post_json`` with a call counter in front of it."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = aws_mod.build_clients(CREDS, REGION).http_post_json

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls += 1
        return self._real(url, payload, timeout)


@pytest.mark.asyncio
async def test_a_refused_guest_is_reported_not_sent_and_the_batch_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 043 case end to end on the provider: bounded attempts, a ``not_sent`` receipt.

    A closed port whose readiness probe also fails is the case the retry bound
    exists for: the probe rides the failed path, so it is evidence and not a
    veto, and four attempts (~3.5 s) is the whole intervention before the
    episode fails exactly as it did before the policy existed.
    """

    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    transport = _CountingTransport()
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=transport)
        provider._clients.http_get = aws_mod.build_clients(CREDS, REGION).http_get
        stubs.ec2 = _NoConsoleEc2()
        provider._clients.ec2 = stubs.ec2
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()", "second()"])

    assert transport.calls == aws_mod.TRANSPORT_RETRY_ATTEMPTS
    assert "guest down: readiness check failed" in str(raised.value)
    assert len(stubs.slept) == aws_mod.TRANSPORT_RETRY_ATTEMPTS - 1, "no settle on a failed batch"
    assert all(0 < delay <= 2.5 for delay in stubs.slept), stubs.slept
    error = raised.value
    assert error.delivery == guest_loss.DELIVERY_NOT_SENT
    message = str(error)
    assert "delivery=not_sent" in message
    assert "connection refused" in message
    assert "no input of this batch was applied" in message
    assert "transport retries exhausted (4)" in message
    assert "batch not retried" in message
    # Value-free, exactly as before: no address, port, URL, command or class name.
    for leaked in ("127.0.0.1", "first()", "second()", "HTTPConnectionPool", "Errno", "/execute"):
        assert leaked not in message


@pytest.mark.asyncio
async def test_a_refusal_after_earlier_actions_says_the_batch_is_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``not_sent`` describes ONE request. Earlier statements of the batch applied."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    transport = _CountingTransport()
    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path, post=transport)
            provider._clients.ec2 = _NoConsoleEc2()
            await provider.execute(["ok()"])
            # The guest dies between statements: the next connect is refused.
            server.shutdown()
            server.server_close()
            with pytest.raises(GuestExecutionError) as raised:
                await provider.execute(["first()", "second()"])
    finally:
        thread.join(timeout=5)

    message = str(raised.value)
    assert raised.value.delivery == guest_loss.DELIVERY_NOT_SENT
    assert "guest action 1 was not sent" in message
    # statement 1 of the SECOND batch failed to connect: nothing of it applied.
    assert "no input of this batch was applied" in message

    # Now the mid-batch case, via the pure function: a refusal at index 2.
    text = guest_loss.describe_failed_action(
        2, guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "refused")
    )
    assert "guest action 3 was not sent" in text
    assert "2 earlier action(s) of this batch applied, input may be partial" in text


class _OkHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"returncode": 0, "output": "", "error": ""}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "client_timeout"),
    [("sleep", 0.5), ("close", 2.0), ("reset", 2.0), ("500", 2.0)],
)
async def test_an_ambiguous_failure_is_never_replayed(
    mode: str, client_timeout: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fail-closed half. The server counts what ARRIVES, not what we meant to send.

    One listener that serves one connection and then closes: a replay would hit
    a closed port, so the counter that matters is the transport's call count,
    and the server's own ``received`` proves the first request got there.
    """

    server = _OneShotServer(mode)
    monkeypatch.setattr(aws_mod, "GUEST_PORT", server.port)
    monkeypatch.setattr(aws_mod, "GUEST_COMMAND_TIMEOUT_S", client_timeout)
    # ``execute`` binds the timeout at import in the provider module namespace.
    transport = _CountingTransport()
    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path, post=transport)
            provider._clients.ec2 = _NoConsoleEc2()
            with pytest.raises(GuestExecutionError) as raised:
                await provider.execute(["first()", "second()"])
    finally:
        server.close()

    assert transport.calls == 1, "an ambiguous batch must be attempted exactly once"
    assert stubs.http_calls == [], "ambiguous delivery must not enter readiness/retry recovery"
    assert stubs.slept == []
    assert len(server.received) == 1
    assert raised.value.delivery == guest_loss.DELIVERY_AMBIGUOUS
    message = str(raised.value)
    assert "outcome is unknown; batch not retried" in message
    assert "delivery=ambiguous" in message
    assert "not sent" not in message


@pytest.mark.asyncio
async def test_refused_connect_recovers_with_one_delivered_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A real refused socket, then real readiness and POST; no paid guest needed."""

    port = _closed_port()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    received: list[dict[str, Any]] = []

    class Handler(_OkHandler):
        def do_GET(self) -> None:
            assert self.path == aws_mod._READY_PATH
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"returncode": 0, "output": "", "error": ""}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    transport = _CountingTransport()
    real_get = aws_mod.build_clients(CREDS, REGION).http_get
    probes: list[str] = []

    def recovering_get(url: str, timeout: float) -> int:
        nonlocal server, thread
        probes.append(url)
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return real_get(url, timeout)

    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path, post=transport)
            provider._clients.http_get = recovering_get
            await provider.execute(["first()"])
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    assert transport.calls == 2
    assert len(received) == 1
    assert len(probes) == 1
    assert len(stubs.slept) == 2, "one retry delay, then the batch's settle"
    assert 0 < stubs.slept[0] <= 2.5
    assert stubs.slept[1] == provider._action_delay_s
    assert (
        "attempt=2 delivery=not_sent error_class=ConnectionError readiness_passed=True"
        in caplog.text
    )
    assert provider._console_stages_done == set()


@pytest.mark.asyncio
async def test_a_refused_guest_exhausts_only_the_current_statement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An answering readiness endpoint cannot authorize more than the bound."""

    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    transport = _CountingTransport()
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=transport)
        provider._clients.ec2 = _NoConsoleEc2()
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()", "second()"])

    assert transport.calls == aws_mod.TRANSPORT_RETRY_ATTEMPTS
    assert len(stubs.http_calls) == 1
    assert stubs.http_timeouts == [aws_mod._EXECUTE_READINESS_TIMEOUT_S]
    assert len(stubs.slept) == aws_mod.TRANSPORT_RETRY_ATTEMPTS - 1
    assert raised.value.delivery == guest_loss.DELIVERY_NOT_SENT
    assert str(raised.value) == (
        guest_loss.describe_failed_action(
            0, guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "refused")
        )
        + "; transport retries exhausted (4)"
    )
    assert (
        "attempt=2 delivery=not_sent error_class=ConnectionError readiness_passed=True"
        in caplog.text
    )
    assert (
        "attempt=3 delivery=not_sent error_class=ConnectionError readiness_passed=True"
        in caplog.text
    )
    assert (
        "attempt=4 delivery=not_sent error_class=ConnectionError readiness_passed=True"
        in caplog.text
    )


# ---------------------------------------------------------------------------
# The pre-delivery retry: classifier-gated, bounded, and visible
# ---------------------------------------------------------------------------


def _no_route(origin: str = _GUEST, path: str = "/execute") -> requests.ConnectionError:
    """The chain ``requests`` produces when the host has no route to the guest.

    ``EHOSTUNREACH`` is the cause the recorded production message -- "connection
    could not be established, delivery=not_sent" -- names, and no other cause
    produces that wording: a closed port says *connection refused*, a blackholed
    SYN says *connect timed out*, and a dead acceptor behind a live listener
    connects and then reports an unknown outcome. Built from the same urllib3
    shape the production transport emits (``NewConnectionError`` under
    ``MaxRetryError`` under ``requests.ConnectionError``), carrying the
    ``PreparedRequest`` the classifier requires as proof of the target.
    """

    reason = u3.NewConnectionError(
        None,  # pyright: ignore[reportArgumentType]
        "Failed to establish a new connection",
    )
    # urllib3 raises that ``from`` the OS error, so the fault is a ``__cause__``
    # link (measured: a real refused loopback post walks
    # ConnectionError -> MaxRetryError -> NewConnectionError -> ConnectionRefusedError).
    # Dropping the link would lose the errno the evidence exists to record.
    reason.__cause__ = OSError(errno.EHOSTUNREACH, "No route to host")
    return requests.ConnectionError(
        u3.MaxRetryError(None, path, reason),  # type: ignore[arg-type]
        request=requests.Request("POST", origin + path, json={}).prepare(),
    )


def test_a_failure_record_names_the_element_and_never_the_text() -> None:
    """Evidence is the closed vocabulary plus an errno: no exception text at all."""

    record = guest_loss.failure_record(_no_route(_GUEST), origin=_GUEST)
    assert record == {
        "delivery": guest_loss.DELIVERY_NOT_SENT,
        "kind": "connect_failed",
        "errno": errno.EHOSTUNREACH,
        "error_class": "ConnectionError",
    }
    assert "No route to host" not in json.dumps(record)


@pytest.mark.asyncio
async def test_an_unreachable_path_recovers_when_the_readiness_probe_also_fails(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The production failure: EHOSTUNREACH on the host->guest hop, twice.

    The readiness probe is a GET over the SAME path, so a path blip fails it too.
    Gating the retry on that probe -- what this code did before -- turned a
    two-second route failure into a dead episode, which is the failure this test
    exists to prevent.
    """

    origin = f"http://127.0.0.1:{aws_mod.GUEST_PORT}"
    attempts: list[str] = []
    remaining = {"failures": 2}

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        attempts.append("".join(payload["command"]))
        if remaining["failures"]:
            remaining["failures"] -= 1
            raise _no_route(origin)
        return {"returncode": 0, "output": "", "error": ""}

    def probe(url: str, timeout: float) -> int:
        # A probe that CANNOT reach the guest: the path is down for it as well.
        raise _no_route(origin, aws_mod._READY_PATH)

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=post)
        provider._clients.http_get = probe
        await provider.execute(["first()"])

    assert len(attempts) == 3, "two provably-unsent failures, then the delivery"
    assert provider.transport_retry_summary() == {
        "retries": 2,
        "recovered": 1,
        "causes": ["connect_failed"],
    }
    assert len(stubs.slept) == 3, "two retry delays, then the batch's settle"
    assert provider._cache_root is not None
    ledger = json.loads((provider._cache_root / "guest-transport-retries.json").read_text())
    assert [event["attempt"] for event in ledger["events"]] == [2, 3]
    assert [event["cause"] for event in ledger["events"]] == ["connect_failed"] * 2
    assert [event["delivery"] for event in ledger["events"]] == ["not_sent"] * 2
    assert all(event["delay_ms"] > 0 for event in ledger["events"])
    # The probe is taken ONCE per statement and stamped with the attempt it was
    # measured on, so a later event's copy is not read as a fresh measurement.
    assert [event["readiness"]["measured_at_attempt"] for event in ledger["events"]] == [1, 1]
    assert all(event["readiness"]["passed"] is False for event in ledger["events"])
    assert all(event["readiness"]["kind"] == "connect_failed" for event in ledger["events"])
    # ``connect_failed`` alone cannot tell EHOSTUNREACH from ENETUNREACH.
    assert all(event["errno"] == errno.EHOSTUNREACH for event in ledger["events"])
    assert ledger["recovered"] == 1
    assert "attempt=2 delivery=not_sent" in caplog.text
    assert "cause=connect_failed" in caplog.text
    assert "readiness_passed=False" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", ["synthetic", "refused"])
async def test_a_recovered_retry_applies_the_action_exactly_once(
    first_failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A server that counts ARRIVALS: a pre-delivery failure must not double it.

    The first attempt dies before a byte leaves this host and the second is a
    real POST to a real listener; the server's own counter is the proof that one
    command ran. Two arms, because the exactly-once claim must not rest on a
    hand-built exception alone: ``synthetic`` raises the EHOSTUNREACH chain up
    front, and ``refused`` refuses a REAL socket (nothing listening) and opens
    the listener from the readiness probe -- the same construction the recorded
    failures had, where the path came back between the failure and the retry.
    """

    received: list[list[str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload["command"])
            body = json.dumps({"returncode": 0, "output": "", "error": ""}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    if first_failure == "refused":
        # Nothing is listening yet: the first attempt is a real ECONNREFUSED.
        port = _closed_port()
        server: ThreadingHTTPServer | None = None
        thread: threading.Thread | None = None
    else:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    origin = f"http://127.0.0.1:{port}"

    real_post = aws_mod.build_clients(CREDS, REGION).http_post_json
    real_get = aws_mod.build_clients(CREDS, REGION).http_get
    left = {"failures": 1}

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if left["failures"]:
            left["failures"] -= 1
            if first_failure == "refused":
                return real_post(url, payload, timeout)
            raise _no_route(origin)
        return real_post(url, payload, timeout)

    def probe(url: str, timeout: float) -> int:
        nonlocal server, thread
        if first_failure == "refused" and server is None:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
        return real_get(url, timeout)

    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path, post=post)
            provider._clients.http_get = probe
            await provider.execute(["first()"])
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    assert len(received) == 1, "the action must be applied exactly once"
    applied = base64.b64decode("".join(received[0][3:])).decode("utf-8")
    assert "first()" in applied
    assert provider.transport_retry_summary() == {
        "retries": 1,
        "recovered": 1,
        "causes": ["refused" if first_failure == "refused" else "connect_failed"],
    }


@pytest.mark.asyncio
async def test_a_failure_after_the_send_is_never_retried_even_when_a_probe_answers(
    tmp_path: Path,
) -> None:
    """The fail-closed half of the policy, with a readiness probe that says 200.

    The request reached the server (the handler ran), so the outcome is unknown
    and a retry could apply it twice. The probe answering is irrelevant: the
    classifier's verdict, not reachability, decides. Call count is 1.
    """

    applied: list[str] = []

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        applied.append("".join(payload["command"]))
        raise requests.ReadTimeout("response lost after the request was written")

    with _Stubs(http_codes=[200]) as stubs:
        provider = _provider_over(stubs, tmp_path, post=post)
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()"])

    assert len(applied) == 1, "an ambiguous delivery must be attempted exactly once"
    assert stubs.http_calls == [], "an ambiguous delivery must not even probe"
    assert stubs.slept == []
    assert provider.transport_retry_summary() == {"retries": 0, "recovered": 0, "causes": []}
    assert provider._cache_root is not None
    assert not (provider._cache_root / "guest-transport-retries.json").exists()
    assert raised.value.delivery == guest_loss.DELIVERY_AMBIGUOUS
    message = str(raised.value)
    assert "outcome is unknown; batch not retried" in message
    assert "delivery=ambiguous" in message


@pytest.mark.asyncio
async def test_the_retry_bound_is_spent_then_the_episode_fails_as_today(
    tmp_path: Path,
) -> None:
    """Exhaustion is bounded, monotone, and ends in the error the harness knows."""

    origin = f"http://127.0.0.1:{aws_mod.GUEST_PORT}"
    attempts: list[str] = []

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        attempts.append(url)
        raise _no_route(origin)

    def probe(url: str, timeout: float) -> int:
        raise _no_route(origin, aws_mod._READY_PATH)

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=post)
        provider._clients.http_get = probe
        provider._clients.ec2 = stubs.ec2 = _NoConsoleEc2()
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()", "second()"])

    assert len(attempts) == aws_mod.TRANSPORT_RETRY_ATTEMPTS
    delays = stubs.slept
    assert len(delays) == aws_mod.TRANSPORT_RETRY_ATTEMPTS - 1
    assert delays == sorted(delays), "backoff must grow, never shrink"
    assert sum(delays) < 5.0, "the whole intervention is bounded to a few seconds"
    error = raised.value
    assert error.delivery == guest_loss.DELIVERY_NOT_SENT
    for expected in (
        "connection could not be established",
        "no input of this batch was applied",
        "delivery=not_sent",
        "transport retries exhausted (4)",
    ):
        assert expected in str(error), str(error)
    assert provider._cache_root is not None
    ledger = json.loads((provider._cache_root / "guest-transport-retries.json").read_text())
    assert [event["attempt"] for event in ledger["events"]] == [2, 3, 4]
    assert ledger["recovered"] == 0
    assert provider.transport_retry_summary()["retries"] == 3


@pytest.mark.asyncio
async def test_a_pre_action_path_failure_attempts_the_action_instead_of_dying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The recorded sequence, end to end: the receipt probes fail, the action POSTs.

    This is the majority of the recorded ``was not sent`` deaths, and none of
    them ever made a request: the pre-action receipt check probed for ~3 s, gave
    up, and the old branch synthesized a transport verdict for an action it had
    not attempted. Here the probes are REAL refusals over a closed port, the
    wait is spent, the evidence records which path element failed -- and the
    action is then compiled, attempted and delivered exactly once.
    """

    port = _closed_port()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    received: list[list[str]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload["command"])
            body = json.dumps({"returncode": 0, "output": "", "error": ""}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    real_post = aws_mod.build_clients(CREDS, REGION).http_post_json

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        nonlocal server, thread
        # The guest becomes reachable exactly at the action attempt -- the shape
        # the recorded bundle had (a health POST reached the guest 2 s after the
        # recovery record). A probe never sees this listener.
        if server is None:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
        return real_post(url, payload, timeout)

    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path, post=post)
            provider._supervisor_unit = "action-server.service"
            provider._clients.http_get = aws_mod.build_clients(CREDS, REGION).http_get
            await provider.execute(["first()"])
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    assert len(received) == 1, "the action must be attempted, and applied once"
    assert sum(stubs.slept) == (aws_mod._PRE_ACTION_RECOVERY_BUDGET_S + provider._action_delay_s)
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    assert evidence["read_failed"] is True
    assert evidence["path_unreachable"] is True
    outcomes = evidence["probe_outcomes"]
    assert len(outcomes) == 10, "one probe plus the backoff schedule"
    assert all(outcome["kind"] == "refused" for outcome in outcomes)
    assert all(outcome["errno"] == errno.ECONNREFUSED for outcome in outcomes)
    assert provider.transport_retry_summary()["retries"] == 0
    assert "under the transport classifier" in caplog.text


@pytest.mark.asyncio
async def test_the_refusal_message_claims_no_cause_it_cannot_know(
    tmp_path: Path,
) -> None:
    """Pinned: "was not attempted", no delivery verdict, and no false cause.

    This branch is reached by a server that ANSWERED (503 here) as well as by an
    ambiguous probe or a failed restart verification, so naming an unreachable
    control server would be false in every case that gets here -- and a false
    cause in a terminal message is exactly what sent an earlier investigation to
    the wrong layer.

    The health clause is rendered HERE, not swallowed: the EC2 stub answers
    ``describe_instances`` (``_NoConsoleEc2`` does not, which is how the false
    wording stayed green) and the guest POST answers 503, so the branch that
    builds ``guest down: <cause>`` actually runs.
    """

    action_posts: list[str] = []
    response = requests.Response()
    response.status_code = 503
    refusal = requests.HTTPError("guest answered 503", response=response)

    class _RunningEc2(_NoConsoleEc2):
        def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "Reservations": [
                    {
                        "Instances": [
                            {
                                "InstanceId": kwargs["InstanceIds"][0],
                                "State": {"Name": "running"},
                            }
                        ]
                    }
                ]
            }

    def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            decoded = base64.b64decode("".join(payload["command"][3:])).decode("utf-8")
        except Exception:
            decoded = ""
        if decoded:
            action_posts.append(decoded)
        raise refusal

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=post)
        provider._supervisor_unit = "action-server.service"
        provider._clients.http_get = lambda url, timeout: 503
        provider._clients.ec2 = _RunningEc2()
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()"])

    message = str(raised.value)
    assert "guest action 1 was not attempted" in message
    assert "delivery=" not in message, "no delivery happened, so there is no verdict"
    assert raised.value.delivery is None
    assert "unreachable" not in message, "the server answered"
    assert "connection could not be established" not in message
    assert "connection refused" not in message
    assert "supervisor recovery budget exhausted" in message
    # The health clause is the branch this test exists to keep honest.
    assert "guest probe answered, not ready (status=503)" in message, message
    assert action_posts == [], "a reachable-but-unhealthy server must not receive the action"
    assert sum(stubs.slept) == aws_mod._PRE_ACTION_RECOVERY_BUDGET_S
    assert provider._cache_root is not None
    record = json.loads((provider._cache_root / "guest-health-execute.json").read_text())
    assert record["instance_state"] == "running"
    assert record["guest_probe"] == "error", "the probe branch must actually run"
    assert record["cause"] == ("instance running; guest probe answered, not ready (status=503)")


@pytest.mark.asyncio
async def test_the_persisted_cause_names_the_answer_not_an_unreachable_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real 503-answering server: "server unreachable" is a false cause.

    The persisted cause is the line a post-mortem reads before choosing which
    layer to investigate. It used to say "server unreachable" for EVERY
    exhaustion -- including this one, where a real listener answered every probe
    -- and that wording is what sent an earlier investigation at the guest.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.send_response(503)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", server.server_address[1])
    try:
        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path)
            provider._supervisor_unit = "action-server.service"
            provider._clients.http_get = aws_mod.build_clients(CREDS, REGION).http_get
            result = provider._recover_guest_supervisor()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.state == "read_failed"
    assert result.path_unreachable is False, "the server answered"
    assert provider._cache_root is not None
    evidence = json.loads((provider._cache_root / "guest-supervisor-recovery.json").read_text())
    cause = evidence["cause"]
    assert "unreachable" not in cause.lower(), cause
    assert "status=503" in cause, cause
    assert cause.startswith("supervisor recovery budget exhausted"), cause


@pytest.mark.asyncio
async def test_an_attempt_that_spends_the_window_stops_the_retrying(tmp_path: Path) -> None:
    """An attempt-count bound alone allowed ~4 x 90 s per statement.

    QA measured the real cost: with a DROP and the production socket deadline,
    four attempts plus probe and backoff ran ~366 s against base's ~92 s. The
    wall-clock window is what turns a persistent fault back into one attempt's
    cost. The fake clock makes the 90 s instant and the assertion exact.
    """

    sent: list[str] = []

    with _Stubs() as stubs:

        def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            sent.append("".join(payload["command"]))
            stubs.now += aws_mod.GUEST_COMMAND_TIMEOUT_S
            raise _no_route(f"http://127.0.0.1:{aws_mod.GUEST_PORT}")

        provider = _provider_over(stubs, tmp_path, post=post)
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()"])

    assert len(sent) == 1, "a failure that already spent the window must not be retried"
    assert stubs.slept == [], "no retry delay may be slept after the window is spent"
    assert raised.value.delivery == guest_loss.DELIVERY_NOT_SENT
    assert "connection could not be established" in str(raised.value)
    assert "transport retries exhausted" not in str(raised.value)
    assert provider._cache_root is not None
    assert not (provider._cache_root / "guest-transport-retries.json").exists()


@pytest.mark.asyncio
async def test_a_slow_second_attempt_ends_the_schedule_mid_way(tmp_path: Path) -> None:
    """The window is checked per failure, not only against the first attempt."""

    sent: list[str] = []

    with _Stubs() as stubs:

        def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            sent.append("".join(payload["command"]))
            if len(sent) == 2:
                stubs.now += aws_mod.TRANSPORT_RETRY_WINDOW_S + 5.0
            raise _no_route(f"http://127.0.0.1:{aws_mod.GUEST_PORT}")

        provider = _provider_over(stubs, tmp_path, post=post)
        with pytest.raises(GuestExecutionError):
            await provider.execute(["first()"])

    assert len(sent) == 2
    assert len(stubs.slept) == 1, "only the first retry delay fits"
    assert provider._cache_root is not None
    ledger = json.loads((provider._cache_root / "guest-transport-retries.json").read_text())
    assert [event["attempt"] for event in ledger["events"]] == [2]


@pytest.mark.asyncio
async def test_retry_preserves_earlier_progress_and_stops_on_ambiguous_delivery(
    tmp_path: Path,
) -> None:
    """Recovery resumes at the failing statement, never at the batch boundary."""

    origin = "http://127.0.0.1:5000"
    reason = u3.NewConnectionError(None, "refused")  # type: ignore[arg-type]
    refusal = requests.ConnectionError(
        u3.MaxRetryError(None, "/execute", reason),  # type: ignore[arg-type]
        request=requests.Request("POST", origin + "/execute").prepare(),
    )
    for ambiguous in (False, True):
        attempts: list[str] = []
        applied: list[str] = []

        def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
            command = base64.b64decode("".join(payload["command"][3:])).decode("utf-8")
            attempts.append(command)
            if "second()" in command and attempts.count(command) == 1:
                raise refusal
            applied.append(command)
            if ambiguous and "second()" in command:
                raise requests.ReadTimeout("response lost")
            return {"returncode": 0, "output": "", "error": ""}

        with _Stubs() as stubs:
            provider = _provider_over(stubs, tmp_path, post=post)
            provider._clients.ec2 = _NoConsoleEc2()
            if ambiguous:
                with pytest.raises(GuestExecutionError) as raised:
                    await provider.execute(["first()", "second()", "third()"])
                assert raised.value.delivery == guest_loss.DELIVERY_AMBIGUOUS
                assert "outcome is unknown; batch not retried" in str(raised.value)
            else:
                await provider.execute(["first()", "second()", "third()"])
        assert sum("first()" in command for command in applied) == 1
        assert sum("second()" in command for command in applied) == 1
        assert sum("second()" in command for command in attempts) == 2
        assert sum("third()" in command for command in applied) == (0 if ambiguous else 1)
        assert len(stubs.http_calls) == 1


@pytest.mark.asyncio
async def test_a_nonzero_exit_keeps_its_wording_and_has_no_delivery_verdict(
    tmp_path: Path,
) -> None:
    """The guest ANSWERED: that is not a transport failure, so nothing is reclassified."""

    with _Stubs() as stubs:
        stubs.guest_default = {"returncode": 7, "output": "", "error": ""}
        provider = _provider_over(stubs, tmp_path)
        provider._clients.ec2 = _NoConsoleEc2()
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["x()"])

    assert "exit 7" in str(raised.value)
    assert "input may be partial" in str(raised.value)
    assert raised.value.delivery is None
    assert provider._clients.ec2.calls == []  # no console capture for a nonzero exit


# ---------------------------------------------------------------------------
# Console capture
# ---------------------------------------------------------------------------


class _NoConsoleEc2:
    """An EC2 stand-in that records every call and fails any console read.

    Used where the console is incidental: it proves WHICH calls were made while
    never letting a test depend on a stub queue's order.
    """

    def __init__(self, *, output: Any = None, screenshot: Any = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._output = output
        self._screenshot = screenshot

    def get_console_output(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_console_output", kwargs))
        if isinstance(self._output, Exception):
            raise self._output
        if self._output is None:
            return {"Output": ""}
        return {"Output": self._output}

    def get_console_screenshot(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_console_screenshot", kwargs))
        if isinstance(self._screenshot, Exception):
            raise self._screenshot
        if self._screenshot is None:
            return {}
        return {"ImageData": self._screenshot}


def _client_error(code: str) -> Exception:
    from botocore.exceptions import ClientError

    return ClientError(
        {"Error": {"Code": code, "Message": "arn:aws:iam::123456789012:user/x"}}, "Op"
    )


def _jpeg_b64(size: int = 2048) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"J" * size + b"\xff\xd9").decode()


def _read_record(directory: Path, stage: str) -> dict[str, Any]:
    return json.loads((directory / guest_loss.console_filename(stage)).read_text())


def test_a_capture_persists_the_console_tail_and_the_screenshot(tmp_path: Path) -> None:
    ec2 = _NoConsoleEc2(output="boot ok\nStopped Session c1 of User gdm\n", screenshot=_jpeg_b64())

    record = guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="execute-not_sent"
    )

    assert record["status"] == guest_loss.STATUS_CAPTURED
    on_disk = _read_record(tmp_path, guest_loss.STAGE_EXECUTE)
    assert on_disk["stage"] == "execute" and on_disk["trigger"] == "execute-not_sent"
    assert on_disk["instance_id"] == INSTANCE
    assert "Stopped Session c1 of User gdm" in on_disk["console_output"]["text"]
    image = (tmp_path / guest_loss.screenshot_filename(guest_loss.STAGE_EXECUTE)).read_bytes()
    assert (
        image.startswith(b"\xff\xd8\xff") and len(image) == on_disk["console_screenshot"]["bytes"]
    )
    # Strictly read-only: Latest asks for fresh output; WakeUp (a keystroke) is never sent.
    assert ("get_console_output", {"InstanceId": INSTANCE, "Latest": True}) in ec2.calls
    assert ("get_console_screenshot", {"InstanceId": INSTANCE}) in ec2.calls
    # No temp file left behind by the atomic write.
    assert not list(tmp_path.glob("*.tmp"))


def test_the_request_shapes_validate_against_the_real_botocore_service_model(
    tmp_path: Path,
) -> None:
    """A fake can accept a wrong parameter name; botocore's Stubber cannot.

    Two clients, one stubber each, because the two reads run on concurrent
    threads and a single Stubber queue is ORDER-sensitive. Also proves botocore
    decodes ``Output`` from base64 before the capture sees it.
    """

    console = boto3.client(
        "ec2", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y"
    )
    screen = boto3.client(
        "ec2", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y"
    )
    console_stub, screen_stub = Stubber(console), Stubber(screen)
    console_stub.add_response(
        "get_console_output",
        {"InstanceId": INSTANCE, "Output": base64.b64encode(b"kernel line\n").decode()},
        {"InstanceId": INSTANCE, "Latest": True},
    )
    screen_stub.add_response(
        "get_console_screenshot",
        {"InstanceId": INSTANCE, "ImageData": _jpeg_b64(64)},
        {"InstanceId": INSTANCE},
    )

    class _Routed:
        get_console_output = console.get_console_output
        get_console_screenshot = screen.get_console_screenshot

    with console_stub, screen_stub:
        record = guest_loss.capture_guest_console(
            _Routed(), INSTANCE, tmp_path, stage=guest_loss.STAGE_OBSERVE, trigger="t"
        )
        console_stub.assert_no_pending_responses()
        screen_stub.assert_no_pending_responses()

    assert record["console_output"]["text"] == "kernel line\n"
    assert record["console_screenshot"]["status"] == guest_loss.STATUS_CAPTURED


def test_a_missing_permission_is_recorded_and_degrades_quietly(tmp_path: Path) -> None:
    """The documented scoped key predates these calls: ``denied``, not a failure."""

    ec2 = _NoConsoleEc2(
        output=_client_error("UnauthorizedOperation"),
        screenshot=_client_error("UnauthorizedOperation"),
    )

    record = guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t"
    )

    assert record["console_output"] == {"status": "denied", "error_code": "UnauthorizedOperation"}
    assert record["console_screenshot"] == {
        "status": "denied",
        "error_code": "UnauthorizedOperation",
    }
    on_disk = (tmp_path / guest_loss.console_filename(guest_loss.STAGE_EXECUTE)).read_text()
    # The AWS message (an ARN / account id) is never persisted, only the code.
    assert "123456789012" not in on_disk
    assert not (tmp_path / guest_loss.screenshot_filename(guest_loss.STAGE_EXECUTE)).exists()


def test_a_hung_call_is_abandoned_at_the_deadline(tmp_path: Path) -> None:
    release = threading.Event()

    class _Hung(_NoConsoleEc2):
        def get_console_output(self, **kwargs: Any) -> dict[str, Any]:
            release.wait(20)
            return {"Output": "late"}

    ec2 = _Hung(screenshot=_jpeg_b64(16))
    started = time.monotonic()
    try:
        record = guest_loss.capture_guest_console(
            ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t", budget_s=0.4
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 5, f"capture waited {elapsed:.1f}s on a hung call"
    assert record["console_output"] == {"status": "timeout"}
    # The OTHER read still landed: the two share a deadline, not a queue.
    assert record["console_screenshot"]["status"] == guest_loss.STATUS_CAPTURED


def test_oversized_artifacts_are_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(guest_loss, "MAX_CONSOLE_TEXT_CHARS", 100)
    monkeypatch.setattr(guest_loss, "MAX_SCREENSHOT_BYTES", 1000)
    text = "OLD-" * 100 + "FRESHEST-LINE"
    ec2 = _NoConsoleEc2(output=text, screenshot=_jpeg_b64(5000))

    record = guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t"
    )

    kept = record["console_output"]
    assert kept["truncated"] is True and kept["original_chars"] == len(text)
    assert len(kept["text"]) == 100 and kept["text"].endswith("FRESHEST-LINE")  # the TAIL survives
    assert record["console_screenshot"]["status"] == guest_loss.STATUS_OVERSIZE
    assert not (tmp_path / guest_loss.screenshot_filename(guest_loss.STAGE_EXECUTE)).exists()


def test_a_wildly_oversized_screenshot_is_refused_before_it_is_decoded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(guest_loss, "MAX_SCREENSHOT_BYTES", 1000)

    def boom(*_a: Any, **_k: Any) -> bytes:
        raise AssertionError("decoded an over-limit payload")

    monkeypatch.setattr(guest_loss.base64, "b64decode", boom)
    ec2 = _NoConsoleEc2(screenshot="A" * 10_000)

    record = guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t"
    )

    assert record["console_screenshot"]["status"] == guest_loss.STATUS_OVERSIZE


# Synthetic, secret-LOOKING values; none is a real credential.
_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
_SESSION = "IQoJb3JpZ2luX2VjEXAMPLESESSIONTOKENVALUE0123456789"
_CLIENT_PW = "Syn-Th3tic-Cl13nt-pw"
_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAEXAMPLEEXAMPLE\n-----END RSA PRIVATE KEY-----"
)


def _secretful_console() -> str:
    return "\n".join(
        [
            "[    1.2] Started osworld service.",
            f"cloud-init: AWS_ACCESS_KEY_ID={_ACCESS_KEY}",
            f"cloud-init: aws_secret_access_key = {_SECRET_KEY}",
            f"cloud-init: token {_SESSION}",
            f"sudo: password was {_CLIENT_PW}",
            f"echoed b64: {base64.b64encode(_CLIENT_PW.encode()).decode()}",
            "Authorization: Bearer sk-syntheticTOKENvalue0123456789abcdef",
            "db: mongodb+srv://svc:hunter2hunter2@cluster0.example.net/app",
            _PEM,
            "[ 4198.9] reboot: Power down",
        ]
    )


def test_console_text_is_scrubbed_of_known_values_and_credential_shapes(tmp_path: Path) -> None:
    """Console output can carry user-data/cloud-init text. None of it may reach disk."""

    ec2 = _NoConsoleEc2(output=_secretful_console())
    known = (_ACCESS_KEY, _SECRET_KEY, _SESSION, _CLIENT_PW)

    guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t", known_values=known
    )

    persisted = (tmp_path / guest_loss.console_filename(guest_loss.STAGE_EXECUTE)).read_text()
    for secret in (
        _ACCESS_KEY,
        _SECRET_KEY,
        _SESSION,
        _CLIENT_PW,
        base64.b64encode(_CLIENT_PW.encode()).decode(),
        "hunter2hunter2",
        "sk-syntheticTOKENvalue0123456789abcdef",
        "MIIEowIBAAKCAQEAEXAMPLEEXAMPLE",
    ):
        assert secret not in persisted, secret
    # What the diagnosis needs is still there.
    assert "Started osworld service" in persisted
    assert "Power down" in persisted


def test_a_secret_straddling_the_size_bound_is_not_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scrub BEFORE bounding: a clip first would sever a secret across the cut.

    The known value sits across the cut line; clipping first would leave a suffix
    no exact-value match could find.
    """

    monkeypatch.setattr(guest_loss, "MAX_CONSOLE_TEXT_CHARS", 40)
    text = "x" * 30 + _CLIENT_PW + "y" * 30
    ec2 = _NoConsoleEc2(output=text)

    guest_loss.capture_guest_console(
        ec2,
        INSTANCE,
        tmp_path,
        stage=guest_loss.STAGE_EXECUTE,
        trigger="t",
        known_values=(_CLIENT_PW,),
    )

    persisted = (tmp_path / guest_loss.console_filename(guest_loss.STAGE_EXECUTE)).read_text()
    assert _CLIENT_PW not in persisted
    for fragment in (_CLIENT_PW[:6], _CLIENT_PW[-6:]):
        assert fragment not in persisted, fragment


# ---------------------------------------------------------------------------
# Review round 1, C2: a known value inside a LARGER encoded blob
# ---------------------------------------------------------------------------

# Built from parts so no literal "name=value" credential sits in the source.
_KEY = "pass" + "word"
_BLOB_PW = "Zq7-pw-Marker-9x41"


def _persist(
    tmp_path: Path, text: str, known: tuple[str, ...] = (_BLOB_PW,), stage: str = "execute"
) -> tuple[str, dict[str, Any]]:
    guest_loss.capture_guest_console(
        _NoConsoleEc2(output=text),
        INSTANCE,
        tmp_path,
        stage=stage,
        trigger="t",
        known_values=known,
    )
    path = tmp_path / guest_loss.console_filename(stage)
    return path.read_text(), json.loads(path.read_text())


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.mark.parametrize("prefix", ["", "x", "xx", "xxx"])
@pytest.mark.parametrize("suffix", ["", "\n", "t!"])
def test_a_secret_inside_a_short_base64_blob_is_not_persisted_at_any_alignment(
    tmp_path: Path, prefix: str, suffix: str
) -> None:
    """The decode path: SHORT blobs (< the opaque-run floor), all 3 byte alignments."""

    blob = _b64((prefix + "k=" + _BLOB_PW + suffix).encode())
    assert len(blob) < guest_loss._OPAQUE_B64_MIN  # premise: only the decode path can catch it

    persisted, record = _persist(tmp_path, f"user-data blob: {blob}\nnext line")

    assert _BLOB_PW not in persisted and blob not in persisted
    assert "[encoded blob withheld" in record["console_output"]["text"]
    assert "next line" in record["console_output"]["text"]


@pytest.mark.parametrize("prefix", ["", "x", "xx", "xxx"])
def test_a_secret_inside_a_long_base64_blob_is_not_persisted(tmp_path: Path, prefix: str) -> None:
    """The reviewer's shape, and the opaque-run rule: long mixed-case blob."""

    blob = _b64((prefix + _KEY + "=" + _BLOB_PW + "\n").encode())
    persisted, _ = _persist(tmp_path, f"user-data blob: {blob}")

    assert _BLOB_PW not in persisted and blob not in persisted


@pytest.mark.parametrize("width", [20, 64, 76])
def test_a_wrapped_blob_is_not_persisted(tmp_path: Path, width: int) -> None:
    """`base64` wraps at 76 columns; a secret can straddle the break."""

    blob = _b64(("A" * 50 + " " + _KEY + "=" + _BLOB_PW + " " + "B" * 50).encode())
    wrapped = "\n".join(blob[i : i + width] for i in range(0, len(blob), width))
    persisted, _ = _persist(tmp_path, f"hdr\n{wrapped}\n[end]")

    assert _BLOB_PW not in persisted
    assert blob[:30] not in persisted and blob[-30:] not in persisted
    assert "hdr" in persisted and "[end]" in persisted


@pytest.mark.parametrize("width", [8, 10, 12])
def test_a_short_wrapped_blob_keeps_no_decodable_secret_tail(tmp_path: Path, width: int) -> None:
    blob = _b64(("xk=" + _BLOB_PW).encode())
    assert len(blob) < guest_loss._OPAQUE_B64_MIN
    wrapped = "\n".join(blob[i : i + width] for i in range(0, len(blob), width))
    _, record = _persist(tmp_path, wrapped)
    text = record["console_output"]["text"]
    assert "[encoded blob withheld" in text
    assert blob not in "".join(text.split())


def test_a_urlsafe_alphabet_blob_is_not_persisted(tmp_path: Path) -> None:
    raw = ("\xfb\xff" + _KEY + "=" + _BLOB_PW + "?>>").encode("latin-1")
    blob = base64.urlsafe_b64encode(raw).decode()
    assert "-" in blob or "_" in blob  # premise: the URL-safe alphabet is actually exercised
    persisted, _ = _persist(tmp_path, f"blob {blob} end")

    assert _BLOB_PW not in persisted and blob not in persisted


def test_a_percent_encoded_value_in_a_larger_string_is_not_persisted(tmp_path: Path) -> None:
    value = "Sy/n+Th3!tic=Pw"
    from urllib.parse import quote

    encoded = quote(value, safe="")
    query = "GET /cb?next=/home&" + "secret" + "=" + encoded + "&x=1"
    persisted, _ = _persist(tmp_path, query, known=(value,))

    assert value not in persisted and quote(value, safe="") not in persisted


# Review round 2: quote(..., safe="") leaves alphanumerics literal. Encoding
# EVERY byte of a containing object is a different spelling, and the shared
# value scrub cannot establish safety of that opaque object.
def _percent_all_bytes(raw: bytes, case: str = "upper") -> str:
    return "".join(
        f"%{byte:02x}" if case == "lower" or (case == "mixed" and index % 2) else f"%{byte:02X}"
        for index, byte in enumerate(raw)
    )


def test_a_fully_percent_encoded_containing_object_is_withheld(tmp_path: Path) -> None:
    """QA's exact shape: all bytes encoded, including the password's alphanumerics."""

    from urllib.parse import unquote

    opaque = _percent_all_bytes(("xx" + _KEY + "=" + _BLOB_PW + "\n").encode())
    _, record = _persist(tmp_path, "user-data: " + opaque)
    text = record["console_output"].get("text", "")
    # The old literal-only assertion passed while unquote recovered the value.
    assert _BLOB_PW not in text
    assert _BLOB_PW not in unquote(text), "percent decoding recovered the registered value"
    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD


@pytest.mark.parametrize("prefix", ["", "x", "xx"])
@pytest.mark.parametrize("case", ["upper", "lower", "mixed"])
@pytest.mark.parametrize("wrap", [0, 13])
def test_percent_objects_are_withheld_across_case_prefix_and_wrapping(
    tmp_path: Path, prefix: str, case: str, wrap: int
) -> None:
    from urllib.parse import unquote

    opaque = _percent_all_bytes((prefix + _KEY + "=" + _BLOB_PW + "\n").encode(), case)
    if wrap:
        # Deliberately split INSIDE %HH too, not just between encoded bytes.
        opaque = "\n  ".join(opaque[i : i + wrap] for i in range(0, len(opaque), wrap))
    _, record = _persist(tmp_path, "user-data: " + opaque)
    text = record["console_output"].get("text", "")
    assert _BLOB_PW not in unquote(
        "".join(text.split())
    ), "wrapped percent blob remained recoverable"
    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD


def test_percent_blob_withholding_precedes_tail_clipping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(guest_loss, "MAX_CONSOLE_TEXT_CHARS", 9)
    opaque = _percent_all_bytes(("xx" + _KEY + "=" + _BLOB_PW + "\n").encode())
    _, record = _persist(tmp_path, opaque)
    # A clip-first pass leaves <4 %HH bytes, below the opaque-object threshold.
    # The whole input must be refused, not just a credential-free short suffix.
    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD
    assert "text" not in record["console_output"]


@pytest.mark.parametrize("case", ["04x", "04X"])
def test_fully_unicode_escaped_containing_objects_are_withheld(tmp_path: Path, case: str) -> None:
    """Direct neighbour: every ASCII byte can also be spelled as a JSON unicode escape."""

    raw = "xx" + _KEY + "=" + _BLOB_PW + "\n"
    opaque = "".join(chr(92) + "u" + format(ord(char), case) for char in raw)
    _, record = _persist(tmp_path, opaque)
    text = record["console_output"].get("text", "")
    assert _BLOB_PW not in json.loads(
        '"' + text + '"'
    ), "JSON escapes recovered the registered value"
    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD


def test_shell_hex_escaped_containing_objects_keep_no_registered_value(tmp_path: Path) -> None:
    """Inspection control: the existing shared scrub already contains this spelling."""

    import codecs

    raw = "xx" + _KEY + "=" + _BLOB_PW + "\n"
    opaque = "".join(chr(92) + "x" + format(ord(char), "02x") for char in raw)
    _, record = _persist(tmp_path, opaque)
    text = record["console_output"].get("text", "")
    assert _BLOB_PW not in codecs.decode(text, "unicode_escape")


def test_isolated_percent_escapes_do_not_withhold_ordinary_diagnostics(tmp_path: Path) -> None:
    text = "[1] Boot complete: 100%. Next path /a%20b ; progress 50% ; no blob."
    _, record = _persist(tmp_path, text)
    assert record["console_output"]["status"] == guest_loss.STATUS_CAPTURED
    assert record["console_output"]["text"] == text


@pytest.mark.parametrize("case", [str.lower, str.upper])
def test_a_hex_encoded_larger_object_is_not_persisted(tmp_path: Path, case: Any) -> None:
    hexed = case((_KEY + "=" + _BLOB_PW).encode().hex())
    persisted, _ = _persist(tmp_path, f"dump: {hexed} end")

    assert _BLOB_PW not in persisted and hexed not in persisted
    assert _BLOB_PW.encode().hex() not in persisted.lower()


def test_an_opaque_encoded_run_is_withheld_even_if_nothing_decodes_out_of_it(
    tmp_path: Path,
) -> None:
    opaque = _b64(bytes(range(256)))[:80]
    persisted, record = _persist(tmp_path, f"key material: {opaque}")

    assert opaque not in persisted
    assert "[encoded blob withheld" in record["console_output"]["text"]


def test_a_blob_straddling_the_clip_boundary_is_withheld_before_clipping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Withhold whole, THEN clip: a clip first would leave a decodable fragment."""

    monkeypatch.setattr(guest_loss, "MAX_CONSOLE_TEXT_CHARS", 60)
    blob = _b64(("xx" + _KEY + "=" + _BLOB_PW + "\n").encode())
    text = "a" * 50 + blob + "b" * 50
    persisted, _ = _persist(tmp_path, text)

    assert _BLOB_PW not in persisted
    assert blob not in persisted
    assert all(blob[i : i + 12] not in persisted for i in range(0, len(blob) - 12, 4))


def test_ordinary_console_text_survives_the_blob_pass(tmp_path: Path) -> None:
    """The diagnosis must stay readable: units, uuids, versions, dotted names."""

    text = "\n".join(
        [
            "[    5.1] systemd[1]: Started osworld service.",
            "[    5.2] EXT4-fs (nvme0n1p1): mounted filesystem "
            "3f2a9c1e-77b4-4d6a-9b0e-1c2d3e4f5a6b r/w",
            "[ 4198.9] systemd[1]: Stopped Session c1 of User gdm.",
            "NetworkManager-wait-online.service: Deactivated successfully.",
            "/usr/lib/systemd/system-generators/zfs-list-cache",
        ]
    )
    _, record = _persist(tmp_path, text)

    # Long slash-separated paths can look exactly like base64, so the
    # conservative blob policy may withhold that path, not the diagnosis.
    kept = record["console_output"]["text"]
    for line in text.splitlines()[:-1]:
        assert line in kept


def test_text_too_large_to_scan_is_withheld(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(guest_loss, "_MAX_SCAN_CHARS", 100)
    persisted, record = _persist(tmp_path, "z " * 200)

    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD
    assert "z z z" not in persisted


def test_text_that_cannot_be_vouched_for_is_withheld_not_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A scrubber that fails to remove a known value turns into a WITHHELD record."""

    import local_operator.redaction_shapes as shapes

    monkeypatch.setattr(shapes, "scrub_secrets", lambda text, values=(): text)  # a broken scrubber
    ec2 = _NoConsoleEc2(output=f"leak {_CLIENT_PW}")

    record = guest_loss.capture_guest_console(
        ec2,
        INSTANCE,
        tmp_path,
        stage=guest_loss.STAGE_EXECUTE,
        trigger="t",
        known_values=(_CLIENT_PW,),
    )

    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD
    assert _CLIENT_PW not in (tmp_path / guest_loss.console_filename("execute")).read_text()


def test_text_is_withheld_when_the_scrubber_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An older harness without ``redaction_shapes``: record absence, never raw text."""

    import sys

    monkeypatch.setitem(
        sys.modules, "local_operator.redaction_shapes", None
    )  # import -> ImportError
    ec2 = _NoConsoleEc2(output="anything at all")

    record = guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t"
    )

    assert record["console_output"]["status"] == guest_loss.STATUS_WITHHELD
    assert "anything at all" not in (tmp_path / guest_loss.console_filename("execute")).read_text()


def test_empty_and_malformed_responses_are_recorded_as_such(tmp_path: Path) -> None:
    ec2 = _NoConsoleEc2(output="   \n", screenshot="@@not base64@@")

    record = guest_loss.capture_guest_console(
        ec2, INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t"
    )

    assert record["console_output"] == {"status": "empty"}
    assert record["console_screenshot"] == {"status": "error", "error_code": "undecodable"}


def test_capture_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Whatever breaks -- the client, the destination, the arguments -- returns a record."""

    class _Exploding:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError("client exploded")

    results = [
        guest_loss.capture_guest_console(
            _Exploding(), INSTANCE, tmp_path, stage=guest_loss.STAGE_EXECUTE, trigger="t"
        ),
        # destination is a FILE: mkdir/write fail
        guest_loss.capture_guest_console(
            _NoConsoleEc2(output="x"),
            INSTANCE,
            _file(tmp_path),
            stage=guest_loss.STAGE_EXECUTE,
            trigger="t",
        ),
        guest_loss.capture_guest_console(
            _NoConsoleEc2(), INSTANCE, None, stage=guest_loss.STAGE_EXECUTE, trigger="t"
        ),
        guest_loss.capture_guest_console(
            _NoConsoleEc2(), INSTANCE, tmp_path, stage="../escape", trigger="t"
        ),
        guest_loss.capture_guest_console(
            _NoConsoleEc2(),
            None,  # type: ignore[arg-type]
            tmp_path,
            stage=guest_loss.STAGE_EXECUTE,
            trigger="t",
        ),
    ]

    assert [r["status"] for r in results][2:4] == ["no_destination", "error"]
    assert all("status" in r for r in results)
    # An invalid stage must not be turned into a path.
    assert not list(tmp_path.parent.glob("escape*"))


def _file(directory: Path) -> Path:
    blocker = directory / "not-a-directory"
    blocker.write_text("x")
    return blocker


# ---------------------------------------------------------------------------
# Wiring: when it runs, when it must not, and that it never gets in the way
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_transport_failure_captures_the_console_before_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=_CountingTransport())
        provider._clients.ec2 = _NoConsoleEc2(
            output="Stopped Session c1 of User gdm", screenshot=_jpeg_b64()
        )
        with pytest.raises(GuestExecutionError):
            await provider.execute(["x()"])
        cache = _cache_root(tmp_path)

    record = _read_record(cache, guest_loss.STAGE_EXECUTE)
    assert record["trigger"] == "execute-not_sent"
    assert "Stopped Session c1 of User gdm" in record["console_output"]["text"]
    assert (cache / guest_loss.screenshot_filename(guest_loss.STAGE_EXECUTE)).exists()


@pytest.mark.asyncio
async def test_a_hanging_or_exploding_console_never_masks_the_error_or_blocks_termination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The load-bearing property. The original verdict is raised, and teardown still runs.

    ``terminate`` is driven through the SAME provider object, on a Stubber, right
    after the failed ``execute`` -- the order a wedge produces.
    """

    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    monkeypatch.setattr(guest_loss, "CAPTURE_BUDGET_S", 0.3)
    release = threading.Event()

    class _Hostile(_NoConsoleEc2):
        def get_console_output(self, **kwargs: Any) -> dict[str, Any]:
            release.wait(20)
            return {"Output": "late"}

        def get_console_screenshot(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=_CountingTransport())
        hostile = _Hostile()
        # Real EC2 for teardown, hostile console reads on the same object.
        ec2 = stubs.ec2
        hostile.describe_instances = ec2.describe_instances  # type: ignore[attr-defined]
        hostile.terminate_instances = ec2.terminate_instances  # type: ignore[attr-defined]
        provider._clients.ec2 = hostile
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("running"), {"Filters": aws_mod._tag_filters(EPISODE)}
        )
        stubs.ec2_stub.add_response("terminate_instances", {}, {"InstanceIds": [INSTANCE]})
        stubs.ec2_stub.add_response(
            "describe_instances", _instance("terminated"), {"InstanceIds": [INSTANCE]}
        )
        started = time.monotonic()
        try:
            with pytest.raises(GuestExecutionError) as raised:
                await provider.execute(["x()"])
            elapsed = time.monotonic() - started
            code = await provider.terminate(f"lop-ep-{EPISODE}")
        finally:
            release.set()
        stubs.ec2_stub.assert_no_pending_responses()

    assert raised.value.delivery == guest_loss.DELIVERY_NOT_SENT  # the ORIGINAL error
    assert elapsed < 10, f"the failure path waited {elapsed:.1f}s on the console"
    assert code == "instance-terminated"


@pytest.mark.asyncio
async def test_each_stage_is_captured_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    ec2 = _NoConsoleEc2(output="o")
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path, post=_CountingTransport())
        provider._clients.ec2 = ec2
        for _ in range(3):
            with pytest.raises(GuestExecutionError):
                await provider.execute(["x()"])

    assert [name for name, _ in ec2.calls].count("get_console_output") == 1
    assert [name for name, _ in ec2.calls].count("get_console_screenshot") == 1


@pytest.mark.asyncio
async def test_a_healthy_observation_and_a_healthy_execute_make_no_console_call(
    tmp_path: Path,
) -> None:
    """The capture is a lost-guest path only: a healthy episode spends nothing."""

    ec2 = _NoConsoleEc2(output="o")
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._clients.ec2 = ec2
        await provider.execute(["ok()"])
        assert (await provider.observe())["screenshot"] == b"png"

    assert ec2.calls == []


@pytest.mark.asyncio
async def test_a_teardown_only_provider_does_nothing_without_an_instance_or_cache_root() -> None:
    """A rescue worker has no instance id in hand and no cache root: nothing to capture."""

    ec2 = _NoConsoleEc2(output="o")
    with _Stubs() as stubs:
        provider = AwsProvider.for_teardown(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients
        )
        provider._clients.ec2 = ec2
        provider._capture_guest_console(guest_loss.STAGE_EXECUTE, "t")

    assert ec2.calls == []


# ---------------------------------------------------------------------------
# observe: the first frameless read names a refused guest, and nothing else changes
# ---------------------------------------------------------------------------

_REFUSED = (
    "An error occurred while trying to get the screenshot: "
    "HTTPConnectionPool(host='x', port=5000): Max retries exceeded"
)


class _RefusedEnv(_BlindEnv):
    """``get_screenshot`` as the pinned upstream runs it against a refused port.

    The log record's ARGUMENT is the real exception (``logger.error("...: %s",
    e)``), which is what lets the capture classify it without reading its text.
    """

    def __init__(self, error: BaseException) -> None:
        super().__init__([])
        self._error = error

    def _get_obs(self) -> dict[str, Any]:
        import logging

        logger = logging.getLogger(aws_mod._SCREENSHOT_LOGGER)
        for _ in range(3):
            logger.handle(
                logger.makeRecord(
                    aws_mod._SCREENSHOT_LOGGER,
                    logging.ERROR,
                    _CONTROLLER,
                    1,
                    "An error occurred while trying to get the screenshot: %s",
                    (self._error,),
                    None,
                )
            )
        logger.handle(
            logger.makeRecord(
                aws_mod._SCREENSHOT_LOGGER,
                logging.ERROR,
                _CONTROLLER,
                1,
                "Failed to get screenshot.",
                (),
                None,
            )
        )
        return {"screenshot": None, "instruction": "do it"}


@pytest.mark.asyncio
async def test_a_refused_screenshot_read_is_named_in_the_cause_and_captures_the_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port = _closed_port()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    refused = _post(f"http://127.0.0.1:{port}/screenshot", 2.0)
    ec2 = _NoConsoleEc2(output="Stopped Session c1 of User gdm", screenshot=_jpeg_b64(32))
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._clients.ec2 = ec2
        provider._env = _RefusedEnv(refused)
        observed = await provider.observe()

    # The observation's shape is exactly what it was: frameless, plus a cause.
    assert observed["screenshot"] is None
    cause = observed[aws_mod.SCREENSHOT_CAUSE_KEY]
    assert cause.startswith(
        "screenshot unavailable: upstream_failures=4 kinds=transport,transport,transport,exhausted"
    )
    # NEW: the head of the line says nothing is listening -- on the FIRST retry record.
    assert "not_sent=3:refused" in cause
    assert cause.index("not_sent=") < cause.index("first_ms=")
    assert len(cause) <= aws_mod.MAX_CAUSE_RECORDS * 0 + 512
    record = _read_record(_cache_root(tmp_path), guest_loss.STAGE_OBSERVE)
    assert record["trigger"] == "observe-frameless"
    assert "Stopped Session c1 of User gdm" in record["console_output"]["text"]


@pytest.mark.asyncio
async def test_a_slow_screenshot_read_is_not_called_not_sent(tmp_path: Path) -> None:
    """A read timeout means the server WAS listening: the cause must not say otherwise."""

    server = _OneShotServer("sleep")
    try:
        slow = _post(f"http://127.0.0.1:{server.port}/screenshot", 0.5)
    finally:
        server.close()
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._clients.ec2 = _NoConsoleEc2()
        provider._env = _RefusedEnv(slow)
        observed = await provider.observe()

    assert "not_sent" not in observed[aws_mod.SCREENSHOT_CAUSE_KEY]


@pytest.mark.asyncio
async def test_the_exception_text_is_still_never_read_into_the_cause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The value-free guarantee on the cause survives the new field."""

    port = _closed_port()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", port)
    refused = _post(f"http://127.0.0.1:{port}/screenshot", 2.0)
    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        provider._clients.ec2 = _NoConsoleEc2()
        provider._env = _RefusedEnv(refused)
        cause = (await provider.observe())[aws_mod.SCREENSHOT_CAUSE_KEY]

    for leaked in ("127.0.0.1", "Errno", "HTTPConnectionPool", "Connection refused"):
        assert leaked not in cause


# ---------------------------------------------------------------------------
# End to end: the real adapter, the real EpisodeRunner, a guest that refuses
# ---------------------------------------------------------------------------


class _FrameEnv(_FakeEnv):
    """A DesktopEnv stand-in that serves a valid native-size PNG frame."""

    def __init__(self) -> None:
        super().__init__()
        shade = 17
        self._png = write_png_rgb(
            NATIVE_SCREEN.width,
            NATIVE_SCREEN.height,
            bytes((shade, shade * 3 % 256, shade * 7 % 256))
            * (NATIVE_SCREEN.width * NATIVE_SCREEN.height),
        )

    def _get_obs(self) -> dict[str, Any]:
        return {"screenshot": self._png, "instruction": "do it"}


class _RefusingGuestProvider(AwsProvider):
    """The REAL ``AwsProvider.execute``/``observe``, allocated without EC2.

    ``allocate`` is replaced by the minimum it would have left behind (a live
    env, an address, an instance id, the cache root); everything this slice
    changed -- ``execute``'s classification, the console capture -- is the
    production code, over the production HTTP transport, against a port nothing
    listens on.
    """

    async def allocate(self, plan: Any, task: Any, *, cache_root: Path) -> None:
        self._cache_root = cache_root
        self._env = _FrameEnv()
        self._env.pkgs_prefix = "{command}"
        self._public_ip = "127.0.0.1"
        self._instance_id = INSTANCE


@pytest.mark.asyncio
async def test_a_refused_guest_through_the_real_adapter_and_runner_is_recorded_as_not_sent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, episode_id: str
) -> None:
    """The 043 shape through the whole chain: adapter RPC error -> runner -> sealed bundle.

    The actions are compiled by the real adapter, posted by the real transport
    to a refused port, classified, and the diagnostic the RUNNER seals names the
    delivery. The console is captured into the episode cache root BEFORE the
    runner's cleanup terminates anything.
    """

    from lop_osworld_v2_adapter.adapter import OSWorldV2Adapter

    from local_operator.evaluation.runner.episode import EpisodeRunner
    from tests.unit.evaluation.adapters.osworld import fixtures
    from tests.unit.evaluation.adapters.osworld.test_fake_end_to_end import (
        _AdapterSupervisorShim,
        _selector,
        _spec_with_task,
    )
    from tests.unit.evaluation.runner.conftest import ScriptedModel, build_config

    monkeypatch.setattr(aws_mod, "GUEST_PORT", _closed_port())
    ec2 = _NoConsoleEc2(output="Stopped Session c1 of User gdm", screenshot=_jpeg_b64(64))
    workspace = tmp_path / "workspace"
    (workspace / "tasks").mkdir(parents=True)
    (workspace / "tasks" / "task_plain.py").write_text(fixtures.PLAIN)

    def make_provider() -> AwsProvider:
        with _Stubs() as stubs:
            stubs.clients.ec2 = ec2
            # The PRODUCTION transport (module-level ``requests.post``), not the
            # Stubs' scripted one: the refusal must come from a real socket.
            stubs.clients.http_post_json = aws_mod.build_clients(CREDS, REGION).http_post_json
            return _RefusingGuestProvider(
                CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients
            )

    adapter = OSWorldV2Adapter(provider_factory=make_provider, workspace_root=workspace)
    selector = _selector(tmp_path, workspace, adapter)
    runner = EpisodeRunner(
        _spec_with_task(episode_id),
        build_config(tmp_path, observation_retry_delay=0.0),
        selector=selector,
        model=ScriptedModel(["type", "finish"]),
        launch=lambda _selector: _AdapterSupervisorShim(adapter, selector),
    )

    outcome = await runner.run()

    assert outcome.status == "failed", outcome.diagnostic
    assert outcome.diagnostic is not None
    # The receipt the operator reads: it says the batch was NOT sent.
    assert "was not sent (connection refused" in outcome.diagnostic
    assert "delivery=not_sent" in outcome.diagnostic
    assert "no input of this batch was applied" in outcome.diagnostic
    # And the guest left evidence behind, in the episode cache root.
    cache = tmp_path / "osworld-cache"
    records = list(cache.glob("*/guest-console-execute.json"))
    assert len(records) == 1, list(cache.rglob("*"))
    persisted = json.loads(records[0].read_text())
    assert persisted["trigger"] == "execute-not_sent"
    assert "Stopped Session c1 of User gdm" in persisted["console_output"]["text"]
