"""Total listing deadlines cover socket reads hidden inside header parsing.

All network traffic is loopback. The server keeps dripping until the client
closes, so recovery cannot be explained by a test releasing a response gate.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpcore
import httpx
import pytest

from local_operator.model import deadline, deadline_http
from local_operator.model.discovery import fetch_models


@contextmanager
def _listing_server():
    finished = threading.Event()
    seen = []
    body = json.dumps({"data": [{"id": "synthetic", "display_name": "Synthetic"}]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self):
            seen.append(self.path)
            try:
                if "headers" in self.path:
                    # No header terminator ever arrives. A body-iterator check
                    # cannot see this case; the backend must bound every read.
                    self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                    while True:
                        self.connection.sendall(b"x")
                        time.sleep(0.02)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if "body" in self.path:
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        time.sleep(0.02)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", finished, seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_owned_listing_deadline_closes_trickling_socket_and_recovers(phase):
    with _listing_server() as (base, finished, seen):
        assert (
            fetch_models("anthropic", api_key="synthetic", base_url=f"{base}/{phase}", timeout=0.3)
            is None
        )
        assert seen, "the expired operation must have actually reached the loopback peer"
        assert finished.wait(2), "expiry must close the socket, not just abandon a worker"
        # No server gate is released. A healthy next listing must work while
        # the malicious peer would otherwise still be dripping.
        rows = fetch_models("anthropic", api_key="synthetic", base_url=f"{base}/healthy", timeout=2)
        assert rows is not None and len(rows) == 1


def test_deadline_transport_preserves_environment_proxy_routing(monkeypatch):
    with _listing_server() as (base, _, seen):
        monkeypatch.setattr(deadline_http, "getproxies", lambda: {"http": base})
        monkeypatch.setattr(deadline_http, "proxy_bypass", lambda _: False)
        with httpx.Client(
            transport=deadline_http.ListingDeadlineTransport(time.monotonic() + 2)
        ) as client:
            response = client.get("http://synthetic.invalid/models")
        assert response.status_code == 200
        assert seen == ["http://synthetic.invalid/models"]


def test_deadline_transport_preserves_no_proxy_bypass(monkeypatch):
    with _listing_server() as (base, _, seen):
        monkeypatch.setattr(
            deadline_http, "getproxies", lambda: {"http": "http://unreachable.invalid:1"}
        )
        monkeypatch.setattr(deadline_http, "proxy_bypass", lambda _: True)
        with httpx.Client(
            transport=deadline_http.ListingDeadlineTransport(time.monotonic() + 2)
        ) as client:
            assert client.get(f"{base}/models").status_code == 200
        assert seen == ["/models"]


def test_tls_return_stream_keeps_the_same_absolute_deadline(monkeypatch):
    calls = []

    class Stream(httpcore.NetworkStream):
        def read(self, max_bytes, timeout=None):
            calls.append(timeout)
            return b"ok"

        def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            calls.append(timeout)
            return Stream()

    monkeypatch.setattr(deadline_http.time, "monotonic", lambda: 10)
    stream = deadline_http._DeadlineStream(Stream(), 12)
    tls = stream.start_tls(httpx.create_ssl_context(), timeout=20)
    monkeypatch.setattr(deadline_http.time, "monotonic", lambda: 11)
    assert tls.read(2, timeout=20) == b"ok"
    assert calls == [2, 1]
    monkeypatch.setattr(deadline_http.time, "monotonic", lambda: 13)
    with pytest.raises(httpcore.ReadTimeout):
        tls.read(2)


def test_worker_budget_does_not_reset_at_later_metadata_legs(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(deadline.time, "monotonic", lambda: clock[0])

    def resolve():
        assert deadline.remaining(20) == 2
        clock[0] = 11
        assert deadline.remaining(20) == 1
        clock[0] = 13
        with pytest.raises(TimeoutError):
            deadline.check()

    deadline.run_before(12, resolve)
    assert deadline.current() is None


@pytest.mark.parametrize(
    ("bypass", "url", "expected"),
    [
        ("127.0.0.1:8765", "http://127.0.0.1:8765", True),
        ("127.0.0.1:8765", "http://127.0.0.1:8766", False),
        ("::1", "http://[::1]:8765", True),
        ("[::1]:8765", "http://[::1]:8765", True),
        ("[::1]:8765", "http://[::1]:8766", False),
        (".example.com", "http://example.com", False),
        (".example.com", "http://sub.example.com", True),
        ("example.com", "http://sub.example.com", True),
        ("example.com", "http://evilexample.com", False),
        ("localhost", "http://sub.localhost", False),
        ("http://example.com:8765", "https://example.com:8765", False),
        ("http://example.com:8765", "http://example.com:8765", True),
        ("example.com,*", "http://other.invalid", True),
    ],
)
def test_no_proxy_keeps_host_port_scheme_and_ipv6_semantics(monkeypatch, bypass, url, expected):
    monkeypatch.setattr(
        deadline_http, "getproxies", lambda: {"http": "http://proxy.invalid:1", "no": bypass}
    )
    transport = deadline_http.ListingDeadlineTransport(time.monotonic() + 2)
    try:
        assert transport._bypass(httpx.URL(url)) is expected
    finally:
        transport.close()


def test_port_qualified_no_proxy_routes_actual_request_to_target(monkeypatch):
    with _listing_server() as (base, _, seen):
        authority = httpx.URL(base).netloc.decode()
        monkeypatch.setattr(
            deadline_http, "getproxies", lambda: {"http": "http://proxy.invalid:1", "no": authority}
        )
        with httpx.Client(
            transport=deadline_http.ListingDeadlineTransport(time.monotonic() + 2)
        ) as client:
            assert client.get(f"{base}/models").status_code == 200
        assert seen == ["/models"]


@pytest.mark.asyncio
async def test_async_resolution_and_to_thread_inherit_same_deadline(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from local_operator.harness.types import ModelSpec
    from local_operator.model import configure
    from local_operator.session.runtime import serving
    from tests.unit.session.runtime.test_serving import make_handle

    handle, session = make_handle()
    readings = []

    async def resolver(spec):
        readings.append(deadline.current())
        readings.append(await asyncio.to_thread(deadline.current))
        return spec

    monkeypatch.setattr(
        session, "_stream_fn", SimpleNamespace(resolve_context_model=resolver), raising=False
    )
    monkeypatch.setattr(configure, "warm_model_info", lambda *_: None)
    monkeypatch.setattr(
        configure, "build_model_spec", lambda p, m: ModelSpec(provider=p, model_id=m)
    )
    monkeypatch.setattr(session, "set_model", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(handle, "_refresh_state", lambda **kw: None)
    monkeypatch.setattr(handle, "_schedule_model_access_publish", lambda: None)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(serving, "_switch_executor", lambda: pool)
    try:
        await handle.set_model("meta-muse", "jeverno-1.1")
    finally:
        pool.shutdown(wait=True)
    assert len(readings) == 2 and readings[0] is not None
    assert readings[0] == readings[1]
    assert deadline.current() is None
