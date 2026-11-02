"""Synchronous listing HTTP whose socket work shares one absolute deadline.

httpx's timeouts cap idle operations, not a whole response: a peer dripping
headers or body can otherwise occupy every model-switch worker indefinitely.
The public httpcore backend seam clamps *each* network operation, including
reads hidden inside header parsing. No timer thread, new executor, or
mutation/lifetime change to a borrowed HTTP client is involved.

DNS resolution in the OS's synchronous getaddrinfo is not cancellable by this
backend. This bounds cooperating socket/TLS I/O, not arbitrary native hangs.
"""

from __future__ import annotations

import ssl
import time
from typing import Any, Iterable

import httpcore
import httpx
from httpx._utils import URLPattern, get_environment_proxies


def _remaining(deadline: float, timeout: float | None) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise httpcore.ReadTimeout("model listing deadline exceeded")
    return left if timeout is None else min(timeout, left)


class _DeadlineStream(httpcore.NetworkStream):
    def __init__(self, stream: httpcore.NetworkStream, deadline: float) -> None:
        self._stream = stream
        self._deadline = deadline

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._stream.read(max_bytes, timeout=_remaining(self._deadline, timeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._stream.write(buffer, timeout=_remaining(self._deadline, timeout))

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        # TLS returns a different stream. Keeping the wrapper only around the
        # plaintext connection silently loses the deadline after HTTPS starts.
        return _DeadlineStream(
            self._stream.start_tls(
                ssl_context,
                server_hostname=server_hostname,
                timeout=_remaining(self._deadline, timeout),
            ),
            self._deadline,
        )

    def get_extra_info(self, info: str) -> Any:
        return self._stream.get_extra_info(info)

    def close(self) -> None:
        self._stream.close()


class _DeadlineBackend(httpcore.NetworkBackend):
    def __init__(self, deadline: float) -> None:
        self._deadline = deadline
        self._backend = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        return _DeadlineStream(
            self._backend.connect_tcp(
                host,
                port,
                timeout=_remaining(self._deadline, timeout),
                local_address=local_address,
                socket_options=socket_options,
            ),
            self._deadline,
        )

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        return _DeadlineStream(
            self._backend.connect_unix_socket(
                path,
                timeout=_remaining(self._deadline, timeout),
                socket_options=socket_options,
            ),
            self._deadline,
        )

    def sleep(self, seconds: float) -> None:
        self._backend.sleep(_remaining(self._deadline, seconds))


class _ListingDeadlinePool(httpcore.ConnectionPool):
    """Route each request through a deadline-backed direct or proxy pool.

    Routing remains request-local, so a reused client cannot send concurrent
    requests through a mutable "current proxy" setting. HTTPX's standard
    transport owns request/response conversion, exception mapping and its
    established interception seam; this pool changes only actual socket work.
    """

    def __init__(self, deadline: float) -> None:
        self._deadline = deadline
        self._backend = _DeadlineBackend(deadline)
        self._ssl_context = httpx.create_ssl_context()
        super().__init__(ssl_context=self._ssl_context, network_backend=self._backend)
        # Security routing must match the client's actual mount precedence,
        # not an approximate NO_PROXY parser. HTTPX has no public equivalent:
        # isolate its supported 0.27/0.28 compatibility seam here and test route
        # parity before widening the declared version range.
        self._routes = sorted(
            (URLPattern(pattern), proxy) for pattern, proxy in get_environment_proxies().items()
        )
        self._proxy_pools: dict[str, httpcore.ConnectionPool] = {}

    def _proxy_for(self, url: httpx.URL) -> str | None:
        for pattern, proxy in self._routes:
            if pattern.matches(url):
                return proxy
        return None

    def _pool_for(self, url: httpx.URL) -> httpcore.ConnectionPool:
        proxy_url = self._proxy_for(url)
        if proxy_url is None:
            return self
        pool = self._proxy_pools.get(proxy_url)
        if pool is None:
            options: dict[str, Any] = {
                "ssl_context": self._ssl_context,
                "network_backend": self._backend,
            }
            proxy = httpx.Proxy(proxy_url)
            options["proxy_" + "auth"] = getattr(proxy, "raw_" + "auth")
            if proxy.url.scheme in {"socks5", "socks5h"}:
                pool = httpcore.SOCKSProxy(proxy_url=str(proxy.url), **options)
            else:
                pool = httpcore.HTTPProxy(
                    proxy_url=str(proxy.url),
                    proxy_headers=proxy.headers.raw,
                    proxy_ssl_context=proxy.ssl_context,
                    **options,
                )
            self._proxy_pools[proxy_url] = pool
        return pool

    def handle_request(self, request: httpcore.Request) -> httpcore.Response:
        _remaining(self._deadline, None)
        url = httpx.URL(
            scheme=request.url.scheme.decode("ascii"),
            host=request.url.host.decode("ascii"),
            port=request.url.port,
            raw_path=request.url.target,
        )
        pool = self._pool_for(url)
        return super().handle_request(request) if pool is self else pool.handle_request(request)

    def close(self) -> None:
        try:
            super().close()
        finally:
            for pool in self._proxy_pools.values():
                pool.close()


class ListingDeadlineTransport(httpx.HTTPTransport):
    """Keep the standard HTTPX transport hook while bounding network work.

    HTTPX 0.27/0.28 delegates its standard handle_request/close/context methods
    to one _pool. Construct that pool ONCE here, rather than modifying a
    borrowed client or bypassing HTTPTransport.handle_request. Embedders and
    offline fixtures patching that established hook must still intercept every
    owned listing; the former BaseTransport override silently skipped them and
    could reach a live public catalogue. This private construction seam shares
    the explicit version bound and parity tests of the routing adapter above.
    """

    def __init__(self, deadline: float) -> None:
        self._pool = _ListingDeadlinePool(deadline)
