"""Synchronous listing HTTP whose socket work shares one absolute deadline.

httpx's timeouts cap idle operations, not a whole response: a peer dripping
headers or body can otherwise occupy every model-switch worker indefinitely.
The public httpcore backend seam clamps *each* network operation, including
reads hidden inside header parsing. No timer thread, new executor, private
httpx pool mutation, or borrowed client's lifetime is involved.

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


class ListingDeadlineTransport(httpx.BaseTransport):
    """One listing's HTTP transport; pagination reuses its absolute deadline.

    Environment proxy routing remains enabled explicitly: httpx disables its
    automatic proxy mounts when handed a custom transport. Each selected proxy
    gets the same deadline backend, TLS defaults and auth as the direct pool.
    """

    def __init__(self, deadline: float) -> None:
        self._deadline = deadline
        self._backend = _DeadlineBackend(deadline)
        self._ssl_context = httpx.create_ssl_context()
        # Security routing must match the client's actual mount precedence,
        # not an approximate NO_PROXY parser. HTTPX has no public equivalent:
        # isolate its supported 0.27/0.28 compatibility seam here and test route
        # parity before widening the declared version range.
        self._routes = sorted(
            (URLPattern(pattern), proxy) for pattern, proxy in get_environment_proxies().items()
        )
        self._pools: dict[str, httpcore.ConnectionPool] = {}

    def _proxy_for(self, url: httpx.URL) -> str | None:
        for pattern, proxy in self._routes:
            if pattern.matches(url):
                return proxy
        return None

    def _bypass(self, url: httpx.URL) -> bool:
        return self._proxy_for(url) is None

    def _pool_for(self, url: httpx.URL) -> httpcore.ConnectionPool:
        proxy_url = self._proxy_for(url)
        key = proxy_url or ""
        pool = self._pools.get(key)
        if pool is None:
            options: dict[str, Any] = {
                "ssl_context": self._ssl_context,
                "network_backend": self._backend,
            }
            if proxy_url:
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
            else:
                pool = httpcore.ConnectionPool(**options)
            self._pools[key] = pool
        return pool

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        assert isinstance(request.stream, httpx.SyncByteStream)
        # httpcore's public request helper reads AND closes the response stream
        # in a finally. A deadline therefore returns the worker slot and closes
        # the socket even when a listing peer never finishes its document.
        try:
            _remaining(self._deadline, None)
            response = self._pool_for(request.url).request(
                request.method,
                httpcore.URL(
                    scheme=request.url.raw_scheme,
                    host=request.url.raw_host,
                    port=request.url.port,
                    target=request.url.raw_path,
                ),
                headers=request.headers.raw,
                content=iter(request.stream),
                extensions=request.extensions,
            )
        except Exception as error:
            # Preserve the public HTTPX error contract for callers that supply
            # domain-specific fallback handling. Specific classes precede bases.
            for source, target in (
                (httpcore.PoolTimeout, httpx.PoolTimeout),
                (httpcore.ConnectTimeout, httpx.ConnectTimeout),
                (httpcore.ReadTimeout, httpx.ReadTimeout),
                (httpcore.WriteTimeout, httpx.WriteTimeout),
                (httpcore.ConnectError, httpx.ConnectError),
                (httpcore.ReadError, httpx.ReadError),
                (httpcore.WriteError, httpx.WriteError),
                (httpcore.LocalProtocolError, httpx.LocalProtocolError),
                (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
                (httpcore.ProxyError, httpx.ProxyError),
                (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
            ):
                if isinstance(error, source):
                    raise target(str(error), request=request) from error
            raise
        return httpx.Response(
            response.status,
            headers=response.headers,
            content=response.content,
            extensions=response.extensions,
        )

    def close(self) -> None:
        for pool in self._pools.values():
            pool.close()
