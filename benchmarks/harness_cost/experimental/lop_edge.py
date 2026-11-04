"""Per-run TLS-terminating egress edge: every paid Local Operator request, one meter.

WHY (METERING-ARCHITECTURE.md §1): released Local Operator v0.94.22 has no
base-URL override for first-party Anthropic and a hard-coded Radient decisions
endpoint, so the only seam that covers the model, the classifier, title helpers,
failover clients and forks — in every process of the tree, without patching the
product — is the network boundary. The child gets ``HTTPS_PROXY`` naming this
edge and ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` naming a bundle that holds ONLY
this run's CA; httpx/requests clients with default ``trust_env`` then CONNECT
here, and the edge terminates TLS with a leaf it minted for the requested host.
A kernel egress sandbox (``sandbox_profile``) makes anything that ignores the
proxy environment fail with EPERM instead of escaping the meter.

Routing is an exact ``(method, host, path)`` map:

* ``POST api.anthropic.com/v1/messages`` → the SAME
  pricing core the other six harnesses use (``meter_proxy`` at
  ``/r/<run_id>/v1/...``), which injects the OpenRouter key.
* ``POST api.radienthq.com/v1/decisions`` → a priced decisions route that
  reserves on the shared ledger's ``classifier`` channel, forwards to the one
  configured decisions upstream with the host-held Radient bearer (no redirects
  followed) and settles only from the reviewed top-level billed ``cost``
  (``radient_meter.classifier.qualify_classifier``).
* Unbilled metadata ``GET``s: none are forwarded yet. The reviewed allowlist
  comes from the zero-paid S4 capture; until then every read is a visible
  (non-latching) denial, so nothing unreviewed is ever proxied.
* Everything else → 403 and a denial row; a denied request that is not a
  ``GET``/``HEAD`` latches the shared ledger (run void), because its bill and
  intent are unknown.

CONSTRAINTS:
* The CA and leaf private keys never touch disk: they are handed to OpenSSL
  through anonymous pipes (``/dev/fd``), and only the CA certificate is written.
* Proxy-Authorization must be ``Basic <run_id>:<nonce>`` of a live run (else
  407); ``finish()`` retires the nonce so a finished run's identity cannot be
  replayed.
* Inbound ``authorization``/``x-api-key``/proxy headers are stripped; real
  credentials are injected only on the exact model/decisions routes.
* Rows record method/host/path/status/body hash and billing facts — never
  headers, bodies or credential values (Radient usage ids are stored hashed).
* HTTP/1.1 only: nothing in ``local_operator/`` enables HTTP/2 at 500883ef.
* Package egress (container lane only, ``package_policy``): LoP's tool
  children inherit ``HTTPS_PROXY`` naming THIS edge, and the launcher refuses
  any other proxy, so the agent-phase package route the other six harnesses
  get at the bench gate is served here under the SAME ``package_egress``
  policy and log shape: a CONNECT or plain-HTTP request whose host the policy
  allows is tunnelled unmodified (no MITM, unmetered, logged per connection).
  Model hosts are always denied by that policy, so they keep falling through
  to the metered routes below; with no policy (desktop lane) nothing changes.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import ssl
import sys
import time
from collections.abc import Iterable, Mapping
from decimal import Decimal
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

import meter_proxy
from radient_meter.classifier import (
    MODEL as CLASSIFIER_MODEL,
    RATE,
    qualify_classifier,
    validate_classifier_request,
)
from radient_meter.json_types import parse_json

if TYPE_CHECKING:
    # Imported lazily below: only the container lane (driver.start_edge) runs
    # the edge with a package policy, so the desktop lane (runner_v2, which
    # copies a fixed file set) never depends on package_egress.py.
    import package_egress

DECISIONS_URL = "https://api.radienthq.com/v1/decisions"
# ``/v1/messages/count_tokens`` is deliberately absent: LoP 500883ef never calls
# it and the meter does not price it, so a request there is an unlisted POST
# (403 + latch) rather than a refused-but-unlatched "model" row.
MODEL_ROUTES = frozenset({("POST", "api.anthropic.com", "/v1/messages")})
DECISIONS_ROUTE = ("POST", "api.radienthq.com", "/v1/decisions")

# The child authenticates to Radient with a value that is worthless outside
# this edge (vendors.py tiers: authstore → RADIENT_API_KEY). The edge strips
# it and injects the host-held bearer on the decisions route only.
DUMMY_RADIENT_KEY = "bench-dummy"
NO_PROXY = "127.0.0.1,localhost,::1"
DENIED_BODY = b'{"error":"route not metered by the benchmark edge"}'
LEDGER_REFUSED_BODY = b'{"error":"budget ledger unavailable, exhausted, or unreconciled"}'
# The edge re-frames relayed bodies itself (chunked), so the upstream's own
# framing/encoding headers must not reach the child.
_HOP_BY_HOP_RESPONSE = frozenset(
    {"content-length", "transfer-encoding", "connection", "content-encoding"}
)

# Bounds from the reviewed conservative limits (public-port fixtures):
# classifier request bytes, decisions response bytes, question/criteria counts.
CLASSIFIER_MAX_BYTES = 131072
RESPONSE_MAX_BYTES = 2 * 1024 * 1024
MAX_QUESTIONS = 16
MAX_CRITERIA = 256
REQUEST_MAX_BYTES = 32 * 1024 * 1024
MAX_HOSTS = 64
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST = re.compile(rf"^(?=.{{1,253}}$){_LABEL}(?:\.{_LABEL})*$")
_STRIP = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authorization",
    "proxy-connection",
    "authorization",
    "x-api-key",
    "accept-encoding",
    "te",
    "upgrade",
}


def _pem_pipe(data: bytes) -> int:
    """Hand PEM bytes to OpenSSL without a file: the read end of a filled pipe.

    A PEM key is far below the pipe buffer, so the write cannot block.
    """
    read, write = os.pipe()
    try:
        os.write(write, data)
    finally:
        os.close(write)
    return read


class RunCA:
    """A per-run CA whose private keys live only in this process's memory."""

    def __init__(self) -> None:
        self._key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.datetime.now(datetime.UTC)
        name = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, f"harness-cost edge {secrets.token_hex(4)}")]
        )
        public = self._key.public_key()
        # Python 3.13+ verifies with VERIFY_X509_STRICT, which requires these
        # CA extensions; without them the child's handshake fails outright.
        self.certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(public)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(public), critical=False
            )
            .sign(self._key, hashes.SHA256())
        )
        self._contexts: dict[str, ssl.SSLContext] = {}

    def certificate_pem(self) -> bytes:
        return self.certificate.public_bytes(serialization.Encoding.PEM)

    def write_certificate(self, path: Path) -> Path:
        """Write the CA CERTIFICATE (never a key): the child's only trust anchor."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.certificate_pem())
        return path

    def server_context(self, host: str) -> ssl.SSLContext:
        """A TLS server context presenting a leaf valid for exactly ``host``."""
        if host in self._contexts:
            return self._contexts[host]
        if len(self._contexts) >= MAX_HOSTS:
            raise ValueError("too many distinct tunnel hosts")
        key = ec.generate_private_key(ec.SECP256R1())
        now = datetime.datetime.now(datetime.UTC)
        try:
            alt: x509.GeneralName = x509.IPAddress(ipaddress.ip_address(host))
        except ValueError:
            alt = x509.DNSName(host)
        leaf = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(self.certificate.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([alt]), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(self._key.public_key()),
                critical=False,
            )
            .sign(self._key, hashes.SHA256())
        )
        chain = leaf.public_bytes(serialization.Encoding.PEM) + self.certificate_pem()
        key_pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_alpn_protocols(["http/1.1"])
        cert_fd, key_fd = _pem_pipe(chain), _pem_pipe(key_pem)
        try:
            context.load_cert_chain(f"/dev/fd/{cert_fd}", f"/dev/fd/{key_fd}")
        finally:
            os.close(cert_fd)
            os.close(key_fd)
        self._contexts[host] = context
        return context


def sandbox_profile(ports: Iterable[int], unix_sockets: Iterable[str] = ()) -> str:
    """A ``sandbox-exec`` profile allowing outbound TCP only to loopback ``ports``.

    Outbound Unix-domain connections are denied except to the exact absolute
    socket paths listed: a host daemon's socket (the operator's own lop
    gateway, Docker, a local proxy) would otherwise be a way around the egress
    fence, since that daemon makes its requests from outside the sandbox. The
    zero-paid S4 run needed none for ``lop exec`` (its own IPC listens inside
    the run HOME and is reached by binding, not by outbound connect). Ports
    and paths are validated because they are interpolated into the profile.
    """
    rules = ["(version 1)", "(allow default)", "(deny network-outbound)"]
    for path in unix_sockets:
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or any(ch in path for ch in '"\\()\n\r\0')
        ):
            raise ValueError("unix socket allowlist entries must be plain absolute paths")
        rules.append(f'(allow network-outbound (remote unix-socket (path-literal "{path}")))')
    for port in ports:
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError("sandbox ports must be integers in 1..65535")
        rules.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
    return "".join(rules)


class _Reject(Exception):
    """A request the edge answers itself; ``latch`` voids the run."""

    def __init__(self, status: int, reason: str, latch: bool = False) -> None:
        super().__init__(reason)
        self.status, self.reason, self.latch = status, reason, latch


async def _read_head(reader: asyncio.StreamReader) -> tuple[str, str, str, dict[str, str]] | None:
    """Parse one HTTP/1.1 request head; duplicate or folded headers are refused."""
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError:
        return None
    except asyncio.LimitOverrunError as exc:
        raise _Reject(431, "request head too large") from exc
    lines = raw[:-4].decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or parts[2] != "HTTP/1.1":
        raise _Reject(400, "malformed request line")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if not sep or not name or name != name.strip() or name.lower() in headers:
            raise _Reject(400, "malformed or duplicate header")
        headers[name.lower()] = value.strip()
    return parts[0], parts[1], parts[2], headers


class EdgeServer:
    """One run's edge: CONNECT MITM, exact routes, shared fail-closed ledger."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        radient_token: str,
        meter_base: str,
        meter_client: httpx.AsyncClient | None = None,
        decisions_client: httpx.AsyncClient | None = None,
        decisions_url: str = DECISIONS_URL,
        package_policy: package_egress.Policy | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id):
            raise ValueError("invalid run identifier")
        if not radient_token:
            raise ValueError("host-side Radient credential required")
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.nonce = secrets.token_urlsafe(24)
        self._token = radient_token
        self._meter_base = meter_base.rstrip("/")
        self._meter = meter_client or httpx.AsyncClient(
            timeout=httpx.Timeout(600, connect=30), trust_env=False
        )
        # trust_env=False: the edge itself must never be routed through a
        # proxy or a CA bundle supplied by its environment.
        self._decisions = decisions_client or httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=30), trust_env=False, follow_redirects=False
        )
        self._decisions_url = decisions_url
        self.package_log_path = self.run_dir / "package-egress.jsonl"
        self._packages: package_egress.PackageProxy | None = None
        if package_policy is not None:
            import package_egress

            self._packages = package_egress.PackageProxy(
                package_policy, package_egress.ConnectionLog(self.package_log_path, via="edge")
            )
        self.ca = RunCA()
        self.ca_path = self.run_dir / "edge-ca.pem"
        self.log_path = self.run_dir / "edge-requests.jsonl"
        # One guard per edge, built now (F6): a misconfigured meter environment
        # refuses to construct the edge instead of failing open per request.
        self._guard = meter_proxy.BudgetGuard()
        self._owns_clients = (meter_client is None, decisions_client is None)
        self._tasks: set[asyncio.Task[None]] = set()
        self.port = 0
        self._finished = False
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    # --- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        self.ca.write_certificate(self.ca_path)
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    def finish(self) -> None:
        """Retire this run's identity: every later Proxy-Authorization is 407."""
        self._finished = True

    async def close(self) -> None:
        """Stop serving; any in-flight request settles or latches before return.

        Handler tasks are cancelled HERE, not left to ``asyncio.run``'s final
        cancellation, so their settle-or-latch ``finally`` blocks run while the
        ledger and clients are still available (F1/N3).
        """
        self.finish()
        if self._server is not None:
            self._server.close()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        for writer in list(self._writers):
            writer.close()
        if self._server is not None:
            await self._server.wait_closed()
        if self._owns_clients[0]:
            await self._meter.aclose()
        if self._owns_clients[1]:
            await self._decisions.aclose()
        self._guard.close()

    def child_env(self) -> dict[str, str]:
        """Environment for the metered child; holds no real credential."""
        proxy = f"http://{self.run_id}:{self.nonce}@127.0.0.1:{self.port}"
        return {
            "HTTPS_PROXY": proxy,
            "https_proxy": proxy,
            "HTTP_PROXY": proxy,
            "http_proxy": proxy,
            "NO_PROXY": NO_PROXY,
            "no_proxy": NO_PROXY,
            "SSL_CERT_FILE": str(self.ca_path),
            "REQUESTS_CA_BUNDLE": str(self.ca_path),
            "RADIENT_API_KEY": DUMMY_RADIENT_KEY,
        }

    # --- accounting --------------------------------------------------------------

    def _row(self, **fields: object) -> None:
        row = {"ts": time.time(), "run_id": self.run_id, **fields}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row, allow_nan=False) + "\n")

    def _latch(self, reason: str) -> None:
        self._guard.latch(self.run_id, reason)

    def _authorized(self, headers: dict[str, str]) -> bool:
        if self._finished:
            return False
        scheme, _, value = headers.get("proxy-authorization", "").partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            supplied = base64.b64decode(value, validate=True)
        except ValueError:
            return False
        expected = f"{self.run_id}:{self.nonce}".encode()
        return hmac.compare_digest(supplied, expected)

    # --- connection handling -------------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        self._writers.add(writer)
        try:
            head = await _read_head(reader)
            if head is None:
                return
            method, target, _, headers = head
            if not self._authorized(headers):
                await self._send(
                    writer,
                    407,
                    b'{"error":"proxy authentication required"}',
                    extra={"Proxy-Authenticate": 'Basic realm="harness-cost-edge"'},
                    close=True,
                )
                return
            if self._packages is not None and await self._package(reader, writer, method,
                                                                  target, headers):
                return
            if method != "CONNECT":
                # Plain-HTTP proxying would carry a request the edge cannot
                # route by TLS host; refuse it like any unlisted route.
                host = httpx.URL(target).host if target.startswith("http://") else ""
                latch = method not in ("GET", "HEAD")
                await self._deny(writer, method, host, "plain-http", latch=latch)
                return
            host, sep, port = target.rpartition(":")
            host = host.lower()
            if not sep or port != "443" or not _HOST.match(host):
                await self._deny(writer, method, host, "connect:" + port, latch=True)
                return
            try:
                context = self.ca.server_context(host)
            except ValueError:
                await self._deny(writer, method, host, "connect", latch=True)
                return
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()
            await writer.start_tls(context)
            await self._tunnel(reader, writer, host)
        except _Reject as exc:
            if exc.latch:
                self._latch(f"edge refused: {exc.reason}")
            with contextlib.suppress(Exception):
                reason = json.dumps({"error": exc.reason}).encode()
                await self._send(writer, exc.status, reason, close=True)
        except (ConnectionError, ssl.SSLError, asyncio.IncompleteReadError):
            pass
        finally:
            if task is not None:
                self._tasks.discard(task)
            self._writers.discard(writer)
            with contextlib.suppress(Exception):
                writer.close()

    async def _package(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        target: str,
        headers: dict[str, str],
    ) -> bool:
        """Serve an allowlisted package request; ``False`` hands it to the metered path.

        A non-model destination the policy denies is logged here as well (the
        trial's package record must show every attempt), then still takes the
        edge's normal deny path, so its latch rules are unchanged.
        """
        assert self._packages is not None
        parsed = self._packages.target(method, target)
        if parsed is None:
            return False
        reason = self._packages.policy.decide(parsed[0], parsed[1])
        if reason is None:
            await self._packages.dispatch(reader, writer, method, target, headers)
            return True
        if reason != "model-host":
            self._packages.log.write(method=method, host=parsed[0].lower(), port=parsed[1],
                                     allowed=False, reason=reason, bytes_up=0, bytes_down=0)
        return False

    async def _tunnel(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, host: str
    ) -> None:
        """Serve keep-alive HTTP/1.1 requests decrypted from one CONNECT tunnel."""
        while not self._finished:
            head = await _read_head(reader)
            if head is None:
                return
            method, target, _, headers = head
            if not target.startswith("/"):
                latch = method not in ("GET", "HEAD")
                await self._deny(writer, method, host, "absolute-form", latch=latch)
                return
            # Host header must name the tunnel host: otherwise a request could
            # borrow a routed tunnel to reach a different upstream identity.
            if headers.get("host", host).lower().removesuffix(":443") != host:
                await self._deny(writer, method, host, target, latch=True)
                return
            if "transfer-encoding" in headers:
                raise _Reject(411, "chunked request bodies are not metered", latch=True)
            try:
                length = int(headers.get("content-length", "0"))
            except ValueError as exc:
                raise _Reject(400, "invalid content-length", latch=True) from exc
            if not 0 <= length <= REQUEST_MAX_BYTES:
                raise _Reject(413, "request body too large", latch=True)
            body = await reader.readexactly(length)
            key = (method, host, target)
            if key in MODEL_ROUTES:
                await self._model(writer, key, headers, body)
            elif key == DECISIONS_ROUTE:
                await self._decisions_route(writer, key, headers, body)
            else:
                await self._deny(
                    writer, method, host, target, latch=method not in ("GET", "HEAD"), body=body
                )
                return
            if headers.get("connection", "").lower() == "close":
                return

    async def _deny(
        self,
        writer: asyncio.StreamWriter,
        method: str,
        host: str,
        target: str,
        *,
        latch: bool,
        body: bytes = b"",
    ) -> None:
        path, _, query = target.partition("?")
        if latch:
            self._latch(f"denied {method} {host}{path}")
        self._row(
            denied=True,
            method=method,
            host=host,
            path=path,
            has_query=bool(query),
            body_sha256=hashlib.sha256(body).hexdigest(),
            latched=latch,
        )
        await self._send(writer, 403, DENIED_BODY, close=True)

    @staticmethod
    def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in headers.items() if k not in _STRIP}

    async def _model(
        self,
        writer: asyncio.StreamWriter,
        key: tuple[str, str, str],
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Forward to the shared pricing core; the meter injects the OpenRouter key."""
        method, host, path = key
        url = f"{self._meter_base}/r/{self.run_id}{path}"
        request = self._meter.build_request(
            method, url, headers=self._forward_headers(headers), content=body
        )
        try:
            response = await self._meter.send(request, stream=True)
        except httpx.HTTPError:
            # The meter is local; failing to reach it means nothing was sent upstream.
            self._row(method=method, host=host, path=path, route="model", channel="model",
                      status=502, body_sha256=hashlib.sha256(body).hexdigest())
            await self._send(writer, 502, b'{"error":"meter unavailable"}', close=True)
            return
        try:
            self._row(method=method, host=host, path=path, route="model", channel="model",
                      status=response.status_code, body_sha256=hashlib.sha256(body).hexdigest())
            await self._stream(writer, response)
        finally:
            await response.aclose()

    async def _decisions_route(
        self,
        writer: asyncio.StreamWriter,
        key: tuple[str, str, str],
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Priced classifier route on the shared ledger's ``classifier`` channel."""
        method, host, path = key
        digest = hashlib.sha256(body).hexdigest()
        base = {"method": method, "host": host, "path": path, "route": "decisions",
                "channel": "classifier", "body_sha256": digest}
        try:
            if len(body) > CLASSIFIER_MAX_BYTES:
                raise ValueError("classifier request byte bound")
            request = parse_json(body)
            if not isinstance(request, dict):
                raise ValueError("classifier body must be an object")
            validate_classifier_request(
                request, max_questions=MAX_QUESTIONS, max_criteria=MAX_CRITERIA
            )
        except (ValueError, KeyError, OSError):
            # An unpriceable request the product expected to send: refusing it
            # changes the run, so the run is void as well as refused.
            self._latch("unpriced classifier request")
            self._row(**base, status=402, refused="unpriced classifier request")
            await self._send(writer, 402, b'{"error":"unpriced classifier request"}')
            return
        # Input tokens are bounded by 4x UTF-8 bytes plus framing slack, the
        # same deliberate over-reservation the model meter uses; the reviewed
        # contract prices output at zero.
        reserve = float((Decimal(4 * len(body) + 65536) * RATE))
        guard = self._guard
        try:
            # ``acquire`` voids the run on any refusal (F5): a refused
            # classifier call changes what the product does next.
            token = await guard.acquire(self.run_id, "classifier", reserve, CLASSIFIER_MODEL)
        except (ValueError, KeyError, OSError):
            self._row(**base, status=402, refused="ledger exhausted, latched or busy")
            await self._send(writer, 402, LEDGER_REFUSED_BODY)
            return
        # Settle-or-latch (F1): from here to the settle below, any exit --
        # including CancelledError at shutdown -- abandons the reservation.
        settled = False
        try:
            await self._decide(writer, guard, token, base, headers, body)
            settled = True
        finally:
            if not settled:
                guard.abandon(token, "decisions request ended before settlement")

    async def _decide(
        self,
        writer: asyncio.StreamWriter,
        guard: meter_proxy.BudgetGuard,
        token: str,
        base: Mapping[str, object],
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Dispatch one admitted decision and settle it (every return path settles)."""
        forward = self._forward_headers(headers)
        forward["authorization"] = f"Bearer {self._token}"
        forward["accept-encoding"] = "identity"
        started = time.time()
        try:
            async with self._decisions.stream(
                "POST", self._decisions_url, headers=forward, content=body
            ) as response:
                content = b""
                # Decoded bytes: the body is re-framed with its own length and
                # no content-encoding, so the child must receive plain JSON.
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) > RESPONSE_MAX_BYTES:
                        raise ValueError("decisions response too large")
                status = response.status_code
                ctype = response.headers.get("content-type", "application/json")
        except Exception:
            # Any failure after dispatch may follow a bill: the reservation is
            # kept as unknown exposure and the ledger latches.
            guard.settle(token, None)
            self._row(**base, status=502, qualified=False, faults=["transport-or-size"])
            await self._send(writer, 502, b'{"error":"decisions upstream failed; reconcile"}')
            return
        try:
            payload = parse_json(content)
            if not isinstance(payload, dict):
                raise ValueError("decisions body is not an object")
        except ValueError:
            guard.settle(token, None)
            self._row(**base, status=status, qualified=False, faults=["non-json-response"])
        else:
            observation = qualify_classifier(payload, status)
            actual = observation.actual_usd
            # Settle only a stated billed amount; informational usage.cost is
            # deliberately ignored by the reviewed contract, so it latches.
            guard.settle(token, None if actual is None else float(actual),
                         qualified=observation.qualified_complete)
            record = payload.get("usage_record_id")
            self._row(
                **base,
                status=status,
                qualified=observation.qualified_complete,
                faults=list(observation.faults),
                actual_usd=None if actual is None else str(actual),
                served_model=observation.served_model,
                latency_s=round(time.time() - started, 3),
                usage_record_sha256=hashlib.sha256(record.encode()).hexdigest()
                if isinstance(record, str) else None,
            )
        await self._send(writer, status, content, ctype=ctype)

    # --- response writing ----------------------------------------------------------

    @staticmethod
    async def _send(
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        *,
        ctype: str = "application/json",
        extra: dict[str, str] | None = None,
        close: bool = False,
    ) -> None:
        reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else "Status"
        lines = [
            f"HTTP/1.1 {status} {reason}",
            f"Content-Type: {ctype}",
            f"Content-Length: {len(body)}",
        ]
        lines += [f"{k}: {v}" for k, v in (extra or {}).items()]
        if close:
            lines.append("Connection: close")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body)
        await writer.drain()

    @staticmethod
    async def _stream(writer: asyncio.StreamWriter, response: httpx.Response) -> None:
        """Relay a (possibly SSE) meter response with chunked framing, unbuffered."""
        status = response.status_code
        reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else "Status"
        lines = [f"HTTP/1.1 {status} {reason}"]
        for k, v in response.headers.items():
            if k.lower() not in _HOP_BY_HOP_RESPONSE:
                lines.append(f"{k}: {v}")
        lines.append("Transfer-Encoding: chunked")
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode("latin-1"))
        async for chunk in response.aiter_raw():
            if chunk:
                writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                await writer.drain()
        writer.write(b"0\r\n\r\n")
        await writer.drain()


async def _serve_from_stdin() -> None:
    """Run one edge from a JSON config on stdin; print the child env as one JSON line.

    The Radient bearer arrives on stdin (never argv/env/disk) and stays in this
    process. ``decisions_url`` exists only so the zero-paid rig can point the
    upstream at a loopback fixture; a paid run uses the default.
    """
    config = json.loads(sys.stdin.readline())
    policy = None
    if config.get("package_allowlist"):
        import package_egress

        policy = package_egress.Policy.load(config["package_allowlist"])
    server = EdgeServer(
        run_id=config["run_id"],
        run_dir=Path(config["run_dir"]),
        radient_token=config["radient_token"],
        meter_base=config["meter_base"],
        decisions_url=config.get("decisions_url", DECISIONS_URL),
        package_policy=policy,
    )
    await server.start()
    sys.stdout.write(json.dumps({"port": server.port, "env": server.child_env()}) + "\n")
    sys.stdout.flush()
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    loop.add_reader(sys.stdin.fileno(), lambda: done.done() or done.set_result(None))
    try:
        await done  # parent closes stdin to finish the run
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(_serve_from_stdin())
