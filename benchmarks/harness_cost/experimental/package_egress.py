"""Agent-phase package egress: an exact-host forward proxy, never a model route.

WHY (container S4 oracle re-qualification, 2026-11-04): a paid trial's ``main``
sits on an internal network with no route, so ``apt-get``/``pip``/``uv`` in
the AGENT phase failed for every harness (nginx-request-logging and
kv-store-grpc scored 0 even for the oracle). Re-screening the cohort would bias
it, so agents get a narrow package route instead, identical for all seven
harnesses:

* the six non-LoP harnesses reach this proxy in the bench gate
  (``gate_relay.py --package-proxy``) through ``HTTP(S)_PROXY``, with
  ``NO_PROXY`` exempting the gate host so model traffic still goes straight to
  the meter/bridge ports;
* Local Operator's proxy variables must keep naming its metering edge (the
  launcher refuses anything else), so the edge applies THIS policy to the same
  tool-child traffic (``lop_edge.EdgeServer(package_policy=...)``).

Both sides share ``Policy`` and ``PackageProxy`` below, so the allowlist, the
model-host denials, the port rule and the log row shape are one implementation.

CONSTRAINTS:
* Exact host names only (no wildcards, no IP literals) from the frozen
  ``coding/package-egress-allowlist.json``; ports 80 and 443 only.
* Every model/LLM host is denied BEFORE the allowlist is consulted, so listing
  one by mistake cannot open an unmetered model route.
* TLS is tunnelled, never intercepted: package bytes are not model traffic and
  are not priced. They are logged per connection (host, port, bytes,
  allowed/denied) and disclosed in the trial record.
* The default upstream opener resolves the name itself and refuses any
  non-public address, so an allowlisted name that resolves to the host
  gateway, a private range or loopback cannot reach the run's own services.
* stdlib only: the gate runs it under the frozen toolchain interpreter.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import re
import socket
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from urllib.parse import urlsplit

PORTS = frozenset({80, 443})
HEAD_MAX_BYTES = 16 * 1024
CONNECT_TIMEOUT_S = 30
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST = re.compile(rf"^(?=.{{1,253}}$){_LABEL}(?:\.{_LABEL})*$")

# Model/LLM API domains: the host itself and every subdomain are denied even
# when allowlisted. Covers the providers the seven harnesses can speak to plus
# the common hosted-inference APIs a tool might try instead of the meter.
MODEL_DOMAINS = frozenset({
    "anthropic.com", "claude.ai", "claude.com",
    "openai.com", "chatgpt.com", "openai.azure.com", "cognitiveservices.azure.com",
    "models.inference.ai.azure.com", "openrouter.ai", "radienthq.com", "deepseek.com",
    "generativelanguage.googleapis.com", "mistral.ai", "x.ai", "groq.com", "cohere.com",
    "cohere.ai", "together.xyz", "together.ai", "fireworks.ai", "perplexity.ai",
    "deepinfra.com", "cerebras.ai", "sambanova.ai", "replicate.com", "ai21.com",
    "moonshot.ai", "moonshot.cn", "z.ai", "bigmodel.cn", "dashscope.aliyuncs.com",
    "api-inference.huggingface.co", "router.huggingface.co", "models.github.ai",
    "githubcopilot.com", "novita.ai", "hyperbolic.xyz",
})
# Regional endpoints that cannot be listed as one domain.
MODEL_PATTERNS = (
    re.compile(r"(^|[.-])aiplatform\.googleapis\.com$"),
    re.compile(r"^bedrock(-[a-z-]+)?\.[a-z0-9-]+\.amazonaws\.com(\.cn)?$"),
)


def model_host(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in MODEL_DOMAINS) or any(
        p.search(host) for p in MODEL_PATTERNS
    )


def _ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


class Policy:
    """Exact-host allowlist with the model deny list layered on top."""

    def __init__(self, allow: Iterable[str]) -> None:
        hosts = list(allow)
        for host in hosts:
            if not isinstance(host, str) or _ip_literal(host) or not _HOST.fullmatch(host):
                raise ValueError(f"allowlist entries must be exact lowercase host names: {host!r}")
        self.allow = frozenset(hosts)

    @classmethod
    def load(cls, path: Path | str) -> Policy:
        data = json.loads(Path(path).read_text())
        return cls(data["allow"])

    def decide(self, host: str, port: int) -> str | None:
        """``None`` to allow, else the denial reason recorded in the log."""
        name = host.lower().rstrip(".")
        if _ip_literal(name):
            return "ip-literal"
        if not _HOST.fullmatch(name):
            return "malformed"
        if model_host(name):
            return "model-host"
        if port not in PORTS:
            return "port"
        if name not in self.allow:
            return "not-allowlisted"
        return None


def logged_host(host: str, reason: str | None) -> str:
    """The host as it may appear in the published trial record.

    A ``malformed`` authority is client-controlled bytes (e.g. a proxy URL's
    ``user:password@host``), so only a short digest is kept: enough to group
    repeats, never the credential (R8 N7). Checked names are logged verbatim.
    """
    if reason == "malformed":
        return "redacted-sha256:" + hashlib.sha256(host.encode("utf-8", "replace")).hexdigest()[:16]
    return host.lower().rstrip(".")


def public_address(addr: str) -> bool:
    ip = ipaddress.ip_address(addr)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def open_upstream(host: str, port: int):
    """Connect to ``host`` on a PUBLIC address only (see module constraints)."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addrs = [info[4][0] for info in infos if public_address(info[4][0])]
    if not addrs:
        raise PermissionError("allowlisted host resolved to a non-public address")
    return await asyncio.open_connection(addrs[0], port)


class ConnectionLog:
    """Append-only JSON lines, one per proxied connection (no headers, no paths)."""

    def __init__(self, path: Path | str, via: str) -> None:
        self.path = Path(path)
        self.via = via

    def write(self, **fields: object) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"ts": time.time(), "via": self.via, **fields}) + "\n")


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> int:
    # Half-close on EOF (not close): the other direction may still be carrying
    # the response, as in gate_relay._pipe.
    total = 0
    try:
        while chunk := await reader.read(65536):
            total += len(chunk)
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError, asyncio.CancelledError):
        pass
    return total


async def read_head(reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str]] | None:
    """``(method, target, headers)``; reads exactly the head, leaving the body buffered."""
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None
    if len(raw) > HEAD_MAX_BYTES:
        return None
    lines = raw.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        return None
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if line:
            name, sep, value = line.partition(":")
            if not sep:
                return None
            headers[name.strip().lower()] = value.strip()
    return parts[0], parts[1], headers


# Hop-by-hop and proxy-only headers never reach a registry (a proxy credential,
# e.g. LoP's edge nonce, must not leak upstream).
_DROP = frozenset({"proxy-authorization", "proxy-connection", "connection", "keep-alive", "te",
                   "upgrade"})

Opener = Callable[[str, int], Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]]]


class PackageProxy:
    """CONNECT tunnels and absolute-form plain-HTTP requests to allowlisted hosts."""

    def __init__(self, policy: Policy, log: ConnectionLog, opener: Opener | None = None) -> None:
        self.policy = policy
        self.log = log
        self.opener = opener or open_upstream

    @staticmethod
    def target(method: str, target: str) -> tuple[str, int, str] | None:
        """``(host, port, origin-form path)`` or ``None`` when not proxyable."""
        if method == "CONNECT":
            host, sep, port = target.rpartition(":")
            if not sep or not port.isdigit():
                return None
            return host.strip("[]"), int(port), ""
        if not target.startswith("http://"):
            # Origin-form or https:// absolute-form: no destination to check.
            return None
        url = urlsplit(target)
        try:
            port = url.port or 80
        except ValueError:
            return None
        if not url.hostname or url.username or url.password:
            return None
        return url.hostname, port, (url.path or "/") + (f"?{url.query}" if url.query else "")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await read_head(reader)
            if head is None:
                return
            await self.dispatch(reader, writer, *head)
        finally:
            writer.close()

    async def dispatch(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        target: str,
        headers: dict[str, str],
    ) -> None:
        """Serve one already-parsed request head; the caller closes ``writer``."""
        started = time.monotonic()
        parsed = self.target(method, target)
        host, port, path = parsed or ("", 0, "")
        reason = "malformed" if parsed is None else self.policy.decide(host, port)
        row: dict[str, object] = {"method": method, "host": logged_host(host, reason),
                                  "port": port}
        host = host.lower().rstrip(".")
        if reason is not None:
            self.log.write(**row, allowed=False, reason=reason, bytes_up=0, bytes_down=0)
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(
                self.opener(host, port), CONNECT_TIMEOUT_S
            )
        except (OSError, asyncio.TimeoutError) as exc:
            self.log.write(**row, allowed=True, error=f"{type(exc).__name__}: {exc}"[:200],
                           bytes_up=0, bytes_down=0)
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                         b"Connection: close\r\n\r\n")
            await writer.drain()
            return
        sent = 0
        try:
            if method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                # One request per connection: ``Connection: close`` upstream.
                # A pipelined follow-up still reaches only THIS host (the TCP
                # destination was checked), and apt retries what it loses.
                lines = [f"{method} {path} HTTP/1.1"]
                # Host is ALWAYS the checked destination, never the client's:
                # on a shared CDN edge plain-HTTP routing follows Host, so a
                # forwarded client Host could reach another tenant (R8 F25).
                lines += [f"{k}: {v}" for k, v in headers.items()
                          if k not in _DROP and k != "host"]
                lines.append(f"Host: {host}" if port == 80 else f"Host: {host}:{port}")
                lines.append("Connection: close")
                request = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
                up_writer.write(request)
                await up_writer.drain()
                sent = len(request)
            up, down = await asyncio.gather(_pump(reader, up_writer), _pump(up_reader, writer))
        finally:
            up_writer.close()
        self.log.write(**row, allowed=True, bytes_up=sent + up, bytes_down=down,
                       duration_s=round(time.monotonic() - started, 3))
