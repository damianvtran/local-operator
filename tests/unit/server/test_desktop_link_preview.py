"""Link previews: the card, the codes, and every refusal the surface can answer.

The fixture origin is an ``httpx.MockTransport`` driven through the REAL fetch
loop (``WebFetchService._follow``), so each assertion exercises the production
redirect/SSRF path rather than a stub of it; DNS is pinned the way
``tests/unit/web_fetch/test_service.py`` pins it, because the guard under test is
the guard that resolves names. The routes are driven through ASGI, which is what
the renderer talks to.
"""

from __future__ import annotations

import asyncio
import operator
import socket
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, SupportsIndex, overload

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.server import desktop
from local_operator.server.app import app
from local_operator.server.routes import desktop_link_preview
from local_operator.team_bridge.link_preview import cache as link_preview_cache
from local_operator.team_bridge.link_preview import extract
from local_operator.team_bridge.link_preview import service as preview_service
from local_operator.team_bridge.link_preview.cache import (
    NEGATIVE_TTL_S,
    POSITIVE_TTL_S,
    PreviewCache,
    PreviewScope,
)
from local_operator.team_bridge.link_preview.store import (
    InvalidArtifactId,
    PreviewStore,
)
from local_operator.team_bridge.link_preview.throttle import ResolveThrottle
from local_operator.web_fetch import service as fetch_service
from local_operator.web_fetch.models import WebFetchSettings
from local_operator.web_fetch.service import PolicyRefusal, WebFetchService
from local_operator.web_search.io import WebReadIO

TOKEN = "desktop-preview-test-bearer"
ACCOUNT = "acc_0123456789abcdef"
OTHER_ACCOUNT = "acc_fedcba9876543210"

RESOLVE = "/v1/desktop/team-bridge/link-preview/resolve"
ARTIFACT = "/v1/desktop/team-bridge/link-preview/artifact"

#: A 2x2 PNG. Real bytes, because the store hashes what it is given and the
#: artifact route re-sniffs it: a fake "image" would test nothing that matters.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000020000000100806000000f1a1b1d1"
    "0000001649444154789c63fcffff3f0303030313101b0c00a5e8003f9a2e0e7c"
    "0000000049454e44ae426082"
)
SVG = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><script/></svg>'
GIF = b"GIF89a" + (3).to_bytes(2, "little") + (7).to_bytes(2, "little") + b"\x00" * 8


@pytest.fixture(autouse=True)
def _plane(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Open the desktop plane and point it at a throwaway config root."""
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", TOKEN)
    monkeypatch.delenv(desktop.ORIGINS_ENV, raising=False)
    manager = ConfigManager(config_dir=tmp_path / "cfg")
    previous = getattr(app.state, "config_manager", None)
    app.state.config_manager = manager
    try:
        yield manager
    finally:
        app.state.config_manager = previous


@pytest.fixture(autouse=True)
def _fresh_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own full per-account bucket.

    The route reads the PROCESS throttle, which outlives a test, so without this
    a test's outcome depended on how many resolves the tests before it spent and
    how fast the box refilled them — measured: adding the round-2 parametrised
    cases turned thirteen unrelated tests into ``429 rate_limited``. Tests of the
    throttle itself still install their own, which wins because it is set later.
    """
    monkeypatch.setattr(desktop_link_preview, "get_resolve_throttle", ResolveThrottle)


@pytest.fixture(autouse=True)
def _pinned_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every name the fixtures use; IP literals keep their own identity.

    ``preview.test`` is the fixture origin, which must look PUBLIC so the guard
    actually runs its judgement (patching it out entirely would leave the very
    policy under test untested).
    """

    def _resolve(hostname: str) -> list[str]:
        if hostname == "preview.test":
            return ["93.184.216.34"]
        return [hostname]

    monkeypatch.setattr(fetch_service, "_resolve_host_ips", _resolve)


#: The production resolver, captured at import time — i.e. BEFORE the autouse
#: fixture above replaces it. Only a test of the resolver's own refusal behaviour
#: wants this; everything else wants the pinned fixture.
_RESOLVE_HOST_IPS_REAL = fetch_service._resolve_host_ips


class _Clock:
    """A monotonic clock a test can move, so a six-hour TTL costs no seconds."""

    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def _clock() -> _Clock:
    return _Clock()


@pytest.fixture(autouse=True)
def preview_cache_for_test(_clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> PreviewCache:
    """A fresh cache per test, behind the seam the route reads.

    The production cache is process-wide on purpose, which would make one test's
    page answer another test's request and hide the very fetch a test is counting.
    """
    cache = PreviewCache(clock=_clock)
    monkeypatch.setattr(link_preview_cache, "get_preview_cache", lambda: cache)
    return cache


def _account_wire(account_id: str = ACCOUNT) -> Callable[..., Any]:
    """A stand-in for ``wire_for``: the signed-in identity, and nothing else."""

    class _Wire:
        async def current_access(self):
            return SimpleNamespace(account_id=account_id, kind="oauth", credential_id=1)

    return lambda _config_dir: _Wire()


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.team_bridge import read_context

    monkeypatch.setattr(read_context, "wire_for", _account_wire())


def _route_transport(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[str]:
    """Install a fixture origin as the route's transport; return the request log.

    The route builds ``WebFetchService`` itself, so the seam is the symbol it
    imports — the transport is the only thing replaced, and every hop, retry and
    refusal still runs production code.
    """
    seen: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return handler(request)

    origin = httpx.MockTransport(_record)
    monkeypatch.setattr(
        preview_service,
        "WebFetchService",
        # The REAL service with the fixture origin behind it: only the transport is
        # replaced, and every other argument the preview passes (``trust_env``
        # included) still reaches the production constructor.
        lambda settings, *, transport=None, trust_env=True: WebFetchService(
            settings, transport=transport or origin, trust_env=trust_env
        ),
    )
    return seen


def _drip_origin() -> tuple[Any, str]:
    """A real loopback origin that sends headers, then one byte every 100 ms."""
    import http.server
    import threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.send_header("content-length", "100000")
            self.end_headers()
            try:
                self.wfile.write(b"<html><head><title>")
                self.wfile.flush()
                for _ in range(100):
                    time.sleep(0.1)
                    self.wfile.write(b"x")
                    self.wfile.flush()
            except OSError:
                return

        # The base signature names its first parameter ``format`` (it is
        # interpolated with ``args``); an override that took only ``*args``
        # would be position-only there and pyright refuses the mismatch.
        def log_message(self, format: str, *args: Any) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def _html(head: str, body: str = "<p>hello</p>") -> bytes:
    return f"<!doctype html><html><head>{head}</head><body>{body}</body></html>".encode()


async def _post(client: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
    return await client.post(RESOLVE, json=body)


def _client(transport: httpx.MockTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://127.0.0.1",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


def _artifact_path(artifact_id: str) -> str:
    """The artifact route's URL. The account receipt is a HEADER, never a query.

    S-3 of the security round is exactly this: a GET's URL is written down by every
    intermediary on the way, so ``expected_account_id`` travels in
    ``X-Link-Preview-Account``. A caller that puts it in the query string instead
    gets a 422 from the header being absent, which the tests below pin.
    """
    return f"{ARTIFACT}/{artifact_id}"


def _receipt(account_id: str = ACCOUNT) -> dict[str, str]:
    """The account receipt header every artifact read carries."""
    return {"X-Link-Preview-Account": account_id}


# ---------------------------------------------------------------------------
# The happy path, end to end through both routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_page_resolves_into_a_card_and_the_artifact_route_serves_its_bytes(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    page = _html(
        "<title>Fallback title</title>"
        '<meta property="og:title" content="Team Bridge ships previews">'
        '<meta property="og:description" content="A short description.">'
        '<meta property="og:site_name" content="Local Operator">'
        '<link rel="icon" href="/assets/favicon.png">'
        '<meta property="og:image" content="https://preview.test/assets/og.png">'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/assets/og.png":
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
        if request.url.path == "/assets/favicon.png":
            return httpx.Response(200, content=GIF, headers={"content-type": "image/png"})
        return httpx.Response(
            200, content=page, headers={"content-type": "text/html; charset=utf-8"}
        )

    _route_transport(monkeypatch, handler)
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/page", "index": 0, "expected_account_id": ACCOUNT}
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["kind"] == "ready", result
        preview = result["preview"]
        # og:title wins over <title>, and the camelCase wire name is what the
        # renderer reads.
        assert preview["title"] == "Team Bridge ships previews"
        assert preview["description"] == "A short description."
        assert preview["siteName"] == "Local Operator"
        assert preview["url"].startswith("http://preview.test/page")
        assert preview["favicon"]["mime"] == "image/gif"
        assert preview["image"]["mime"] == "image/png"
        assert (preview["image"]["width"], preview["image"]["height"]) == (32, 16)

        # The artifact route hands back the stored bytes under the SNIFFED mime,
        # never the label the origin claimed (that favicon said image/png and is
        # a GIF on the wire).
        favicon = await client.get(_artifact_path(preview["favicon"]["id"]), headers=_receipt())
        assert favicon.status_code == 200
        assert favicon.headers["content-type"] == "image/gif"
        assert favicon.headers["x-content-type-options"] == "nosniff"
        assert favicon.content == GIF

        image = await client.get(_artifact_path(preview["image"]["id"]), headers=_receipt())
        assert image.status_code == 200
        assert image.content == PNG
        assert image.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_a_utf8_title_round_trips_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    title = "Привет, мир — Καλημέρα κόσμε"
    body = _html(f"<title>{title}</title>").decode("utf-8")
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "text/html; charset=UTF-8"}
        ),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/utf8", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"]["preview"]["title"] == title


@pytest.mark.asyncio
async def test_a_declared_non_utf8_charset_is_honoured_not_guessed(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    title = "Café résumé naïve"
    body = _html(f"<title>{title}</title>")
    # The page is Latin-1 and SAYS so in the header. A blind UTF-8 read (or a
    # blind latin-1 read of a UTF-8 page) is the bug this pins.
    payload = body.decode("utf-8").encode("iso-8859-1")
    assert payload != body  # the fixture is genuinely not UTF-8
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=payload, headers={"content-type": "text/html; charset=iso-8859-1"}
        ),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/latin", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"]["preview"]["title"] == title


@pytest.mark.asyncio
async def test_a_meta_declared_charset_is_used_when_the_header_names_none(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    title = "Grüße aus Köln"
    body = _html(f'<meta charset="iso-8859-1"><title>{title}</title>')
    payload = body.decode("utf-8").encode("iso-8859-1")
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=payload, headers={"content-type": "text/html"}),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/meta", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"]["preview"]["title"] == title


# ---------------------------------------------------------------------------
# Negatives, each asserted by raw status and body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_cap_answers_before_a_single_request(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=b""))
    async with _client() as client:
        for index in (3, 4, 99):
            response = await _post(
                client,
                {"url": "http://preview.test/page", "index": index, "expected_account_id": ACCOUNT},
            )
            assert response.status_code == 200
            assert response.json()["result"] == {"kind": "absent", "reason": "cap"}
    assert seen == []


@pytest.mark.asyncio
async def test_a_non_http_scheme_is_absent_without_a_request(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=b""))
    async with _client() as client:
        for url in ("file:///etc/passwd", "javascript:alert(1)", "ftp://preview.test/x"):
            response = await _post(client, {"url": url, "index": 0, "expected_account_id": ACCOUNT})
            assert response.status_code == 200
            assert response.json()["result"] == {"kind": "absent", "reason": "unsupported-scheme"}
    assert seen == []


def _spy_dns(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every name the address guard asks DNS about; return the log.

    Replaces the pinned fixture for one test by putting the PRODUCTION
    ``_resolve_host_ips`` back and spying one layer lower, on
    ``socket.getaddrinfo`` — so the resolver's own handling (the IP-literal
    short-circuit, the gaierror split, the undecodable-name refusal) is what runs,
    and only the network is replaced. The spy IDNA-encodes a ``str`` host first,
    which is what the real ``getaddrinfo`` does before it sends anything: ``.`` or
    ``a..b`` raises the same ``UnicodeError`` QA measured against real DNS (it
    used to become a 500).
    """
    looked_up: list[str] = []

    def _getaddrinfo(host: str, port: object, *args: object, **kwargs: object) -> list[Any]:
        looked_up.append(host)
        host.encode("idna")
        address = "10.0.0.7" if host == "private.test" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0))]

    monkeypatch.setattr(fetch_service, "_resolve_host_ips", _RESOLVE_HOST_IPS_REAL)
    monkeypatch.setattr(fetch_service.socket, "getaddrinfo", _getaddrinfo)
    return looked_up


#: Inputs a message can carry that are NOT absolute http(s) URLs with a host,
#: and the coded absence each must answer. Scheme-less spellings are refused as
#: ``unsupported-scheme`` (no http/https scheme at all) rather than having
#: ``https://`` prepended — QA measured ``README.md`` being fetched as the site
#: ``https://README.md`` after a real DNS lookup. A URL that names http(s) but no
#: usable host, or does not parse, is ``blocked-host``: the code the renderer
#: draws as "could not be used" with no Retry.
_REFUSED_TARGETS = [
    ("README.md", "unsupported-scheme"),
    ("example.com/login", "unsupported-scheme"),
    ("//evil.example/p", "unsupported-scheme"),
    ("./notes/plan.md", "unsupported-scheme"),
    ("#section", "unsupported-scheme"),
    ("mailto:x@y", "unsupported-scheme"),
    ("javascript:", "unsupported-scheme"),
    ("https:///no-host", "blocked-host"),
    ("https://./x", "blocked-host"),
    ("https://[::1/unterminated", "blocked-host"),
    ("https://preview.test:99999/x", "blocked-host"),
    # A COLLAPSED SCHEME is refused too, and that is a deliberate narrowing: the
    # shared fetcher repairs ``https:/host`` to ``https://host``, but a preview's
    # contract is an absolute URL, and keeping the repair here would be a second
    # spelling of "guess what the operator meant". The renderer already shows the
    # original text, so nothing is lost but a fetcher-shaped reinterpretation.
    ("https:/preview.test/x", "blocked-host"),
    # A TAB, CR OR LF ANYWHERE in the target is refused (security S-3): ``urlsplit``
    # strips them (WHATWG), so the guard saw ``preview.test``, but the fetcher's
    # collapsed-scheme repair saw ``https:`` + ``\t//...`` and built a URL with no
    # host — a retryable error whose Retry could never work. Refusing the raw
    # spelling makes the guard and the fetcher read one string the same way.
    ("https:\t//preview.test/x", "blocked-host"),
    ("https://pre\nview.test/x", "blocked-host"),
    # A host over DNS's 253-octet total, though every label is a legal 63 (S-4):
    # the per-label IDNA check passes it, so only a total-length check keeps it
    # from costing a lookup.
    ("https://" + ".".join(["a" * 63] * 5) + "/x", "blocked-host"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("url", "reason"), _REFUSED_TARGETS)
async def test_a_non_absolute_url_is_a_coded_absence_with_no_lookup_or_fetch(
    url: str, reason: str, monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    looked_up = _spy_dns(monkeypatch)
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=_html("")))
    async with _client() as client:
        response = await _post(client, {"url": url, "index": 0, "expected_account_id": ACCOUNT})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"kind": "absent", "reason": reason}
    assert looked_up == []
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "lookups"),
    [
        # The control: an ordinary absolute https URL is still looked up and fetched.
        ("https://preview.test/page", ["preview.test"]),
        # A non-ASCII path is a real URL, not a parse error: it is fetched.
        ("https://preview.test/caf\u00e9/notes", ["preview.test"]),
        # A public IP literal is a real host; it is judged directly, with no DNS.
        ("http://93.184.216.34/page", []),
        # A Unicode (IDN) host is looked up AS the punycode the request will carry
        # (QA Q2): it used to be looked up in Unicode and then crash encoding the
        # Host header, so it could never preview.
        ("https://caf\u00e9.test/p", ["xn--caf-dma.test"]),
    ],
)
async def test_an_absolute_http_url_with_a_host_is_still_resolved(
    url: str, lookups: list[str], monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    looked_up = _spy_dns(monkeypatch)
    seen = _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=_html("<title>Fixture</title>"),
        ),
    )
    async with _client() as client:
        response = await _post(client, {"url": url, "index": 0, "expected_account_id": ACCOUNT})
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["kind"] == "ready", result
    assert result["preview"]["title"] == "Fixture"
    assert looked_up == lookups
    assert len(seen) >= 1


#: Absolute http(s) URLs that pass the preview's own spelling guard but that no
#: request can be built from: httpx (the parser the request is built with) refuses
#: the host, or a character in the URL. Each used to escape every handler as
#: ``httpx.InvalidURL`` — a 500 the UI maps to a Retry that can never succeed
#: (security S-1). The fetcher now refuses them as ``PolicyRefusal("host")`` BEFORE
#: the lookup, so each is a coded absence that cost no DNS and no request.
_UNBUILDABLE_TARGETS = [
    "https://ex\u00adample.com/",  # a soft hyphen inside the host
    "https://\uff25\uff38\uff21\uff2d\uff30\uff2c\uff25.com/",  # fullwidth letters
    "http://0177.0.0.1/",  # octal IPv4: platforms disagree on what it means
    "https://preview.test/page#x\x00",  # a NUL in the fragment
    # An A-label httpx cannot decode passed the guard, cost a lookup and then
    # raised at request time — a 500 (security S-1a); a port httpx accepts but
    # no socket can use is the same class (review R2-1).
    "https://xn--zz.preview.test/",
    "https://preview.test:99999/",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("url", _UNBUILDABLE_TARGETS)
async def test_a_url_no_request_can_be_built_from_is_blocked_host_with_no_lookup(
    url: str, monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    looked_up = _spy_dns(monkeypatch)
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=_html("")))
    async with _client() as client:
        response = await _post(client, {"url": url, "index": 0, "expected_account_id": ACCOUNT})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert looked_up == []
    assert seen == []


@pytest.mark.asyncio
async def test_an_idn_host_carries_its_punycode_in_the_host_header(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The name that is looked up, sent as ``Host`` and used for SNI is ONE name.

    HTTP and TLS carry ASCII only, so the Unicode spelling could never be sent;
    and resolving one spelling (Python's IDNA 2003 codec) while presenting another
    (httpx's IDNA 2008) would pin the socket to one name's address and verify the
    certificate against a different name.
    """
    _spy_dns(monkeypatch)
    hosts: list[tuple[str | None, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append((request.headers.get("host"), request.extensions.get("sni_hostname")))
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=_html("<title>Café</title>")
        )

    _route_transport(monkeypatch, handler)
    async with _client() as client:
        response = await _post(
            client,
            {"url": "https://caf\u00e9.test/p", "index": 0, "expected_account_id": ACCOUNT},
        )
    assert response.json()["result"]["kind"] == "ready"
    assert hosts == [("xn--caf-dma.test", "xn--caf-dma.test")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decoration",
    [
        '<link rel="icon" href="https://a..b/x.ico">',
        '<meta property="og:image" content="https://./x.png">',
        '<meta property="og:image" content="https://ex\u00adample.com/x.png">',
        # A port or an A-label the request parser cannot use (R2-1, S-1d): these
        # used to raise past ``_store_artifact`` as a bare ``ValueError``.
        '<link rel="icon" href="https://public.test:99999/x.ico">',
        '<link rel="icon" href="https://public.test:65536/x.ico">',
        '<meta property="og:image" content="https://xn--zz.public.test/x.png">',
    ],
)
async def test_a_malformed_icon_or_image_host_drops_the_decoration_not_the_card(
    decoration: str, monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The page author writes the icon and image URLs, so they are hostile input.

    A host the resolver cannot even encode (``a..b``, ``.``) or that no request can
    be built from (a soft hyphen) used to raise past ``_store_artifact`` and turn
    every reader's card for that page into a 500 (review R1-1, security S-1). It
    is now the fetcher's own ``PolicyRefusal``, which the sub-fetch already drops.
    """
    _spy_dns(monkeypatch)
    page = _html("<title>Decorated</title>" + decoration)
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=page),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "https://preview.test/page", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["kind"] == "ready", result
    assert result["preview"]["title"] == "Decorated"
    assert "favicon" not in result["preview"]
    assert "image" not in result["preview"]


@pytest.mark.asyncio
async def test_a_redirect_to_a_malformed_host_is_a_cached_blocked_host(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """An unusable host is a DELIBERATELY cached absence (review R1-2, S-2, QA Q1).

    It is the fetcher's ``PolicyRefusal("host")`` now — the same class as a name
    that does not exist — so it takes the same 30-minute negative-cache path: the
    spelling is a property of the origin's redirect, and asking again changes
    nothing. A second resolve is a cache hit that costs the origin no request.
    """
    _spy_dns(monkeypatch)
    seen = _route_transport(
        monkeypatch,
        lambda request: httpx.Response(302, headers={"location": "https://a..b/next"}),
    )
    body = {"url": "https://preview.test/moved", "index": 0, "expected_account_id": ACCOUNT}
    async with _client() as client:
        first = await _post(client, body)
        second = await _post(client, body)
    assert first.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert second.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert seen == ["/moved"]


#: ``Location`` values that used to escape as a 500 or as a retryable error the
#: card's Retry could never fix (review R2-1, security S-1b/c, QA Q3). The origin
#: writes the header, so each must be a coded ``blocked-host`` after ONE request
#: and no lookup of the unusable hop. Raw bytes: what a real origin sends.
_UNUSABLE_LOCATIONS = [
    b"https://u:[p]@preview.test/x",
    b"https://public.test:99999/x",
    b"https://public.test:65536/x",
    b"https://xn--zz.public.test/x",
    b"http://[::1/x",
    b"http://0177.0.0.1/x",
    b"https:\\\\evil.test/x",
    b"https://ex\xc2\xadample.test/x",
    b"https://pre\tview.test/x",
    b"https://preview.test/a\x00b",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("location", _UNUSABLE_LOCATIONS)
async def test_an_unusable_redirect_location_is_blocked_host_after_one_request(
    location: bytes, monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    looked_up = _spy_dns(monkeypatch)
    seen = _route_transport(
        monkeypatch, lambda request: httpx.Response(302, headers=[(b"location", location)])
    )
    async with _client() as client:
        response = await _post(
            client,
            {"url": "https://preview.test/moved", "index": 0, "expected_account_id": ACCOUNT},
        )
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert seen == ["/moved"]
    assert looked_up == ["preview.test"]


@pytest.mark.asyncio
async def test_the_shared_fetcher_still_absolutises_what_a_preview_refuses(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The scheme prepend STAYS in the shared fetcher; only the preview refuses.

    ``normalize_url`` is shared by ``fetch``, ``fetch_bytes`` and the ``web_fetch``
    tool, where a model genuinely means ``example.com`` and the prepend is the
    documented repair. Rewriting it would change those callers instead, so this
    test pins the boundary: the same string is absolutised by the fetcher and
    refused by the preview, in one place where a future "simplification" that
    moved the rule into ``normalize_url`` would go red.
    """
    from local_operator.web_fetch.service import normalize_url

    assert normalize_url("README.md") == "https://README.md"
    looked_up = _spy_dns(monkeypatch)
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=_html("")))
    async with _client() as client:
        response = await _post(
            client, {"url": "README.md", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "absent", "reason": "unsupported-scheme"}
    assert looked_up == []
    assert seen == []


@pytest.mark.asyncio
async def test_a_private_ip_literal_is_refused_without_a_fetch(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    looked_up = _spy_dns(monkeypatch)
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=b""))
    async with _client() as client:
        response = await _post(
            client, {"url": "http://10.0.0.7/admin", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.status_code == 200
    assert response.json()["result"] == {"kind": "absent", "reason": "local-or-private"}
    assert looked_up == []
    assert seen == []


@pytest.mark.asyncio
async def test_a_loopback_target_is_refused_as_local_or_private(signed_in: None) -> None:
    async with _client() as client:
        response = await _post(
            client,
            {"url": "http://127.0.0.1:9/secrets", "index": 0, "expected_account_id": ACCOUNT},
        )
    assert response.status_code == 200
    assert response.json()["result"] == {"kind": "absent", "reason": "local-or-private"}


@pytest.mark.asyncio
async def test_a_redirect_to_a_private_address_is_re_validated_at_the_hop(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The public URL is followed; its destination is judged BEFORE we connect."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/page", "the private hop must never be requested"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/steal"})

    seen = _route_transport(monkeypatch, handler)
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/page", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "absent", "reason": "local-or-private"}
    assert seen == ["/page"]


@pytest.mark.asyncio
async def test_credentials_in_the_url_are_refused_before_a_single_request(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """URL userinfo would become ``Authorization: Basic`` on the wire.

    httpx turns it into a header, so honouring such a URL would send a credential
    the MESSAGE AUTHOR chose to a third party — in a request this process signs.
    Refused by the address guard, before any lookup or connect, which the empty
    request log is what proves.
    """
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        response = await _post(client, _resolve_body("http://someone:hunter2@preview.test/page"))
    assert response.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert seen == []


@pytest.mark.asyncio
async def test_a_redirect_that_introduces_credentials_is_refused_at_the_hop(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The hop target is judged BEFORE we connect, exactly like a private address."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/page", "the credentialed hop must never be requested"
        return httpx.Response(302, headers={"location": "http://someone:hunter2@preview.test/deep"})

    seen = _route_transport(monkeypatch, handler)
    async with _client() as client:
        response = await _post(client, _resolve_body())
    assert response.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert seen == ["/page"]


@pytest.mark.asyncio
async def test_a_declared_oversize_page_is_too_large(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            content=b"<html></html>",
            headers={"content-type": "text/html", "content-length": str(9 * 1024 * 1024)},
        ),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/big", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "absent", "reason": "too-large"}


@pytest.mark.asyncio
async def test_a_non_html_declared_type_is_refused(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}
        ),
    )
    async with _client() as client:
        response = await _post(
            client,
            {"url": "http://preview.test/doc.pdf", "index": 0, "expected_account_id": ACCOUNT},
        )
    assert response.json()["result"] == {"kind": "absent", "reason": "content-type"}


@pytest.mark.asyncio
async def test_markup_that_carries_none_is_not_html(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=b"\x00\x01binary", headers={"content-type": "text/html"}
        ),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/bin", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "absent", "reason": "not-html"}


@pytest.mark.asyncio
async def test_html_with_nothing_to_show_is_no_metadata(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            content=b"<html><head></head><body>x</body></html>",
            headers={"content-type": "text/html"},
        ),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/empty", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "absent", "reason": "no-metadata"}


@pytest.mark.asyncio
async def test_an_svg_og_image_is_rejected_and_the_card_still_ships(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """An SVG is a document: it is never stored, never served from our origin."""

    page = _html(
        "<title>Notes</title>"
        '<meta property="og:image" content="https://preview.test/assets/og.svg">'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".svg"):
            # Labelled as a PNG, deliberately: Content-Type is never trusted.
            return httpx.Response(200, content=SVG, headers={"content-type": "image/png"})
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    _route_transport(monkeypatch, handler)
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/page", "index": 0, "expected_account_id": ACCOUNT}
        )
    result = response.json()["result"]
    assert result["kind"] == "ready"
    assert result["preview"]["title"] == "Notes"
    assert "image" not in result["preview"]


@pytest.mark.asyncio
async def test_an_absent_display_string_is_absent_from_the_json_not_empty(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The renderer's contract says ``string | null``, so a missing one is a missing key."""
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            content=_html("<title>Only a title</title>"),
            headers={"content-type": "text/html"},
        ),
    )
    async with _client() as client:
        response = await _post(client, _resolve_body())
    preview = response.json()["result"]["preview"]
    assert preview["title"] == "Only a title"
    assert "description" not in preview
    assert "siteName" not in preview


@pytest.mark.asyncio
async def test_an_oversize_image_is_dropped_not_truncated(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    page = _html(
        "<title>Big</title>" '<meta property="og:image" content="http://preview.test/big.png">'
    )
    huge = PNG + b"\x00" * (preview_service.IMAGE_MAX_BYTES + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/big.png":
            return httpx.Response(200, content=huge, headers={"content-type": "image/png"})
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    _route_transport(monkeypatch, handler)
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/page", "index": 0, "expected_account_id": ACCOUNT}
        )
    result = response.json()["result"]
    assert result["kind"] == "ready"
    assert "image" not in result["preview"]


@pytest.mark.asyncio
async def test_the_total_deadline_answers_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A REAL slow-drip origin cannot outlast the budget.

    MockTransport cannot prove this: timeouts are enforced by the transport, so a
    handler that sleeps forever simply blocks the mock. This runs a real socket
    that sends headers at once and then drips a byte every 100 ms — the shape a
    hostile origin uses to hold a promise open — against the real httpx transport.
    """
    server, base = _drip_origin()
    try:
        monkeypatch.setattr(preview_service, "TOTAL_DEADLINE_S", 0.6)
        monkeypatch.setattr(preview_service, "PAGE_DEADLINE_S", 0.5)
        # THE ADDRESS POLICY IS PROVEN ELSEWHERE and it stays in force on the
        # preview path (that is the whole point of forcing allow_private off);
        # this test measures the DEADLINE, so the one thing it patches away is
        # the refusal of its own fixture address. Everything else — real socket,
        # real drip, real transport — is untouched.
        monkeypatch.setattr(fetch_service, "validate_public_url", lambda url, **_kw: None)
        settings = WebFetchSettings(enrich=False, timeout_seconds=10.0, max_attempts=1)
        started = time.monotonic()
        resolution = await preview_service.resolve_preview(
            url=f"{base}/drip",
            index=0,
            scope=PreviewScope(account_id=ACCOUNT),
            settings=settings,
            store=PreviewStore(tmp_path / "artifacts"),
            cache=PreviewCache(),
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
    assert elapsed < 3.0, f"the resolution outlived its budget: {elapsed:.2f}s"
    assert resolution.kind == "error"
    assert isinstance(resolution.retryable, bool)


@pytest.mark.asyncio
async def test_an_expired_budget_makes_no_request_at_all(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    monkeypatch.setattr(preview_service, "TOTAL_DEADLINE_S", -1.0)
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=b""))
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/x", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "error", "retryable": True}
    assert seen == []


@pytest.mark.asyncio
async def test_an_upstream_failure_is_a_retryable_error(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(503, content=b"down", headers={"content-type": "text/html"}),
    )
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/down", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.json()["result"] == {"kind": "error", "retryable": True}


# ---------------------------------------------------------------------------
# The gate: who may ask, and what an id may be
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_bearer_is_refused_by_the_plane(signed_in: None) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            RESOLVE,
            json={"url": "http://preview.test/x", "index": 0, "expected_account_id": ACCOUNT},
        )
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_another_accounts_preview_is_refused_before_any_fetch(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    seen = _route_transport(monkeypatch, lambda request: httpx.Response(200, content=b""))
    async with _client() as client:
        response = await _post(
            client,
            {
                "url": "http://preview.test/page",
                "index": 0,
                "expected_account_id": "acc_someoneelse",
            },
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert seen == []


@pytest.mark.asyncio
async def test_a_malformed_account_id_is_a_shape_error(signed_in: None) -> None:
    async with _client() as client:
        response = await _post(
            client,
            {"url": "http://preview.test/page", "index": 0, "expected_account_id": "../../etc"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_an_unsigned_install_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.team_bridge import read_context
    from local_operator.team_bridge.client import BridgeSignInRequired

    def _refuse(_config_dir):
        class _Wire:
            async def current_access(self):
                raise BridgeSignInRequired("sign in")

        return _Wire()

    monkeypatch.setattr(read_context, "wire_for", _refuse)
    async with _client() as client:
        response = await _post(
            client, {"url": "http://preview.test/page", "index": 0, "expected_account_id": ACCOUNT}
        )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "not_signed_in"


@pytest.mark.asyncio
async def test_an_extra_body_field_is_refused(signed_in: None) -> None:
    async with _client() as client:
        response = await _post(
            client,
            {
                "url": "http://preview.test/page",
                "index": 0,
                "expected_account_id": ACCOUNT,
                "sender": "someone",
            },
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_a_malformed_artifact_id_never_reaches_the_filesystem(
    signed_in: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each shape answers for the RIGHT reason, and no store path is ever built.

    The earlier version accepted 404 *or* 422 for every case, which cannot tell the
    route's shape gate from the store's refusal — an id that reached
    ``PreviewStore._path`` would have passed it. Two assertions per case do the job:
    the exact status, and a spy proving ``_path`` (the only place an id becomes a
    filename) was never called. The spy raises, so a regression fails on the id that
    caused it rather than on a missing blob.
    """
    reached: list[str] = []

    def _spy(self: PreviewStore, artifact_id: str) -> None:
        reached.append(artifact_id)
        raise AssertionError(f"artifact id reached the filesystem: {artifact_id!r}")

    monkeypatch.setattr(PreviewStore, "_path", _spy)
    cases = {
        # Not the grammar at all: the path parameter is shaped, so FastAPI answers
        # before a handler runs.
        "not-a-hash": 422,
        # An encoded traversal is decoded to a path the router does not match at
        # all, so it is a 404 from the router rather than a 422 from the shape —
        # either way it never reaches this handler.
        "..%2F..%2Fetc%2Fpasswd": 404,
        "A" * 32: 422,  # uppercase is not the pattern
        "0" * 31: 422,  # too short
        "0" * 33: 422,  # too long
        "0" * 64: 422,
        # WELL-SHAPED, and not an id this scope owns: the route's own refusal.
        "0" * 32: 404,
    }
    async with _client() as client:
        for artifact_id, expected in cases.items():
            response = await client.get(ARTIFACT + "/" + artifact_id, headers=_receipt())
            assert response.status_code == expected, (
                artifact_id,
                response.status_code,
                response.text,
            )
    assert reached == []


@pytest.mark.asyncio
async def test_the_account_receipt_is_not_accepted_in_the_query_string(signed_in: None) -> None:
    """S-3: the receipt is a header, so the old query spelling must not work.

    A GET's URL is what every intermediary records, so the account id may not travel
    there. This pins the move in the direction that would otherwise regress
    silently: the route ignores the query string and the header is what it reads.
    """
    async with _client() as client:
        in_query_only = await client.get(
            ARTIFACT + "/" + "a" * 32 + f"?expected_account_id={ACCOUNT}"
        )
        in_header = await client.get(ARTIFACT + "/" + "a" * 32, headers=_receipt())
    assert in_query_only.status_code == 422
    assert in_header.status_code == 404


@pytest.mark.asyncio
async def test_an_unknown_artifact_id_is_a_404(signed_in: None) -> None:
    async with _client() as client:
        response = await client.get(_artifact_path("a" * 32), headers=_receipt())
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "artifact_not_found"


# ---------------------------------------------------------------------------
# Nothing that could identify a reading ends up in a log line
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_log_line_carries_the_url_host_or_query(
    monkeypatch: pytest.MonkeyPatch, signed_in: None, caplog: pytest.LogCaptureFixture
) -> None:
    """A correspondent's link must not survive in a log file.

    The query string is the sharp end: it routinely carries a capability token,
    so a debug line that quotes the URL leaks something the message never showed.
    """
    caplog.set_level("DEBUG", logger="local_operator")
    _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            content=_html("<title>Logged</title>"),
            headers={"content-type": "text/html"},
        ),
    )
    async with _client() as client:
        await _post(
            client,
            {
                "url": "http://preview.test/zzqq-path?zzqqquery=1",
                "index": 0,
                "expected_account_id": ACCOUNT,
            },
        )
    # Scoped to OUR loggers on purpose: httpx's own request line ("HTTP Request:
    # GET <url>") is a library record that ``logger.quiet_wire_clients`` pins shut
    # at the level the daemon runs at (LOG_LEVEL defaults to WARNING), and it is
    # the same line every other outbound request in this process already writes.
    # What must never carry the URL is the code this feature owns.
    text = "\n".join(
        record.getMessage() for record in caplog.records if record.name.startswith("local_operator")
    )
    assert "zzqq-path" not in text
    assert "zzqqquery" not in text
    assert "preview.test" not in text


# ---------------------------------------------------------------------------
# The pieces under the service: extraction, sniffing, storage
# ---------------------------------------------------------------------------


def test_sniffing_rejects_a_document_even_when_it_is_labelled_an_image() -> None:
    assert extract.sniff_raster(SVG) is None
    assert extract.sniff_raster(b"<html><title>x</title>") is None
    assert extract.sniff_raster(b"") is None


@pytest.mark.parametrize(
    ("body", "mime", "size"),
    [
        (PNG, "image/png", (32, 16)),
        (GIF, "image/gif", (3, 7)),
        (
            # VP8X extended: 24-bit canvas dimensions minus one.
            b"RIFF"
            + (30).to_bytes(4, "little")
            + b"WEBP"
            + b"VP8X"
            + b"\x00" * 8
            + (99).to_bytes(3, "little")
            + (49).to_bytes(3, "little"),
            "image/webp",
            (100, 50),
        ),
        (
            # VP8X at 1x1: the minimum a real file carries.
            b"RIFF"
            + (30).to_bytes(4, "little")
            + b"WEBP"
            + b"VP8X"
            + b"\x00" * 8
            + b"\x00\x00\x00"
            + b"\x00\x00\x00",
            "image/webp",
            (1, 1),
        ),
    ],
)
def test_dimensions_come_from_the_header_never_from_decoding(
    body: bytes, mime: str, size: tuple[int, int]
) -> None:
    raster = extract.sniff_raster(body)
    assert raster is not None
    assert raster.mime == mime
    assert (raster.width, raster.height) == size


def test_a_jpeg_size_is_read_from_its_start_of_frame_segment() -> None:
    # SOI, APP0 (len 16), SOF0 (len 17, precision 8, height 200, width 300).
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    sof0 = (
        b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + (200).to_bytes(2, "big")
        + (300).to_bytes(2, "big")
        + b"\x03"
        + b"\x00" * 9
    )
    raster = extract.sniff_raster(b"\xff\xd8\xff" + app0 + sof0 + b"\xff\xd9")
    assert raster is not None
    assert raster.mime == "image/jpeg"
    assert (raster.width, raster.height) == (300, 200)


def test_the_parser_is_bounded_by_its_tag_budget() -> None:
    """The bound is DETECTABLE: a marker past the budget must not be collected.

    The earlier version asserted only that a hostile document did not raise, which a
    parser with no bound at all also satisfies. Here the same ``og:title`` appears
    once inside the budget (and IS collected) and once past ``_MAX_TAGS`` (and is
    not): the pair proves the walk stops where the constant says it does.
    """
    inside = '<meta property="og:title" content="Collected">'
    past = '<meta property="og:title" content="Should not be collected">'
    filler = "<div></div>" * (extract._MAX_TAGS + 50)
    metadata = extract.extract_metadata(
        f"<html><head>{inside}</head><body>{filler}{past}</body></html>",
        base_url="http://preview.test/",
    )
    assert metadata.title == "Collected"


class _Audited(bytes):
    """Bytes that record the furthest byte index anyone read from them.

    A WALL-CLOCK ASSERTION CANNOT FAIL HERE. Code review round 2 measured the cap's
    own cost at 0.07-0.2 s against a 0.3 s bound, so "it was fast" passed even when
    the scan read the whole 2 MiB body. This counts the reads instead, so the test
    fails the moment the bound is removed however fast the machine is.
    """

    def __new__(cls, data: bytes) -> "_Audited":
        return super().__new__(cls, data)

    def __init__(self, data: bytes) -> None:
        self.furthest = -1

    # The overloads mirror ``bytes.__getitem__`` exactly (``SupportsIndex`` and
    # ``slice``), so this stays a real subclass of the type the sniffer is written
    # against rather than a duck type that the guard could not run on.
    @overload
    def __getitem__(self, item: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, item: slice) -> bytes: ...

    def __getitem__(self, item: SupportsIndex | slice) -> int | bytes:
        if isinstance(item, slice):
            self.furthest = max(self.furthest, item.stop if item.stop is not None else len(self))
        else:
            index = operator.index(item)
            self.furthest = max(self.furthest, index if index >= 0 else len(self) + index)
        return super().__getitem__(item)


def _jpeg_with_sof_at(offset: int) -> bytes:
    """A JPEG whose SOF header sits at ``offset`` bytes, with filler before it.

    The filler is ``FF D0`` restart markers: a legal two-byte sequence with no
    length field, so the walk steps over it two bytes at a time — the exact shape
    that made the unbounded version slow, and the only way to place a header past a
    64 KiB segment-length field.
    """
    units = max(0, (offset - 3) // 2)
    sof0 = (
        b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + (200).to_bytes(2, "big")
        + (300).to_bytes(2, "big")
        + b"\x03"
        + b"\x00" * 9
    )
    return b"\xff\xd8\xff" + b"\xff\xd0" * units + sof0 + b"\xff\xd9"


def test_the_jpeg_walk_reads_no_further_than_its_cap() -> None:
    """BYTES, not seconds: the hostile body is never walked past the bound.

    The reviewer's exact body — ``FF D8 FF`` then repeated ``FF D0``, no SOF ever.
    Bounded by :data:`extract._JPEG_SOF_SCAN_BYTES` because the walk is a Python loop
    over attacker-supplied bytes (2 MiB of this measured 1.75 s before the cap).
    Dimensions are optional in the contract, so "unknown" is the right answer.
    """
    hostile = _Audited(b"\xff\xd8\xff" + b"\xff\xd0" * (1024 * 1024))
    raster = extract.sniff_raster(hostile)
    assert raster is not None and raster.mime == "image/jpeg"
    assert (raster.width, raster.height) == (None, None)
    assert hostile.furthest <= extract._JPEG_SOF_SCAN_BYTES, hostile.furthest
    assert hostile.furthest < len(hostile), "the walk read the whole body"


def test_a_sof_past_the_cap_reports_unknown_and_a_sof_inside_it_reports_its_size() -> None:
    """The pair that makes the cap DETECTABLE: same header, either side of the bound.

    If the cap were removed (or raised past the second offset), the far case would
    report 300x200 and this test would fail — which the wall-clock version could not
    do. The far case is the documented EXIF limitation: unknown, never wrong.
    """
    near = extract.sniff_raster(_jpeg_with_sof_at(1024))
    far_offset = extract._JPEG_SOF_SCAN_BYTES + 4096
    far = extract.sniff_raster(_jpeg_with_sof_at(far_offset))
    assert near is not None and (near.width, near.height) == (300, 200)
    assert far is not None and far.mime == "image/jpeg"
    assert (far.width, far.height) == (None, None)


def test_a_relative_icon_resolves_against_the_served_url() -> None:
    markup = '<html><head><link rel="apple-touch-icon" href="./icon.png"></head></html>'
    metadata = extract.extract_metadata(markup, base_url="https://example.com/a/b")
    assert metadata.icon_url == "https://example.com/a/icon.png"


def test_the_store_is_scope_addressed_and_rejects_a_foreign_id(tmp_path) -> None:
    store = PreviewStore(tmp_path / "artifacts")
    first = store.put(PNG, artifact_id="a" * 32)
    assert first == "a" * 32
    assert store.get(first) == PNG
    assert store.size_of(first) == len(PNG)
    # The id names a SLOT (this scope, this page, this kind), not a body: a card
    # refetched after its TTL must show the bytes just fetched, not the old ones.
    store.put(GIF, artifact_id="a" * 32)
    assert store.get(first) == GIF
    with pytest.raises(InvalidArtifactId):
        store.get("../../etc/passwd")
    with pytest.raises(InvalidArtifactId):
        store.get("A" * 32)
    with pytest.raises(InvalidArtifactId):
        store.put(PNG, artifact_id="../../etc/passwd")


def test_the_store_sweeps_least_recently_used_first(tmp_path) -> None:
    store = PreviewStore(tmp_path / "artifacts", max_files=4, max_bytes=10_000_000)
    ids = []
    for index in range(4):
        body = PNG + bytes([index])
        ids.append(store.put(body, artifact_id=f"{index:032x}"))
        # Age each blob deliberately: mtime is the LRU key, and the filesystem's
        # own resolution would otherwise make them all look simultaneous.
        target = store.root / ids[-1]
        stamp = target.stat().st_mtime - (100 - index * 10)
        import os

        os.utime(target, (stamp, stamp))
    # Touch the oldest so it is no longer the coldest.
    store.get(ids[0])
    store.put(PNG + b"fresh", artifact_id="f" * 32)
    survivors = {path.name for path in store.root.iterdir()}
    assert len(survivors) <= 4
    assert ids[0] in survivors, "the recently-read blob was evicted"
    assert ids[1] not in survivors, "the least recently used blob survived"


def test_policy_refusals_carry_a_machine_reason() -> None:
    for url, reason in (
        ("file:///etc/passwd", "scheme"),
        ("http://127.0.0.1/", "private"),
        ("http://10.0.0.1/", "private"),
        # Credentials in the URL are a ``host`` refusal: never forwarded as an
        # ``Authorization`` header, and refused before any lookup.
        ("http://someone:hunter2@example.com/x", "host"),
    ):
        with pytest.raises(PolicyRefusal) as caught:
            fetch_service.validate_public_url(url, allow_private=False)
        assert caught.value.reason == reason
        # Still a FetchError, so every existing catcher is unchanged.
        assert isinstance(caught.value, fetch_service.FetchError)


@pytest.mark.asyncio
async def test_fetch_bytes_returns_the_body_undecoded() -> None:
    """The whole reason this entry point exists: bytes survive where text does not."""
    payload = "<title>Café</title>".encode("iso-8859-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=payload,
            headers={"content-type": "text/html; charset=iso-8859-1"},
        )

    service = WebFetchService(
        WebFetchSettings(enrich=False), transport=httpx.MockTransport(handler)
    )
    fetched = await service.fetch_bytes(
        "http://preview.test/page", timeout_seconds=5, max_bytes=4096
    )
    assert fetched.body == payload
    assert fetched.headers["content-type"] == "text/html; charset=iso-8859-1"
    assert fetched.complete is True


# ---------------------------------------------------------------------------
# The cache: scoped to an account, TTL'd, shared in flight, bounded
# ---------------------------------------------------------------------------

CARD_HEAD = (
    "<title>Fallback title</title>"
    '<meta property="og:title" content="A card">'
    '<link rel="icon" href="/favicon.png">'
)


def _card_handler(page: bytes) -> Callable[[httpx.Request], httpx.Response]:
    """A page with a favicon, and nothing else on the origin."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/favicon.png":
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
        return httpx.Response(200, content=page, headers={"content-type": "text/html"})

    return handler


def _resolve_body(
    url: str = "http://preview.test/page", account_id: str = ACCOUNT
) -> dict[str, Any]:
    return {"url": url, "index": 0, "expected_account_id": account_id}


@pytest.mark.asyncio
async def test_the_cache_reports_the_bytes_it_holds(
    monkeypatch: pytest.MonkeyPatch,
    signed_in: None,
    preview_cache_for_test: PreviewCache,
) -> None:
    """The "resident size reported" requirement, asserted rather than asserted-about."""

    _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        first = await _post(client, _resolve_body())
        assert first.json()["result"]["kind"] == "ready", first.text
        stats = preview_cache_for_test.stats()
        assert stats.entries == 1
        assert stats.resolutions == 1
        assert stats.resident_bytes == len(PNG)

        # A second resolve is a HIT: no new resolution, no new bytes.
        second = await _post(client, _resolve_body())
        assert second.json()["result"] == first.json()["result"]
        stats = preview_cache_for_test.stats()
        assert stats.hits == 1
        assert stats.resolutions == 1
        # The bytes are held ONCE, and a HIT must not add or drop a byte: this is
        # the post-hit half of the resident-size assertion (review R2-7 restored it
        # after the previous delta dropped it).
        assert stats.resident_bytes == len(PNG)


def _write_half_written_owner_row(config_dir: Path) -> None:
    """A committed enrollment row as a crashed writer leaves it: an int column NULL.

    Written straight to the store's own path with NO ``NOT NULL`` on
    ``credential_id``, because that is the shape that reaches the decoder:
    ``_owner_from_row`` does ``int(row["credential_id"])``, and ``int(None)``
    raises ``TypeError`` out of ``active_owner_readonly`` (which only wraps what
    SQLite itself raises, as ``StoreRefused``). Reviewer R2-2 measured that as a
    500 on both routes; the scope is supposed to degrade to the account alone.
    """
    path = Path(config_dir) / "team_bridge" / "state.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE enrollments (owner_key TEXT PRIMARY KEY, credential_id INTEGER, "
            "account_id TEXT, tenant_id TEXT, endpoint_id TEXT, canonical_base_url TEXT, "
            "enrollment_generation INTEGER, consent_version TEXT, consent_ref TEXT, "
            "designated_session TEXT, state TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO enrollments VALUES (?,?,?,?,?,?,?,?,?,?,'committed')",
            (
                "half-written-owner",
                None,
                ACCOUNT,
                "tenant-partial",
                "endpoint-partial",
                "https://hub.preview.test",
                1,
                None,
                None,
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_a_half_written_owner_row_cannot_500_either_route(
    monkeypatch: pytest.MonkeyPatch, signed_in: None, _plane: ConfigManager
) -> None:
    """A damaged owner row narrows the scope to the account; it never fails a route.

    The tenant is the SECOND fence, so losing it must leave the account fence
    intact — resolve still answers its normal resolution and artifact still
    answers 404 for an id it holds no blob for. Reviewer R2-2 reproduced 500s on
    BOTH before this: the ``sqlite3.Error`` arm caught nothing (the store already
    wraps those) while the ``TypeError`` the real corruption raises escaped.
    """
    _write_half_written_owner_row(Path(_plane.config_dir))
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        resolved = await _post(client, _resolve_body())
        artifact = await client.get(_artifact_path("a" * 32), headers=_receipt())
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["result"]["kind"] == "ready", resolved.text
    # The fetch really ran: the degraded scope fetched instead of short-circuiting.
    assert seen == ["/page", "/favicon.png"]
    assert artifact.status_code == 404, artifact.text


@pytest.mark.asyncio
async def test_a_credentialed_url_is_refused_even_when_the_plain_card_is_fresh(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The refusal runs on the RAW url, BEFORE the key is built (QA Q2-1).

    ``canonical_url`` drops the userinfo pair so a credential never enters a
    long-lived key, which makes the credentialed spelling and the plain one ONE
    key. With the refusal only in the fetcher, a cache HIT never reaches the
    fetcher at all — so a credentialed URL was answered from the plain spelling's
    fresh card. Measured here: the card is cached FIRST, and the refusal must
    still be a refusal, with no request at all.
    """

    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        plain = await _post(client, _resolve_body())
        assert plain.json()["result"]["kind"] == "ready", plain.text
        fetched = len(seen)

        credentialed = await _post(
            client, _resolve_body("http://someone:hunter2@preview.test/page")
        )
    assert credentialed.status_code == 200, credentialed.text
    assert credentialed.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
    assert len(seen) == fetched, "the refusal must cost no request at all"


@pytest.mark.asyncio
async def test_a_credentialed_refusal_does_not_poison_the_plain_spelling(
    monkeypatch: pytest.MonkeyPatch, signed_in: None, preview_cache_for_test: PreviewCache
) -> None:
    """The refusal must not answer for the URL others are entitled to (QA Q2-1).

    Because the two spellings share one canonical key, a ``blocked-host`` written
    under it would be served to the plain URL as a negative answer for the whole
    ``NEGATIVE_TTL_S`` with no fetch — one message's malformed link denying every
    later reader of the same page its card. The refusal returns BEFORE the cache,
    so nothing is stored; the plain URL then fetches for itself.
    """

    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        credentialed = await _post(
            client, _resolve_body("http://someone:hunter2@preview.test/page")
        )
        assert credentialed.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
        assert seen == [], "the refusal must not reach the origin"

        plain = await _post(client, _resolve_body())
        assert plain.json()["result"]["kind"] == "ready", plain.text
    # One entry, written by the plain resolve: the refusal stored nothing.
    assert preview_cache_for_test.stats().entries == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attack", "victim"),
    [
        ("http://preview.test/page#\x01", "http://preview.test/page"),
        ("http://preview.test/page#x\x00", "http://preview.test/page"),
        ("http://preview.test/page#\x7f", "http://preview.test/page"),
        ("http://preview.test/page?a=\x01", "http://preview.test/page?a=%01"),
        ("http://preview.test/page?b=\x01&a=1", "http://preview.test/page?a=1&b=%01"),
    ],
)
async def test_an_unrequestable_spelling_does_not_poison_its_canonical_twin(
    attack: str,
    victim: str,
    monkeypatch: pytest.MonkeyPatch,
    signed_in: None,
    preview_cache_for_test: PreviewCache,
) -> None:
    """A refusal decided from the string alone is never cached (security S-5).

    The key drops the fragment and re-encodes the query, so ``page#\x01`` and
    ``page`` are ONE key. A cached ``blocked-host`` for the unrequestable spelling
    would deny the legitimate URL its card for ``NEGATIVE_TTL_S`` — the same class
    as the credential refusal (Q2-1), closed the same way: refused on the raw
    target, before the key exists.
    """
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        attacked = await _post(client, _resolve_body(attack))
        assert attacked.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
        assert seen == [], "the refusal must not reach the origin"

        legitimate = await _post(client, _resolve_body(victim))
        assert legitimate.json()["result"]["kind"] == "ready", legitimate.text
    assert preview_cache_for_test.stats().entries == 1


@pytest.mark.asyncio
async def test_the_store_write_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """``PreviewStore.put`` is file I/O, so the service hands it to a worker thread.

    The write is a temporary file, an ``os.replace`` and a directory sweep —
    measured at 30-60 ms per artifact — and it sits inside the request path, where
    on the loop it would stall every other renderer's resolve. Asserted by THREAD
    IDENTITY rather than by reading the source: the write must land somewhere
    other than the thread running the app.
    """
    threads: list[int] = []
    real_put = PreviewStore.put

    def _spy(self: PreviewStore, body: bytes, *, artifact_id: str) -> str:
        threads.append(threading.get_ident())
        return real_put(self, body, artifact_id=artifact_id)

    monkeypatch.setattr(PreviewStore, "put", _spy)
    _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    loop_thread = threading.get_ident()
    async with _client() as client:
        response = await _post(client, _resolve_body())
    assert response.json()["result"]["kind"] == "ready", response.text
    assert threads, "the favicon was never stored"
    assert all(ident != loop_thread for ident in threads)


CACHE = "/v1/desktop/team-bridge/link-preview/cache"


@pytest.mark.asyncio
async def test_the_cache_route_is_bearer_gated_like_its_siblings() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get(CACHE)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_the_cache_route_reports_counts_and_bytes_and_names_nothing(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The "resident size is reported" contract line, answered by a real route."""
    _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        resolved = await _post(client, _resolve_body())
        assert resolved.json()["result"]["kind"] == "ready", resolved.text
        response = await client.get(CACHE)
    assert response.status_code == 200
    assert response.json()["result"] == {
        "entries": 1,
        "residentBytes": len(PNG),
        "hits": 0,
        "misses": 1,
        "evictions": 0,
        "resolutions": 1,
    }
    # Counts and bytes only: no URL, host or account id can appear in the answer.
    assert "http" not in response.text
    assert ACCOUNT not in response.text


@pytest.mark.asyncio
async def test_one_accounts_card_is_never_served_into_anothers_session(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """Scope isolation, end to end: a second account re-fetches and cannot read A's id."""

    from local_operator.team_bridge import read_context

    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        as_a = await _post(client, _resolve_body())
        assert as_a.json()["result"]["kind"] == "ready", as_a.text
        a_favicon = as_a.json()["result"]["preview"]["favicon"]["id"]
        assert seen.count("/page") == 1

        # The install is now signed in as B, and B asks for the SAME url.
        monkeypatch.setattr(read_context, "wire_for", _account_wire(OTHER_ACCOUNT))
        as_b = await _post(client, _resolve_body(account_id=OTHER_ACCOUNT))
        assert as_b.json()["result"]["kind"] == "ready", as_b.text
        b_favicon = as_b.json()["result"]["preview"]["favicon"]["id"]

        # A second origin fetch, not A's cached card...
        assert seen.count("/page") == 2
        # ...and a different id, because the id IS the scope's claim on the slot.
        assert b_favicon != a_favicon

        # B asking for A's id gets a 404 — not the bytes, and not a 403 that would
        # confirm the id exists somewhere.
        denied = await client.get(_artifact_path(a_favicon), headers=_receipt(OTHER_ACCOUNT))
        assert denied.status_code == 404, denied.text
        assert denied.json()["detail"]["code"] == "artifact_not_found"

        # B's own id serves, and it is the same public bytes B's own fetch stored.
        allowed = await client.get(_artifact_path(b_favicon), headers=_receipt(OTHER_ACCOUNT))
        assert allowed.status_code == 200, allowed.text
        assert allowed.content == PNG

        # The receipt is still checked before ownership: A's id under A's receipt
        # is refused on the identity, not answered with bytes.
        mismatch = await client.get(_artifact_path(a_favicon), headers=_receipt(ACCOUNT))
        assert mismatch.status_code == 409
        assert mismatch.json()["detail"]["code"] == "account_mismatch"


@pytest.mark.asyncio
async def test_a_repeat_inside_the_ttl_is_free_and_expires_after_it(
    monkeypatch: pytest.MonkeyPatch,
    signed_in: None,
    _clock: _Clock,
) -> None:
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        await _post(client, _resolve_body())
        await _post(client, _resolve_body())
        assert seen.count("/page") == 1, "a fresh card was fetched twice"

        _clock.advance(POSITIVE_TTL_S + 1)
        await _post(client, _resolve_body())
        assert seen.count("/page") == 2, "an expired card was served from the cache"


@pytest.mark.asyncio
async def test_an_absent_answer_is_reused_but_expires_sooner_than_a_card(
    monkeypatch: pytest.MonkeyPatch,
    signed_in: None,
    _clock: _Clock,
) -> None:
    """A page with nothing to draw is a NEGATIVE answer with its own shorter TTL."""

    seen = _route_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=_html(""), headers={"content-type": "text/html"}
        ),
    )
    async with _client() as client:
        first = await _post(client, _resolve_body())
        assert first.json()["result"] == {"kind": "absent", "reason": "no-metadata"}
        again = await _post(client, _resolve_body())
        assert again.json()["result"] == first.json()["result"]
        assert seen.count("/page") == 1

        # Shorter than the positive TTL, which is the property that matters: the
        # judgement "nothing here" is made from markup that changes weekly.
        assert NEGATIVE_TTL_S < POSITIVE_TTL_S
        _clock.advance(NEGATIVE_TTL_S + 1)
        await _post(client, _resolve_body())
        assert seen.count("/page") == 2


@pytest.mark.asyncio
async def test_a_name_with_no_address_is_a_blocked_host_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    signed_in: None,
    _clock: _Clock,
    preview_cache_for_test: PreviewCache,
) -> None:
    """A permanently dead link must not be re-resolved on every scroll.

    A gaierror meaning "no such name" (``EAI_NONAME``) is a POLICY answer — the
    name has no address, and no retry changes that — so it surfaces as
    ``blocked-host`` and is remembered as a NEGATIVE entry, reused for the whole
    negative TTL before one more lookup is allowed. The OTHER gaierrors are not:
    ``EAI_AGAIN`` is a resolver asking to be asked again and stays a retryable
    ``error``, as does a DNS TIMEOUT — see
    ``test_a_transient_lookup_failure_stays_retryable_and_is_never_cached`` and
    ``test_the_total_deadline_answers_instead_of_hanging`` (review R2-1) — because
    a slow or busy resolver is not a dead host.
    """
    # The autouse fixture pins every name; this test is about the resolver's own
    # answer, so the production resolver goes back in and only DNS itself is faked.
    monkeypatch.setattr(fetch_service, "_resolve_host_ips", _RESOLVE_HOST_IPS_REAL)
    lookups: list[str] = []

    def _no_such_name(hostname: str, *args: Any, **kwargs: Any) -> Any:
        lookups.append(hostname)
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", _no_such_name)
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    body = _resolve_body("http://nowhere.invalid/page")
    async with _client() as client:
        first = await _post(client, body)
        assert first.json()["result"] == {"kind": "absent", "reason": "blocked-host"}
        # No retryable flag at all: a dead name is an answer, not a blip.
        assert "retryable" not in first.json()["result"]

        again = await _post(client, body)
        assert again.json()["result"] == first.json()["result"]
        assert lookups == ["nowhere.invalid"], "the negative answer was not reused"

        # Still inside the 30 min negative TTL: still no second lookup.
        _clock.advance(NEGATIVE_TTL_S - 1)
        third = await _post(client, body)
        assert third.json()["result"] == first.json()["result"]
        assert lookups == ["nowhere.invalid"]

        # Past it, one more lookup is allowed — and answered the same way.
        _clock.advance(2)
        await _post(client, body)
        assert lookups == ["nowhere.invalid", "nowhere.invalid"]
    assert seen == [], "a refused name reached the transport"
    assert preview_cache_for_test.stats().entries == 1


@pytest.mark.asyncio
async def test_a_transient_lookup_failure_stays_retryable_and_is_never_cached(
    monkeypatch: pytest.MonkeyPatch, signed_in: None, preview_cache_for_test: PreviewCache
) -> None:
    """``EAI_AGAIN`` is a blip, NOT a verdict on the host (review R2-1).

    The negative cache is what makes this more than a wording preference: a
    transient resolver failure classified as ``blocked-host`` would be remembered
    for the whole 30-minute ``NEGATIVE_TTL_S``, so one bad second at the resolver
    would hide a live page from every later reader. Measured across two resolves:
    both answer retryable, the transport is never reached, and nothing is written
    to the cache — so the next scroll asks the resolver again.
    """
    monkeypatch.setattr(fetch_service, "_resolve_host_ips", _RESOLVE_HOST_IPS_REAL)
    lookups: list[str] = []

    def _ask_again(hostname: str, *args: Any, **kwargs: Any) -> Any:
        lookups.append(hostname)
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", _ask_again)
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    body = _resolve_body("http://busy.invalid/page")
    async with _client() as client:
        first = await _post(client, body)
        second = await _post(client, body)
    assert first.json()["result"] == {"kind": "error", "retryable": True}, first.text
    # The second resolve looked the name up AGAIN: no negative entry was written.
    assert second.json()["result"] == first.json()["result"]
    assert len(lookups) > 1, lookups
    assert seen == [], "a failed lookup reached the transport"
    assert preview_cache_for_test.stats().entries == 0


@pytest.mark.asyncio
async def test_concurrent_resolves_of_one_url_hit_the_origin_exactly_once(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        responses = await asyncio.gather(*[_post(client, _resolve_body()) for _ in range(6)])
    assert [response.json()["result"]["kind"] for response in responses] == ["ready"] * 6
    assert seen.count("/page") == 1, f"the origin was fetched {seen.count('/page')} times"


# ---------------------------------------------------------------------------
# The two settings a preview refuses to honour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_private_target_is_refused_even_when_the_user_allows_private_fetches(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """``web_fetch.allow_private`` is the tool's escape hatch; a pasted link is not.

    The setting exists so a user can ask for their own local dev server. A preview
    fetches whatever a correspondent pasted, so honouring it here would let a
    stranger's link reach the loopback or the cloud metadata service from a
    session that holds credentials.
    """

    monkeypatch.setattr(
        desktop_link_preview,
        "load_fetch_settings",
        lambda manager: WebFetchSettings(allow_private=True),
    )
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        for url in (
            "http://127.0.0.1:9/x",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://10.0.0.7/",
        ):
            response = await _post(client, _resolve_body(url))
            assert response.json()["result"] == {
                "kind": "absent",
                "reason": "local-or-private",
            }, (url, response.text)
    assert seen == [], "a private target reached the transport"


def test_the_preview_fetcher_forces_the_guard_on_without_mutating_the_settings() -> None:
    settings = WebFetchSettings(allow_private=True)
    service = preview_service.preview_fetcher(settings)
    assert service.settings.allow_private is False
    assert service.trust_env is False
    # The caller's own settings object is not rewritten: the tool keeps the escape
    # hatch it was configured with.
    assert settings.allow_private is True


@pytest.mark.asyncio
async def test_the_previews_client_is_built_without_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HTTP(S)_PROXY`` in the environment must not decide where a preview goes."""

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("NO_PROXY", "preview.test")

    captured: list[dict[str, Any]] = []
    pooled = WebReadIO.client

    @asynccontextmanager
    async def _capture(self, key, **kwargs):
        captured.append(dict(kwargs))
        async with pooled(self, key, **kwargs) as client:
            yield client

    monkeypatch.setattr(WebReadIO, "client", _capture)

    transport = httpx.MockTransport(_card_handler(_html(CARD_HEAD)))
    service = preview_service.preview_fetcher(WebFetchSettings(enrich=False), transport=transport)
    fetched = await service.fetch_bytes(
        "http://preview.test/page", max_bytes=4096, timeout_seconds=5.0
    )
    assert fetched.status == 200
    assert captured, "the fetch never built a pooled client"
    assert captured[0]["trust_env"] is False


# ---------------------------------------------------------------------------
# Code review round 1 (majors): the kill switch, the sniff's thread, the flood
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_fetch_off_switch_stops_previews_with_zero_requests(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """``web_fetch.enabled: false`` must answer — and touch nothing.

    Security round 1 (S-1) measured this surface happily making two outbound
    requests while the operator's fetch switch was off. The switch is the one a
    privacy- or airgap-minded user flips, and it must hold for EVERY spelling of
    fetching, so the assertion is not the answer's shape but the request log's
    length: zero.
    """
    monkeypatch.setattr(
        desktop_link_preview,
        "load_fetch_settings",
        lambda manager: WebFetchSettings(enabled=False),
    )
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        response = await _post(client, _resolve_body("http://preview.test/page"))
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"kind": "absent", "reason": "disabled"}
    assert seen == [], f"a switched-off install still fetched: {seen}"


@pytest.mark.asyncio
async def test_the_resolver_itself_honours_the_off_switch(signed_in: None) -> None:
    """The gate lives in the orchestration boundary, not only in the route.

    Any other caller of ``resolve_preview`` (a future relay, a script) gets the same
    refusal, which is the point of putting it below the HTTP surface.
    """
    asked: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(200, content=_html(CARD_HEAD), headers={"content-type": "text/html"})

    resolution = await preview_service.resolve_preview(
        url="http://preview.test/page",
        index=0,
        scope=PreviewScope(account_id=ACCOUNT, tenant_id=""),
        settings=WebFetchSettings(enabled=False),
        store=PreviewStore.from_config(Path("/tmp/link-preview-off-switch")),
        cache=PreviewCache(),
        fetcher=WebFetchService(
            WebFetchSettings(enabled=False), transport=httpx.MockTransport(_handler)
        ),
    )
    assert resolution.kind == "absent"
    assert resolution.reason == "disabled"
    assert asked == []


@pytest.mark.asyncio
async def test_the_sniff_never_runs_on_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """Both sniff sites run in a worker thread, measured by thread identity.

    A byte-wise walk over attacker-supplied data on the loop stalls every other
    renderer's resolve for the walk's duration (round 1 measured 1.75 s for a
    hostile 2 MiB JPEG before the scan was capped). A wall-clock assertion would go
    green again the moment the walk got cheap, so this pins the property directly:
    the sniff must not execute on the loop's own thread, at either call site.
    """
    import threading

    threads: list[str] = []
    real = extract.sniff_raster

    def _recording_sniff(body: bytes):
        threads.append(threading.current_thread().name)
        return real(body)

    monkeypatch.setattr(extract, "sniff_raster", _recording_sniff)
    monkeypatch.setattr(preview_service, "sniff_raster", _recording_sniff)
    monkeypatch.setattr(desktop_link_preview, "sniff_raster", _recording_sniff)

    page = _html(
        '<title>Threads</title><link rel="icon" href="/favicon.png">'
        '<meta property="og:image" content="/favicon.png">'
    )
    _route_transport(monkeypatch, _card_handler(page))
    async with _client() as client:
        resolved = await _post(client, _resolve_body("http://preview.test/page"))
        assert resolved.status_code == 200, resolved.text
        favicon_id = resolved.json()["result"]["preview"]["favicon"]["id"]
        served = await client.get(_artifact_path(favicon_id), headers=_receipt())
        assert served.status_code == 200

    assert threads, "no sniff ran at all"
    loop_thread = threading.main_thread().name
    assert all(name != loop_thread for name in threads), threads


@pytest.mark.asyncio
async def test_a_burst_of_resolves_is_refused_by_the_account_bucket(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """The server-side flood bound: past the burst, the route answers 429.

    ``index < 3`` is the renderer's numbering, so a client that renumbers its links
    gets an unbounded stream of resolves (round 1). This drives the bucket past its
    depth with the clock pinned, asserts the refusal and its ``Retry-After``, and
    proves the refused request cost no fetch.
    """
    # A pinned clock means no elapsed time, hence no refill, at any rate.
    frozen = ResolveThrottle(burst=3, clock=lambda: 1_000.0)
    monkeypatch.setattr(desktop_link_preview, "get_resolve_throttle", lambda: frozen)
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        answers = []
        # DISTINCT urls, so every one of them is a cold miss and therefore spends a
        # token. Three repeats of one url would be two cache hits and only one charge
        # — which is the behaviour the next test pins.
        for name in ("a1", "a2", "a3"):
            answers.append(await _post(client, _resolve_body(f"http://preview.test/{name}")))
        refused = await _post(client, _resolve_body("http://preview.test/b"))
    assert [answer.status_code for answer in answers] == [200, 200, 200]
    assert refused.status_code == 429
    detail = refused.json()["detail"]
    assert detail["code"] == "rate_limited"
    assert detail["retryable"] is True
    assert "Retry-After" in refused.headers
    # The refused URL never reached the origin, and nothing else did for it either.
    assert "/b" not in seen, seen


@pytest.mark.asyncio
async def test_a_warm_re_render_costs_no_tokens(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """A CACHE HIT IS FREE — the user-visible bug this round fixes.

    The bucket used to be charged at the HTTP boundary, so re-rendering an already
    cached screen (seven messages × three links = twenty-one resolves, all warm)
    exhausted a burst of twenty and started refusing real work. The charge now lives
    where a fetch does: one token for this throttled-to-1 bucket, spent by the FIRST
    resolve, and the warm repeat must still answer.
    """
    starving = ResolveThrottle(burst=1, clock=lambda: 1_000.0)
    monkeypatch.setattr(desktop_link_preview, "get_resolve_throttle", lambda: starving)
    seen = _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        first = await _post(client, _resolve_body("http://preview.test/page"))
        warm = await _post(client, _resolve_body("http://preview.test/page"))
        cold = await _post(client, _resolve_body("http://preview.test/other"))
    assert first.status_code == 200 and first.json()["result"]["kind"] == "ready"
    assert warm.status_code == 200, warm.text
    assert warm.json()["result"]["kind"] == "ready"
    # The bucket held exactly one token: the warm repeat was free, the second COLD
    # url was not.
    assert cold.status_code == 429
    assert seen.count("/page") == 1, seen


@pytest.mark.asyncio
async def test_a_cap_refusal_costs_no_token(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    """Over-cap indexes are the renderer's numbering, not a fetch: they are free."""
    starving = ResolveThrottle(burst=1, clock=lambda: 1_000.0)
    monkeypatch.setattr(desktop_link_preview, "get_resolve_throttle", lambda: starving)
    _route_transport(monkeypatch, _card_handler(_html(CARD_HEAD)))
    async with _client() as client:
        over_cap = await _post(
            client,
            {
                "url": "http://preview.test/page",
                "index": 3,
                "expected_account_id": ACCOUNT,
            },
        )
        real = await _post(client, _resolve_body("http://preview.test/page"))
    assert over_cap.json()["result"] == {"kind": "absent", "reason": "cap"}
    assert real.status_code == 200 and real.json()["result"]["kind"] == "ready"


def test_the_account_bucket_refills_and_stays_bounded() -> None:
    """The bucket is a rate, not a lifetime quota, and its map cannot grow."""
    now = {"t": 0.0}
    throttle = ResolveThrottle(burst=2, refill_per_second=1.0, clock=lambda: now["t"])
    assert throttle.allow("acc_a") and throttle.allow("acc_a")
    assert not throttle.allow("acc_a")
    # 1 s of refill buys exactly one more token.
    now["t"] += 1.0
    assert throttle.allow("acc_a")
    assert not throttle.allow("acc_a")
    assert 0.0 < throttle.retry_after_s("acc_a") <= 1.0

    small = ResolveThrottle(burst=1, max_accounts=4, clock=lambda: now["t"])
    for index in range(40):
        small.allow(f"acc_{index}")
    assert len(throttle._level) <= 1
    assert len(small._level) <= 4


def test_forgetting_an_account_never_refunds_a_spent_burst() -> None:
    """Eviction may only drop an IDLE-FULL bucket, or it hands tokens back.

    Dropping a DEPLETED bucket and recreating it later at full burst is a way to buy
    tokens by being forgotten (round 2). So a full map of depleted buckets admits no
    new account at all, and once a bucket has refilled itself it is safe to forget.
    """
    now = {"t": 0.0}
    clock = lambda: now["t"]  # noqa: E731 - a one-line test clock
    tight = ResolveThrottle(burst=2, refill_per_second=1.0, max_accounts=1, clock=clock)
    assert clock() == 0.0
    assert [tight.allow("acc_a") for _ in range(2)] == [True, True]
    assert not tight.allow("acc_a"), "the bucket should be empty"
    # A second account cannot buy its way in by evicting a depleted bucket.
    assert not tight.allow("acc_b")
    assert "acc_a" in tight._level
    # S-4: reading the wait must not CREATE a bucket — a read that grew the map
    # would be the same bug the eviction rule exists to prevent, and mutating this
    # to ``setdefault`` used to leave the suite green.
    # M1: the refusal is "the map is full", not "this account is empty", so the
    # advertised wait is until a slot frees (acc_a needs its whole burst of 2 back
    # at 1 token/s = 2 s), NOT the 0.05 s floor a missing bucket used to report.
    assert "acc_b" not in tight._level
    assert tight.retry_after_s("acc_b") == 2.0, "Retry-After must name the real wait"
    assert "acc_b" not in tight._level
    # Once acc_a has refilled to its whole burst, forgetting it changes nothing, so
    # the slot may be handed over.
    now["t"] += 10.0
    assert tight.allow("acc_b")
    assert "acc_a" not in tight._level
    assert not tight.allow("acc_a"), "the evicted account must not return with a full burst"
