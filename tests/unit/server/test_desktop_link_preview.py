"""Link previews: the card, the codes, and every refusal the surface can answer.

The fixture origin is an ``httpx.MockTransport`` driven through the REAL fetch
loop (``WebFetchService._follow``), so each assertion exercises the production
redirect/SSRF path rather than a stub of it; DNS is pinned the way
``tests/unit/web_fetch/test_service.py`` pins it, because the guard under test is
the guard that resolves names. The routes are driven through ASGI, which is what
the renderer talks to.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest

from local_operator.config import ConfigManager
from local_operator.server import desktop
from local_operator.server.app import app
from local_operator.team_bridge.link_preview import extract, service as preview_service
from local_operator.team_bridge.link_preview.store import (
    InvalidArtifactId,
    PreviewStore,
)
from local_operator.web_fetch import service as fetch_service
from local_operator.web_fetch.models import WebFetchSettings
from local_operator.web_fetch.service import PolicyRefusal, WebFetchService

TOKEN = "desktop-preview-test-bearer"
ACCOUNT = "acc_0123456789abcdef"

RESOLVE = "/v1/desktop/team-bridge/link-preview/resolve"

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

    transport = httpx.MockTransport(_record)
    monkeypatch.setattr(
        preview_service,
        "WebFetchService",
        lambda settings: WebFetchService(settings, transport=transport),
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

        def log_message(self, *args: Any) -> None:
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
        favicon = await client.get(
            f"/v1/desktop/team-bridge/link-preview/artifact/{preview['favicon']['id']}"
        )
        assert favicon.status_code == 200
        assert favicon.headers["content-type"] == "image/gif"
        assert favicon.headers["x-content-type-options"] == "nosniff"
        assert favicon.content == GIF

        image = await client.get(
            f"/v1/desktop/team-bridge/link-preview/artifact/{preview['image']['id']}"
        )
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
            response = await _post(
                client, {"url": url, "index": 0, "expected_account_id": ACCOUNT}
            )
            assert response.status_code == 200
            assert response.json()["result"] == {"kind": "absent", "reason": "unsupported-scheme"}
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
async def test_an_oversize_image_is_dropped_not_truncated(
    monkeypatch: pytest.MonkeyPatch, signed_in: None
) -> None:
    page = _html(
        '<title>Big</title>'
        '<meta property="og:image" content="http://preview.test/big.png">'
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
        settings = WebFetchSettings(
            enrich=False, allow_private=True, timeout_seconds=10.0, max_attempts=1
        )
        started = time.monotonic()
        resolution = await preview_service.resolve_preview(
            url=f"{base}/drip",
            index=0,
            settings=settings,
            store=PreviewStore(tmp_path / "artifacts"),
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
async def test_a_malformed_artifact_id_never_reaches_the_filesystem(signed_in: None) -> None:
    async with _client() as client:
        for bad in ("not-a-hash", "..%2F..%2Fetc%2Fpasswd", "A" * 64, "0" * 63, "0" * 65, "0" * 32):
            response = await client.get(f"/v1/desktop/team-bridge/link-preview/artifact/{bad}")
            assert response.status_code in (404, 422), (bad, response.status_code)


@pytest.mark.asyncio
async def test_an_unknown_artifact_id_is_a_404(signed_in: None) -> None:
    async with _client() as client:
        response = await client.get(f"/v1/desktop/team-bridge/link-preview/artifact/{'a' * 64}")
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
            b"RIFF" + (30).to_bytes(4, "little") + b"WEBP" + b"VP8X" + b"\x00" * 8
            + (99).to_bytes(3, "little") + (49).to_bytes(3, "little"),
            "image/webp",
            (100, 50),
        ),
        (
            # VP8X at 1x1: the minimum a real file carries.
            b"RIFF" + (30).to_bytes(4, "little") + b"WEBP" + b"VP8X" + b"\x00" * 8
            + b"\x00\x00\x00" + b"\x00\x00\x00",
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


def test_the_parser_is_bounded_and_survives_a_hostile_document() -> None:
    markup = "<html><head>" + "<meta property='og:title' content='Nope'>" * 5000 + "</head></html>"
    metadata = extract.extract_metadata(markup, base_url="http://preview.test/")
    # The tag budget stopped the walk; whatever it collected is still a valid
    # (empty) answer rather than an error or a hang.
    assert isinstance(metadata.title, str)


def test_a_relative_icon_resolves_against_the_served_url() -> None:
    markup = '<html><head><link rel="apple-touch-icon" href="./icon.png"></head></html>'
    metadata = extract.extract_metadata(markup, base_url="https://example.com/a/b")
    assert metadata.icon_url == "https://example.com/a/icon.png"


def test_the_store_is_content_addressed_and_rejects_a_foreign_id(tmp_path) -> None:
    store = PreviewStore(tmp_path / "artifacts")
    first = store.put(PNG)
    second = store.put(PNG)
    assert first == second
    assert store.get(first) == PNG
    with pytest.raises(InvalidArtifactId):
        store.get("../../etc/passwd")
    with pytest.raises(InvalidArtifactId):
        store.get("A" * 64)


def test_the_store_sweeps_least_recently_used_first(tmp_path) -> None:
    store = PreviewStore(tmp_path / "artifacts", max_files=4, max_bytes=10_000_000)
    ids = []
    for index in range(4):
        body = PNG + bytes([index])
        ids.append(store.put(body))
        # Age each blob deliberately: mtime is the LRU key, and the filesystem's
        # own resolution would otherwise make them all look simultaneous.
        target = store.root / ids[-1]
        stamp = target.stat().st_mtime - (100 - index * 10)
        import os

        os.utime(target, (stamp, stamp))
    # Touch the oldest so it is no longer the coldest.
    store.get(ids[0])
    store.put(PNG + b"fresh")
    survivors = {path.name for path in store.root.iterdir()}
    assert len(survivors) <= 4
    assert ids[0] in survivors, "the recently-read blob was evicted"
    assert ids[1] not in survivors, "the least recently used blob survived"


def test_policy_refusals_carry_a_machine_reason() -> None:
    for url, reason in (
        ("file:///etc/passwd", "scheme"),
        ("http://127.0.0.1/", "private"),
        ("http://10.0.0.1/", "private"),
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
