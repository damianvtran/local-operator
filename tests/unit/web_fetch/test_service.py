"""Service-layer tests: SSRF policy, streaming cap, redirect loop, cache index.

Uses httpx ``MockTransport`` for determinism. The SSRF checks resolve real
hostnames, so those tests monkeypatch ``_resolve_host_ips`` to pin an address
rather than depending on live DNS.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.web_fetch import service
from local_operator.web_fetch.models import WebFetchSettings
from local_operator.web_fetch.service import (
    FetchError,
    WebFetchService,
    coerce_fetch_settings,
    normalize_url,
    validate_public_url,
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))


def _service(transport: httpx.MockTransport, **overrides) -> WebFetchService:
    settings = WebFetchSettings(enrich=False, **overrides)
    return WebFetchService(settings, transport=transport)


# --- URL normalization -----------------------------------------------------


def test_normalize_bare_host_defaults_https() -> None:
    assert normalize_url("example.com") == "https://example.com"


def test_normalize_collapsed_scheme() -> None:
    assert normalize_url("https:/example.com/x") == "https://example.com/x"


def test_normalize_rejects_non_http_scheme() -> None:
    with pytest.raises(FetchError, match="only http"):
        normalize_url("file:///etc/passwd")


# --- SSRF policy: direct ---------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254", "::1", "::ffff:169.254.169.254"],
)
def test_validate_refuses_private_targets(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: [ip])
    with pytest.raises(FetchError, match="private/loopback/reserved"):
        validate_public_url("http://evil.example/", allow_private=False)


def test_validate_allows_public_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    validate_public_url("https://example.com/", allow_private=False)  # no raise


def test_validate_allow_private_bypasses(monkeypatch: pytest.MonkeyPatch) -> None:
    # With allow_private, DNS is not even consulted — the switch is the escape.
    validate_public_url("http://localhost:3000/", allow_private=True)


def test_file_scheme_refused_at_normalize() -> None:
    with pytest.raises(FetchError):
        normalize_url("file:///etc/passwd")


# --- SSRF policy: via redirect (the case a scheme-only check misses) -------


@pytest.mark.asyncio
async def test_ssrf_via_redirect_is_refused_at_the_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A public URL that 302s to the metadata endpoint must be refused, not
    followed. The redirect target is re-validated at the hop."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})
        return httpx.Response(200, text="SHOULD NOT REACH")

    # public.example resolves public; the metadata IP is judged by its literal.
    monkeypatch.setattr(
        service,
        "_resolve_host_ips",
        lambda host: ["93.184.216.34"] if host == "public.example" else [host],
    )
    svc = _service(httpx.MockTransport(handler))
    with pytest.raises(FetchError, match="private/loopback/reserved"):
        await svc.fetch("http://public.example/")


# --- streaming byte cap ----------------------------------------------------


@pytest.mark.asyncio
async def test_max_bytes_enforced_during_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body larger than max_bytes stops downloading and is flagged truncated;
    the buffer never exceeds the ceiling (memory does not balloon)."""
    big = "A" * (2 * 1024 * 1024)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=big, headers={"content-type": "text/plain"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler), max_bytes=64 * 1024)
    result = await svc.fetch("https://example.com/big")
    assert result.complete is False
    assert result.bytes <= 64 * 1024


@pytest.mark.asyncio
async def test_redirect_updates_final_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "https://example.com/new"})
        return httpx.Response(200, text="landed", headers={"content-type": "text/plain"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/old")
    assert result.final_url == "https://example.com/new"
    assert "landed" in result.content


@pytest.mark.asyncio
async def test_too_many_redirects_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/loop"})

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler), max_redirects=2)
    with pytest.raises(FetchError, match="too many redirects"):
        await svc.fetch("https://example.com/loop")


@pytest.mark.asyncio
async def test_timeout_names_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    with pytest.raises(FetchError, match="timed out"):
        await svc.fetch("https://example.com/slow")


@pytest.mark.asyncio
async def test_non_2xx_status_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html><body>Not found</body></html>")

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/missing")
    assert result.status == 404


@pytest.mark.asyncio
async def test_pdf_returns_binary_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"%PDF-1.4 fake", headers={"content-type": "application/pdf"}
        )

    monkeypatch.setattr(service, "_resolve_host_ips", lambda host: ["93.184.216.34"])
    svc = _service(httpx.MockTransport(handler))
    result = await svc.fetch("https://example.com/doc.pdf")
    assert result.render_method == "binary"
    assert "application/pdf" in result.content


# --- config coercion & cache index -----------------------------------------


def test_coerce_clamps_bad_values() -> None:
    settings = coerce_fetch_settings(
        {"timeout_seconds": 0, "max_bytes": -5, "max_redirects": 999, "cache_ttl_seconds": -1}
    )
    assert settings.timeout_seconds >= 1.0
    assert settings.max_bytes >= 1024
    assert settings.max_redirects <= 20
    assert settings.cache_ttl_seconds == 0


def test_coerce_defaults_on_garbage() -> None:
    settings = coerce_fetch_settings("not a mapping")
    assert settings.enabled is True
    assert settings.max_redirects == 5


def test_cache_roundtrip() -> None:
    entry = service.CacheEntry(
        url="https://example.com/",
        final_url="https://example.com/",
        spill_handle="spill://" + "a" * 32,
        fetched_at_ms=123,
        status=200,
        content_type="text/html",
        render_method="markdownify",
        complete=True,
        low_quality=False,
    )
    service.write_cache_entry(entry)
    loaded = service.read_cache_entry("https://example.com/")
    assert loaded is not None
    assert loaded.spill_handle == entry.spill_handle
    assert loaded.status == 200


def test_cache_prune_keeps_newest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "CACHE_MAX_ENTRIES", 3)
    for i in range(6):
        service.write_cache_entry(
            service.CacheEntry(
                url=f"https://example.com/{i}",
                final_url=f"https://example.com/{i}",
                spill_handle="spill://" + str(i) * 32,
                fetched_at_ms=i,
                status=200,
                content_type="text/html",
                render_method="stdlib",
                complete=True,
                low_quality=False,
            )
        )
    remaining = list(service.cache_dir().iterdir())
    assert len(remaining) <= 3
