"""``download_asset``: bounded, redirect-refusing, header-honest.

Each test drives a real ``httpx.AsyncClient`` over ``httpx.MockTransport`` —
the streaming path is exactly what production runs, and the 32 MiB cap is a
behaviour a mocked client cannot verify (it must observe the abort mid-stream).
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.clients._http import APIError
from local_operator.imagegen import media

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_a_200_downloads_bytes_and_reads_the_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    async with _client(handler) as client:
        asset = await media.download_asset(
            "https://img.test/a.png", client=client, width=8, height=8
        )
    assert asset.data == PNG
    assert asset.content_type == "image/png"
    assert asset.source_url == "https://img.test/a.png"
    assert (asset.width, asset.height) == (8, 8)


@pytest.mark.asyncio
async def test_a_generic_header_falls_back_to_the_payloads_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=PNG, headers={"content-type": "application/octet-stream"}
        )

    async with _client(handler) as client:
        asset = await media.download_asset(
            "https://img.test/a", client=client, fallback_content_type="image/webp"
        )
    assert asset.content_type == "image/webp"


@pytest.mark.asyncio
async def test_a_redirect_is_refused_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.test/a"})

    async with _client(handler) as client:
        with pytest.raises(APIError) as caught:
            await media.download_asset("https://img.test/a", client=client)
    assert "redirect" in str(caught.value).lower()
    assert caught.value.status_code == 302


@pytest.mark.asyncio
async def test_an_http_error_raises_with_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "gone"})

    async with _client(handler) as client:
        with pytest.raises(APIError) as caught:
            await media.download_asset("https://img.test/a", client=client)
    assert caught.value.status_code == 404


@pytest.mark.asyncio
async def test_the_byte_cap_refuses_rather_than_truncates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (media.ARTIFACT_MAX_BYTES + 1))

    async with _client(handler) as client:
        with pytest.raises(APIError) as caught:
            await media.download_asset("https://img.test/big", client=client)
    assert "cap" in str(caught.value)


@pytest.mark.asyncio
async def test_a_transport_failure_reads_as_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    async with _client(handler) as client:
        with pytest.raises(APIError) as caught:
            await media.download_asset("https://img.test/a", client=client)
    assert caught.value.code == "network"
