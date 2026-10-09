"""The i18n catalogue route, and the capability + language that advertise it.

Only `en` ships in M0, so the shipped-tree assertions are about shape (map,
hash, ETag, 304, 404s) rather than about translated content; the fixture test
proves a populated catalogue traverses the same path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve


@pytest.mark.asyncio
async def test_catalogue_route_serves_map_and_hash(test_app_client) -> None:
    response = await test_app_client.get("/v1/i18n/catalogues/en/wire.errors")
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["messages"] == {}
    raw = (Path(catalogues.__file__).parent / "catalogues" / "en" / "wire.errors.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert payload["result"]["content_sha256"] == sha
    assert response.headers["etag"] == f'"{sha}"'
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_conditional_request_short_circuits_to_304(test_app_client) -> None:
    first = await test_app_client.get("/v1/i18n/catalogues/en/wire.errors")
    etag = first.headers["etag"]
    second = await test_app_client.get(
        "/v1/i18n/catalogues/en/wire.errors", headers={"If-None-Match": etag}
    )
    assert second.status_code == 304
    assert second.headers["etag"] == etag
    assert not second.content


@pytest.mark.asyncio
async def test_weak_and_list_validators_still_revalidate_to_304(test_app_client) -> None:
    # RFC 9110 §13.1.2 weak comparison (round-1 n5): a client echoing the tag
    # weak, in a list, or sending `*` must still short-circuit.
    first = await test_app_client.get("/v1/i18n/catalogues/en/wire.errors")
    etag = first.headers["etag"]
    for header in (f"W/{etag}", f'"other", W/{etag}', "*"):
        response = await test_app_client.get(
            "/v1/i18n/catalogues/en/wire.errors", headers={"If-None-Match": header}
        )
        assert response.status_code == 304, header
    # A non-matching tag still fetches.
    response = await test_app_client.get(
        "/v1/i18n/catalogues/en/wire.errors", headers={"If-None-Match": '"nope"'}
    )
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/v1/i18n/catalogues/fr/wire.errors",  # locale exists in the wave list, not shipped
        "/v1/i18n/catalogues/en/nope",  # unknown namespace
        "/v1/i18n/catalogues/en/..",  # traversal-shaped component
    ],
)
async def test_unknown_names_are_404(test_app_client, path: str) -> None:
    response = await test_app_client.get(path)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_populated_fixture_catalogue_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_app_client
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    messages = {"demo.words.hello": "Hello, {name}!"}
    raw = json.dumps(messages).encode()
    (root / "en" / "demo.words.json").write_bytes(raw)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    response = await test_app_client.get("/v1/i18n/catalogues/en/demo.words")
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["messages"] == messages
    assert payload["result"]["content_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.asyncio
async def test_capabilities_advertise_i18n_and_a_resolved_language(
    test_app_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resolve, "config_value", lambda: None)
    response = await test_app_client.get("/v1/capabilities")
    result = response.json()["result"]
    assert result["features"]["i18n"] >= 1
    assert result["language"] in resolve.SUPPORTED_LOCALES


@pytest.mark.asyncio
async def test_language_follows_resolution_and_the_shipped_filter(
    test_app_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resolve, "config_value", lambda: None)
    monkeypatch.setattr(resolve, "os_language", lambda **kwargs: None)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))

    monkeypatch.setenv("LOP_LANG", "fr")
    response = await test_app_client.get("/v1/capabilities")
    assert response.json()["result"]["language"] == "fr"

    # An unshipped locale never surfaces, even when explicitly requested.
    monkeypatch.setenv("LOP_LANG", "es")
    response = await test_app_client.get("/v1/capabilities")
    assert response.json()["result"]["language"] == "en"
