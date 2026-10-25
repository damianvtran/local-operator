"""The i18n catalogue route, and the capability + language that advertise it.

Chinese ships after its 2026-10-24 security disposition, so the route serves
its audited committed bytes rather than treating the candidate as absent. Route
assertions cover shape and cache revalidation, not semantic approval or full
CLI localization. A locale the generated set does not ship stays 404 even with
committed-shaped bytes on disk or an old candidate ETag; the public preview
contract for ordinary present catalogues is unchanged.
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
async def test_a_wave1_locale_is_served_once_translated(test_app_client) -> None:
    # Wave 1: `fr` used to 404 for want of a file; it now serves its bytes.
    response = await test_app_client.get("/v1/i18n/catalogues/fr/wire.errors")
    assert response.status_code == 200
    raw = (Path(catalogues.__file__).parent / "catalogues" / "fr" / "wire.errors.json").read_bytes()
    assert response.json()["result"]["content_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.asyncio
async def test_audited_urdu_catalogue_is_served_from_committed_bytes(test_app_client) -> None:
    response = await test_app_client.get("/v1/i18n/catalogues/ur/wire.errors")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["messages"] == catalogues.load_catalogue("ur", "wire.errors")
    sha = catalogues.catalogue_sha256("ur", "wire.errors")
    assert result["content_sha256"] == sha
    assert response.headers["etag"] == f'"{sha}"'


@pytest.mark.asyncio
async def test_wave3_hindi_exec_catalogue_and_conditional_request(test_app_client) -> None:
    # Pin populated Hindi exec through the mounted route, not only the loader.
    locale = "hi"
    path = f"/v1/i18n/catalogues/{locale}/wire.exec"
    response = await test_app_client.get(path)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["messages"] == catalogues.load_catalogue(locale, "wire.exec")
    assert len(result["messages"]) == 17
    sha = catalogues.catalogue_sha256(locale, "wire.exec")
    assert result["content_sha256"] == sha
    assert response.headers["etag"] == f'"{sha}"'
    cached = await test_app_client.get(path, headers={"If-None-Match": response.headers["etag"]})
    assert cached.status_code == 304
    assert not cached.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("namespace", "old_sha"),
    [
        ("wire.exec", "afe5a5d81a9f60309d6d9575f3da6d09858e3041f1c7833da613f8e44997c238"),
        ("wire.slash", "643981b50181193f5361735c27f93b2d333deba2d85624ccf3375545215f3a79"),
    ],
)
async def test_released_chinese_candidate_serves_current_bytes_not_old_etag(
    test_app_client, namespace: str, old_sha: str
) -> None:
    # These are the preserved candidate's real byte hashes, not arbitrary
    # validators. The released locale now serves its AUDITED catalogues, so a
    # client holding the old candidate's tag gets fresh bytes with a 200 — a
    # failed historical asset must never revalidate as 304.
    path = f"/v1/i18n/catalogues/zh-CN/{namespace}"
    current = await test_app_client.get(path)
    assert current.status_code == 200
    result = current.json()["result"]
    assert result["messages"] == catalogues.load_catalogue("zh-CN", namespace)
    current_sha = catalogues.catalogue_sha256("zh-CN", namespace)
    assert result["content_sha256"] == current_sha
    stale = await test_app_client.get(path, headers={"If-None-Match": f'"{old_sha}"'})
    if old_sha == current_sha:
        # wire.exec's released bytes ARE the preserved candidate's bytes (the
        # audit re-attested them unchanged), so the historical tag is also the
        # current one and a 304 is the correct revalidation.
        assert stale.status_code == 304
    else:
        # wire.slash was remediated: the historical tag must not revalidate,
        # and the answer carries the current bytes and current ETag.
        assert stale.status_code == 200
        assert stale.json()["result"]["content_sha256"] == current_sha
        assert stale.headers["etag"] == f'"{current_sha}"'
    wildcard = await test_app_client.get(path, headers={"If-None-Match": "*"})
    assert wildcard.status_code == 304


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
        # Shipped Urdu carries no wire.exec file; the raw route must not
        # synthesize one from English or return a synthetic empty namespace.
        "/v1/i18n/catalogues/ur/wire.exec",
        "/v1/i18n/catalogues/xx/wire.errors",  # unknown locale
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


@pytest.mark.asyncio
async def test_the_per_locale_bundle_serves_eager_wire_messages_and_a_deferred_manifest(
    test_app_client,
) -> None:
    response = await test_app_client.get("/v1/i18n/catalogues/en")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["locale"] == "en"
    messages = result["messages"]
    # An eager namespace is flattened into the map, keys intact.
    assert "wire.errors.desktop.runtime_unreachable" in messages
    assert messages["wire.errors.desktop.runtime_unreachable"] == (
        catalogues.load_catalogue("en", "wire.errors.desktop")[
            "wire.errors.desktop.runtime_unreachable"
        ]
    )
    # The known-large namespace is described, never carried, in the eager map.
    # (`wire.settings.account_policy` is a CHILD namespace with its own small
    # file and its own keys, so compare against this namespace's own key set.)
    settings_keys = catalogues.load_catalogue("en", "wire.settings")
    assert not any(key in messages for key in settings_keys)
    assert result["namespaces"]["wire.errors.desktop"] == catalogues.catalogue_sha256(
        "en", "wire.errors.desktop"
    )
    deferred = {entry["namespace"]: entry for entry in result["deferred"]}
    assert set(deferred) == {"wire.settings"}
    assert deferred["wire.settings"]["content_sha256"] == catalogues.catalogue_sha256(
        "en", "wire.settings"
    )
    assert deferred["wire.settings"]["keys"] == len(
        catalogues.load_catalogue("en", "wire.settings")
    )
    assert deferred["wire.settings"]["bytes"] == len(
        catalogues.catalogue_bytes("en", "wire.settings")
    )
    assert response.headers["etag"] == f'"{result["content_sha256"]}"'
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.asyncio
async def test_the_bundle_carries_a_translated_identity_and_revalidates(
    test_app_client,
) -> None:
    response = await test_app_client.get("/v1/i18n/catalogues/fr")
    assert response.status_code == 200
    result = response.json()["result"]
    french = catalogues.load_catalogue("fr", "wire.errors.settings")
    identity = "wire.errors.settings.not_registered"
    assert result["messages"][identity] == french[identity]
    assert (
        result["messages"][identity]
        != catalogues.load_catalogue("en", "wire.errors.settings")[identity]
    )

    cached = await test_app_client.get(
        "/v1/i18n/catalogues/fr", headers={"If-None-Match": response.headers["etag"]}
    )
    assert cached.status_code == 304
    assert not cached.content


@pytest.mark.asyncio
async def test_the_urdu_bundle_carries_committed_translations(test_app_client) -> None:
    response = await test_app_client.get("/v1/i18n/catalogues/ur")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["locale"] == "ur"
    identity = "wire.errors.settings.not_registered"
    assert (
        result["messages"][identity]
        == catalogues.load_catalogue("ur", "wire.errors.settings")[identity]
    )
    assert (
        result["messages"][identity]
        != catalogues.load_catalogue("en", "wire.errors.settings")[identity]
    )
    assert result["namespaces"]["wire.errors.settings"] == catalogues.catalogue_sha256(
        "ur", "wire.errors.settings"
    )


@pytest.mark.asyncio
async def test_a_bundle_excludes_client_owned_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_app_client
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "wire.demo.json").write_text(
        json.dumps({"wire.demo.hello": "Hello"}), encoding="utf-8"
    )
    (root / "en" / "tui.demo.json").write_text(
        json.dumps({"tui.demo.hello": "Hello"}), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    response = await test_app_client.get("/v1/i18n/catalogues/en")
    assert response.status_code == 200
    assert response.json()["result"]["messages"] == {"wire.demo.hello": "Hello"}
    assert "tui.demo" not in response.json()["result"]["namespaces"]


@pytest.mark.asyncio
async def test_a_shipped_locale_with_no_wire_files_is_an_empty_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_app_client
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    response = await test_app_client.get("/v1/i18n/catalogues/en")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["messages"] == {}
    assert result["namespaces"] == {}
    assert result["deferred"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["xx", "de"])
async def test_an_unshipped_locale_bundle_is_404(test_app_client, locale: str) -> None:
    # Unknown or unsupported locales must not answer 200-empty: eligibility is
    # the same gate the per-namespace route applies, before any disk work.
    response = await test_app_client.get(f"/v1/i18n/catalogues/{locale}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_held_locale_bundle_is_404_even_with_a_catalogue_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_app_client
) -> None:
    # The bundle gate is the generated shipped set, not disk presence. Every
    # supported locale ships today, so narrow the set in isolation: the file
    # exists but an ineligible tag must still be a 404 before any disk work.
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en",))
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "zh-CN").mkdir()
    (root / "zh-CN" / "wire.demo.json").write_text(
        json.dumps({"wire.demo.hello": "Held"}), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    response = await test_app_client.get("/v1/i18n/catalogues/zh-CN")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_malformed_bundle_is_500_not_a_half_typed_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_app_client
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "wire.bad.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    response = await test_app_client.get("/v1/i18n/catalogues/en")
    assert response.status_code == 500
