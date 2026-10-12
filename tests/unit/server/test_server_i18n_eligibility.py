"""Held bytes must be rejected before hash/validator work on the public route."""

import hashlib
import json

import pytest

from local_operator.i18n import catalogues, resolve


@pytest.mark.asyncio
@pytest.mark.parametrize("validator", [False, True])
async def test_held_locale_refused_before_hash(tmp_path, monkeypatch, test_app_client, validator):
    folder = tmp_path / "zh-CN"
    folder.mkdir()
    # Synthetic credential-shaped bytes exercise non-disclosure, not redaction.
    sentinel = "sk-" + "eligibility_fixture_not_a_real_credential_1234567890"
    raw = json.dumps({"demo.words.secret": sentinel}).encode()
    (folder / "demo.words.json").write_bytes(raw)
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))

    def forbidden_hash(*args):
        pytest.fail("held locale reached catalogue hash")

    monkeypatch.setattr(catalogues, "catalogue_sha256", forbidden_hash)
    headers = {"If-None-Match": f'"{hashlib.sha256(raw).hexdigest()}"'} if validator else {}
    response = await test_app_client.get("/v1/i18n/catalogues/zh-CN/demo.words", headers=headers)
    assert response.status_code == 404
    assert sentinel not in response.text
    assert "etag" not in response.headers
    assert "cache-control" not in response.headers


@pytest.mark.asyncio
async def test_kill_switch_refuses_nonenglish_but_preserves_english(monkeypatch, test_app_client):
    monkeypatch.setenv("LOP_I18N", "0")
    response = await test_app_client.get("/v1/i18n/catalogues/fr/wire.errors")
    assert response.status_code == 404
    english = await test_app_client.get("/v1/i18n/catalogues/en/wire.errors")
    assert english.status_code == 200
    cached = await test_app_client.get(
        "/v1/i18n/catalogues/en/wire.errors", headers={"If-None-Match": english.headers["etag"]}
    )
    assert cached.status_code == 304
