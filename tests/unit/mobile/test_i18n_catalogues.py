"""Independent relay catalogue routing, cache validators and public-data scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from local_operator.i18n import catalogues, resolve
from local_operator.i18n.http import etag_matches
from local_operator.mobile.daemon import MobileDaemon, build_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(
        build_app(MobileDaemon(port=0, password="fixture-password")), follow_redirects=False
    )


def test_public_catalogue_not_session_authority(client) -> None:
    assert client.get("/api/sessions").status_code == 401
    response = client.get("/api/i18n/catalogues/en/wire.exec")
    assert response.status_code == 200
    result = response.json()["result"]
    raw = (catalogues.catalogue_root() / "en/wire.exec.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert result == {"messages": json.loads(raw), "content_sha256": sha}
    assert response.json()["status"] == 200
    assert response.headers["etag"] == f'"{sha}"'
    assert response.headers["cache-control"] == "no-cache"
    assert "set-cookie" not in response.headers
    assert client.get("/api/sessions").status_code == 401
    assert client.post("/login", data={"password": "fixture-password"}).status_code == 303
    capabilities = client.get("/api/sessions").json()["capabilities"]
    assert capabilities["features"]["i18n"] >= 1
    assert capabilities["i18n"] == {"catalogues": 1}
    assert client.get("/logout").status_code == 303
    assert client.get("/api/i18n/catalogues/en/wire.exec").status_code == 200


def test_raw_byte_revision_and_revalidation(client, monkeypatch, tmp_path) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    path = root / "en/wire.fixture.json"
    raw = b'{\n  "wire.fixture.label": "Original"\n}\n'
    path.write_bytes(raw)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    response = client.get("/api/i18n/catalogues/en/wire.fixture")
    etag = response.headers["etag"]
    sha = hashlib.sha256(raw).hexdigest()
    assert etag == f'"{sha}"'
    for header in [etag, "W/" + etag, f'"different", W/{etag}', "*"]:
        cached = client.get(
            "/api/i18n/catalogues/en/wire.fixture", headers={"if-none-match": header}
        )
        assert cached.status_code == 304 and cached.content == b""
        assert cached.headers["etag"] == etag
        assert cached.headers["cache-control"] == "no-cache"
        assert etag_matches(header, sha)
    assert (
        client.get(
            "/api/i18n/catalogues/en/wire.fixture", headers={"if-none-match": sha}
        ).status_code
        == 200
    )
    # Identical map, different file bytes: the revision is opaque raw-file
    # identity, never a client JSON.stringify/semantic-message digest.
    path.write_text(json.dumps(json.loads(raw)))
    changed = client.get("/api/i18n/catalogues/en/wire.fixture", headers={"if-none-match": etag})
    assert changed.status_code == 200
    assert changed.headers["etag"] != etag
    assert changed.json()["result"]["messages"] == response.json()["result"]["messages"]


@pytest.mark.parametrize(
    "path",
    [
        "/en/relay.login",
        "/en/tui.welcome",
        "/en/nope",
        "/en/wire.nope",
        "/unknown/wire.exec",
        "/en/wire..fixture",
        "/en/%2e%2e%2fwire.exec",
    ],
)
def test_only_valid_existing_wire_catalogues_public(client, path: str) -> None:
    assert client.get("/api/i18n/catalogues" + path).status_code == 404


def test_malformed_catalogue_fails_loudly(client, monkeypatch, tmp_path) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en/wire.invalid.json").write_text('{"wire.invalid.label": 42}')
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    assert client.get("/api/i18n/catalogues/en/wire.invalid").status_code == 500


@pytest.mark.parametrize("disabled", [False, True])
def test_ineligible_bytes_never_reach_hash_or_conditional_response(
    client, monkeypatch, tmp_path, disabled: bool
) -> None:
    root = tmp_path / "catalogues"
    # Every supported locale now ships (zh-CN included after its disposition),
    # so the ineligible-by-shipment arm uses a synthetic tag with
    # committed-shaped bytes; eligibility must come from the generated set,
    # never from disk presence.
    locale = "fr" if disabled else "xx"
    (root / locale).mkdir(parents=True)
    (root / locale / "wire.held.json").write_text('{"wire.held.label": "Held fixture copy"}')
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.delenv("LOP_I18N", raising=False)
    if disabled:
        # Make this fixture explicitly otherwise eligible, independently of
        # the evolving committed ship set. A 200 control proves the later 404
        # tests the operational switch rather than ordinary locale rejection.
        monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
        control = client.get(f"/api/i18n/catalogues/{locale}/wire.held")
        assert control.status_code == 200
        assert control.json()["result"]["messages"] == {"wire.held.label": "Held fixture copy"}
        assert "etag" in control.headers
        monkeypatch.setenv("LOP_I18N", "0")

    def forbidden_hash(*args, **kwargs):
        raise AssertionError("ineligible bytes reached hashing")

    monkeypatch.setattr(catalogues, "catalogue_sha256", forbidden_hash)
    for validator in [None, "*"]:
        headers = {} if validator is None else {"if-none-match": validator}
        response = client.get(f"/api/i18n/catalogues/{locale}/wire.held", headers=headers)
        assert response.status_code == 404
        assert "etag" not in response.headers
        assert "Held fixture copy" not in response.text


def test_released_chinese_catalogue_is_served(client) -> None:
    # The disposition made zh-CN eligible: the mobile route serves its audited
    # committed bytes with the file-hash ETag, not a 404 or an English copy.
    response = client.get("/api/i18n/catalogues/zh-CN/wire.exec")
    assert response.status_code == 200
    result = response.json()["result"]
    raw = (catalogues.catalogue_root() / "zh-CN/wire.exec.json").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert result == {"messages": json.loads(raw), "content_sha256": sha}
    assert response.headers["etag"] == f'"{sha}"'
    assert response.headers["cache-control"] == "no-cache"


def test_adapter_does_not_import_core_route_models() -> None:
    source = Path(__file__).resolve().parents[3] / "local_operator/mobile/daemon.py"
    text = source.read_text()
    assert "from local_operator.server.routes.i18n" not in text
    assert "from local_operator.server.models.schemas import CRUDResponse" not in text
