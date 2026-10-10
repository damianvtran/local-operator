"""Catalogue loader: names, hashes, traversal refusal, shape validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from local_operator.i18n import catalogues


def test_shipped_wire_namespaces_exist() -> None:
    # M0 ships the six `wire.*` namespace STUBS (RFC §5): the files exist so
    # later slices populate them; being empty is the correct current state.
    assert catalogues.locales() == ("en",)
    assert catalogues.namespaces("en") == (
        "wire.errors",
        "wire.exec",
        "wire.incidents",
        "wire.notices",
        "wire.settings",
        "wire.slash",
    )
    for namespace in catalogues.namespaces("en"):
        assert catalogues.load_catalogue("en", namespace) == {}


def test_hash_is_over_file_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    raw = b'{\n  "demo.words.hello": "Hello"\n}\n'
    (root / "en" / "demo.words.json").write_bytes(raw)
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    assert catalogues.catalogue_sha256("en", "demo.words") == hashlib.sha256(raw).hexdigest()
    # The map is parsed, the hash is NOT: re-serialising the same map would
    # hash differently, which would break the ledger's staleness rule.
    assert catalogues.load_catalogue("en", "demo.words") == {"demo.words.hello": "Hello"}


def test_missing_locale_and_namespace_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "demo.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("fr", "demo")
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("en", "nope")


@pytest.mark.parametrize("namespace", ["..", "../etc/passwd", "a/b", ".hidden", "x\x00y"])
def test_traversal_shaped_names_are_refused(namespace: str) -> None:
    with pytest.raises(catalogues.CatalogueNotFound):
        catalogues.load_catalogue("en", namespace)


def test_invalid_shapes_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "list.json").write_text("[1, 2]", encoding="utf-8")
    (root / "en" / "value.json").write_text('{"k": 3}', encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "list")
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "value")


def test_malformed_file_raises_catalogue_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Core hardening (S2 M1): decode/parse failures surface as the ONE
    # catalogue exception the render path handles, so a garbled file degrades
    # a render rather than raising a bare ValueError through it.
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "demo.json").write_text("{not json", encoding="utf-8")
    (root / "en" / "demo.bad.json").write_bytes(b"\xff\xfe{}")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "demo")
    with pytest.raises(catalogues.CatalogueInvalid):
        catalogues.load_catalogue("en", "demo.bad")


def test_context_sidecar_is_not_a_namespace_and_reads_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "en" / "demo.json").write_text(json.dumps({"demo.x": "y"}), encoding="utf-8")
    sidecar = {"demo.x": {"role": "help text", "maxCells": 43}}
    (root / "en" / "demo.context.json").write_text(json.dumps(sidecar), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    assert catalogues.namespaces("en") == ("demo",)
    assert catalogues.context("en", "demo") == sidecar
    # Absent sidecar is a valid state ({}), not an error.
    assert catalogues.context("en", "missing") == {}
