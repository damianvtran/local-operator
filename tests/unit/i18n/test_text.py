"""`i18n.text` — the three contracts the shared render helper documents.

`tr(namespace)` is the module every extraction slice resolves its copy
through; beyond the unknown-KEY degradation the T1 slice test pins, these
tests hold it to its docstring: ONE catalogue read per namespace (shared
across resolvers and stable across resolutions), a cache keyed by the
catalogue ROOT (a re-pointed root resolves from its own tree), and no-raise
degradation for every broken shape — unknown namespace, missing key,
malformed file, unrenderable message.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, text
from local_operator.i18n.messages import Msg


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    text._sources.cache_clear()
    yield
    text._sources.cache_clear()


def test_one_catalogue_read_per_namespace_shared_across_resolvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real = catalogues.load_catalogue

    def counting(locale: str, namespace: str) -> dict[str, str]:
        calls.append(namespace)
        return real(locale, namespace)

    monkeypatch.setattr(catalogues, "load_catalogue", counting)
    first = text.tr("tui.welcome")
    second = text.tr("tui.welcome")
    assert calls == ["tui.welcome"], "two resolvers must share one read"
    assert first(Msg("tui.welcome.tip.resume", {}))
    assert second(Msg("tui.welcome.tip.paste", {}))
    assert calls == ["tui.welcome"], "resolutions must add no reads"
    text.tr("wire.slash")
    assert calls == ["tui.welcome", "wire.slash"], "each namespace reads once"


def test_the_cache_is_keyed_by_the_catalogue_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (("a", "A"), ("b", "B")):
        catalogue = tmp_path / name / "en" / "demo.words.json"
        catalogue.parent.mkdir(parents=True)
        catalogue.write_text(json.dumps({"demo.words.hello": value}), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path / "a")
    assert text.tr("demo.words")(Msg("demo.words.hello", {})) == "A"
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path / "b")
    assert text.tr("demo.words")(Msg("demo.words.hello", {})) == "B"


def test_every_broken_shape_degrades_to_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_namespace = text.tr("demo.missing")
    assert missing_namespace(Msg("demo.missing.key", {})) == "demo.missing.key"
    welcome = text.tr("tui.welcome")
    assert welcome(Msg("tui.welcome.nope", {})) == "tui.welcome.nope"
    # An unrenderable message — `step.full` needs `{aida}` — is the
    # MessageError boundary, not the catalogue boundary.
    assert welcome(Msg("tui.welcome.step.full", {})) == "tui.welcome.step.full"
    (tmp_path / "en").mkdir(parents=True)
    (tmp_path / "en" / "demo.bad.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    assert text.tr("demo.bad")(Msg("demo.bad.key", {})) == "demo.bad.key"
