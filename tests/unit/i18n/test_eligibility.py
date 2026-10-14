"""Production gates must not activate disk-only catalogues or format fallback as Russian."""

import json
import logging
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, messages, resolve, text
from local_operator.i18n.messages import Msg


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data = {
        "demo.words.number": "{n, number}",
        "demo.words.jobs": "{n, plural, one {# job} other {# jobs}}",
        "demo.words.secret": "English",
    }
    for locale, values in (("en", data), ("zh-CN", {"demo.words.secret": "HELD"})):
        folder = tmp_path / locale
        folder.mkdir()
        (folder / "demo.words.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "ru", "fr"))
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    text._sources.cache_clear()
    yield
    text._sources.cache_clear()


def test_ship_set_is_cached_once(tmp_path, monkeypatch):
    manifest = tmp_path / "shipped.json"
    manifest.write_text(json.dumps({"locales": ["fr"]}))
    monkeypatch.setattr(resolve, "_data_path", lambda name: manifest)
    resolve.shipped_locales.cache_clear()
    try:
        assert resolve.shipped_locales() == ("en", "fr")
        manifest.write_text(json.dumps({"locales": ["ru"]}))
        assert resolve.shipped_locales() == ("en", "fr")
        assert resolve.shipped_locales.cache_info().maxsize == 1
        resolve.shipped_locales.cache_clear()
        assert resolve.shipped_locales() == ("en", "ru")
    finally:
        resolve.shipped_locales.cache_clear()


def test_held_disk_bytes_never_activate(corpus):
    runs = []
    text.register_refresh(lambda: runs.append(text.active_locale()))
    text.set_active_locale("zh-CN")
    assert text.active_locale() == "en"
    assert runs == []
    msg = Msg("demo.words.secret")
    assert msg.envelope(locale="zh-CN")["text"] == "English"
    renderer = text.tr("demo.words")
    assert renderer(msg) == renderer(msg) == "English"
    assert catalogues.load_catalogue("zh-CN", "demo.words")[msg.code] == "HELD"


@pytest.mark.parametrize("missing", ["namespace", "key"])
@pytest.mark.parametrize(
    "code,n,expected",
    [("number", 1234.5, "1,234.5"), ("jobs", 1, "1 job"), ("jobs", 21, "21 jobs")],
)
def test_english_fallback_uses_english_semantics(corpus, missing, code, n, expected, caplog):
    if missing == "key":
        root = catalogues.catalogue_root() / "ru"
        root.mkdir()
        (root / "demo.words.json").write_text("{}", encoding="utf-8")
    msg = Msg(f"demo.words.{code}", {"n": n})
    text.set_active_locale("ru")
    with caplog.at_level(logging.DEBUG):
        assert messages.render(msg.code, msg.params, locale="ru") == expected
        assert text.tr("demo.words")(msg) == expected
    for logger in (messages.__name__, text.__name__):
        records = [r.getMessage() for r in caplog.records if r.name == logger]
        assert any(msg.code in r and "ru" in r and "en" in r for r in records)
        assert all(expected not in r for r in records)


def test_kill_switch_overrides_explicit_and_cached_locale(corpus, monkeypatch):
    root = catalogues.catalogue_root() / "fr"
    root.mkdir()
    (root / "demo.words.json").write_text(json.dumps({"demo.words.secret": "Bonjour"}))
    msg = Msg("demo.words.secret")
    renderer = text.tr("demo.words")
    text.set_active_locale("fr")
    assert renderer(msg) == "Bonjour"
    monkeypatch.setenv("LOP_I18N", "0")
    assert renderer(msg) == "English"
    assert msg.envelope(locale="fr")["text"] == "English"
    text.set_active_locale("fr")
    assert text.active_locale() == "en"
