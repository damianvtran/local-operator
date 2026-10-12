"""`i18n.text` — the contracts the shared render helper documents.

`tr(namespace)` is the module every extraction slice resolves its copy
through; beyond the unknown-KEY degradation the T1 slice test pins, these
tests hold it to its docstring: ONE catalogue read per (root, namespace,
locale) shared across resolvers and stable across resolutions, a cache keyed
by the catalogue ROOT and by LOCALE (a re-pointed root resolves from its own
tree; a moved locale reads the new tree), no-raise degradation for every
broken shape — unknown namespace, missing key, malformed file, unrenderable
message, unshipped locale — and the LIVE path the C slice added:
``set_active_locale`` flips resolution, re-runs registered refreshers exactly
once per move, and a same-locale call is a no-op.
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
    # The active locale and the refresher registry are PROCESS state (the
    # imported widgets own real entries). The registry is emptied for the
    # test body — a locale move here must not re-materialise the widgets'
    # constants against this file's temp catalogue roots — and restored
    # afterwards, moved back to `en` while it is still empty so the restore
    # runs no refresher against a monkeypatched root.
    saved = list(text._refreshers)
    text._refreshers.clear()
    yield
    text._sources.cache_clear()
    text.set_active_locale("en")
    text._refreshers[:] = saved


@pytest.mark.parametrize("count", [0, 1, 2, 3, 1_000_000])
def test_hub_group_english_clauses_keep_existing_decimal_bytes(count: int) -> None:
    from local_operator.i18n.keys import tui_transcript

    text.set_active_locale("en")
    render = text.tr("tui.transcript")
    # Numeric selectors make translated noun forms correct. Their separately
    # rendered label must not acquire grouping separators in existing EN frames.
    for message, suffix in (
        (tui_transcript.group_actions(count=count, label=str(count)), "actions"),
        (
            tui_transcript.group_reply_requested(count=count, label=str(count)),
            "reply requested",
        ),
        (tui_transcript.group_steered(count=count, label=str(count)), "steered"),
    ):
        assert render(message) == f"{count} {suffix}"


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
    # Construction is LAZY since the live-language slice: root and locale are
    # read per resolution, so the one read lands on the first resolution and
    # every later one is served from the cache.
    assert calls == [], "constructing resolvers reads nothing"
    assert first(Msg("tui.welcome.tip.resume", {}))
    assert calls == ["tui.welcome"], "one read serves the first resolution"
    assert second(Msg("tui.welcome.tip.paste", {}))
    assert calls == ["tui.welcome"], "resolutions must add no reads"
    assert text.tr("wire.slash")(Msg("wire.slash.help", {}))
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
    # A resolver built under `a` (the cache is what must distinguish the two
    # roots, not resolver identity) still follows a re-point: root and locale
    # are read per call.
    resolver = text.tr("demo.words")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path / "a")
    assert resolver(Msg("demo.words.hello", {})) == "A"
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path / "b")
    assert resolver(Msg("demo.words.hello", {})) == "B"


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


def test_set_active_locale_flips_resolution_and_runs_refreshers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in (("en", "Hello"), ("fr", "Bonjour")):
        catalogue = tmp_path / name / "demo.words.json"
        catalogue.parent.mkdir(parents=True)
        catalogue.write_text(json.dumps({"demo.words.hello": value}), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    runs: list[str] = []
    text.register_refresh(lambda: runs.append(text.active_locale()))
    resolver = text.tr("demo.words")
    assert resolver(Msg("demo.words.hello", {})) == "Hello"
    assert text.active_locale() == "en"
    text.set_active_locale("fr")
    assert runs == ["fr"], "one move, one refresh pass, after the locale moved"
    assert resolver(Msg("demo.words.hello", {})) == "Bonjour"
    text.set_active_locale("fr")
    assert runs == ["fr"], "a same-locale call must not re-run refreshers"
    text.set_active_locale("en")
    assert runs == ["fr", "en"]
    assert resolver(Msg("demo.words.hello", {})) == "Hello"


def test_a_shipped_locale_missing_its_namespace_falls_back_to_english(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "en").mkdir(parents=True)
    (tmp_path / "en" / "demo.words.json").write_text(
        json.dumps({"demo.words.hello": "Hello"}), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    text.set_active_locale("fr")
    assert text.tr("demo.words")(Msg("demo.words.hello", {})) == "Hello"
    text.set_active_locale("en")
    assert text.tr("demo.words")(Msg("demo.words.hello", {})) == "Hello"
