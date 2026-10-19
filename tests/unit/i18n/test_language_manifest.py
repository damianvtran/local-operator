"""The language manifest and the shrink-only translation coverage gate.

``i18n/languages.json`` is the single machine-readable list of languages the
product ships or intends to ship. Agents read it to know every locale a new or
changed key must be translated into; this module fails when a language in that
manifest is missing a key that exists in English.

The repository is not fully translated today, so ``i18n/coverage-allowance.json``
records the missing ``(locale, namespace)`` pairs with a reason each. The
allowance is SHRINK-ONLY, mirroring ``i18n/baseline.json``'s ceilings: a newly
missing pair fails until it is translated or deliberately added with a reason,
and a listed pair that has become complete fails until its entry is removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n.resolve import SUPPORTED_LOCALES

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "i18n" / "languages.json"
ALLOWANCE = REPO / "i18n" / "coverage-allowance.json"
STYLE_GUIDES = REPO / "i18n" / "style-guides"
SCHEMA = 1
_LANGUAGE_FIELDS = {"code", "native", "english", "dir", "status"}
_ALLOWANCE_FIELDS = {"locale", "namespace", "reason"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _languages(manifest: dict[str, Any]) -> list[dict[str, str]]:
    assert manifest.get("schema") == SCHEMA, f"{MANIFEST}: unsupported schema"
    languages = manifest.get("languages")
    assert isinstance(languages, list) and languages, f"{MANIFEST}: no languages"
    for language in languages:
        assert set(language) == _LANGUAGE_FIELDS, f"{MANIFEST}: bad fields in {language!r}"
        assert language["native"].strip() and language["english"].strip()
        assert language["dir"] in {"ltr", "rtl"}
        assert language["status"] in {"source", "supported"}
    return languages


def _allowance_pairs(allowance: dict[str, Any]) -> set[tuple[str, str]]:
    assert allowance.get("schema") == SCHEMA, f"{ALLOWANCE}: unsupported schema"
    entries = allowance.get("pairs")
    assert isinstance(entries, list), f"{ALLOWANCE}: pairs must be a list"
    pairs: set[tuple[str, str]] = set()
    for entry in entries:
        assert set(entry) == _ALLOWANCE_FIELDS, f"{ALLOWANCE}: bad fields in {entry!r}"
        assert entry["reason"].strip(), f"{ALLOWANCE}: {entry!r} has no reason"
        pair = (entry["locale"], entry["namespace"])
        assert pair not in pairs, f"{ALLOWANCE}: duplicate pair {pair!r}"
        pairs.add(pair)
    return pairs


def _english_messages() -> dict[str, dict[str, str]]:
    return {
        namespace: catalogues.load_catalogue("en", namespace)
        for namespace in catalogues.namespaces("en")
    }


def _checked_pairs(languages: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Every ``(target locale, populated English namespace)`` pair."""
    populated = [namespace for namespace, messages in _english_messages().items() if messages]
    return {
        (language["code"], namespace)
        for language in languages
        if language["status"] != "source"
        for namespace in populated
    }


def _missing_pairs(languages: list[dict[str, str]]) -> set[tuple[str, str]]:
    """Pairs where a target locale lacks at least one English key.

    An absent locale directory and an absent namespace file are both all-missing;
    a present file is compared key-by-key so a partial translation still counts
    as missing. Empty English stubs are skipped: there is nothing to translate.
    """
    english = _english_messages()
    present_locales = set(catalogues.locales())
    missing: set[tuple[str, str]] = set()
    for language in languages:
        locale = language["code"]
        if language["status"] == "source":
            continue
        carried = set(catalogues.namespaces(locale)) if locale in present_locales else set()
        for namespace, expected in english.items():
            if not expected:
                continue
            got = catalogues.load_catalogue(locale, namespace) if namespace in carried else {}
            if set(expected) - set(got):
                missing.add((locale, namespace))
    return missing


def _gate(
    languages: list[dict[str, str]], allowance: dict[str, Any]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    """Return ``(unlisted missing, complete allowance entries, unknown entries)``.

    ``complete`` is restricted to pairs the gate actually checks, so an entry
    naming a removed locale/namespace is reported as ``unknown`` instead of
    masquerading as complete.
    """
    missing = _missing_pairs(languages)
    allowed = _allowance_pairs(allowance)
    checked = _checked_pairs(languages)
    unlisted = missing - allowed
    unknown = allowed - checked
    complete = (allowed & checked) - missing
    return unlisted, complete, unknown


def test_manifest_is_schema_versioned_and_matches_the_runtime_list() -> None:
    languages = _languages(_json(MANIFEST))
    codes = [language["code"] for language in languages]
    assert len(codes) == len(set(codes)), f"{MANIFEST}: duplicate locale codes"
    # The runtime's recognition vocabulary and the manifest are two views of
    # one list; drift would let an agent translate into a locale the resolver
    # cannot select (or miss one it can).
    assert set(codes) == set(SUPPORTED_LOCALES), (
        f"{MANIFEST} and local_operator/i18n/resolve.SUPPORTED_LOCALES disagree: "
        f"manifest-only={sorted(set(codes) - set(SUPPORTED_LOCALES))}, "
        f"runtime-only={sorted(set(SUPPORTED_LOCALES) - set(codes))}"
    )
    assert set(catalogues.locales()) <= set(codes), (
        f"catalogue directories are not declared in {MANIFEST}: "
        f"{sorted(set(catalogues.locales()) - set(codes))}"
    )
    sources = [language for language in languages if language["status"] == "source"]
    assert [language["code"] for language in sources] == ["en"]
    directions = {language["code"]: language["dir"] for language in languages}
    assert directions["ur"] == "rtl"
    assert all(direction == "ltr" for code, direction in directions.items() if code != "ur")


def test_every_manifest_language_has_a_style_guide() -> None:
    # "Adding a language" is the manifest row PLUS its binding style guide
    # (i18n/style-guides/en.md records the source voice the others translate
    # from). A manifest entry without one asks translators to guess.
    for language in _languages(_json(MANIFEST)):
        guide = STYLE_GUIDES / f"{language['code']}.md"
        assert guide.is_file(), f"{MANIFEST}: {language['code']} has no {guide.relative_to(REPO)}"


def test_every_missing_key_is_covered_by_the_shrink_only_allowance() -> None:
    languages = _languages(_json(MANIFEST))
    unlisted, complete, unknown = _gate(languages, _json(ALLOWANCE))
    assert not unlisted, (
        "Untranslated (locale, namespace) pairs are missing from "
        f"i18n/coverage-allowance.json: {sorted(unlisted)}. Translate them into every "
        "language in i18n/languages.json, or add each pair with a reason."
    )
    assert not complete, (
        "Shrink-only allowance entries are now complete and must be removed: "
        f"{sorted(complete)}"
    )
    assert not unknown, (
        "i18n/coverage-allowance.json names pairs that do not exist in the manifest "
        f"or English catalogues: {sorted(unknown)}"
    )


@pytest.fixture()
def synthetic_catalogues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir()
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    return root


_LANGUAGES = [
    {"code": "en", "native": "English", "english": "English", "dir": "ltr", "status": "source"},
    {"code": "fr", "native": "Français", "english": "French", "dir": "ltr", "status": "supported"},
]


def _allowance(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "pairs": [
            {"locale": locale, "namespace": namespace, "reason": "synthetic control"}
            for locale, namespace in pairs
        ],
    }


def test_gate_fails_on_a_new_missing_pair(synthetic_catalogues: Path) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    assert _gate(_LANGUAGES, _allowance()) == ({("fr", "demo")}, set(), set())


def test_gate_accepts_an_allowanced_missing_pair(synthetic_catalogues: Path) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    assert _gate(_LANGUAGES, _allowance(("fr", "demo"))) == (set(), set(), set())


def test_gate_rejects_an_allowance_entry_that_has_become_complete(
    synthetic_catalogues: Path,
) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    (synthetic_catalogues / "fr" / "demo.json").write_text(json.dumps({"demo.title": "Titre"}))
    unlisted, complete, unknown = _gate(_LANGUAGES, _allowance(("fr", "demo")))
    assert (unlisted, complete, unknown) == (set(), {("fr", "demo")}, set())


def test_gate_rejects_an_unknown_allowance_entry(synthetic_catalogues: Path) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    unlisted, complete, unknown = _gate(_LANGUAGES, _allowance(("de", "demo")))
    assert (unlisted, complete, unknown) == ({("fr", "demo")}, set(), {("de", "demo")})


def test_allowance_requires_a_reason(synthetic_catalogues: Path) -> None:
    allowance = _allowance(("fr", "demo"))
    allowance["pairs"][0]["reason"] = "  "
    with pytest.raises(AssertionError, match="has no reason"):
        _allowance_pairs(allowance)
