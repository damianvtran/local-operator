"""The language manifest and the shrink-only translation coverage gate.

``i18n/languages.json`` is the single machine-readable list of languages the
product ships or intends to ship. Agents read it to know every locale a new or
changed key must be translated into; this module fails when a language in that
manifest is missing a key that exists in English.

The repository is not fully translated today, so ``i18n/coverage-allowance.json``
records the exact missing key IDs for each ``(locale, namespace)`` pair, with a
reason each. The allowance is SHRINK-ONLY, mirroring ``i18n/baseline.json``'s
ceilings: any missing key not listed fails until it is translated or
deliberately added with a reason, and any listed key that is no longer missing
fails until it is removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from local_operator.i18n import catalogues
from local_operator.i18n.resolve import SUPPORTED_LOCALES

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "i18n" / "languages.json"
ALLOWANCE = REPO / "i18n" / "coverage-allowance.json"
STYLE_GUIDES = REPO / "i18n" / "style-guides"
SCHEMA = 1
_LANGUAGE_FIELDS = {"code", "native", "english", "dir", "status"}
_ALLOWANCE_FIELDS = {"locale", "namespace", "reason", "owner", "payoff", "missing"}


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


def _allowance_map(allowance: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    """The allowance as ``(locale, namespace) -> exact missing key IDs``.

    The key list is the whole point of the entry: a pair-level allowance would
    silently admit a new English key added to a namespace that was already
    incomplete. Entries are therefore sorted, duplicate-free and namespace
    scoped so the file stays reviewable and a key cannot be smuggled into the
    wrong pair. An empty list is invalid: a pair with no missing keys is
    complete and must be removed.
    """
    assert allowance.get("schema") == SCHEMA, f"{ALLOWANCE}: unsupported schema"
    entries = allowance.get("pairs")
    assert isinstance(entries, list), f"{ALLOWANCE}: pairs must be a list"
    listed: dict[tuple[str, str], set[str]] = {}
    for entry in entries:
        assert set(entry) == _ALLOWANCE_FIELDS, f"{ALLOWANCE}: bad fields in {entry!r}"
        assert entry["reason"].strip(), f"{ALLOWANCE}: {entry!r} has no reason"
        assert entry["owner"].strip(), f"{ALLOWANCE}: {entry!r} has no owner"
        assert entry["payoff"].strip(), f"{ALLOWANCE}: {entry!r} has no payoff"
        pair = (entry["locale"], entry["namespace"])
        assert pair not in listed, f"{ALLOWANCE}: duplicate pair {pair!r}"
        missing = entry["missing"]
        assert isinstance(missing, list), f"{ALLOWANCE}: {pair!r} missing must be a list"
        assert missing, f"{ALLOWANCE}: {pair!r} has no missing keys; remove the complete entry"
        assert all(
            isinstance(key, str) and key for key in missing
        ), f"{ALLOWANCE}: {pair!r} has a non-string or empty key"
        assert len(set(missing)) == len(missing), f"{ALLOWANCE}: {pair!r} has duplicate keys"
        assert missing == sorted(missing), f"{ALLOWANCE}: {pair!r} keys are not sorted"
        assert all(
            key.startswith(pair[1] + ".") for key in missing
        ), f"{ALLOWANCE}: {pair!r} lists a key outside its namespace"
        listed[pair] = set(missing)
    return listed


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


def _missing_keys(languages: list[dict[str, str]]) -> dict[tuple[str, str], set[str]]:
    """Actual missing key IDs per ``(locale, namespace)``.

    An absent locale directory and an absent namespace file are both all-missing;
    a present file is compared key-by-key so a partial translation still counts
    as missing. Empty English stubs are skipped: there is nothing to translate.
    """
    english = _english_messages()
    present_locales = set(catalogues.locales())
    missing: dict[tuple[str, str], set[str]] = {}
    for language in languages:
        locale = language["code"]
        if language["status"] == "source":
            continue
        carried = set(catalogues.namespaces(locale)) if locale in present_locales else set()
        for namespace, expected in english.items():
            if not expected:
                continue
            got = catalogues.load_catalogue(locale, namespace) if namespace in carried else {}
            absent = set(expected) - set(got)
            if absent:
                missing[(locale, namespace)] = absent
    return missing


class GateResult(NamedTuple):
    """Key-level findings: ``(locale, namespace, key)`` sets plus unknown pairs."""

    unlisted: set[tuple[str, str, str]]
    stale: set[tuple[str, str, str]]
    unknown: set[tuple[str, str]]


def _gate(languages: list[dict[str, str]], allowance: dict[str, Any]) -> GateResult:
    """Compare the allowance's exact key sets with what the tree actually carries.

    ``stale`` is restricted to pairs the gate actually checks, so an entry naming
    a removed locale/namespace is reported as ``unknown`` instead of masquerading
    as a key that has been translated.
    """
    actual = _missing_keys(languages)
    listed = _allowance_map(allowance)
    checked = _checked_pairs(languages)
    unknown = set(listed) - checked
    unlisted = {
        (locale, namespace, key)
        for (locale, namespace), keys in actual.items()
        for key in keys - listed.get((locale, namespace), set())
    }
    stale = {
        (locale, namespace, key)
        for (locale, namespace), keys in listed.items()
        if (locale, namespace) in checked
        for key in keys - actual.get((locale, namespace), set())
    }
    return GateResult(unlisted, stale, unknown)


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
    result = _gate(languages, _json(ALLOWANCE))
    assert not result.unlisted, (
        "Untranslated (locale, namespace, key) entries are missing from "
        f"i18n/coverage-allowance.json: {sorted(result.unlisted)}. Translate each key "
        "into every language in i18n/languages.json, or record the debt with a reason."
    )
    assert not result.stale, (
        "Allowance keys that are no longer missing must be removed: " f"{sorted(result.stale)}"
    )
    assert not result.unknown, (
        "i18n/coverage-allowance.json names pairs that do not exist in the manifest "
        f"or English catalogues: {sorted(result.unknown)}"
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


def _allowance(*entries: tuple[str, str, list[str]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "pairs": [
            {
                "locale": locale,
                "namespace": namespace,
                "reason": "synthetic control",
                "owner": "synthetic lane",
                "payoff": "2026-12-31",
                "missing": missing,
            }
            for locale, namespace, missing in entries
        ],
    }


def test_gate_fails_on_a_new_missing_key(synthetic_catalogues: Path) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    result = _gate(_LANGUAGES, _allowance())
    assert result.unlisted == {("fr", "demo", "demo.title")}
    assert not result.stale and not result.unknown


def test_gate_accepts_an_allowanced_missing_key(synthetic_catalogues: Path) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    result = _gate(_LANGUAGES, _allowance(("fr", "demo", ["demo.title"])))
    assert result == GateResult(set(), set(), set())


def test_gate_rejects_a_new_key_in_an_already_incomplete_namespace(
    synthetic_catalogues: Path,
) -> None:
    """The reviewer's discriminating control at unit level.

    The pair is already allowanced as incomplete, so a pair-level gate would
    pass; only the key-level set comparison can see the newly added English key.
    """
    en = synthetic_catalogues / "en" / "demo.json"
    en.write_text(json.dumps({"demo.title": "Title"}))
    allowance = _allowance(("fr", "demo", ["demo.title"]))
    en.write_text(json.dumps({"demo.title": "Title", "demo.new": "New"}))
    result = _gate(_LANGUAGES, allowance)
    assert result.unlisted == {("fr", "demo", "demo.new")}
    assert not result.stale and not result.unknown


def test_gate_rejects_an_allowance_key_that_is_no_longer_missing(
    synthetic_catalogues: Path,
) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    (synthetic_catalogues / "fr" / "demo.json").write_text(json.dumps({"demo.title": "Titre"}))
    result = _gate(_LANGUAGES, _allowance(("fr", "demo", ["demo.title"])))
    assert result.stale == {("fr", "demo", "demo.title")}
    assert not result.unlisted and not result.unknown


def test_gate_rejects_an_unknown_allowance_entry(synthetic_catalogues: Path) -> None:
    (synthetic_catalogues / "en" / "demo.json").write_text(json.dumps({"demo.title": "Title"}))
    result = _gate(_LANGUAGES, _allowance(("de", "demo", ["demo.title"])))
    assert result.unlisted == {("fr", "demo", "demo.title")}
    assert result.unknown == {("de", "demo")}
    assert not result.stale


def test_allowance_requires_a_reason(synthetic_catalogues: Path) -> None:
    allowance = _allowance(("fr", "demo", ["demo.title"]))
    allowance["pairs"][0]["reason"] = "  "
    with pytest.raises(AssertionError, match="has no reason"):
        _allowance_map(allowance)


@pytest.mark.parametrize("field", ["owner", "payoff"])
def test_allowance_requires_owner_and_payoff(field: str, synthetic_catalogues: Path) -> None:
    allowance = _allowance(("fr", "demo", ["demo.title"]))
    allowance["pairs"][0][field] = "  "
    with pytest.raises(AssertionError, match=f"has no {field}"):
        _allowance_map(allowance)


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ([], "no missing keys"),
        (["demo.title", "demo.title"], "duplicate keys"),
        (["demo.title", "demo.alt"], "not sorted"),
        (["other.title"], "outside its namespace"),
    ],
)
def test_allowance_rejects_malformed_missing_lists(missing: list[str], match: str) -> None:
    allowance = _allowance(("fr", "demo", missing))
    with pytest.raises(AssertionError, match=match):
        _allowance_map(allowance)


def test_allowance_rejects_a_duplicate_pair() -> None:
    allowance = _allowance(
        ("fr", "demo", ["demo.one"]),
        ("fr", "demo", ["demo.two"]),
    )
    with pytest.raises(AssertionError, match="duplicate pair"):
        _allowance_map(allowance)


def test_allowance_growth_reports_only_added_key_ids(i18n_check) -> None:
    base = {"pairs": [{"locale": "fr", "namespace": "demo", "missing": ["demo.one"]}]}
    head = {"pairs": [{"locale": "fr", "namespace": "demo", "missing": ["demo.one", "demo.two"]}]}
    assert i18n_check.allowance_growth(base, head) == {("fr", "demo", "demo.two")}


def test_allowance_growth_skips_without_a_base_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, i18n_check
) -> None:
    """An isolated fixture or exported tarball has no base ref; that is a skip.

    Treating an unknown base as a catalogue problem broke every ``check.py``
    caller that runs in a temp tree, so this pins the non-fatal path.
    """
    monkeypatch.setattr(i18n_check, "REPO", tmp_path)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    assert i18n_check.check_allowance_growth("definitely-not-a-ref") == []
