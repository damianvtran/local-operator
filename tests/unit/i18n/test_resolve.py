"""The resolver truth table: normalisation, the source chain, and the filters.

Every OS branch is exercised with injected inputs (`system`/`environ`/`home`),
so the table is deterministic on any host and CI never reads a real
preferences file or shells out by accident.
"""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from local_operator.i18n import resolve

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "want"),
    [
        ("en", "en"),
        ("fr-CA", "fr"),
        ("fr_FR.UTF-8", "fr"),
        ("ru@modifier", "ru"),
        ("zh", "zh-CN"),
        ("zh-CN", "zh-CN"),
        ("zh-Hans", "zh-CN"),
        ("zh-TW", None),
        ("zh-Hant", None),
        ("zh-Hant-TW", None),
        ("de", None),
        ("pt-BR", None),
        ("C", None),
        ("POSIX", None),
        ("", None),
        (None, None),
    ],
)
def test_normalise(tag: str | None, want: str | None) -> None:
    assert resolve.normalise_locale(tag) == want


# ---------------------------------------------------------------------------
# OS readers (injected)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environ", "want"),
    [
        ({"LC_ALL": "fr_FR.UTF-8", "LANG": "de_DE.UTF-8"}, "fr"),
        ({"LC_MESSAGES": "ru_RU.UTF-8", "LANG": "de_DE.UTF-8"}, "ru"),
        ({"LANG": "zh_CN.UTF-8"}, "zh-CN"),
        ({"LANG": "zh_TW.UTF-8"}, None),
        ({"LANG": "de_DE.UTF-8", "LANGUAGE": "es:fr"}, "es"),
        ({"LANGUAGE": "fr:es"}, "fr"),
        ({}, None),
    ],
)
def test_linux_environment(environ: dict[str, str], want: str | None) -> None:
    assert resolve.os_language(system="Linux", environ=environ) == want


def test_macos_plist_preferred(tmp_path: Path) -> None:
    home = tmp_path
    prefs = home / "Library" / "Preferences" / ".GlobalPreferences.plist"
    prefs.parent.mkdir(parents=True)
    with open(prefs, "wb") as handle:
        plistlib.dump({"AppleLanguages": ["fr-CA", "en-US"]}, handle)
    assert resolve.os_language(system="Darwin", environ={}, home=home) == "fr"


def test_macos_plist_with_only_unsupported_entries_is_none(tmp_path: Path) -> None:
    home = tmp_path
    prefs = home / "Library" / "Preferences" / ".GlobalPreferences.plist"
    prefs.parent.mkdir(parents=True)
    with open(prefs, "wb") as handle:
        plistlib.dump({"AppleLanguages": ["de-DE"]}, handle)
    assert resolve.os_language(system="Darwin", environ={}, home=home) is None


def test_macos_defaults_fallback_parses_quoted_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `defaults read -g` fallback when the plist is absent."""

    class Result:
        returncode = 0
        stdout = '(\n    "fr-CA",\n    "en-US"\n)\n'

    monkeypatch.setattr(resolve.subprocess, "run", lambda *a, **k: Result(), raising=True)
    got = resolve.os_language(system="Darwin", environ={}, home=tmp_path)
    assert got == "fr"


def test_windows_reader_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolve, "_windows_language", lambda: "fr")
    assert resolve.os_language(system="Windows", environ={}) == "fr"


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_config_tag_wins_over_env_and_os() -> None:
    import local_operator.i18n.resolve as mod

    original = mod.shipped_locales
    mod.shipped_locales = lambda: ("en", "fr", "ru")  # type: ignore[assignment]
    try:
        got = resolve.resolve_language("ru", environ={"LOP_LANG": "fr"}, system="Linux")
        assert got == "ru"
    finally:
        mod.shipped_locales = original  # type: ignore[assignment]


def test_env_wins_over_os() -> None:
    got = resolve.resolve_language(
        "auto", environ={"LOP_LANG": "fr", "LANG": "es_ES.UTF-8"}, system="Linux"
    )
    assert got == "en"  # fr/es are not shipped yet: the filter below is the last word


def test_env_beats_os_when_config_is_auto() -> None:
    # With a shipped set widened, the order is observable: LOP_LANG wins over
    # the OS environment.
    import local_operator.i18n.resolve as mod

    original = mod.shipped_locales
    mod.shipped_locales = lambda: ("en", "fr", "es")  # type: ignore[assignment]
    try:
        got = resolve.resolve_language(
            "auto", environ={"LOP_LANG": "fr", "LANG": "es_ES.UTF-8"}, system="Linux"
        )
        assert got == "fr"
    finally:
        mod.shipped_locales = original  # type: ignore[assignment]


def test_unsupported_config_falls_through_to_env() -> None:
    import local_operator.i18n.resolve as mod

    original = mod.shipped_locales
    mod.shipped_locales = lambda: ("en", "fr")  # type: ignore[assignment]
    try:
        assert resolve.resolve_language("de", environ={"LOP_LANG": "fr"}) == "fr"
    finally:
        mod.shipped_locales = original  # type: ignore[assignment]


def test_unshipped_locale_falls_back_to_en() -> None:
    # fr is supported but NOT shipped today: every chain path lands on en.
    assert resolve.resolve_language("fr", environ={}) == "en"
    assert resolve.resolve_language("auto", environ={"LOP_LANG": "fr"}) == "en"
    assert resolve.resolve_language("auto", environ={"LANG": "fr_FR.UTF-8"}, system="Linux") == "en"


def test_os_is_consulted_last_and_filtered() -> None:
    assert resolve.resolve_language("auto", environ={}, system="Linux") == "en"
    assert resolve.resolve_language("auto", environ={"LANG": "ru_RU.UTF-8"}, system="Linux") == "en"


# ---------------------------------------------------------------------------
# The config read (the real path, isolated)
# ---------------------------------------------------------------------------


def test_config_value_reads_language_from_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store's shape: every setting lives under the `values:` mapping, so
    # `language` is `values.language` (a nested path, unlike the flat-dotted
    # `display.*` keys). A top-level `language:` is warned about by the store
    # itself and does nothing — this test pins the shape the resolver reads.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "config.yml").write_text("values:\n  language: fr\n", encoding="utf-8")
    import local_operator.i18n.resolve as mod

    original = mod.shipped_locales
    mod.shipped_locales = lambda: ("en", "fr")  # type: ignore[assignment]
    try:
        assert resolve.resolve_language() == "fr"
    finally:
        mod.shipped_locales = original  # type: ignore[assignment]


def test_missing_config_resolves_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LOP_LANG", raising=False)
    assert resolve.resolve_language() == "en"


def test_kill_switch_forces_en_over_config_and_env() -> None:
    # RFC §9: `LOP_I18N=0` is the operational control and outranks every other
    # source, config included.
    import local_operator.i18n.resolve as mod

    original = mod.shipped_locales
    mod.shipped_locales = lambda: ("en", "fr")  # type: ignore[assignment]
    try:
        got = resolve.resolve_language(
            "fr", environ={"LOP_I18N": "0", "LOP_LANG": "fr"}, system="Linux"
        )
        assert got == "en"
        # Any other value (including `1`) leaves the chain alone.
        got = resolve.resolve_language(
            "fr", environ={"LOP_I18N": "1", "LOP_LANG": "fr"}, system="Linux"
        )
        assert got == "fr"
    finally:
        mod.shipped_locales = original  # type: ignore[assignment]


def test_shipped_locales_reads_generated_set() -> None:
    # The committed data ships `en` alone in M0 (the mechanism is dark).
    assert resolve.shipped_locales() == ("en",)


def test_language_options_lead_with_auto() -> None:
    options = resolve.language_options()
    assert options[0] == (resolve.DEFAULT_LANGUAGE, resolve.DEFAULT_LANGUAGE)
    assert all(locale in resolve.SUPPORTED_LOCALES for _, locale in options[1:])
