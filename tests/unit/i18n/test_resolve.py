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
    assert got == "fr"  # wave 1 ships fr and es: the env pick survives the shipped filter


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
    # Chinese is supported but security-held: every chain path lands on en
    # until a separately reviewed disposition clears that hold.
    assert resolve.resolve_language("zh-CN", environ={}) == "en"
    assert resolve.resolve_language("auto", environ={"LOP_LANG": "zh-CN"}) == "en"
    assert resolve.resolve_language("auto", environ={"LANG": "zh_CN.UTF-8"}, system="Linux") == "en"


def test_os_is_consulted_last_and_filtered() -> None:
    assert resolve.resolve_language("auto", environ={}, system="Linux") == "en"
    # The shipped filter retains earlier waves and admits hi's complete audit.
    # Chinese's whole candidate is held outside the active package; its OS
    # tag still falls back to en without changing the resolution contract.
    assert resolve.resolve_language("auto", environ={"LANG": "ru_RU.UTF-8"}, system="Linux") == "ru"
    assert resolve.resolve_language("auto", environ={"LANG": "hi_IN.UTF-8"}, system="Linux") == "hi"
    assert resolve.resolve_language("auto", environ={"LANG": "zh_CN.UTF-8"}, system="Linux") == "en"
    # Urdu's audited relay slice ships it, so its OS tag now resolves.
    assert resolve.resolve_language("auto", environ={"LANG": "ur_PK.UTF-8"}, system="Linux") == "ur"


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
    # The whole Chinese candidate is held outside this branch, not reduced
    # to nine passed groups. Historical exec ledger-only records still do
    # not unship earlier waves, which no longer carry exec files. Urdu ships
    # on its audited relay slice alone; its other namespaces stay English.
    assert resolve.shipped_locales() == ("en", "fr", "es", "ru", "vi", "ur", "hi")


def test_held_chinese_config_and_env_resolve_to_english() -> None:
    assert resolve.resolve_language("zh-CN", environ={}, system="Linux") == "en"
    assert resolve.resolve_language("auto", environ={"LOP_LANG": "zh-CN"}, system="Linux") == "en"
    assert "zh-CN" not in {locale for _, locale in resolve.language_options()}


def test_language_options_lead_with_auto() -> None:
    options = resolve.language_options()
    assert options[0] == (resolve.DEFAULT_LANGUAGE, resolve.DEFAULT_LANGUAGE)
    assert all(locale in resolve.SUPPORTED_LOCALES for _, locale in options[1:])


# ---------------------------------------------------------------------------
# The model-facing preference line (SYS1)
# ---------------------------------------------------------------------------


def test_preferred_language_line_defaults_to_english() -> None:
    line = resolve.preferred_language_line("auto", environ={}, system="Linux")
    assert line == (
        "The user's preferred language is English (en). Reply in that language "
        "by default; if the conversation context clearly establishes another "
        "language, follow that instead."
    )


def test_preferred_language_line_accepts_supported_but_unshipped_locales() -> None:
    """The DELIBERATE divergence from ``resolve_language``, pinned by name.

    A supported locale whose translation wave has not shipped resolves to
    ``en`` for every live SURFACE (it cannot render unaudited catalogues), but
    the model line must state the user's actual preference: an OS-Chinese user
    gets en UI copy today AND a model told to answer in Chinese. A later
    "unify the two filters" edit fails HERE rather than silently muting the
    feature until zh-CN's hold clears.
    """
    assert resolve.resolve_language("zh-CN", environ={}, system="Linux") == "en"

    line = resolve.preferred_language_line("zh-CN", environ={}, system="Linux")
    assert "Chinese (Simplified) (zh-CN)" in line


def test_preferred_language_line_reads_the_env_override() -> None:
    line = resolve.preferred_language_line("auto", environ={"LOP_LANG": "vi"}, system="Linux")
    assert "Vietnamese (vi)" in line


def test_preferred_language_line_reads_the_os_when_nothing_wins_earlier() -> None:
    line = resolve.preferred_language_line("auto", environ={"LANG": "es_ES.UTF-8"}, system="Linux")
    assert "Spanish (es)" in line


def test_preferred_language_line_normalises_or_falls_back_to_en() -> None:
    # Unknown tags, Traditional Chinese (v1 falls back to en per the RFC) and
    # empty values all render the English line — never a crash, never a tag
    # the naming table does not carry.
    for tag in ("xx", "zh-TW", "zh-Hant", ""):
        assert "English (en)" in resolve.preferred_language_line(tag, environ={}, system="Linux")


def test_preferred_language_line_kill_switch_forces_en() -> None:
    line = resolve.preferred_language_line("fr", environ={"LOP_I18N": "0"}, system="Linux")
    assert "English (en)" in line


def test_preferred_language_line_never_raises_when_the_config_read_fails() -> None:
    import local_operator.i18n.resolve as mod

    def _boom() -> None:
        raise RuntimeError("config unreadable")

    original = mod.config_value
    mod.config_value = _boom  # type: ignore[assignment]
    try:
        line = resolve.preferred_language_line(None, environ={}, system="Linux")
    finally:
        mod.config_value = original  # type: ignore[assignment]
    assert "English (en)" in line


def test_every_supported_locale_has_an_english_display_name() -> None:
    # The naming table is static while SUPPORTED_LOCALES is code-owned: a new
    # supported locale must not silently render as "English" in the model line.
    for locale in resolve.SUPPORTED_LOCALES:
        assert locale in resolve._LANGUAGE_NAMES, locale


def test_preferred_language_line_shape_is_uniform_across_locales() -> None:
    # Uniform INCLUDING en: the prefix's shape must not change with the
    # language, so a language switch cannot read as a structural prompt edit.
    for locale in resolve.SUPPORTED_LOCALES:
        line = resolve.preferred_language_line(locale, environ={}, system="Linux")
        assert line.startswith("The user's preferred language is ")
        assert f"({locale})" in line
        assert line.endswith("follow that instead.")
