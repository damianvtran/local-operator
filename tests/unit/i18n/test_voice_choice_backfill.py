"""Voice display copy must localize without changing the model/storage vocabulary.

This intentionally starts red on carried English copy: comparing only catalogue
and registry would let English-in-both silently pass. Source IDs, not guessed
translations, define the contract; translators own the target wording.
"""

from __future__ import annotations

import json

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.i18n import catalogues, resolve, text

LOCALES = ("fr", "es", "zh-CN", "ru", "vi", "ur", "hi")
CHOICES = {
    "tone": ("warm", "neutral", "bright", "calm", "authoritative"),
    "expressiveness": ("low", "medium", "high"),
}
KEYS = tuple(
    f"wire.settings.speech_voice_{group}.choice.{value}.{field}"
    for group, values in CHOICES.items()
    for value in values
    for field in ("label", "description")
)


@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Use real catalogues/refreshing, but never the operator's config resolver."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    monkeypatch.delenv("LOP_LANG", raising=False)
    monkeypatch.setattr(resolve, "config_value", lambda: None)
    monkeypatch.setattr(text, "_refreshers", [settings_io._refresh_registry_copy])
    original = text.active_locale()
    original_root = catalogues.catalogue_root()
    try:
        yield
    finally:
        # Restore the real root before refreshing frozen global records; pytest
        # restores monkeypatch attributes only after this fixture's finalizer.
        monkeypatch.setattr(catalogues, "_CATALOGUES", original_root)
        text.set_active_locale(original)
        settings_io._refresh_registry_copy()


def _assert_display_and_ids(locale, manager):
    english = catalogues.load_catalogue("en", "wire.settings")
    target = catalogues.load_catalogue(locale, "wire.settings")
    assert len(KEYS) == 16
    missing = [key for key in KEYS if key not in target]
    assert not missing, f"{locale}: missing target copy: {missing}"
    for group, values in CHOICES.items():
        setting = settings_io.BY_KEY[f"speech.voice.{group}"]
        assert tuple(choice.value for choice in setting.choices) == values
        for choice in setting.choices:
            prefix = f"wire.settings.speech_voice_{group}.choice.{choice.value}"
            assert str(choice.label) == target[f"{prefix}.label"]
            assert str(choice.description) == target[f"{prefix}.description"]
            assert settings_io.coerce(setting, choice.value) == choice.value
            assert settings_io.validate(setting, choice.value) is None
            settings_io.write_setting(manager, setting, choice.value, notify=False)
            assert settings_io.read_setting(manager, setting) == choice.value
            # Re-open the persisted config, not merely an in-memory snapshot.
            assert (
                settings_io.read_setting(ConfigManager(manager.config_dir), setting) == choice.value
            )
            if choice.label not in values:
                assert settings_io.validate(setting, str(choice.label)) is not None
        for field in ("label", "description"):
            keys = [
                f"wire.settings.speech_voice_{group}.choice.{value}.{field}" for value in values
            ]
            # A cognate may legitimately equal English. Reject wholesale English
            # fallback, not native wording: independent audits own each key.
            assert any(
                target[key] != english[key] for key in keys
            ), f"{locale}: English display copy remains for {group} {field}: {keys}"


@pytest.mark.parametrize("locale", LOCALES)
def test_carried_voice_choices_reach_registry(isolated_registry, tmp_path, locale):
    assert locale in resolve.shipped_locales()
    text.set_active_locale(locale)
    assert text.active_locale() == locale
    _assert_display_and_ids(locale, ConfigManager(tmp_path / "stored"))


def test_registry_seam_and_english_fallback_negative_control(
    isolated_registry, tmp_path, monkeypatch
):
    """Synthetic markers prove wiring, without pretending to translate French.

    The same fixture must pass on French and fail on forced English; a failure
    cannot be attributed to the pre-existing target catalogue debt.
    """
    root = tmp_path / "catalogues"
    english = catalogues.load_catalogue("en", "wire.settings")
    for locale in ("en", "fr"):
        (root / locale).mkdir(parents=True)
        values = dict(english)
        if locale == "fr":
            values.update({key: f"localized-marker-{index}" for index, key in enumerate(KEYS)})
        (root / locale / "wire.settings.json").write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    text.set_active_locale("en")
    text.set_active_locale("fr")
    manager = ConfigManager(tmp_path / "stored")
    _assert_display_and_ids("fr", manager)
    text.set_active_locale("en")
    with pytest.raises(AssertionError):
        _assert_display_and_ids("fr", manager)
