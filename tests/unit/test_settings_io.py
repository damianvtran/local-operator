"""Tests for the settings schema registry and its write facade.

The two that matter most, and why:

- **Anti-drift.** Every ``Setting.default`` is compared against what its actual
  consumer defaults to. A registry that claims ``retry.maxRetries`` defaults to
  10 while ``RetrySettings`` uses 4 would paint a page of confident lies, and
  the divergence would appear silently on the next change to either side.
- **The ``display.*`` flat-key round trip.** ``display.shimmer`` is a literal
  dotted TOP-LEVEL key, not a nested one. A facade that split it would write a
  ``display:`` mapping nothing reads — a failure that looks exactly like
  success from every angle except the one that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from local_operator import settings_io
from local_operator.config import DEFAULT_CONFIG, ConfigManager
from local_operator.settings_io import Kind


@pytest.fixture()
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path)


def _consumer_defaults() -> dict[str, object]:
    """What each setting's REAL consumer falls back to when the key is absent.

    Built by asking the consumers themselves rather than by restating their
    values here — a second hard-coded table would drift from the first and the
    test would then guard nothing.
    """
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness.jobs import DEFAULT_MAX_RUNNING_JOBS
    from local_operator.providers.failover import (
        CONNECTIVITY_BACKOFF_CAP_MS,
        CONNECTIVITY_MAX_RETRIES,
        RetrySettings,
    )
    from local_operator.web_fetch.models import DEFAULT_WEB_FETCH_CONFIG
    from local_operator.web_search.models import DEFAULT_WEB_SEARCH_CONFIG

    retry = RetrySettings()
    compaction = CompactionSettings()
    consumers: dict[str, object] = {
        "retry.enabled": retry.enabled,
        "retry.maxRetries": retry.max_retries,
        "retry.baseDelayMs": retry.base_delay_ms,
        "retry.connectivityMaxRetries": CONNECTIVITY_MAX_RETRIES,
        "retry.connectivityBackoffCapMs": CONNECTIVITY_BACKOFF_CAP_MS,
        "retry.modelFallback": retry.model_fallback,
        "retry.usageAwareFallback": retry.usage_aware_fallback,
        "retry.usageReservePercent": retry.usage_reserve_percent,
        "retry.fallbackChains": dict(retry.fallback_chains),
        "subagents.max_running": DEFAULT_MAX_RUNNING_JOBS,
        "providers.openai.api": DEFAULT_CONFIG.values["providers"]["openai"]["api"],
    }
    for field in type(compaction).model_fields:
        consumers[f"compaction.{field}"] = getattr(compaction, field)
    for key, value in DEFAULT_WEB_SEARCH_CONFIG.items():
        consumers[f"web_search.{key}"] = value
    for key, value in DEFAULT_WEB_FETCH_CONFIG.items():
        consumers[f"web_fetch.{key}"] = value
    for key, value in DEFAULT_CONFIG.values.items():
        if not isinstance(value, dict):
            consumers.setdefault(key, value)
    return consumers


@pytest.mark.parametrize("setting", settings_io.SETTINGS, ids=lambda s: s.key)
def test_every_default_matches_its_consumer(setting) -> None:
    """A registry default that disagrees with its consumer is a painted lie."""
    consumers = _consumer_defaults()
    if setting.key not in consumers:
        pytest.skip(f"{setting.key} has no single-value consumer to compare against")
    assert setting.default == consumers[setting.key], (
        f"{setting.key}: registry says {setting.default!r}, "
        f"consumer defaults to {consumers[setting.key]!r}"
    )


def test_display_keys_are_flat_dotted() -> None:
    """THE trap. Every display flag's path is ONE element containing a dot."""
    for key in settings_io.flat_dotted_keys():
        setting = settings_io.BY_KEY[key]
        assert len(setting.path) == 1, f"{key} was split into {setting.path}"
        assert "." in setting.path[0]
    assert set(settings_io.flat_dotted_keys()) >= {
        "display.shimmer",
        "display.nerd_icons",
        "display.terminal_title",
        "display.images",
        "display.notifications",
    }


def test_display_flag_round_trips_through_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: a written display flag is READ BACK by the fast path.

    Writing it as a nested ``display:`` mapping would pass a test that only
    checked the file, and fail here — ``tui.settings.settings_get`` reads the
    flat key and nothing reads the nested one. The reload is asserted too: the
    facade must drop the process cache or the running TUI keeps the old value.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.tui.settings import settings_get, settings_reload

    settings_reload()
    manager = ConfigManager(tmp_path)
    assert settings_get("display.shimmer") is True

    settings_io.write_setting(manager, settings_io.BY_KEY["display.shimmer"], False)

    # Flat on disk, under the literal dotted key.
    stored = yaml.safe_load((tmp_path / "config.yml").read_text())["values"]
    assert stored["display.shimmer"] is False
    assert "display" not in stored, "wrote a nested mapping nothing reads"

    # And visible to the reader WITHOUT a manual reload, because the facade
    # invalidated the cache itself.
    assert settings_get("display.shimmer") is False


def test_nested_write_preserves_siblings(manager: ConfigManager) -> None:
    """``_load_config`` back-fills missing TOP-LEVEL keys only, so a partial
    ``retry`` block never regains its siblings. A replacing writer would delete
    them and nothing would notice until a failover did not happen."""
    settings_io.write_setting(manager, settings_io.BY_KEY["retry.maxRetries"], 4)
    retry = manager.get_config_value("retry")
    assert retry["maxRetries"] == 4
    for sibling in ("enabled", "baseDelayMs", "modelFallback", "fallbackChains"):
        assert sibling in retry, f"{sibling} was destroyed by the write"


def test_deeply_nested_write_preserves_siblings(manager: ConfigManager) -> None:
    """Three levels down (``providers.openai.api``) follows the same rule."""
    manager.set_config_value("providers", {"openai": {"api": "responses", "extra": 1}})
    settings_io.write_setting(
        manager, settings_io.BY_KEY["providers.openai.api"], "chat_completions"
    )
    providers = manager.get_config_value("providers")
    assert providers["openai"]["api"] == "chat_completions"
    assert providers["openai"]["extra"] == 1


def test_reset_deletes_rather_than_writing_the_default(manager: ConfigManager) -> None:
    """Absence and an explicit value differ for the tri-state, and a config
    carrying only what its owner chose stays readable by hand."""
    setting = settings_io.BY_KEY["display.shimmer"]
    settings_io.write_setting(manager, setting, False)
    assert "display.shimmer" in manager.get_config().values
    settings_io.reset_setting(manager, setting)
    assert "display.shimmer" not in manager.get_config().values
    assert settings_io.read_setting(manager, setting) is True


def test_nerd_icons_auto_writes_nothing(manager: ConfigManager) -> None:
    """The tri-state's ``auto`` is ABSENCE, not an explicit null.

    ``settings_get`` returns None only when the key is missing, so storing
    ``null`` would report an explicit choice where the user asked for the
    automatic one.
    """
    setting = settings_io.BY_KEY["display.nerd_icons"]
    settings_io.write_setting(manager, setting, True)
    assert manager.get_config().values["display.nerd_icons"] is True
    settings_io.write_setting(manager, setting, None)
    assert "display.nerd_icons" not in manager.get_config().values


def test_validation_rejects_out_of_range(manager: ConfigManager) -> None:
    """Bounds are enforced here because the consumers clamp SILENTLY: a stored
    500 that the tool reads as 120 is the config and the behaviour disagreeing
    with nothing on screen admitting it."""
    setting = settings_io.BY_KEY["web_search.timeout_seconds"]
    assert settings_io.validate(setting, 500.0) is not None
    with pytest.raises(ValueError):
        settings_io.write_setting(manager, setting, 500.0)
    assert "web_search" not in manager.get_config().values or (
        manager.get_config_value("web_search", {}).get("timeout_seconds") != 500.0
    )


def test_validation_rejects_unknown_enum_and_list_members(manager: ConfigManager) -> None:
    assert settings_io.validate(settings_io.BY_KEY["compaction.strategy"], "nope") is not None
    assert settings_io.validate(settings_io.BY_KEY["web_search.providers"], ["bing"]) is not None
    assert settings_io.validate(settings_io.BY_KEY["web_search.providers"], []) is not None


def test_retired_settings_cannot_be_written(manager: ConfigManager) -> None:
    setting = settings_io.BY_KEY["session_retention_max_sessions"]
    with pytest.raises(ValueError):
        settings_io.write_setting(manager, setting, 5)
    with pytest.raises(ValueError):
        settings_io.reset_setting(manager, setting)


def test_coerce_parses_what_a_user_types() -> None:
    assert settings_io.coerce(settings_io.BY_KEY["retry.maxRetries"], " 7 ") == 7
    assert settings_io.coerce(settings_io.BY_KEY["display.shimmer"], "off") is False
    assert settings_io.coerce(settings_io.BY_KEY["web_search.providers"], "brave, exa") == [
        "brave",
        "exa",
    ]
    # Stable de-duplication, matching coerce_search_settings: a repeated
    # provider is a typo, not a request to weight it twice.
    assert settings_io.coerce(settings_io.BY_KEY["web_search.providers"], "exa,exa") == ["exa"]
    with pytest.raises(ValueError):
        settings_io.coerce(settings_io.BY_KEY["retry.maxRetries"], "ten")


def test_cascade_round_trips_and_survives_normalization(manager: ConfigManager) -> None:
    """A chain written here must be one ``providers/failover.py`` will accept:
    a page that stored a shape the failover layer drops would show a cascade
    that does not exist."""
    from local_operator.providers.failover import RetrySettings

    settings_io.write_chains(
        manager,
        {"default": ["anthropic/claude-opus-5", "openrouter/deepseek/deepseek-chat"]},
    )
    assert settings_io.read_chains(manager) == {
        "default": ["anthropic/claude-opus-5", "openrouter/deepseek/deepseek-chat"]
    }
    resolved = RetrySettings.from_settings(manager.get_config().values)
    assert list(resolved.fallback_chains["default"]) == [
        "anthropic/claude-opus-5",
        "openrouter/deepseek/deepseek-chat",
    ]


def test_empty_chain_is_dropped(manager: ConfigManager) -> None:
    """``_normalize_chains`` drops it on read, so storing it would put a row in
    the file that the page shows and the failover layer does not have."""
    settings_io.write_chains(manager, {"gone": [], "kept": ["anthropic/claude-opus-5"]})
    assert set(settings_io.read_chains(manager)) == {"kept"}


def test_read_chains_survives_a_hand_broken_config(manager: ConfigManager) -> None:
    """A malformed chain must render as absent, never raise: the page that
    would let a user FIX it is the one thing that must not crash on it."""
    manager.set_config_value("retry", {"fallbackChains": "not-a-mapping"})
    assert settings_io.read_chains(manager) == {}
    manager.set_config_value("retry", "not-a-mapping")
    assert settings_io.read_chains(manager) == {}
    assert settings_io.read_setting(manager, settings_io.BY_KEY["retry.maxRetries"]) == 10


def test_validate_hop_requires_a_selector() -> None:
    assert settings_io.validate_hop("anthropic/claude-opus-5") is None
    assert settings_io.validate_hop("anthropic") is not None
    assert settings_io.validate_hop("") is not None


def test_every_key_is_unique_and_resolvable() -> None:
    keys = [setting.key for setting in settings_io.SETTINGS]
    assert len(keys) == len(set(keys)), "duplicate key in the registry"
    for key in keys:
        assert settings_io.resolve_key(key) is not None
    assert settings_io.resolve_key("nope") is None


def test_every_setting_belongs_to_a_declared_section() -> None:
    names = {section.name for section in settings_io.SECTIONS}
    for setting in settings_io.SETTINGS:
        assert setting.section in names, setting.key
    # And no section is empty, which would paint a header with nothing under it.
    for section in settings_io.SECTIONS:
        assert settings_io.settings_for(section.name), section.name


def test_enum_settings_declare_choices_including_their_default() -> None:
    """A default outside its own choice list cannot be selected back."""
    for setting in settings_io.SETTINGS:
        if setting.kind is Kind.ENUM:
            values = [choice.value for choice in setting.choices]
            assert values, setting.key
            assert setting.default in values, setting.key


def test_display_defaults_matches_the_tui_reader() -> None:
    """The page and the fast-path reader must agree on what 'unset' means."""
    from local_operator.tui import settings as tui_settings

    assert settings_io.display_defaults() == tui_settings._DEFAULT_NOTES


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """``_write_config`` writes through a temp file and renames it; a stray
    ``.config.*.yml.tmp`` beside a config the user is about to hand-edit is its
    own small confusion."""
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["retry.maxRetries"], 3)
    assert [path.name for path in tmp_path.iterdir()] == ["config.yml"]


def test_write_preserves_a_widened_file_mode(tmp_path: Path) -> None:
    """0600 at CREATION only. A user who widened config.yml on purpose must not
    find it narrowed again by every toggle — the rule ``_load_config`` already
    states for the directory."""
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["retry.maxRetries"], 3)
    config_file = tmp_path / "config.yml"
    config_file.chmod(0o644)
    settings_io.write_setting(manager, settings_io.BY_KEY["retry.maxRetries"], 4)
    assert config_file.stat().st_mode & 0o777 == 0o644
