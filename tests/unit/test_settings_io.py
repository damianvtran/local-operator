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
    from local_operator.model.configure import ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
    from local_operator.providers.failover import (
        CONNECTIVITY_BACKOFF_CAP_MS,
        CONNECTIVITY_MAX_RETRIES,
        RetrySettings,
    )
    from local_operator.session.retention import DEFAULT_REAP_UNUSED
    from local_operator.session.runtime.control import DEFAULT_BACKGROUND_ON_RESUME
    from local_operator.session.runtime.owned import DEFAULT_UNATTENDED_GATE_TIMEOUT_H
    from local_operator.spawn.policy import (
        DEFAULT_FORK_CMUX_PLACEMENT,
        DEFAULT_FORK_MODE,
    )
    from local_operator.tui.theme import DEFAULT_THEME
    from local_operator.web_fetch.models import DEFAULT_WEB_FETCH_CONFIG
    from local_operator.web_search.models import DEFAULT_WEB_SEARCH_CONFIG

    retry = RetrySettings()
    compaction = CompactionSettings()
    consumers: dict[str, object] = {
        # The theme the TUI actually boots on. The registry restates "dark"
        # rather than importing this (that import would put `rich` on the CLI's
        # path), so THIS is what stops the two drifting — as they had: the
        # registry said `""` while the app used `dark`, so the page reported a
        # user explicitly on `dark` as having changed the setting (round 1, M1).
        "tui.theme": DEFAULT_THEME,
        "retry.enabled": retry.enabled,
        "retry.maxRetries": retry.max_retries,
        "retry.baseDelayMs": retry.base_delay_ms,
        "retry.connectivityMaxRetries": CONNECTIVITY_MAX_RETRIES,
        "retry.connectivityBackoffCapMs": CONNECTIVITY_BACKOFF_CAP_MS,
        "retry.modelFallback": retry.model_fallback,
        "retry.usageAwareFallback": retry.usage_aware_fallback,
        "retry.usageReservePercent": retry.usage_reserve_percent,
        "retry.usageAwareAccountPick": retry.usage_aware_account_pick,
        "retry.fallbackChains": dict(retry.fallback_chains),
        "session.background_on_resume": DEFAULT_BACKGROUND_ON_RESUME,
        "runtime.unattended_gate_timeout": DEFAULT_UNATTENDED_GATE_TIMEOUT_H,
        "session.reap_unused": DEFAULT_REAP_UNUSED,
        "subagents.max_running": DEFAULT_MAX_RUNNING_JOBS,
        "providers.openai.api": DEFAULT_CONFIG.values["providers"]["openai"]["api"],
        # The client-side constant is the real consumer (``_anthropic_cache_ttl_
        # 1h_min_context_tokens`` restates it for a settings mapping without the
        # key); the config default is checked against the same constant below.
        "providers.anthropic.cache_ttl_1h_min_context_tokens": (
            ANTHROPIC_CACHE_TTL_1H_MIN_CONTEXT_TOKENS
        ),
        # The fork keys have REAL single-value consumers — the constants
        # ``/fork`` itself reads — so they are mapped here rather than
        # allow-listed. An allow-list entry would buy a green test while leaving
        # the registry default and the feature's default free to disagree.
        "fork.mode": DEFAULT_FORK_MODE,
        "fork.cmux_placement": DEFAULT_FORK_CMUX_PLACEMENT,
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


#: Settings with no independent consumer constant to compare against, and WHY
#: each one is exempt. An explicit list rather than a blanket
#: ``if key not in consumers: skip`` (review round 1, M1): a skip makes the
#: drift guard reward the thing it exists to catch, because adding a setting
#: with no consumer entry buys a green test instead of a prompt to wire one up.
#: With the list explicit, a NEW unmapped key fails here by name.
#:
#: Every ``display.*`` flag is exempt for a specific reason: ``tui/settings.py``
#: DERIVES its defaults from this registry (``display_defaults()``), so it is
#: not an independent second source and comparing them would be a tautology.
#: ``test_display_defaults_matches_the_tui_reader`` pins that derivation, and
#: the prose in ``_DEFAULT_NOTES`` is what documents the intent.
_NO_SINGLE_VALUE_CONSUMER: dict[str, str] = {
    "display.shimmer": "tui/settings.py derives its defaults from this registry",
    "display.comfortable_rows": "tui/settings.py derives its defaults from this registry",
    "display.nerd_icons": "derived; tri-state None means auto-detect, not a value",
    "display.terminal_title": "tui/settings.py derives its defaults from this registry",
    "display.images": "tui/settings.py derives its defaults from this registry",
    "display.notifications": "tui/settings.py derives its defaults from this registry",
    "subagents.models.lo": "free text; empty means 'keep the parent's model', no constant",
    "subagents.models.med": "free text; empty means 'keep the parent's model', no constant",
    "subagents.models.hi": "free text; empty means 'keep the parent's model', no constant",
}


def test_the_drift_allow_list_is_not_stale() -> None:
    """An allow-listed key that HAS gained a consumer must leave the list.

    The allow-list is a standing exemption, so it needs its own guard: without
    this, wiring a consumer up for an exempt key would leave the exemption in
    place and the drift guard still not watching it.
    """
    consumers = _consumer_defaults()
    now_mapped = sorted(set(_NO_SINGLE_VALUE_CONSUMER) & set(consumers))
    assert not now_mapped, f"these now have a consumer; drop them from the allow-list: {now_mapped}"
    unknown = sorted(set(_NO_SINGLE_VALUE_CONSUMER) - set(settings_io.BY_KEY))
    assert not unknown, f"allow-list names settings that no longer exist: {unknown}"


@pytest.mark.parametrize("setting", settings_io.SETTINGS, ids=lambda s: s.key)
def test_every_default_matches_its_consumer(setting) -> None:
    """A registry default that disagrees with its consumer is a painted lie."""
    consumers = _consumer_defaults()
    if setting.key not in consumers:
        assert setting.key in _NO_SINGLE_VALUE_CONSUMER, (
            f"{setting.key} has no consumer entry in _consumer_defaults(). Add one so its "
            f"default is guarded, or add it to _NO_SINGLE_VALUE_CONSUMER with the reason "
            f"it genuinely has no single-value consumer."
        )
        return
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


def test_a_write_does_not_revert_another_managers_change(tmp_path: Path) -> None:
    """A page write must not clobber keys it never touched (round 1, B1).

    ``set_config_value`` dumps the manager's WHOLE in-memory snapshot, and the
    ``/settings`` page holds one manager for as long as it is open. Without a
    reload before the write, toggling one unrelated row wrote back a stale copy
    of the entire file — reverting the theme, the default model, everything
    another writer had changed since the page opened. Silent, and with no undo.

    Both writers here are real: ``SettingsView`` captures a manager at open,
    and ``OperatorApp._persist_theme`` constructs a fresh one per call, so this
    fires within a single session and not only across two.
    """
    page = ConfigManager(tmp_path)  # captured when /settings opened

    # Something else writes while the page sits open: /theme, /model default,
    # or another session entirely.
    other = ConfigManager(tmp_path)
    other.set_config_value("tui", {"theme": "gruvbox"})
    other.set_config_value("model_name", "claude-opus-4")
    retry = dict(other.get_config_value("retry", {}) or {})
    retry["maxRetries"] = 42
    other.set_config_value("retry", retry)

    # The user toggles one unrelated row on the still-open page.
    settings_io.write_setting(page, settings_io.BY_KEY["display.shimmer"], False)

    disk = ConfigManager(tmp_path)
    assert disk.get_config_value("tui", {}) == {"theme": "gruvbox"}
    assert disk.get_config_value("model_name", None) == "claude-opus-4"
    assert (disk.get_config_value("retry", {}) or {})["maxRetries"] == 42
    # ...and the toggle itself still landed.
    assert disk.get_config_value("display.shimmer", None) is False
    # The page's own manager now agrees with the file, so the next repaint
    # shows what is really there rather than the snapshot it opened with.
    assert settings_io.read_setting(page, settings_io.BY_KEY["model_name"]) == "claude-opus-4"


def test_a_reset_does_not_revert_another_managers_change(tmp_path: Path) -> None:
    """The delete path carries the same staleness trap as the write path."""
    page = ConfigManager(tmp_path)
    page.set_config_value("model_name", "claude-opus-5")

    other = ConfigManager(tmp_path)
    other.set_config_value("tui", {"theme": "gruvbox"})

    settings_io.reset_setting(page, settings_io.BY_KEY["model_name"])

    disk = ConfigManager(tmp_path)
    assert disk.get_config_value("tui", {}) == {"theme": "gruvbox"}
    assert not disk.get_config_value("model_name", "")


def test_editing_one_chain_preserves_effort_in_the_others(manager: ConfigManager) -> None:
    """Structured hops keep their ``effort`` when another chain is edited.

    Round 1, B2. The page rewrites EVERY chain on any edit, and un-labelling a
    display string back to a bare selector silently dropped ``effort`` from
    every structured hop in every untouched chain. ``failover.py`` calls that
    "the one key that makes the entry mean something different" — it is the
    "retry cheaper on failure" routing decision, not decoration.
    """
    manager.set_config_value(
        "retry",
        {
            "fallbackChains": {
                "primary": [
                    {"provider": "openai", "model": "gpt-5", "effort": "high"},
                    "anthropic/claude-opus-5",
                ],
                "backup": ["anthropic/claude-opus-5"],
            }
        },
    )
    # The effort is SHOWN, so the user can see the routing decision they have.
    assert settings_io.read_chains(manager)["primary"][0] == "openai/gpt-5 (high)"

    # The user adds a hop to the unrelated 'backup' chain; the page writes all.
    chains = {key: list(hops) for key, hops in settings_io.read_chains(manager).items()}
    chains["backup"].append("openrouter/qwen/qwen3-coder")
    settings_io.write_chains(manager, chains)

    stored = ConfigManager(manager.config_dir).get_config_value("retry")["fallbackChains"]
    assert stored["primary"][0] == {"provider": "openai", "model": "gpt-5", "effort": "high"}
    assert stored["primary"][1] == "anthropic/claude-opus-5"
    assert stored["backup"] == ["anthropic/claude-opus-5", "openrouter/qwen/qwen3-coder"]

    # And the failover layer still resolves the effort it was given.
    from local_operator.providers.failover import RetrySettings

    resolved = RetrySettings.from_settings(ConfigManager(manager.config_dir).get_config().values)
    assert resolved.fallback_chains["primary"][0] == {
        "provider": "openai",
        "model": "gpt-5",
        "effort": "high",
    }


def test_retyping_a_structured_hop_replaces_it(manager: ConfigManager) -> None:
    """Editing a hop's text DOES drop its effort, and that is correct.

    The preservation in ``write_chains`` is keyed on the label, so it must not
    resurrect an effort onto a hop the user deliberately retyped: they replaced
    that hop, and the page has no field in which to have kept it.
    """
    manager.set_config_value(
        "retry",
        {
            "fallbackChains": {
                "primary": [{"provider": "openai", "model": "gpt-5", "effort": "high"}]
            }
        },
    )
    settings_io.write_chains(manager, {"primary": ["openai/gpt-5-mini"]})
    stored = ConfigManager(manager.config_dir).get_config_value("retry")["fallbackChains"]
    assert stored["primary"] == ["openai/gpt-5-mini"]


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

    # A STRUCTURED hop through the same round trip. Plain strings alone kept
    # this test green while `effort` was being stripped (round 1, B2): the
    # lossy step only exists for the mapping form.
    manager.set_config_value(
        "retry",
        {
            "fallbackChains": {
                "default": [{"provider": "anthropic", "model": "claude-opus-5", "effort": "low"}]
            }
        },
    )
    read = settings_io.read_chains(manager)
    assert read == {"default": ["anthropic/claude-opus-5 (low)"]}
    settings_io.write_chains(manager, read)
    assert settings_io.read_chains(manager) == read
    resolved = RetrySettings.from_settings(manager.get_config().values)
    assert resolved.fallback_chains["default"] == [
        {"provider": "anthropic", "model": "claude-opus-5", "effort": "low"}
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
            # `resolved_choices`, which is what `validate` and the page read:
            # a registry-sourced enum declares nothing in the static field.
            values = [choice.value for choice in setting.resolved_choices]
            assert values, setting.key
            assert setting.default in values, setting.key


def test_theme_is_an_enum_over_the_live_registry(manager: ConfigManager) -> None:
    """``tui.theme`` offers the registry's themes and refuses anything else.

    Round 1, m1. As free TEXT the page accepted a theme that does not exist and
    then displayed a value the app was not using — ``app.py`` catches the
    KeyError and quietly falls back to the default. Sourced from the registry
    rather than a literal list because the palettes add ~30 of them.
    """
    from local_operator.tui.theme import DEFAULT_THEME, available_themes

    setting = settings_io.BY_KEY["tui.theme"]
    assert setting.kind is Kind.ENUM
    offered = [choice.value for choice in setting.resolved_choices]
    assert offered == available_themes()
    assert len(offered) > 2, "the curated palettes are part of the value space"
    assert DEFAULT_THEME in offered

    with pytest.raises(ValueError):
        settings_io.write_setting(manager, setting, "not-a-real-theme")
    settings_io.write_setting(manager, setting, "gruvbox")
    assert settings_io.read_setting(manager, setting) == "gruvbox"

    # A user explicitly on the default is NOT reported as having changed it,
    # which is what the `default=""` drift used to claim.
    settings_io.write_setting(manager, setting, DEFAULT_THEME)
    assert settings_io.is_default(manager, setting)


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


def test_write_fsyncs_the_parent_directory_after_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename is only durable once the parent directory is synced.

    Round 1, m3. Syncing the file's data (which ``_write_config`` already did,
    in the right order) guarantees the bytes; the directory entry that gives
    them the config's name is separate metadata, so a crash in between can
    still surface the old file.
    """
    import os

    real_open, real_fsync = os.open, os.fsync
    dir_fds: set[int] = set()
    synced_dirs: list[int] = []

    def spy_open(path, flags, *args, **kwargs):  # type: ignore[no-untyped-def]
        fd = real_open(path, flags, *args, **kwargs)
        if os.path.isdir(path):
            dir_fds.add(fd)
        return fd

    def spy_fsync(fd):  # type: ignore[no-untyped-def]
        if fd in dir_fds:
            synced_dirs.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "fsync", spy_fsync)

    ConfigManager(tmp_path).set_config_value("model_name", "claude-opus-5")

    assert synced_dirs, "the parent directory was never fsynced after os.replace"


def test_write_survives_a_filesystem_that_cannot_fsync_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused directory fsync must not fail a write that already landed."""
    import os

    real_fsync = os.fsync

    def refuse_dir_fsync(fd):  # type: ignore[no-untyped-def]
        if os.fstat(fd).st_mode & 0o040000:  # S_IFDIR
            raise OSError("directory fsync unsupported")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", refuse_dir_fsync)

    ConfigManager(tmp_path).set_config_value("model_name", "claude-opus-5")
    assert ConfigManager(tmp_path).get_config_value("model_name", None) == "claude-opus-5"


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


@pytest.mark.parametrize(
    "corruption",
    [
        pytest.param("\thosting: broken\n", id="tab-indented-line"),
        pytest.param("values:\n  retry:\n    maxRetries: [1, 2\n", id="truncated-file"),
        pytest.param("", id="zero-byte-file"),
        pytest.param("- a\n- b\n", id="top-level-list"),
    ],
)
@pytest.mark.parametrize("path", ["write", "reset"])
def test_an_unreadable_config_aborts_the_write_instead_of_defaulting_over_it(
    tmp_path: Path, corruption: str, path: str
) -> None:
    """A config that stopped parsing must not be REPLACED BY DEFAULTS.

    The regression this pins is data loss, and it came from the fix for B1
    (review round 2, B3). ``ConfigManager._load_config`` does not raise on a
    malformed config.yml \u2014 it moves the file aside and returns a fresh default
    config \u2014 so the reload before a write silently succeeded, the manager
    became defaults, and the write dumped those defaults over the user's file.
    The ``.bad`` backup then held only the broken edit, so the last good config
    was recoverable from nowhere.

    Asserts on the FILE'S BYTES rather than on parsed values: the property that
    matters is that nothing was written at all, and a value-level assertion
    would pass on a rewrite that happened to round-trip.
    """
    page = ConfigManager(tmp_path)  # holding a GOOD snapshot, as an open page does
    page.set_config_value("model_name", "claude-opus-4")
    page.set_config_value("retry", {"maxRetries": 42})

    # The user hand-edits config.yml in another window and breaks it.
    config_file = tmp_path / "config.yml"
    config_file.write_text(corruption)
    before = config_file.read_bytes()

    setting = settings_io.BY_KEY["display.shimmer"]
    with pytest.raises(settings_io.ConfigUnreadableError):
        if path == "write":
            settings_io.write_setting(page, setting, True)
        else:
            settings_io.reset_setting(page, setting)

    assert config_file.read_bytes() == before, "the broken file was overwritten"
    # And nothing was moved aside, so the user still has the file to repair.
    assert not list(tmp_path.glob("config.yml.bad*"))


def test_a_non_utf8_config_is_refused_as_unreadable_not_as_a_bad_value(
    tmp_path: Path,
) -> None:
    """Review round 3, n2 \u2014 a codec error must reach the unreadable-config slot.

    ``UnicodeDecodeError`` is a ``ValueError`` subclass, so left uncaught it was
    caught by the page's ``except ValueError`` one branch BEFORE the
    ``ConfigUnreadableError`` handler \u2014 and the user saw "'utf-8' codec can't
    decode byte 0xff in position 0" sitting where "the value you typed is
    wrong" goes, on a row whose value was perfectly fine. Reachable from a
    Windows editor or a PowerShell redirect writing UTF-16.

    The bytes were already safe on this path; it is the message that pointed at
    the wrong thing, so the type is what this pins.
    """
    page = ConfigManager(tmp_path)
    page.set_config_value("model_name", "claude-opus-4")
    config_file = tmp_path / "config.yml"
    config_file.write_bytes("values:\n  hosting: anthropic\n".encode("utf-16"))
    before = config_file.read_bytes()

    with pytest.raises(settings_io.ConfigUnreadableError):
        settings_io.write_setting(page, settings_io.BY_KEY["display.shimmer"], True)

    assert config_file.read_bytes() == before, "the non-UTF-8 file was overwritten"
    assert not list(tmp_path.glob("config.yml.bad*"))


def test_a_missing_config_is_not_treated_as_unreadable(tmp_path: Path) -> None:
    """A first run has no file and no prior config to destroy, so it writes."""
    manager = ConfigManager(tmp_path)
    (tmp_path / "config.yml").unlink(missing_ok=True)
    settings_io.write_setting(manager, settings_io.BY_KEY["retry.maxRetries"], 7)
    assert ConfigManager(tmp_path).get_config_value("retry", {})["maxRetries"] == 7


def test_a_concurrent_chain_add_survives_a_page_write(tmp_path: Path) -> None:
    """``write_chains`` merges the caller's own edit rather than replacing all.

    Review round 2, M2. The page builds its chain list from an earlier
    ``read_chains`` and edits one hop in it. Replacing ``retry.fallbackChains``
    wholesale then deleted a chain another session had added in the meantime \u2014
    reloading first read the fresh state and immediately discarded it.
    """
    page = ConfigManager(tmp_path)
    settings_io.write_chains(page, {"primary": ["openai/gpt-5"]})
    base = settings_io.read_chains(page)
    working = {key: list(hops) for key, hops in base.items()}

    other = ConfigManager(tmp_path)
    other_chains = {key: list(hops) for key, hops in settings_io.read_chains(other).items()}
    other_chains["newchain"] = ["anthropic/claude-opus-5"]
    settings_io.write_chains(other, other_chains)

    # The page adds a hop to ITS chain and saves.
    working["primary"].append("groq/llama-3")
    settings_io.write_chains(page, working, base=base)

    stored = settings_io.read_chains(ConfigManager(tmp_path))
    assert "newchain" in stored, "the concurrent chain was deleted"
    assert stored["primary"] == ["openai/gpt-5", "groq/llama-3"], "our own edit was lost"


def test_a_concurrent_effort_edit_is_not_flattened_by_a_page_write(tmp_path: Path) -> None:
    """The sharper half of M2: a stale label missed the lookup and dropped effort.

    The page's label for an untouched hop (``anthropic/claude (low)``) no longer
    matched the freshly-reloaded entry once another session changed its effort,
    so the originals lookup missed and the hop was rewritten as a BARE SELECTOR
    \u2014 losing both the concurrent edit and the effort itself.
    """
    page = ConfigManager(tmp_path)
    page.set_config_value(
        "retry",
        {
            "fallbackChains": {
                "primary": [{"provider": "openai", "model": "gpt-5", "effort": "high"}],
                "backup": [{"provider": "anthropic", "model": "claude", "effort": "low"}],
            }
        },
    )
    page = ConfigManager(tmp_path)
    base = settings_io.read_chains(page)
    working = {key: list(hops) for key, hops in base.items()}

    other = ConfigManager(tmp_path)
    other.set_config_value(
        "retry",
        {
            "fallbackChains": {
                "primary": [{"provider": "openai", "model": "gpt-5", "effort": "high"}],
                "backup": [{"provider": "anthropic", "model": "claude", "effort": "minimal"}],
            }
        },
    )

    # The page edits only `primary`.
    working["primary"].append("groq/llama-3")
    settings_io.write_chains(page, working, base=base)

    stored = ConfigManager(tmp_path).get_config_value("retry")["fallbackChains"]
    assert stored["backup"] == [
        {"provider": "anthropic", "model": "claude", "effort": "minimal"}
    ], "the untouched chain was flattened to a bare selector"
    assert stored["primary"][0] == {"provider": "openai", "model": "gpt-5", "effort": "high"}


def test_a_page_delete_still_removes_the_chain_it_deleted(tmp_path: Path) -> None:
    """The merge must not resurrect a chain the caller deliberately deleted."""
    page = ConfigManager(tmp_path)
    settings_io.write_chains(page, {"keep": ["openai/gpt-5"], "drop": ["anthropic/claude"]})
    base = settings_io.read_chains(page)
    working = {key: list(hops) for key, hops in base.items()}
    del working["drop"]
    settings_io.write_chains(page, working, base=base)
    assert sorted(settings_io.read_chains(ConfigManager(tmp_path))) == ["keep"]


def test_a_hop_typed_in_the_displayed_format_is_refused_rather_than_narrowed(
    manager: ConfigManager,
) -> None:
    """n1: the page shows ``openai/gpt-5 (high)``, so copying it must not
    silently store a hop without its effort."""
    assert settings_io.validate_hop("anthropic/claude (low)") is not None
    assert settings_io.validate_hop("anthropic/claude") is None


def test_an_unresolvable_choice_list_says_so_rather_than_offering_nothing() -> None:
    """m4: an empty value space is a broken host, not a rejected value."""
    setting = settings_io.Setting(
        key="scratch.enum",
        path=("scratch",),
        section="Appearance",
        label="Scratch",
        kind=settings_io.Kind.ENUM,
        default="dark",
        help="",
        choices_source=lambda: (),
    )
    problem = settings_io.validate(setting, "dark")
    assert problem is not None
    assert "could not be read" in problem
    assert not problem.endswith(": ")
