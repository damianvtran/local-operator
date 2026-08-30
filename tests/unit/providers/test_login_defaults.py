"""Login-sets-default-hosting behaviour and default-model resolution.

Covers item 2 (login adopts hosting/model when config is empty) and item 3
(per-provider default model fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.model.defaults import DEFAULT_MODEL_NAMES, default_model_for


def test_default_model_for_known_provider() -> None:
    assert default_model_for("deepseek") == "deepseek-chat"
    assert default_model_for("zai") == "glm-5.3"
    # noop aliases to test, which has no default.
    assert default_model_for("noop") is None


def test_default_model_for_unknown_provider() -> None:
    assert default_model_for("some-custom-host") is None


def test_default_model_names_map_reexported_from_configure() -> None:
    # configure.DEFAULT_MODEL_NAMES must be the SAME object as defaults' — one
    # map, imported cheaply on the startup path (item 3).
    from local_operator.model import configure

    assert configure.DEFAULT_MODEL_NAMES is DEFAULT_MODEL_NAMES


def test_apply_login_defaults_sets_hosting_and_model_when_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    auth_cli._apply_login_defaults("deepseek")

    manager = ConfigManager(tmp_path)
    assert manager.get_config_value("hosting") == "deepseek"
    assert manager.get_config_value("model_name") == "deepseek-chat"


def test_apply_login_defaults_leaves_existing_hosting_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "openai")
    manager.set_config_value("model_name", "gpt-4o")

    # Logging into a second provider must not repoint an existing default.
    auth_cli._apply_login_defaults("deepseek")

    reloaded = ConfigManager(tmp_path)
    assert reloaded.get_config_value("hosting") == "openai"
    assert reloaded.get_config_value("model_name") == "gpt-4o"


def test_apply_login_defaults_repairs_an_unknown_hosting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`login` REPLACES a hosting the registry does not own.

    The unknown-hosting error recommends this exact command, so the "already
    set, leave it alone" rule had to gain an exception: without it the login
    stored a credential, wrote nothing, and the next run failed to boot on the
    same corrupted value — the remedy looping back to the problem it fixes.
    The stale model goes with it: it belonged to the provider being replaced.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "anthropicxyq")
    manager.set_config_value("model_name", "claude-sonnet-4-5")

    auth_cli._apply_login_defaults("deepseek")

    reloaded = ConfigManager(tmp_path)
    assert reloaded.get_config_value("hosting") == "deepseek"
    assert reloaded.get_config_value("model_name") == "deepseek-chat"


def test_apply_login_defaults_leaves_a_legacy_alias_hosting_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A legacy ALIAS is a valid hosting and must not be treated as corrupt.

    `noop` is not a registry id — it maps to `test`. The repair check asks the
    registry (which resolves aliases) rather than testing membership against
    ids, so a working alias config is not silently repointed by a later login.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "noop")

    auth_cli._apply_login_defaults("deepseek")

    assert ConfigManager(tmp_path).get_config_value("hosting") == "noop"


def test_resolve_hosting_model_falls_back_to_default_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hosting set, model empty: resolves the provider default rather than
    raising 'Model name is not configured' (item 3)."""
    import argparse

    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session_factory import resolve_hosting_model

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "deepseek")
    manager.set_config_value("model_name", "")

    args = argparse.Namespace(hosting=None, model=None)
    hosting, model = resolve_hosting_model(None, args, manager)
    assert hosting == "deepseek"
    assert model == "deepseek-chat"


def test_resolve_hosting_model_no_hosting_raises_hosting_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No hosting at all raises the dedicated HostingNotConfiguredError so the
    setup-state gate can classify it (item 1)."""
    import argparse

    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session_factory import (
        HostingNotConfiguredError,
        resolve_hosting_model,
    )

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    args = argparse.Namespace(hosting=None, model=None)
    with pytest.raises(HostingNotConfiguredError):
        resolve_hosting_model(None, args, manager)
