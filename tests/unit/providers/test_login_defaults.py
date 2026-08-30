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


@pytest.mark.parametrize(
    ("provider", "expected_hosting", "expected_model"),
    [
        # Login FLAVOURS: authentication routes, not hosting ids. Each has no
        # default model of its own, which is what made the first version of the
        # repair leave the dead model in place beside the new hosting.
        ("xai-oauth", "xai", "grok-3"),
        ("openai-device", "openai", "gpt-4o"),
        ("zai-oauth", "zai", "glm-5.3"),
        # No default model even after alias resolution: the stale model must be
        # CLEARED, not kept. An empty model_name is a state the resolver handles
        # and explains; a model from a provider that never existed is not.
        ("alibaba-token-plan", "alibaba-token-plan", ""),
        # Ordinary provider with a default: the baseline case.
        ("deepseek", "deepseek", "deepseek-chat"),
    ],
)
def test_repair_never_leaves_a_model_from_the_replaced_provider(
    provider: str, expected_hosting: str, expected_model: str
) -> None:
    """A repair must not produce a config that boots and then fails at stream time.

    The regression this pins: writing the raw login-flavour id as hosting left
    `model_name` untouched when that flavour had no default model, yielding e.g.
    `hosting='xai-oauth' model_name='claude-sonnet-4-5'`. `configure_model`
    ACCEPTS that pair, so boot succeeded and the failure moved to stream time as
    a provider-side unknown-model error -- trading a boot failure the app
    explains for a runtime failure it cannot.
    """
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults(provider, "anthropicxyq", "claude-sonnet-4-5")

    assert plan.repairing is True
    assert plan.hosting == expected_hosting
    # Never None while repairing: the stale model is always overwritten, and ""
    # (clear it) is a deliberate value rather than "leave it alone".
    assert plan.model_name == expected_model
    assert plan.model_name != "claude-sonnet-4-5"


def test_plan_is_the_single_source_of_truth_for_both_login_front_ends() -> None:
    """Both `/login` and `local-operator login` must plan identically.

    The two copies of this rule had drifted on every axis that mattered (which
    hosting id, which brokenness test, which model), which is the class of bug
    the shared planner exists to make impossible. Asserting the front ends call
    the planner rather than re-deriving the policy is what keeps them together.
    """
    import inspect

    from local_operator.providers import auth_cli
    from local_operator.tui import app as tui_app

    for source in (
        inspect.getsource(auth_cli._apply_login_defaults),
        inspect.getsource(tui_app.OperatorApp._apply_login_defaults),
    ):
        assert "plan_login_defaults" in source
        # The policy must not be re-implemented beside the call.
        assert "default_model_for" not in source
        assert "credential_provider_id" not in source


def test_plan_leaves_a_usable_hosting_alone() -> None:
    """Logging into a second provider must not repoint an existing default."""
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("deepseek", "openai", "gpt-4o")
    assert plan.hosting is None
    assert plan.model_name is None
    assert plan.receipt is None


def test_plan_treats_a_legacy_alias_as_usable() -> None:
    """`noop` is not a registry id -- it maps to `test`, and must not be
    mistaken for a corrupted value and silently replaced."""
    from local_operator.providers.login_defaults import (
        is_unusable_hosting,
        plan_login_defaults,
    )

    assert is_unusable_hosting("noop") is False
    assert is_unusable_hosting("anthropicxyq") is True
    # Empty hosting is the separate nothing-configured case, not "unusable".
    assert is_unusable_hosting("") is False
    assert plan_login_defaults("deepseek", "noop", "m").hosting is None


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
