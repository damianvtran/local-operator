"""Bedrock self-selects as the hosting only on its documented signal.

Regression for the v0.108.0 live validation (AWS Build Day prep): on a fresh
config, a Lambda / ECS container / ``LOP_BEDROCK_AUTH=ambient`` worker running
``lop exec`` or the SDK without ``--hosting`` failed "Hosting platform is not
configured", although docs/providers/BEDROCK.md says those signals turn Bedrock
on, and the documented ``bedrock/<id>`` model form did not route either.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.bedrock import auto_select

_AWS_ENV = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "LOP_BEDROCK_AUTH",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_LAMBDA_FUNCTION_NAME",
    "ECS_CONTAINER_METADATA_URI_V4",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


class _FakeConfig:
    """Just what both resolvers read: ``get_config_value`` and ``config_dir``."""

    def __init__(self, root: Path, values: dict[str, Any] | None = None) -> None:
        self.config_dir = root
        self._values = values or {}

    def get_config_value(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)


def _Config(root: Path, values: dict[str, Any] | None = None) -> Any:
    return _FakeConfig(root, values)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in _AWS_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def census(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Pin what the provider census answers, and count how often it is asked."""
    state: dict[str, Any] = {"answer": {"bedrock"}, "calls": 0}

    def fake(**_kwargs: Any) -> set[str] | None:
        state["calls"] += 1
        return state["answer"]

    import local_operator.providers.model_access as model_access

    monkeypatch.setattr(model_access, "credentialed_chat_providers_here", fake)
    return state


def _implied(tmp_path: Path, model: str | None = None, **values: Any):
    return auto_select.implied_bedrock_hosting(
        model, config_dir=tmp_path, config_values=values or {}
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AWS_BEARER_TOKEN_BEDROCK", "fake-key"),
        ("LOP_BEDROCK_AUTH", "ambient"),
        ("AWS_LAMBDA_FUNCTION_NAME", "fn"),
        ("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1:1/creds"),
        ("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/token"),
    ],
)
def test_each_documented_signal_selects_bedrock(
    clean_env, census, tmp_path: Path, name: str, value: str
) -> None:
    clean_env.setenv(name, value)
    assert _implied(tmp_path) == ("bedrock", None)
    assert _implied(tmp_path, "ca.anthropic.claude-sonnet-5-5") == (
        "bedrock",
        "ca.anthropic.claude-sonnet-5-5",
    )


def test_config_auth_ambient_is_a_signal(clean_env, census, tmp_path: Path) -> None:
    assert _implied(tmp_path, providers={"bedrock": {"auth": "ambient"}}) == ("bedrock", None)


def test_config_values_are_read_from_the_runs_own_config_dir(
    clean_env, census, tmp_path: Path
) -> None:
    # No values passed: the helper must read THIS dir's config.yml, not the
    # process-default config dir.
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  providers:\n    bedrock:\n      auth: ambient\n"
    )
    assert auto_select.implied_bedrock_hosting(None, config_dir=tmp_path) == ("bedrock", None)


@pytest.mark.parametrize(
    "laptop_env",
    [
        {"AWS_PROFILE": "dev"},
        {"AWS_ACCESS_KEY_ID": "AKIAFAKE", "AWS_SECRET_ACCESS_KEY": "x"},
    ],
)
def test_plain_laptop_aws_env_never_selects_bedrock(
    clean_env, census, tmp_path: Path, laptop_env: dict[str, str]
) -> None:
    for name, value in laptop_env.items():
        clean_env.setenv(name, value)
    assert _implied(tmp_path) is None
    assert _implied(tmp_path, "ca.anthropic.claude-sonnet-5-5") is None
    assert census["calls"] == 0, "no signal must not even consult the store"


@pytest.mark.parametrize("answer", [{"bedrock", "openai"}, set(), None, {"openai"}])
def test_signal_needs_bedrock_to_be_the_only_credentialed_provider(
    clean_env, census, tmp_path: Path, answer: set[str] | None
) -> None:
    clean_env.setenv("AWS_LAMBDA_FUNCTION_NAME", "fn")
    census["answer"] = answer
    assert _implied(tmp_path) is None


def test_bedrock_prefixed_model_routes_without_any_signal(
    clean_env, census, tmp_path: Path
) -> None:
    assert _implied(tmp_path, "bedrock/ca.anthropic.claude-sonnet-5-5") == (
        "bedrock",
        "ca.anthropic.claude-sonnet-5-5",
    )
    assert _implied(tmp_path, "bedrock/") == ("bedrock", None)
    assert census["calls"] == 0


def _args(**overrides: Any) -> argparse.Namespace:
    base = dict(hosting=None, model=None, agent_name=None, agent_id=None, yolo=False, train=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_resolver_selects_bedrock_with_its_default_model(
    clean_env, census, tmp_path: Path
) -> None:
    from local_operator.model.defaults import default_model_for
    from local_operator.session_factory import resolve_hosting_model

    clean_env.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://127.0.0.1:1/creds")
    config: Any = _Config(tmp_path)
    assert resolve_hosting_model(None, _args(), config) == ("bedrock", default_model_for("bedrock"))
    assert resolve_hosting_model(
        None, _args(model="bedrock/ca.anthropic.claude-haiku-5-5"), config
    ) == ("bedrock", "ca.anthropic.claude-haiku-5-5")


def test_cli_resolver_never_overrides_a_chosen_hosting(clean_env, census, tmp_path: Path) -> None:
    from local_operator.session_factory import resolve_hosting_model

    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "fake-key")
    config: Any = _Config(tmp_path, {"hosting": "openai", "model_name": "gpt-6-astra"})
    assert resolve_hosting_model(None, _args(), config) == ("openai", "gpt-6-astra")
    assert resolve_hosting_model(None, _args(hosting="deepseek"), _Config(tmp_path))[0] == (
        "deepseek"
    )


def test_cli_resolver_without_signal_still_reports_not_configured(
    clean_env, census, tmp_path: Path
) -> None:
    from local_operator.session_factory import (
        HostingNotConfiguredError,
        resolve_hosting_model,
    )

    clean_env.setenv("AWS_PROFILE", "dev")
    with pytest.raises(HostingNotConfiguredError, match="Hosting platform is not configured."):
        resolve_hosting_model(None, _args(), _Config(tmp_path))


def test_sdk_resolver_shares_the_rule(clean_env, census, tmp_path: Path) -> None:
    from local_operator.bootstrap import resolve_hosting_model
    from local_operator.model.defaults import default_model_for

    config: Any = _Config(tmp_path)
    with pytest.raises(ValueError, match="Hosting platform is not configured."):
        resolve_hosting_model(config, None, None, None)
    clean_env.setenv("LOP_BEDROCK_AUTH", "ambient")
    assert resolve_hosting_model(config, None, None, None) == (
        "bedrock",
        default_model_for("bedrock"),
    )
    assert resolve_hosting_model(
        config, None, "bedrock/global.anthropic.claude-opus-5-5", None
    ) == (
        "bedrock",
        "global.anthropic.claude-opus-5-5",
    )
    configured: Any = _Config(tmp_path, {"hosting": "openai", "model_name": "gpt-6-astra"})
    assert resolve_hosting_model(configured, None, None, None) == ("openai", "gpt-6-astra")
