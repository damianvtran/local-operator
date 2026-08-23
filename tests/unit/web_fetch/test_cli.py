"""CLI surface tests: status view, set mutators, and dispatch."""

from __future__ import annotations

import argparse

import pytest

from local_operator.config import ConfigManager
from local_operator.paths import config_dir
from local_operator.web_fetch import cli
from local_operator.web_fetch.service import load_fetch_settings


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))


def test_status_reports_render_backend_and_policy() -> None:
    text = cli.format_fetch_status(ConfigManager(config_dir()))
    assert "Web fetch:" in text
    assert "Render backend:" in text
    assert "Cache TTL:" in text
    assert "Allow private/loopback" in text


def test_set_enabled_off_persists() -> None:
    args = argparse.Namespace(fetch_command="set", key="enabled", value="off")
    assert cli.fetch_command(args) == 0
    assert load_fetch_settings(ConfigManager(config_dir())).enabled is False


def test_set_allow_private_on_persists() -> None:
    args = argparse.Namespace(fetch_command="set", key="allow-private", value="on")
    assert cli.fetch_command(args) == 0
    assert load_fetch_settings(ConfigManager(config_dir())).allow_private is True


def test_set_ttl_persists() -> None:
    args = argparse.Namespace(fetch_command="set", key="ttl", value="0")
    assert cli.fetch_command(args) == 0
    assert load_fetch_settings(ConfigManager(config_dir())).cache_ttl_seconds == 0


def test_set_backend_rejects_bad_value() -> None:
    args = argparse.Namespace(fetch_command="set", key="backend", value="lxml")
    assert cli.fetch_command(args) == 1


def test_status_is_default_command() -> None:
    args = argparse.Namespace(fetch_command=None)
    assert cli.fetch_command(args) == 0
