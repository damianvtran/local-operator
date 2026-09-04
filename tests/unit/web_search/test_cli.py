from __future__ import annotations

import json

from local_operator.cli import build_cli_parser
from local_operator.config import ConfigManager
from local_operator.mcp import config as mcp_config
from local_operator.paths import config_dir
from local_operator.web_search.cli import search_command
from local_operator.web_search.service import load_search_settings


def _args(*parts: str):
    return build_cli_parser().parse_args(["search", *parts])


def test_search_enable_disable_and_order_persist(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    assert search_command(_args("disable", "tavily")) == 0
    assert search_command(_args("enable", "brave")) == 0
    assert search_command(_args("balance", "ordered")) == 0
    assert search_command(_args("order", "brave", "duckduckgo")) == 0

    settings = load_search_settings(ConfigManager(config_dir()))
    assert settings.strategy == "ordered"
    assert settings.providers == ["brave", "duckduckgo"]


def test_setup_tavily_oauth_writes_http_oauth_server(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    mcp_path = tmp_path / "mcp.json"
    monkeypatch.setattr(mcp_config, "_scope_path", lambda _cwd, _scope: mcp_path)

    assert search_command(_args("setup", "tavily", "--oauth")) == 0

    payload = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["tavily"] == {
        "type": "http",
        "url": "https://mcp.tavily.com/mcp/",
        "auth": {"type": "oauth"},
    }
    assert "tavily" in load_search_settings(ConfigManager(config_dir())).providers


def test_setup_tavily_oauth_repairs_global_non_oauth_entry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    # The user-scope mcp.json lives in `config_dir()`, which this test points
    # at `<tmp>/config` — NOT at `$HOME/.local-operator`. The fixture used to
    # write the HOME path and pass only because `_scope_path` rebuilt a home
    # path by hand instead of asking `config_dir()`; that is the bug that let
    # an isolated config dir still write a developer's real mcp.json
    # (round 5, U15 containment).
    mcp_path = tmp_path / "config" / "mcp.json"
    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    mcp_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "tavily": {
                        "type": "http",
                        "url": "https://mcp.tavily.com/mcp",
                        "headers": {"X-Test": "not-oauth"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert search_command(_args("setup", "tavily", "--oauth")) == 0
    assert search_command(_args("setup", "tavily", "--oauth")) == 0

    payload = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["tavily"] == {
        "type": "http",
        "url": "https://mcp.tavily.com/mcp/",
        "auth": {"type": "oauth"},
    }


def test_setup_tavily_oauth_rejects_shadowed_non_oauth_entry(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    project = tmp_path / "project"
    project_config = project / ".local-operator" / "mcp.json"
    project_config.parent.mkdir(parents=True)
    project_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "tavily": {
                        "type": "http",
                        "url": "https://mcp.tavily.com/mcp/",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    assert search_command(_args("setup", "tavily", "--oauth")) == 1
    assert "higher-priority entry" in capsys.readouterr().out


def test_setup_searxng_validates_and_stores_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    assert (
        search_command(_args("setup", "searxng", "--endpoint", "https://search.example.test/")) == 0
    )

    settings = load_search_settings(ConfigManager(config_dir()))
    assert settings.searxng_endpoint == "https://search.example.test"
    assert "searxng" in settings.providers


def test_setup_rejects_oauth_for_non_tavily(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    assert search_command(_args("setup", "brave", "--oauth")) == 1

    assert "--oauth is supported only for Tavily" in capsys.readouterr().out
