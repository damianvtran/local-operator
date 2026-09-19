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
    # `disable` commits an EXCLUSION and leaves the priority list alone (removing
    # the id could empty the prefix into a value the settings registry rejects);
    # `order` is the explicit "use this" verb, so it clears the exclusion for the
    # ids it names -- duckduckgo here, and not tavily.
    assert settings.excluded_providers == ["tavily"]


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
    # Assert the EFFECTIVE chain, not the stored prefix: `enable` no longer appends
    # to `providers`, and membership of the chain is what the user is asking about.
    from local_operator.credentials import CredentialManager
    from local_operator.web_search.providers import resolve_providers

    settings = load_search_settings(ConfigManager(config_dir()))
    assert "tavily" in resolve_providers(settings, CredentialManager(config_dir()))


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

    from local_operator.credentials import CredentialManager
    from local_operator.web_search.providers import resolve_providers

    settings = load_search_settings(ConfigManager(config_dir()))
    assert settings.searxng_endpoint == "https://search.example.test"
    # `setup` calls `enable`, which now means "clear the exclusion" rather than
    # "append to the stored list". Pin the effective chain instead of the list.
    assert "searxng" in resolve_providers(settings, CredentialManager(config_dir()))
    assert "searxng" not in settings.excluded_providers


def test_setup_rejects_oauth_for_non_tavily(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    assert search_command(_args("setup", "brave", "--oauth")) == 1

    assert "--oauth is supported only for Tavily" in capsys.readouterr().out


def test_enable_is_an_un_exclusion_and_the_landing_line_names_the_band(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.credentials import CredentialManager

    assert search_command(_args("disable", "deepseek")) == 0
    assert "excluded" in capsys.readouterr().out

    # No DeepSeek credential here: allowed, and the reply names what it still
    # needs. "enabled (off; not usable yet)" told the user their action had both
    # worked and not worked and named no way forward (round-1 U4).
    assert search_command(_args("enable", "deepseek")) == 0
    landing = capsys.readouterr().out
    assert "no search can use it yet" in landing
    assert "login deepseek" in landing

    settings = load_search_settings(ConfigManager(config_dir()))
    assert settings.excluded_providers == []
    # `enable` must NOT promote the provider into the priority prefix: for a
    # metered provider that would be a spend decision the user did not make.
    assert "deepseek" not in settings.providers

    # With a credential the same verb reports the band it will serve from.
    CredentialManager(config_dir()).set_credential("DEEPSEEK_API_KEY", "stored")
    assert search_command(_args("disable", "deepseek")) == 0
    capsys.readouterr()
    assert search_command(_args("enable", "deepseek")) == 0
    landing = capsys.readouterr().out
    # The landing sentence is the same one the TUI prints (`provider_landing_line`),
    # and it says where the leg will actually serve from.
    assert "deepseek enabled (auto paid; tried after the free providers" in landing


def test_search_list_states_the_chain_bands_and_the_new_vocabulary(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    assert search_command(_args("disable", "brave")) == 0
    capsys.readouterr()

    assert search_command(_args("list")) == 0
    table = capsys.readouterr().out

    # `order:` is the stored prefix; `chain:` is what a search will walk, and it
    # names the free legs first (round-1 D1).
    assert "chain: DuckDuckGo → Tavily → Exa → Parallel → Perplexity" in table
    assert "excluded: brave" in table
    assert "duckduckgo   enabled" in table
    assert "exa          auto free" in table
    assert "brave        excluded" in table
    # A provider that cannot serve and was never excluded is the `needs setup`
    # state: no readiness column repeats it any more (round-1 D6).
    assert "serpapi      needs setup" in table
    # Every state word the table can print is defined in the legend under it, and
    # there is no readiness column left to repeat the state word (round-1 D6/U5).
    assert "States: enabled · enabled (paid) · auto free · auto best-effort" in table
    assert "setup needed" not in table
