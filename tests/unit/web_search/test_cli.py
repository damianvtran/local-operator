from __future__ import annotations

import json

import pytest

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
    assert "tavily" in resolve_providers(settings, CredentialManager.readonly(config_dir()))


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
    assert "searxng" in resolve_providers(settings, CredentialManager.readonly(config_dir()))
    assert "searxng" not in settings.excluded_providers


def test_setup_rejects_oauth_for_non_tavily(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    assert search_command(_args("setup", "brave", "--oauth")) == 1

    assert "--oauth is supported only for Tavily" in capsys.readouterr().out


def test_enable_is_an_un_exclusion_and_the_landing_line_names_the_band(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

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

    # With a credential the same verb reports the band it will serve from. The
    # key goes in as a PROVIDER-CLASS STORE ROW (PR2a): the plaintext file leg is
    # gone, so the store row is what the resolver actually reads.
    from local_operator.providers.registry import store_provider_key

    store_provider_key("DEEPSEEK_API_KEY", "stored")
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
    # The legend carries the MEANINGS, not just the words (round-2 U2-5/D2-4), scoped
    # to the states these rows paint (round-3 D3-3), and there is no readiness column
    # left to repeat the state word (round-1 D6).
    assert "States: enabled = in your priority order" in table
    assert "auto free = with the free providers" in table
    assert "needs setup = cannot serve yet; a listed one is still tried" in table
    assert "excluded = never used, whatever else is configured" in table
    # No deepseek credential in this fixture, so its paid words are not printed -- and
    # not defined either.
    assert "enabled (paid)" not in table and "auto paid" not in table
    assert "setup needed" not in table


def test_order_receipt_states_where_each_named_provider_lands(
    monkeypatch, tmp_path, capsys
) -> None:
    """Round-2 U2-1: "tried first" was false for a named PAID provider.

    `search order` writes the priority prefix, and the resolver now hoists a paid
    entry into the paid band -- so the receipt has to read the landing off the
    resolver rather than promise the same thing for every id.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    from local_operator.providers.registry import store_provider_key

    store_provider_key("DEEPSEEK_API_KEY", "stored")

    assert search_command(_args("order", "duckduckgo", "tavily")) == 0
    free_only = capsys.readouterr().out
    assert "Web search order: duckduckgo, tavily (tried first; any exclusion" in free_only
    assert "paid" not in free_only

    assert search_command(_args("order", "duckduckgo", "deepseek")) == 0
    mixed = capsys.readouterr().out
    assert "(tried first; deepseek is paid and runs after the free legs; any exclusion" in mixed

    assert search_command(_args("order", "deepseek")) == 0
    paid_only = capsys.readouterr().out
    assert "(deepseek is paid and runs after the free legs; any exclusion" in paid_only
    assert "tried first" not in paid_only

    # And the receipt agrees with the chain the same command produced.
    assert search_command(_args("list")) == 0
    table = capsys.readouterr().out
    assert "chain: " in table and "DeepSeek (paid)" in table


def test_the_help_text_describes_both_bands(capsys) -> None:
    """U8's help line must not promise "tried first" unconditionally either."""
    with pytest.raises(SystemExit):
        build_cli_parser().parse_args(["search", "--help"])
    # argparse wraps the help line to the terminal width, so compare the sentence
    # with its whitespace flattened rather than asserting a line break.
    help_text = " ".join(capsys.readouterr().out.split())
    assert "Set the priority prefix (free legs tried first, paid legs run after them)" in help_text
