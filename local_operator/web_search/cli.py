"""CLI configuration experience for built-in web search."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
from pathlib import Path

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.paths import config_dir
from local_operator.web_search.models import PROVIDER_IDS, SearchProviderId

_API_KEY_NAMES: dict[SearchProviderId, str] = {
    "tavily": "TAVILY_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "brave": "BRAVE_API_KEY",
    "exa": "EXA_API_KEY",
    "serpapi": "SERPAPI_API_KEY",
}
TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"


def add_search_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    parent_parser: argparse.ArgumentParser,
) -> None:
    """Install ``search`` and its focused configuration subcommands."""
    search_parser = subparsers.add_parser(
        "search",
        help="Configure and test load-balanced web search",
        parents=[parent_parser],
    )
    commands = search_parser.add_subparsers(dest="search_command")
    commands.add_parser("list", help="Show provider readiness and enabled state")
    commands.add_parser("on", help="Enable the web_search tool")
    commands.add_parser("off", help="Disable the web_search tool")

    enable = commands.add_parser("enable", help="Enable a provider")
    enable.add_argument("provider", choices=PROVIDER_IDS)
    disable = commands.add_parser("disable", help="Disable a provider")
    disable.add_argument("provider", choices=PROVIDER_IDS)

    strategy = commands.add_parser("balance", help="Choose load-balancing strategy")
    strategy.add_argument("strategy", choices=("round_robin", "ordered"))

    order = commands.add_parser("order", help="Set enabled providers and their priority")
    order.add_argument("providers", nargs="+", choices=PROVIDER_IDS)

    setup = commands.add_parser("setup", help="Configure credentials or free access")
    setup.add_argument("provider", choices=PROVIDER_IDS)
    setup_mode = setup.add_mutually_exclusive_group()
    setup_mode.add_argument(
        "--oauth",
        action="store_true",
        help="Configure Tavily's remote OAuth MCP server",
    )
    setup_mode.add_argument(
        "--api-key",
        action="store_true",
        help="Store an API key instead of using the provider's free mode",
    )
    setup.add_argument("--endpoint", help="SearXNG base URL")

    test = commands.add_parser("test", help="Run a real search through the configured chain")
    test.add_argument("query")
    test.add_argument("--provider", choices=PROVIDER_IDS)
    test.add_argument("--max-results", type=int, default=3)
    test.add_argument("--json", action="store_true", dest="as_json")


def _stack() -> tuple[ConfigManager, CredentialManager]:
    base = config_dir()
    return ConfigManager(base), CredentialManager(base)


def format_search_status(manager: ConfigManager, credentials: CredentialManager) -> str:
    """Plain, width-tolerant status table shared by CLI and TUI."""
    from local_operator.web_search.providers import provider_statuses
    from local_operator.web_search.service import load_search_settings

    settings = load_search_settings(manager)
    header = (
        f"Web search: {'on' if settings.enabled else 'off'} | "
        f"balance: {settings.strategy} | order: {', '.join(settings.providers) or 'none'}"
    )
    rows = [header]
    for status in provider_statuses(settings, credentials):
        enabled = "enabled" if status.enabled else "disabled"
        ready = "ready" if status.available else "setup needed"
        rows.append(f"{status.id:<12} {enabled:<8} {ready:<12} {status.access} | {status.detail}")
    rows.append("Setup: local-operator search setup <provider>")
    return "\n".join(rows)


def _store_api_key(provider_id: SearchProviderId, credentials: CredentialManager) -> None:
    key_name = _API_KEY_NAMES.get(provider_id)
    if key_name is None:
        raise ValueError(f"{provider_id} does not use an API key")
    value = getpass.getpass(f"{key_name}: ").strip()
    if not value:
        raise ValueError("API key was empty; nothing changed")
    credentials.set_credential(key_name, value)


def _setup_tavily_oauth() -> int:
    from local_operator.mcp import config as mcp_config

    current = mcp_config.list_effective_servers(Path.cwd())
    existing = current.get("tavily")
    if isinstance(existing, dict) and existing.get("url") == TAVILY_MCP_URL:
        print("Tavily OAuth MCP is already configured. It will reconnect on the next session.")
        return 0
    result = mcp_config.add_server(
        "tavily",
        url=TAVILY_MCP_URL,
        oauth=True,
        scope="global",
    )
    if result == 0:
        print("Tavily OAuth MCP configured. Start or reload a session to sign in.")
    return result


def _setup_provider(args: argparse.Namespace) -> int:
    from local_operator.web_search.providers import PROVIDERS
    from local_operator.web_search.service import (
        set_provider_enabled,
        set_searxng_endpoint,
    )

    manager, credentials = _stack()
    provider_id: SearchProviderId = args.provider
    if args.oauth and provider_id != "tavily":
        print("error: --oauth is supported only for Tavily")
        return 1
    if args.endpoint and provider_id != "searxng":
        print("error: --endpoint is supported only for SearXNG")
        return 1

    try:
        if provider_id == "tavily" and args.oauth:
            result = _setup_tavily_oauth()
            if result != 0:
                return result
        elif provider_id == "tavily" and args.api_key:
            _store_api_key(provider_id, credentials)
        elif provider_id == "perplexity" and args.api_key:
            _store_api_key(provider_id, credentials)
        elif provider_id in ("brave", "exa", "serpapi"):
            _store_api_key(provider_id, credentials)
        elif provider_id == "searxng":
            endpoint = str(args.endpoint or input("SearXNG base URL: ")).strip().rstrip("/")
            if not endpoint.startswith(("http://", "https://")):
                raise ValueError("SearXNG endpoint must start with http:// or https://")
            set_searxng_endpoint(manager, endpoint)
        # DuckDuckGo, Tavily keyless, and Perplexity anonymous need no secret.
        set_provider_enabled(manager, provider_id, True)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1

    mode = PROVIDERS[provider_id].access
    print(f"Enabled {provider_id} ({mode}).")
    return 0


async def _test_search(args: argparse.Namespace) -> int:
    from local_operator.web_search.service import WebSearchService, load_search_settings

    manager, credentials = _stack()
    service = WebSearchService(load_search_settings(manager), credentials)
    try:
        response = await service.search(
            args.query,
            limit=args.max_results,
            forced_provider=args.provider,
        )
    except Exception as error:
        print(f"error: {error}")
        return 1
    if args.as_json:
        print(json.dumps(response.model_dump(mode="json"), indent=2))
        return 0
    print(f"Provider: {response.provider} ({response.auth_mode})")
    if response.answer:
        print(response.answer)
    for index, source in enumerate(response.sources, start=1):
        print(f"{index}. {source.title}\n   {source.url}")
    if response.failures:
        print("Fallbacks: " + "; ".join(response.failures))
    return 0


def search_command(args: argparse.Namespace) -> int:
    """Dispatch ``search`` configuration and smoke-test commands."""
    from local_operator.web_search.service import (
        set_provider_enabled,
        set_provider_order,
        set_search_enabled,
        set_search_strategy,
    )

    command = args.search_command or "list"
    manager, credentials = _stack()
    if command == "list":
        print(format_search_status(manager, credentials))
        return 0
    if command == "on":
        set_search_enabled(manager, True)
        print("Web search enabled. Reload a running session if the tool was previously off.")
        return 0
    if command == "off":
        set_search_enabled(manager, False)
        print("Web search disabled. Reload a running session to remove the tool.")
        return 0
    if command in ("enable", "disable"):
        set_provider_enabled(manager, args.provider, command == "enable")
        print(f"{args.provider} {command}d.")
        return 0
    if command == "balance":
        set_search_strategy(manager, args.strategy)
        print(f"Web search balance strategy: {args.strategy}")
        return 0
    if command == "order":
        set_provider_order(manager, args.providers)
        print("Web search order: " + ", ".join(args.providers))
        return 0
    if command == "setup":
        return _setup_provider(args)
    if command == "test":
        return asyncio.run(_test_search(args))
    print(f"error: unknown search command {command}")
    return 1
