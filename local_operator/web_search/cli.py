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
from local_operator.web_search.models import (
    PROVIDER_IDS,
    SearchProviderId,
    WebSearchSettings,
)

_API_KEY_NAMES: dict[SearchProviderId, str] = {
    "tavily": "TAVILY_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "brave": "BRAVE_API_KEY",
    "exa": "EXA_API_KEY",
    "parallel": "PARALLEL_API_KEY",
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

    enable = commands.add_parser("enable", help="Allow a provider (clears its exclusion)")
    enable.add_argument("provider", choices=PROVIDER_IDS)
    disable = commands.add_parser("disable", help="Never use a provider (excludes it)")
    disable.add_argument("provider", choices=PROVIDER_IDS)

    strategy = commands.add_parser("balance", help="Choose load-balancing strategy")
    strategy.add_argument("strategy", choices=("round_robin", "ordered"))

    # The help line says what the verb does to the CHAIN, in both bands: a paid id is
    # held back to the paid band by the strengthening, so an unconditional "tried
    # first" here is the same false promise the receipt used to make (round-2 U2-1).
    order = commands.add_parser(
        "order",
        help="Set the priority prefix (free legs tried first, paid legs run after them)",
    )
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
    from local_operator.web_search.providers import (
        chain_label,
        provider_state_label,
        provider_statuses,
        state_legend,
    )
    from local_operator.web_search.service import load_search_settings

    settings = load_search_settings(manager)
    statuses = provider_statuses(settings, credentials)
    header = (
        f"Web search: {'on' if settings.enabled else 'off'} | "
        f"balance: {settings.strategy}\n"
        f"order: {', '.join(settings.providers) or 'none'} | "
        f"excluded: {', '.join(settings.excluded_providers) or 'none'}\n"
        # `order:` is the stored prefix; this line is the chain as it will be
        # walked, every leg named in try order. A paid leg is marked, because on an
        # install that LISTS one the order line cannot show where it lands.
        f"chain: {chain_label(statuses)}"
    )
    rows = [header]
    for status in statuses:
        # ONE state column, no readiness column: a provider that cannot serve is
        # never in a chain, so `off`/`setup needed` said the same thing twice on
        # every such row (round-1 D6). The words and meanings come from
        # providers.py, and the legend below defines them in place.
        rows.append(
            f"{status.id:<12} {provider_state_label(status):<16} {status.access} | {status.detail}"
        )
    rows.append("Setup: local-operator search setup <provider>")
    # The legend is scoped to the states these rows actually paint: the full table
    # costs 526 cells and defines states this install does not have (round-3 D3-3).
    rows.append(state_legend(statuses))
    return "\n".join(rows)


def _store_api_key(provider_id: SearchProviderId, credentials: CredentialManager) -> None:
    """Save a search provider's API key as a provider-class store row.

    Writes ``LOP_PROVIDER_<KEY>`` with ``role="provider"`` — the consolidated
    home for provider keys — so ``_credential`` finds it on the store-first leg.
    """
    from local_operator.providers.registry import store_provider_key

    key_name = _API_KEY_NAMES.get(provider_id)
    if key_name is None:
        raise ValueError(f"{provider_id} does not use an API key")
    value = getpass.getpass(f"{key_name}: ").strip()
    if not value:
        raise ValueError("API key was empty; nothing changed")
    store_provider_key(key_name, value, description=f"Web search API key for {provider_id}")


def _setup_tavily_oauth() -> int:
    from local_operator.mcp import config as mcp_config

    current, sources = mcp_config.load_all_mcp_configs(Path.cwd())
    existing = current.get("tavily")
    if existing is not None:
        existing_url = getattr(existing, "url", "").rstrip("/")
        auth = getattr(existing, "auth", None)
        if existing_url == TAVILY_MCP_URL.rstrip("/") and getattr(auth, "type", None) == "oauth":
            print("Tavily OAuth MCP is already configured. It will reconnect on the next session.")
            return 0

        global_path = mcp_config._scope_path(None, "global")
        source = Path(sources["tavily"])
        if source != global_path:
            print(
                f"error: Tavily is configured by {source} without the required "
                "OAuth URL/auth block; update or remove that higher-priority entry"
            )
            return 1

    result = mcp_config.set_http_oauth_server("tavily", TAVILY_MCP_URL, scope="global")
    if result == 0:
        print("Tavily OAuth MCP configured. Start or reload a session to sign in.")
    return result


def _setup_provider(args: argparse.Namespace) -> int:
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
        elif provider_id == "deepseek":
            if args.api_key:
                _store_api_key(provider_id, credentials)
            else:
                # DeepSeek search has no search-specific secret: it bills the
                # same key the model route uses. Saying so is the whole setup, and
                # the availability gate reads the login store, so this only has to
                # point at `login` when neither tier is populated yet.
                print(
                    "DeepSeek search reuses the DeepSeek model key (one model turn "
                    "per search). If it is not set yet, run "
                    "`local-operator login deepseek`; the provider becomes available "
                    "on the next search, and it joins the chain automatically in the "
                    "paid band -- tried only after the free providers. Pass --api-key "
                    "to store a separate key."
                )
        elif provider_id in ("exa", "parallel"):
            if args.api_key:
                _store_api_key(provider_id, credentials)
            else:
                # Their free tier is the DEFAULT now, so setup must not demand a
                # credential: prompting for a key a user does not need, and failing
                # with "API key was empty; nothing changed" when they decline, is the
                # opposite of what this provider needs. `--api-key` is the explicit
                # opt-in to the higher-limit keyed mode.
                print(
                    f"{provider_id} works keyless (free MCP tier, no key needed). Pass "
                    f"--api-key to store a key and use the higher-limit keyed API."
                )
        elif provider_id in ("brave", "serpapi"):
            _store_api_key(provider_id, credentials)
        elif provider_id == "searxng":
            endpoint = str(args.endpoint or input("SearXNG base URL: ")).strip().rstrip("/")
            if not endpoint.startswith(("http://", "https://")):
                raise ValueError("SearXNG endpoint must start with http:// or https://")
            set_searxng_endpoint(manager, endpoint)
        # DuckDuckGo, Tavily keyless, and Perplexity anonymous need no secret.
        # Exa/Parallel always SERVED without a key too; `--api-key` only stores
        # the higher-limit credential for them.
        settings = set_provider_enabled(manager, provider_id, True)
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 1

    if provider_id in ("exa", "parallel"):
        # Their free tier is the DEFAULT, and the key switches transports rather
        # than merely raising a limit, so the landing line has to say which one is
        # now in use -- a keyed exa is billed at the published REST rate, not free.
        if args.api_key:
            print(
                f"Enabled {provider_id} with a stored key (billed at the keyed API's "
                "rate; remove the key to fall back to the free keyless MCP tier)."
            )
        else:
            print(f"Enabled {provider_id} (keyless MCP tier: free, rate-limited).")
        return 0
    if provider_id == "deepseek":
        # The setup sentence above says what the provider IS; where it LANDED comes
        # from the resolver, like every other surface's landing line. The hardcoded
        # `auto paid` this replaces described a LISTED leg as if the user had not
        # listed it, two commands after `search enable deepseek` learned to say
        # `enabled (paid)` on the same install (round-2 U2-4).
        _print_landing(settings, credentials, provider_id)
        return 0
    _print_landing(settings, credentials, provider_id)
    return 0


def _print_landing(
    settings: WebSearchSettings,
    credentials: CredentialManager,
    provider_id: SearchProviderId,
) -> None:
    """Say where a provider landed, so the user never has to guess.

    The sentence comes from ``provider_landing_line`` so this surface and /search
    print the same words for the same state; `search enable` only clears an
    exclusion, so what a user needs is where the provider will serve from -- or,
    when it cannot serve yet, the command that fixes that (round-1 U4).
    """
    from local_operator.web_search.providers import (
        provider_landing_line,
        provider_statuses,
    )

    status = next(
        (row for row in provider_statuses(settings, credentials) if row.id == provider_id),
        None,
    )
    if status is None:
        print(f"{provider_id} enabled.")
        return
    print(f"{provider_landing_line(provider_id, status)}.")


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
    from local_operator.web_search.providers import provider_order_note
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
        settings = set_provider_enabled(manager, args.provider, command == "enable")
        if command == "disable":
            # Name the way back: `disable` is the only verb that can leave a chain
            # with no usable provider, and the fix is one command.
            print(
                f"{args.provider} excluded; it will not be used by any search until "
                f"you run `local-operator search enable {args.provider}`."
            )
        else:
            _print_landing(settings, credentials, args.provider)
        return 0
    if command == "balance":
        set_search_strategy(manager, args.strategy)
        print(f"Web search balance strategy: {args.strategy}")
        return 0
    if command == "order":
        settings = set_provider_order(manager, args.providers)
        # The receipt is DERIVED from the resolver, not hardcoded: naming a paid
        # provider writes a prefix entry that the resolver hoists into the paid band,
        # so "tried first" would be a promise this very command breaks (round-2
        # U2-1). The helper states the landing for each half of the named list.
        print(
            "Web search order: "
            + ", ".join(args.providers)
            + " "
            + provider_order_note(list(args.providers), settings, credentials)
        )
        return 0
    if command == "setup":
        return _setup_provider(args)
    if command == "test":
        return asyncio.run(_test_search(args))
    print(f"error: unknown search command {command}")
    return 1
