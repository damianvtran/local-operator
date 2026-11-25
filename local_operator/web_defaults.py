"""The shipped web-tool defaults, in a module that costs nothing to import.

WHY THIS MODULE EXISTS, AND WHY IT IS THIS SMALL. These two constants used to
live beside their pydantic settings models (``web_fetch.models``,
``web_search.models``), which was the right home for *documentation* and the
wrong one for *cost*: ``local_operator.config`` needs nothing from those modules
except these two dicts, and importing them there put pydantic (40 modules) plus
``email`` (13) on every ``import local_operator.config`` — measured at 117.6 ms
of CPU on this host, of which ``web_fetch.models`` was 82.7 ms. ``cli.py``
imports ``config`` unconditionally, so a verb that never fetches a URL paid it.

So the SOURCE OF TRUTH moves here and the two model modules import *from* this
one and re-export, which keeps exactly one definition of each value — the point
is to move the constants, not to fork them. This module imports nothing at all
beyond the stdlib-typing, so it cannot re-introduce the cost it exists to avoid:
anything added here must stay import-cheap (no pydantic, no yaml, no dotenv).

The values are typed ``dict[str, object]`` and are plain dicts, NOT pydantic
model instances — every consumer already treats them as mappings
(``dict(DEFAULT_WEB_FETCH_CONFIG)``, ``{**cfg, "enrich": False}``, ``.items()``
in ``tests/unit/test_settings_io.py``) and the boundary that turns them into
validated settings is ``WebFetchSettings.model_validate`` /
``WebSearchSettings.model_validate`` at the point of use. A caller that wants
attributes validates first; nothing reads these by attribute.
"""

from __future__ import annotations

#: Defaults for the ``values.web_fetch`` block of ``config.yml``.
#:
#: Keys and their meanings are documented on ``WebFetchSettings`` in
#: :mod:`local_operator.web_fetch.models`, which is the validating view of this
#: same mapping; the two are kept in step by
#: ``tests/unit/test_settings_io.py`` walking this dict's keys.
DEFAULT_WEB_FETCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "timeout_seconds": 20.0,
    "max_bytes": 5 * 1024 * 1024,
    "max_redirects": 5,
    "cache_ttl_seconds": 900,
    "allow_private": False,
    "render_backend": "auto",
    "enrich": True,
    "max_attempts": 3,
    "blocked_retry": True,
}

#: Defaults for the ``values.web_search`` block of ``config.yml``.
DEFAULT_WEB_SEARCH_CONFIG: dict[str, object] = {
    "enabled": True,
    "strategy": "round_robin",
    # A PRIORITY PREFIX: these two credential-free transports are tried first, and
    # the resolver appends the rest of this install's usable providers in band
    # order after them (tavily's keyless tier is rate-limited; DDG is the durable
    # no-account fallback). A fresh install therefore walks
    # duckduckgo, tavily, exa, parallel, perplexity, deepseek -- free legs first,
    # the metered tail strictly last.
    "providers": ["duckduckgo", "tavily"],
    "excluded_providers": [],
    "timeout_seconds": 20.0,
    "searxng_endpoint": "",
    "deepseek_evidence": False,
    # Claude-subscription (Anthropic OAuth) native search. OFF by default,
    # operator decision 2026-11-24: Anthropic draws third-party-harness plan use
    # from extra usage, so spending it on search must be an explicit opt-in.
    # The Anthropic API-key leg does not read this.
    "claude_subscription_search": False,
    "read_enabled": True,
}


#: Every built-in web-search provider id, in the user-facing row order of
#: `search list` and `/settings`. The typed view is
#: ``web_search.models.PROVIDER_IDS``; it lives HERE so ``settings_io`` can offer
#: exactly these ids as ``/settings`` members without importing pydantic. (The
#: members used to be hand-copied tuples, so a new provider id was silently
#: unsettable from ``/settings``.)
WEB_SEARCH_PROVIDER_IDS: tuple[str, ...] = (
    "duckduckgo",
    "tavily",
    "deepseek",
    # Radient's hosted search (Tavily billed to Radient credits), beside the
    # native model search it follows in the chain.
    "radient",
    "perplexity",
    "brave",
    "exa",
    # `parallel` sits beside `exa` because the pair is the same family --
    # credential-free MCP endpoints.
    "parallel",
    "serpapi",
    "searxng",
    # The native-search family (``web_search/native/``): a model vendor's own
    # built-in search, billed to the user's plan or model key. Last because they
    # are the newest rows and none of them is a dedicated search product.
    "openai",
    "anthropic",
    "xai",
    "alibaba",
    "google",
    "kimi",
    "zai",
    "mistral",
    "openrouter",
)
