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
    "read_enabled": True,
}
