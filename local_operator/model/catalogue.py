"""Disk-cached model catalogue for providers with no static registry entry.

WHY THIS EXISTS
---------------
``model/registry.py`` hard-codes windows and prices for providers whose
catalogues are small and stable (Anthropic, OpenAI, Kimi, ...). Aggregators
cannot work that way: OpenRouter alone routes hundreds of models that change
weekly, so its registry entry is a placeholder with ``context_window = -1``
("no data") and zero prices.

``configure_model`` has always been able to read the real numbers from the
provider's ``list_models()`` payload, but only when handed a
``model_info_client`` — and the session factory never handed it one. Every
OpenRouter and Radient session therefore ran on the fallbacks in
``configure.py``, which is wrong in three ways that all matter at runtime:

* **Context window** fell back to ``UNKNOWN_CONTEXT_WINDOW`` (128k). Auto
  compaction derives its threshold from the window, so a 1M-context model
  compacted at ~102k instead of ~800k — an 8x premature summarisation of
  history the model could still hold, on every long session.
* **Prices** stayed 0.0, so the status band could only report ``$—``. Cost is
  one of the few numbers an operator steers by mid-task.
* **``supports_prompt_cache``** stayed False, which gates ``cache_control``
  emission. Models reached THROUGH OpenRouter that need explicit breakpoints
  (the Anthropic family) silently never got them.

WHY IT IS CACHED
----------------
The fix is to call the listing — but a naive call adds a synchronous HTTP
round trip to every session start, and makes an offline start fail for a
reason the user cannot act on. Model catalogues change on the order of days,
so the payload is cached on disk with a TTL and a stale entry is preferred to
no entry:

1. Fresh cache (< TTL)            -> use it, no network.
2. Stale or missing               -> fetch, rewrite the cache, use it.
3. Fetch fails but a stale cache exists -> use the stale copy. A window that
   is a week old is enormously better than pretending a 1M model holds 128k.
4. Fetch fails with no cache      -> return None; the caller keeps its static
   fallbacks. A session MUST still start with no network.

The cache is keyed by provider and holds the raw payload, so the mapping in
``_info_from_listing`` stays the single place that interprets it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("local_operator.model.catalogue")

#: How long a cached catalogue is considered fresh. Model listings move on the
#: order of days; an hour would re-fetch constantly for no new information and
#: a week would hide a genuinely new model for too long.
DEFAULT_TTL_S = 24 * 60 * 60

#: Providers whose static registry entry is a placeholder, so the listing is
#: the ONLY source of their real windows and prices.
LISTING_PROVIDERS = frozenset({"openrouter", "radient"})


def default_cache_dir() -> Path:
    """Same cache root the skills index uses, so there is one place to clear."""
    return Path("~/.local-operator/cache").expanduser()


def _cache_path(provider: str, cache_dir: Path | None) -> Path:
    return (cache_dir or default_cache_dir()) / f"{provider}.models.json"


def _read_cache(path: Path) -> tuple[dict[str, Any] | None, float]:
    """Return ``(payload, age_seconds)``; ``(None, inf)`` when unreadable.

    A corrupt cache is treated as absent rather than raised: this is an
    optimisation store, and a half-written file from a killed process must not
    stop a session from starting.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw["payload"]
        fetched_at = float(raw["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, float("inf")
    if not isinstance(payload, dict):
        return None, float("inf")
    return payload, max(0.0, time.time() - fetched_at)


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Persist a payload, best-effort.

    Written to a sibling temp file and renamed so a concurrent reader never
    sees a partial document (two sessions may start at once).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:  # pragma: no cover - disk full, read-only home
        logger.debug("could not cache %s catalogue: %s", path.name, exc)


def cached_listing(
    provider: str,
    fetch: Callable[[], dict[str, Any]],
    *,
    ttl_s: float = DEFAULT_TTL_S,
    cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """The provider's ``list_models()`` payload, from cache when fresh.

    ``fetch`` returns the payload as a plain dict (``model_dump()`` of the
    client's response). Returns None only when there is no cache AND the fetch
    failed, which is the one case where the caller must keep its static
    fallbacks.
    """
    path = _cache_path(provider, cache_dir)
    payload, age = _read_cache(path)
    if payload is not None and age < ttl_s:
        return payload

    try:
        fresh = fetch()
    except Exception as exc:  # noqa: BLE001 - any client/transport error degrades
        if payload is not None:
            # Stale beats absent: the numbers are days old at worst, whereas
            # the static fallback is wrong by a factor of eight.
            logger.debug("using stale %s catalogue (%.0fs old): %s", provider, age, exc)
            return payload
        logger.debug("no %s catalogue available: %s", provider, exc)
        return None

    _write_cache(path, fresh)
    return fresh
