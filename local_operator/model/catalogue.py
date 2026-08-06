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
  compacted at ~102k instead of ~600k (`min(0.8 * window, 600_000)`, the cap in
  `compaction/thresholds.py`) — a ~5.9x premature summarisation of history the
  model could still hold, on every long session.
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
import os
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
    """Return ``(payload, age_seconds)``; ``(None, inf)`` when unusable.

    A corrupt cache is treated as absent rather than raised: this is an
    optimisation store, and a half-written file from a killed process must not
    stop a session from starting.

    A timestamp in the FUTURE also counts as unusable. Clamping the age to zero
    instead (the obvious move) makes such an entry look permanently fresh, so a
    single clock skew — an NTP correction, a suspend/resume, a file copied from
    another machine — would pin the catalogue forever with no way to recover
    short of deleting the file by hand. Refetching once is the cheap direction
    to be wrong in.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw["payload"]
        fetched_at = float(raw["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, float("inf")
    if not isinstance(payload, dict):
        return None, float("inf")
    age = time.time() - fetched_at
    if age < 0:
        return payload, float("inf")
    return payload, age


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Persist a payload, best-effort.

    Written to a temp file and renamed, because ``rename`` within a directory
    is atomic: a concurrent reader sees either the old document or the new one,
    never a partial write.

    The temp name carries the PID. A name derived only from the target (the
    obvious ``path.with_suffix('.tmp')``) is SHARED by every concurrent writer,
    so two sessions starting together interleave their writes into one file and
    then rename the resulting corrupt document into place — turning the atomic
    rename into a guarantee that the corruption is delivered intact.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "payload": payload}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:  # pragma: no cover - disk full, read-only home
        logger.debug("could not cache %s catalogue: %s", path.name, exc)
        # A failed write can strand the temp file; it would otherwise accumulate
        # one per failing start.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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
            # the static fallback is wrong by nearly a factor of six.
            logger.debug("using stale %s catalogue (%.0fs old): %s", provider, age, exc)
            return payload
        logger.debug("no %s catalogue available: %s", provider, exc)
        return None

    _write_cache(path, fresh)
    return fresh


def invalidate(provider: str, *, cache_dir: Path | None = None) -> None:
    """Drop ``provider``'s cached catalogue so the next call refetches.

    Needed because the payload is written BEFORE anything tries to interpret
    it. A document that validates but cannot be mapped (a non-scalar price, a
    context length arriving as an object) would otherwise be served as a fresh
    cache hit on every start for the whole TTL, repeating the same failure for a
    day with no way to recover but deleting the file by hand.
    """
    try:
        _cache_path(provider, cache_dir).unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - read-only cache dir
        logger.debug("could not invalidate %s catalogue: %s", provider, exc)
