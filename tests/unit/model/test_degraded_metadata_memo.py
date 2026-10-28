"""A resolution whose listing leg FAILED must not be memoised for the day.

``configure._resolve_model_info_cached`` keys its memo on ``(provider, model,
day bucket, ...)``, and every enrichment leg returns its fallback on failure
without raising. Before this file's fix, a 401/DNS/5xx/timeout at the moment of
the FIRST resolution was therefore stored as a normal entry and served for up to
24 h after the network recovered: a 128 000-token default window instead of the
listing's real one, and zero prices. The daemon's boot-time model warm
(``session.index_prewarm.warm_model_metadata``) made that reachable with no user
action, at the moment network and credentials are least likely to be ready.

The seam is ``discovery.fetch_models`` — the one HTTP call under the real
``read_listing`` path — so the cache document, the failure backoff and the memo
all behave exactly as in production.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from local_operator.model import catalogue
from local_operator.model import configure as configure_mod
from local_operator.model import discovery
from local_operator.model.discovery import DiscoveredModel

#: An id the shipped registry cannot describe, so the provider listing is the
#: only source of its window (``_listing_can_correct`` sends it to the network).
MODEL = "listing-only-model-l9"
HEALTHY_WINDOW = 1_500_000


@pytest.fixture
def listing(monkeypatch, tmp_path) -> Iterator[dict[str, bool]]:
    """A radient listing that fails or answers as the test says, from a clean memo."""
    state = {"up": False}
    monkeypatch.setenv("RADIENT_API_KEY", "synthetic-key")
    # Keep the keyless price chain off the network; it is not the leg under test.
    monkeypatch.setattr(configure_mod, "_from_price_catalogue", lambda _p, _m, info, **_k: info)

    def fetch_models(provider_id, **_kwargs):
        if not state["up"]:
            return None  # every transport failure reads as None here
        return [DiscoveredModel(id=MODEL, name=MODEL, context_window=HEALTHY_WINDOW)]

    monkeypatch.setattr(discovery, "fetch_models", fetch_models)
    configure_mod.invalidate_model_info_cache()
    with catalogue._failure_lock:
        catalogue._last_failure.clear()
    yield state
    configure_mod.invalidate_model_info_cache()
    with catalogue._failure_lock:
        catalogue._last_failure.clear()


def _forget_listing_backoff() -> None:
    """Let a minute pass for ``catalogue``'s own one-minute failure backoff.

    That backoff is a separate, deliberate rate bound on the listing fetch; this
    test is about the METADATA memo outliving it by a day, so it clears it the
    way the passing of its window would.
    """
    with catalogue._failure_lock:
        catalogue._last_failure.clear()


def test_a_healthy_listing_resolves_the_listed_window(listing) -> None:
    """Control arm: the seam really does deliver the listing's number."""
    listing["up"] = True
    info = configure_mod.resolve_model_info("radient", MODEL)
    assert info.context_window == HEALTHY_WINDOW


def test_a_failed_resolution_is_not_pinned_after_the_listing_recovers(listing, monkeypatch) -> None:
    """fail -> (warm) resolve -> recover -> open must see the healthy value."""
    clock = [1_700_000_000.0]
    monkeypatch.setattr(configure_mod.time, "time", lambda: clock[0])
    degraded = configure_mod.resolve_model_info("radient", MODEL)
    assert degraded.context_window != HEALTHY_WINDOW

    listing["up"] = True
    _forget_listing_backoff()
    clock[0] += configure_mod.DEGRADED_MEMO_TTL_S + 1
    healed = configure_mod.resolve_model_info("radient", MODEL)
    assert healed.context_window == HEALTHY_WINDOW


def test_a_failed_resolution_is_still_memoised_briefly(listing, monkeypatch) -> None:
    """Inside the short negative window the failed answer IS reused.

    Without it every caller in a failing window — the 1 Hz cost poller, a burst of
    opens — would re-run the whole resolution; ``catalogue``'s backoff already stops
    the fetch, and this keeps the rest of the work bounded the same way.
    """
    clock = [1_700_000_000.0]
    monkeypatch.setattr(configure_mod.time, "time", lambda: clock[0])
    configure_mod.resolve_model_info("radient", MODEL)
    misses = configure_mod._resolve_model_info_cached.cache_info().misses
    clock[0] += 1
    configure_mod.resolve_model_info("radient", MODEL)
    assert configure_mod._resolve_model_info_cached.cache_info().misses == misses


def test_a_complete_answer_keeps_the_day_memo(listing, monkeypatch) -> None:
    """The healthy path is unchanged: memoised for the whole TTL bucket."""
    listing["up"] = True
    clock = [1_700_000_000.0]
    monkeypatch.setattr(configure_mod.time, "time", lambda: clock[0])
    configure_mod.resolve_model_info("radient", MODEL)
    listing["up"] = False
    clock[0] += configure_mod.DEGRADED_MEMO_TTL_S + 1
    info = configure_mod.resolve_model_info("radient", MODEL)
    assert info.context_window == HEALTHY_WINDOW


def test_a_caller_collecting_around_the_resolution_hears_the_failure(listing) -> None:
    """The boot warm stops on the first unavailable listing; it must be told.

    Both on the resolution that failed and on a memo hit of that degraded answer —
    otherwise an outer collector (``index_prewarm._resolve_model``) would see a
    clean answer the second time and keep querying a dead network.
    """
    with catalogue.collect_unavailable() as first:
        configure_mod.resolve_model_info("radient", MODEL)
    assert first
    with catalogue.collect_unavailable() as again:
        configure_mod.resolve_model_info("radient", MODEL)
    assert again
    listing["up"] = True
    configure_mod.invalidate_model_info_cache()
    _forget_listing_backoff()
    with catalogue.collect_unavailable() as healthy:
        configure_mod.resolve_model_info("radient", MODEL)
    assert healthy == []
