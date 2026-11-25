"""Radient-hosted web search: Tavily, billed to the user's Radient credits.

WHY THIS LEG EXISTS. A user signed in to Radient already has a funded account,
and Radient's ``GET /v1/tools/search`` serves Tavily from Radient's own key --
the user needs no Tavily account of their own to get higher-volume search than
the keyless tiers allow. The chain places this leg AFTER every free leg and the
native-plan legs, and BEFORE the user's own search API keys (the operator's
"free -> native -> Radient -> API keys" order, ``providers.resolve_provider_bands``).

CONSTRAINTS this module keeps:

* **Presence is a persisted login, never an environment variable.** The leg
  joins the chain only when ``radient_login_present`` finds a stored Radient
  credential -- an exported ``RADIENT_API_KEY`` must not make a paid leg appear
  that the user never signed in to (the same rule the speech cascades follow,
  ``radient_credentials.has_persisted_radient_credential``).
* **The server's 402 is the credit gate.** There is no client-side balance
  pre-check on the hot path: the server admits a request only when the balance
  covers it (minus a courtesy floor), and a client rule could refuse searches the
  server would serve. A 402 is REMEMBERED for ``RADIENT_CREDIT_TTL_SECONDS`` so
  the chain skips this leg without a round trip until a top-up can have landed.
* **401 is not cached.** A refused bearer usually means a rotated login; the next
  search after ``local-operator login radient`` must reach the server at once.
* **Cost comes from the server.** A successful response carries a ``cost``
  object (``amount_usd``, ``basis: "billed"``, ``usage_record_id``); that is the
  bill, so it is recorded verbatim. Only when it is missing is the published
  $0.01 rate used, labelled as a list price rather than a bill.

The base URL is never hard-coded: ``configured_radient_base_url`` resolves the
configured hub (staging, legacy gateway or canonical) exactly as the rest of the
Radient consumers do.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from local_operator.i18n.keys import web_search as web_search_copy
from local_operator.i18n.messages import Msg, render
from local_operator.web_search.models import (
    SearchCost,
    SearchResponse,
    SearchSource,
    SearchUsage,
    WebSearchSettings,
)

#: How long a 402 keeps the leg skipped, in seconds. Five minutes trades one
#: wasted round trip per window against noticing a top-up: Radient credits are
#: bought in a browser, so a user who just topped up waits at most this long,
#: while a drained account does not pay a request per search.
RADIENT_CREDIT_TTL_SECONDS = 300.0

#: Radient's search route always asks the hub for Tavily: it is the provider
#: Radient holds the key for, and the one the per-search price is quoted against.
RADIENT_SEARCH_PROVIDER = "tavily"

_RADIENT_CREDIT_LOCK = threading.Lock()
#: ``time.monotonic()`` of each remembered 402, keyed by ``(base URL, credential
#: fingerprint)``. KEYED, not one process-wide timestamp: a 402 is a verdict on ONE
#: account at ONE hub, so after ``login radient`` as a funded account (or a switch
#: of hub) the next search must reach the server instead of being skipped for up
#: to the TTL (PR #4553 review round 1, R1-6). The fingerprint is a truncated
#: SHA-256 of the bearer -- never the bearer itself -- so the cache holds no
#: credential material.
_RADIENT_CREDIT_REFUSALS: dict[tuple[str, str], float] = {}


def _english(message: Msg) -> str:
    """A keyed message rendered in English.

    These sentences are raised as the leg's failure reason, and failure reasons
    reach the MODEL in the tool result (and the persisted transcript), which the
    i18n contract keeps in English (docs/I18N.md, "Add a string", step 1).
    """
    return render(message.code, message.params, locale="en")


def credential_fingerprint(token: str) -> str:
    """A short, non-reversible handle for a bearer: the 402 cache's account key."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def radient_credit_refusal_remaining(
    base: str | None = None, fingerprint: str | None = None
) -> float | None:
    """Seconds until a remembered 402 lapses, or ``None`` when none is fresh.

    The search path passes BOTH parts and so asks about exactly the account it
    is about to bill. The status surfaces pass neither (they hold no bearer and
    must not resolve one per render): they read "some account in this process
    was refused", which a successful search clears for its hub
    (:func:`_forget_credit_refusals`), so a stale "out of credits" lasts at most
    until the next Radient search succeeds.
    """
    now = time.monotonic()
    with _RADIENT_CREDIT_LOCK:
        matches = [
            refused_at
            for (entry_base, entry_fp), refused_at in _RADIENT_CREDIT_REFUSALS.items()
            if (base is None or entry_base == base)
            and (fingerprint is None or entry_fp == fingerprint)
        ]
    if not matches:
        return None
    remaining = RADIENT_CREDIT_TTL_SECONDS - (now - max(matches))
    return remaining if remaining > 0 else None


def _remember_credit_refusal(base: str = "", fingerprint: str = "") -> None:
    with _RADIENT_CREDIT_LOCK:
        _RADIENT_CREDIT_REFUSALS[(base, fingerprint)] = time.monotonic()


def _forget_credit_refusals(base: str) -> None:
    """Drop every remembered 402 at ``base``: a served search proves it funded."""
    with _RADIENT_CREDIT_LOCK:
        for key in [key for key in _RADIENT_CREDIT_REFUSALS if key[0] == base]:
            del _RADIENT_CREDIT_REFUSALS[key]


def reset_radient_credit_cache_for_tests() -> None:
    """Forget any remembered 402 (tests only; mirrors the DeepSeek helper)."""
    with _RADIENT_CREDIT_LOCK:
        _RADIENT_CREDIT_REFUSALS.clear()


def radient_login_present(config_dir: Path | None) -> bool:
    """Whether a Radient credential is PERSISTED on this install (never env).

    A sync, bare-SELECT probe because it runs inside the status loop and the
    chain resolver, per provider, per search -- the same shape as
    ``providers._deepseek_login_present``. It mirrors
    ``has_persisted_radient_credential``'s two branches: on the canonical hub any
    stored ``radient`` row (a login or a saved Radient API key -- ``radient-key``
    stores under ``radient`` too) counts; on a legacy gateway only the
    ``RADIENT_API_KEY`` store row does, because that branch never receives the
    central login's bearer. Never raises: an unreadable store reads as "not
    signed in", which keeps a paid leg out of the chain rather than in it.

    The canonical branch reads the SAME store the token comes from
    (``radient_credentials._radient_auth_store``): ``config_dir``-scoped, and
    mesh-aware, so a paired device that borrows the owner's Radient bearer gets
    the leg its search would in fact be served on. The ambient
    ``shared_auth_store`` it used to read ignored both (PR #4553 review round 1,
    R1-5). ``list_credentials`` is a read: it never dials a peer.
    """
    try:
        base = _radient_base_url(config_dir)
        from local_operator.providers.radient_credentials import (
            canonical_radient_destination,
        )

        if not canonical_radient_destination(base):
            from local_operator.providers.registry import provider_secret_value

            return bool(provider_secret_value("RADIENT_API_KEY", base=config_dir))
        from local_operator.providers.radient_credentials import _radient_auth_store

        store = _radient_auth_store(config_dir)
        try:
            return bool(store.list_credentials("radient"))
        finally:
            store.close()
    except Exception:  # noqa: BLE001 -- an unreadable store is "not signed in"
        return False


def _radient_base_url(config_dir: Path | None) -> str:
    """The configured Radient API root (version segment included)."""
    from local_operator.env import resolve_radient_api_base_url

    if config_dir is None:
        return resolve_radient_api_base_url(None)
    from local_operator.config import ConfigManager
    from local_operator.providers.radient_credentials import configured_radient_base_url

    return configured_radient_base_url(ConfigManager(config_dir))


async def _radient_token(config_dir: Path | None, base: str) -> str:
    from local_operator.providers.radient_credentials import resolve_radient_credential

    return (await resolve_radient_credential(config_dir, base)).get_secret_value()


def _search_cost(payload: dict[str, Any]) -> tuple[SearchCost, SearchUsage]:
    """The bill from the server's ``cost`` object, or the list-price fallback."""
    from local_operator.clients.radient_cost import micro_from_amount
    from local_operator.web_search.cost import (
        BASIS_BILLED,
        BASIS_PER_SEARCH,
        PROVIDER_USD_PER_SEARCH,
    )

    cost = payload.get("cost")
    if isinstance(cost, dict):
        # The Radient cost contract allows a decimal STRING amount (the shape that
        # survives JSON without rounding); the shared parser reads both, so a
        # "0.0025" bill is recorded as billed rather than replaced by the list
        # price (PR #4553 review round 1, R1-4).
        micro = micro_from_amount(cost.get("amount_usd"))
        units = cost.get("units")
        if micro is not None and micro >= 0:
            vendor = str(cost.get("provider") or RADIENT_SEARCH_PROVIDER)
            return (
                SearchCost(
                    usd=micro / 1_000_000,
                    basis=f"{BASIS_BILLED} Radient ({vendor})",
                    priced_from_usage=True,
                ),
                SearchUsage(
                    credits=(
                        float(units)
                        if isinstance(units, (int, float)) and not isinstance(units, bool)
                        else None
                    )
                ),
            )
    return (
        SearchCost(
            usd=PROVIDER_USD_PER_SEARCH["radient"],
            basis=f"{BASIS_PER_SEARCH}; Radient sent no cost",
            priced_from_usage=False,
        ),
        SearchUsage(),
    )


async def search_radient(
    client: httpx.AsyncClient,
    config_dir: Path | None,
    _settings: WebSearchSettings,
    query: str,
    limit: int,
) -> SearchResponse:
    """One Radient search, raising on every failure so the chain moves on."""
    base = _radient_base_url(config_dir).rstrip("/")
    token = await _radient_token(config_dir, base)
    if not token:
        raise RuntimeError(_english(web_search_copy.radient_not_signed_in()))
    fingerprint = credential_fingerprint(token)
    if radient_credit_refusal_remaining(base, fingerprint) is not None:
        # Fresh 402 for THIS account at THIS hub: skip the round trip. The token
        # is resolved first (a local store read, no network) so a different
        # account is never skipped on another one's verdict.
        raise RuntimeError(_english(web_search_copy.radient_credits_low()))
    response = await client.get(
        f"{base}/tools/search",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "query": query,
            "max_results": limit,
            "provider": RADIENT_SEARCH_PROVIDER,
            "search_depth": "basic",
            "include_raw": "false",
        },
    )
    if response.status_code == 402:
        _remember_credit_refusal(base, fingerprint)
        raise RuntimeError(_english(web_search_copy.radient_credits_low()))
    if response.status_code == 401:
        raise RuntimeError(_english(web_search_copy.radient_sign_in_again()))
    if not response.is_success:
        from local_operator.web_search.providers import _http_error

        raise _http_error("Radient", response)
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError("Radient returned a non-JSON response") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Radient returned a non-object response")
    # A 200 can still carry the upstream's failure (``WebSearchResponse.Error``
    # server-side). Treating it as an empty success would hide the reason.
    error = str(payload.get("error") or "").strip()
    if error:
        raise RuntimeError(f"Radient search failed: {error}")
    from local_operator.web_search.providers import _source

    sources: list[SearchSource] = [
        source
        for item in payload.get("results") or []
        if isinstance(item, dict)
        and (
            source := _source(
                title=item.get("title"),
                url=item.get("url"),
                snippet=item.get("content"),
                published_date=item.get("published_date"),
            )
        )
        is not None
    ]
    _forget_credit_refusals(base)
    cost, usage = _search_cost(payload)
    raw_cost = payload.get("cost")
    record = raw_cost.get("usage_record_id") if isinstance(raw_cost, dict) else None
    return SearchResponse(
        provider="radient",
        auth_mode="radient-credits",
        sources=sources[:limit],
        # The usage record id is the handle Radient support can find the charge
        # by, so it is what this response reports as its request id.
        request_id=str(record).strip() or None if record else None,
        usage=usage,
        cost=cost,
    )
