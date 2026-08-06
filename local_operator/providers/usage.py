"""Provider usage-quota fetchers, normalized to one ``UsageReport`` shape.

A compact per-provider usage layer with a single normalized output shape.
The design is deliberately small because local-operator is a per-token
billing product: most providers here
are pay-per-token API keys with **no** public balance endpoint (OpenRouter
is the notable exception — its ``/api/v1/auth/key`` returns live credit
data). The natural line is drawn the same way: live quota is implemented only for
subscription/coding-plan products and shows local token tallies for the
rest.

What local-operator DOES fetch:

- ``openrouter`` — `GET /api/v1/auth/key` with the API key → USD credit
  usage/limit plus free-tier state. The one every local-operator user has.
- ``zai`` — `GET /api/monitor/usage/quota/limit` (raw token, no ``Bearer``)
  — a per-token provider that DOES expose a quota endpoint, so it doubles
  as the template for any future provider with a token budget.
- OAuth subscription plans when the user logged in via OAuth: ``anthropic``
  (``/api/oauth/usage``), ``openai``/``openai-device`` (backend
  ``/wham/usage``), ``kimi`` (``/coding/v1/usages``), ``xai`` (billing).

Every fetcher returns the SAME ``UsageReport`` (or ``None`` when the
provider/credential cannot report) so the TUI renders one shape and is
never coupled to a provider's response schema. A missing endpoint is not an
error — it is a ``None`` report that the caller leaves off the table.

Anything that hits the network uses a caller-provided ``httpx.AsyncClient``
so the TUI can share one connection pool and cancel fetches cleanly; no
code path here owns a client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

#: Human labels for the unit a usage amount is measured in.
UNIT_LABELS = {"usd": "USD", "percent": "%", "tokens": "tokens", "requests": "req", "unknown": ""}

#: OpenRouter's key endpoint doubles as the credit statement. Returns 200 with
#: ``data`` for a valid key, 401 for an invalid one.
#:
#: The DOCUMENTED path, deliberately. `/api/v1/auth/key` still answers with an
#: identical body but is no longer in the docs, so it is an undocumented alias —
#: the kind of thing that keeps working right up until it does not, in a fetcher
#: whose failure mode is a silently empty table.
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"

#: Anthropic OAuth usage (subscription plan). OAuth-only; server unreachable
#: with a raw API key.
ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"

#: OpenAI ChatGPT backend usage windows (OAuth device-code / ChatGPT Plus).
OPENAI_WHAM_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

#: Kimi coding-plan usage (OAuth).
KIMI_USAGE_URL = "https://api.kimi.com/coding/v1/usages"

#: Moonshot/Kimi account balance, reachable with the PLAIN API key the registry
#: already stores. Its absence was the widest gap in this module: a KIMI_API_KEY
#: user was told by `/provider` that Kimi reports quota and then got an empty
#: table forever, because the only Kimi fetcher wanted an OAuth token.
MOONSHOT_BALANCE_URL = "https://api.moonshot.ai/v1/users/me/balance"

#: DeepSeek account balance — plain Bearer with the key already in the registry.
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

#: xAI Grok subscription usage (OAuth).
XAI_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing"

#: Providers with a live quota endpoint. Kept as a set so the registry and
#: the TUI can answer "does this provider report usage?" without importing
#: every fetcher.
#:
#: ``zai`` was removed rather than fixed. It had a working fetcher and a passing
#: test, but no ``ProviderDefinition`` — so `/login zai` raised, its env var was
#: never read, and no code path could insert the credential row the fetcher
#: needed. It was unreachable by construction, and a set that advertises an
#: unreachable provider is worse than one that is merely incomplete.
USAGE_PROVIDERS: frozenset[str] = frozenset(
    {
        "openrouter",
        "anthropic",
        "openai",
        "openai-device",
        "kimi",
        "deepseek",
        "xai",
        "xai-oauth",
    }
)


@dataclass
class UsageAmount:
    """One window's numeric usage, with any two of used/limit/remaining."""

    used: float | None = None
    limit: float | None = None
    remaining: float | None = None
    used_fraction: float | None = None  # explicit 0..1, precedence over derived
    unit: str = "usd"  # one of UNIT_LABELS

    def fraction(self) -> float | None:
        """Best-effort consumed fraction 0..1, or None when unmeasurable."""
        if self.used_fraction is not None:
            return self.used_fraction
        if self.used is not None and self.limit and self.limit > 0:
            return self.used / self.limit
        if (
            self.used is not None
            and self.remaining is not None
            and (self.used + self.remaining) > 0
        ):
            denom = self.used + self.remaining
            return self.used / denom
        return None

    def status(self) -> str:
        """ok / warning / exhausted derived from the fraction."""
        frac = self.fraction()
        if frac is None:
            return "unknown"
        if frac >= 1.0:
            return "exhausted"
        if frac >= 0.85:
            return "warning"
        return "ok"


@dataclass
class UsageLimit:
    """One named allowance on one rolling window."""

    id: str
    label: str
    amount: UsageAmount
    window: str = ""
    status: str | None = None  # default derived from amount
    resets_at: str | None = None

    def effective_status(self) -> str:
        return self.status or self.amount.status()


@dataclass
class UsageReport:
    """The normalized quota state for one provider at one instant."""

    provider: str
    fetched_at: int = field(default_factory=lambda: int(time.time() * 1000))
    limits: list[UsageLimit] = field(default_factory=list)
    notes: str | None = None
    #: Pass-through identity so the TUI can annotate which account reported.
    identity: str | None = None


# ---------------------------------------------------------------------------
# Per-provider fetchers. Each returns a ``UsageReport`` or ``None``.
# ---------------------------------------------------------------------------


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _get_json(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], *, timeout: float = 10.0
) -> dict[str, Any] | None:
    """GET and return the JSON body, or None on any transport/status failure.

    A quota failure is never a crash: the caller drops the report.
    """
    try:
        resp = await client.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


async def fetch_openrouter(client: httpx.AsyncClient, api_key: str) -> UsageReport | None:
    """OpenRouter credit statement from ``/api/v1/key``.

    The response carries the key's own usage so it is the closest thing to a
    "how much have I spent / how much is left" without a separate billing
    API. ``is_free_tier`` (unlimited, rate-limited) is surfaced as a note
    rather than a 0/0 budget.
    """
    payload = await _get_json(client, OPENROUTER_KEY_URL, _bearer(api_key))
    if payload is None:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    try:
        usage = float(data.get("usage") or 0.0)
        limit = float(data.get("limit") or 0.0)
    except (TypeError, ValueError):
        return None
    is_free_tier = bool(data.get("is_free_tier"))
    limits: list[UsageLimit] = []
    notes: str | None = None
    if is_free_tier:
        # Free tier: no credit budget, rate-limited. Don't fabricate a 0/0
        # budget that would read as "spent everything".
        notes = "free tier — no credit limit (rate-limited)"
    elif limit > 0:
        limits.append(
            UsageLimit(
                id="openrouter:credits",
                label="Credits",
                amount=UsageAmount(used=usage, limit=limit, unit="usd"),
                window="lifetime",
            )
        )
    else:
        # A key with no limit set on the dashboard: report spend only.
        limits.append(
            UsageLimit(
                id="openrouter:spend",
                label="Spend (no limit set)",
                amount=UsageAmount(used=usage, unit="usd"),
                window="lifetime",
            )
        )
    return UsageReport(provider="openrouter", limits=limits, notes=notes)


async def fetch_moonshot_balance(client: httpx.AsyncClient, api_key: str) -> UsageReport | None:
    """Moonshot/Kimi account balance from the PLAIN API key.

    The counterpart to :func:`fetch_kimi_oauth`, and the one most Kimi users can
    actually reach: the registry stores ``KIMI_API_KEY`` and the coding-plan
    endpoint only accepts an OAuth token, so an API-key user had a provider that
    claimed to report quota and never did.

    Balances are USD. ``available_balance`` is the actionable number — it already
    nets the voucher and cash components the response also breaks out — so it is
    reported as ``remaining`` with no limit. There is no spend figure to pair it
    with, and inventing ``used = limit - remaining`` from a limit we were never
    given would draw a progress bar out of an assumption.
    """
    payload = await _get_json(client, MOONSHOT_BALANCE_URL, _bearer(api_key))
    if payload is None:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    available = _num(data.get("available_balance"))
    if available is None:
        return None
    notes: str | None = None
    voucher = _num(data.get("voucher_balance"))
    cash = _num(data.get("cash_balance"))
    if voucher is not None and cash is not None:
        notes = f"voucher ${voucher:.2f} + cash ${cash:.2f}"
    return UsageReport(
        provider="kimi",
        limits=[
            UsageLimit(
                id="kimi:balance",
                label="Balance",
                amount=UsageAmount(remaining=available, unit="usd"),
                window="lifetime",
            )
        ],
        notes=notes,
    )


async def fetch_deepseek_balance(client: httpx.AsyncClient, api_key: str) -> UsageReport | None:
    """DeepSeek account balance from the key the registry already stores.

    DeepSeek returns one entry per currency, so the currency is part of the limit
    id rather than assumed: a CNY balance rendered as USD would be wrong by roughly
    a factor of seven. ``is_available`` is the provider's own "can this account
    still serve requests" flag and is worth surfacing, because a zero balance and a
    suspended account look identical in the numbers alone.
    """
    payload = await _get_json(client, DEEPSEEK_BALANCE_URL, _bearer(api_key))
    if payload is None:
        return None
    infos = payload.get("balance_infos")
    if not isinstance(infos, list):
        return None
    limits: list[UsageLimit] = []
    for item in infos:
        if not isinstance(item, dict):
            continue
        total = _num(item.get("total_balance"))
        if total is None:
            continue
        currency = str(item.get("currency") or "").upper() or "USD"
        limits.append(
            UsageLimit(
                id=f"deepseek:balance:{currency.lower()}",
                label=f"Balance ({currency})",
                # `unit` is the renderer's vocabulary, not the vendor's: only USD
                # has a symbol here, so a CNY balance is reported unitless with the
                # currency in the label rather than mislabelled as dollars.
                amount=UsageAmount(remaining=total, unit="usd" if currency == "USD" else "unknown"),
                window="lifetime",
            )
        )
    if not limits:
        return None
    notes = None if payload.get("is_available", True) else "account not available for requests"
    return UsageReport(provider="deepseek", limits=limits, notes=notes)


async def fetch_anthropic_oauth(client: httpx.AsyncClient, access_token: str) -> UsageReport | None:
    """Anthropic subscription usage (5h / 7d windows, model-scoped tiers)."""
    payload = await _get_json(client, ANTHROPIC_USAGE_URL, _bearer(access_token))
    if payload is None:
        return None
    limits: list[UsageLimit] = []
    # Generic limits[] carries named windows (session, weekly_all, ...).
    raw_limits = payload.get("limits")
    if isinstance(raw_limits, list):
        for item in raw_limits:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "limit")
            used = item.get("used")
            limit = item.get("limit")
            if used is None:
                continue
            resets = item.get("resets_at")
            limits.append(
                UsageLimit(
                    id=f"anthropic:{kind}",
                    label=f"Claude {kind}",
                    amount=UsageAmount(used=_num(used), limit=_num(limit) or None, unit="percent"),
                    window=str(item.get("window") or kind),
                    resets_at=str(resets) if resets else None,
                )
            )
    # Named top-level windows (five_hour / seven_day / seven_day_opus).
    for key, label in (
        ("five_hour", "5 hour"),
        ("seven_day", "7 day"),
        ("seven_day_opus", "7 day Opus"),
        ("seven_day_sonnet", "7 day Sonnet"),
    ):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("used") is not None:
            limits.append(
                UsageLimit(
                    id=f"anthropic:{key}",
                    label=label,
                    amount=UsageAmount(
                        used=_num(value.get("used")),
                        limit=_num(value.get("limit")) or None,
                        unit="percent",
                    ),
                    window=label,
                )
            )
    return UsageReport(provider="anthropic", limits=limits) if limits else None


async def fetch_openai_oauth(
    client: httpx.AsyncClient, access_token: str, account_id: str | None = None
) -> UsageReport | None:
    """OpenAI ChatGPT/Codex plan rate-limit windows (primary / secondary)."""
    headers = _bearer(access_token)
    headers["User-Agent"] = "LocalOperator/1.0"
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    payload = await _get_json(client, OPENAI_WHAM_USAGE_URL, headers)
    if payload is None:
        return None
    limits: list[UsageLimit] = []
    for key in ("primary", "secondary"):
        window = payload.get(key)
        if not isinstance(window, dict):
            continue
        used_pct = window.get("used_percent")
        minutes = window.get("window_minutes")
        resets = window.get("resets_at") or window.get("resets_at_utc")
        frac = _num(used_pct)
        frac = frac / 100.0 if frac is not None else None
        limits.append(
            UsageLimit(
                id=f"openai:{key}",
                label=key.capitalize(),
                amount=UsageAmount(used_fraction=frac, unit="percent"),
                window=f"{minutes} min" if minutes else "window",
                resets_at=str(resets) if resets else None,
            )
        )
    return UsageReport(provider="openai", limits=limits) if limits else None


async def fetch_kimi_oauth(client: httpx.AsyncClient, access_token: str) -> UsageReport | None:
    """Kimi coding-plan usage windows (OAuth)."""
    payload = await _get_json(client, KIMI_USAGE_URL, _bearer(access_token))
    if payload is None or not isinstance(payload.get("data"), dict):
        return None
    data = payload["data"]
    raw_limits = data.get("limits")
    if not isinstance(raw_limits, list):
        return None
    limits: list[UsageLimit] = []
    for item in raw_limits:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("title") or "quota")
        used = item.get("used")
        limit = item.get("limit")
        if used is None:
            continue
        resets = item.get("reset_time")
        limits.append(
            UsageLimit(
                id=f"kimi:{name}",
                label=name,
                amount=UsageAmount(used=_num(used), limit=_num(limit) or None, unit="unknown"),
                window=str(item.get("window") or "window"),
                resets_at=str(resets) if resets else None,
            )
        )
    return UsageReport(provider="kimi", limits=limits) if limits else None


async def fetch_xai_oauth(client: httpx.AsyncClient, access_token: str) -> UsageReport | None:
    """xAI Grok subscription (weekly credits and monthly included)."""
    headers = _bearer(access_token)
    headers["Accept"] = "application/json"
    headers["X-XAI-Token-Auth"] = "xai-grok-cli"
    try:
        resp = await client.get(XAI_BILLING_URL, headers=headers, timeout=10.0)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except ValueError:
        return None
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        return None
    limits: list[UsageLimit] = []
    # Weekly credits.
    weekly = _num(config.get("creditUsagePercent"))
    if weekly is not None:
        limits.append(
            UsageLimit(
                id="xai:credits:1w",
                label="Weekly credits",
                amount=UsageAmount(used_fraction=weekly / 100.0, unit="percent"),
                window="1 week",
            )
        )
    # Monthly included.
    monthly_limit = config.get("monthlyLimit")
    used = config.get("used")
    if monthly_limit is not None and used is not None:
        limits.append(
            UsageLimit(
                id="xai:included:1mo",
                label="Monthly included",
                amount=UsageAmount(used=_num(used), limit=_num(monthly_limit), unit="unknown"),
                window="1 month",
            )
        )
    return UsageReport(provider="xai", limits=limits) if limits else None


def _num(value: Any, default: float | None = None) -> float | None:
    """Coerce numeric/str fields defensively; None on garbage."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


#: The fetcher a provider id routes to. Naming the kinds keeps the dispatch
#: table and the if-chain in :func:`fetch_usage` in lockstep.
FetcherKind = Literal[
    "openrouter",
    "anthropic-oauth",
    "openai-oauth",
    "kimi-oauth",
    "moonshot-balance",
    "deepseek-balance",
    "xai-oauth",
]

#: Provider id (canonical) -> ``(oauth_fetcher, api_key_fetcher)``.
#:
#: A PAIR, because a provider's two credential kinds often mean two different
#: endpoints rather than two ways of authenticating one. Kimi is the clearest
#: case: the coding-plan usage route wants an OAuth token, while the account
#: balance wants the plain API key the registry already stores, and they live on
#: different hosts and return different shapes. Under the old one-fetcher-per-
#: provider mapping the API-key half was simply unreachable — `/provider`
#: advertised that Kimi and xAI report quota and both rendered an empty table
#: forever, because the single fetcher hard-returned None without a token.
#:
#: ``None`` in either slot means "that credential kind has no route here", which
#: is what :func:`usage_kinds` reports so the UI can say WHICH credential is
#: missing instead of showing nothing.
_FETCHERS: dict[str, tuple[FetcherKind | None, FetcherKind | None]] = {
    "openrouter": (None, "openrouter"),
    "anthropic": ("anthropic-oauth", None),
    "openai": ("openai-oauth", None),
    "openai-device": ("openai-oauth", None),
    "kimi": ("kimi-oauth", "moonshot-balance"),
    "deepseek": (None, "deepseek-balance"),
    "xai": ("xai-oauth", None),
    "xai-oauth": ("xai-oauth", None),
}

#: Providers whose usage needs an OAuth access token and has NO API-key route.
#: Derived, so it can never drift from the dispatch table above — the hand-written
#: version had already drifted and was read by nothing.
OAUTH_USAGE_PROVIDERS: frozenset[str] = frozenset(
    provider for provider, (oauth, api_key) in _FETCHERS.items() if oauth and not api_key
)


async def fetch_usage(
    client: httpx.AsyncClient,
    provider: str,
    api_key: str | None = None,
    access_token: str | None = None,
    account_id: str | None = None,
) -> UsageReport | None:
    """Dispatch a provider id to whichever fetcher its credentials can reach.

    ``api_key`` is the resolved API-key cascade value; ``access_token`` is the
    OAuth access token. The OAuth route is preferred when both are present,
    because for every provider that has both it reports the SUBSCRIPTION the user
    is actually spending (plan windows) while the API-key route reports a
    pay-as-you-go balance that a subscription user does not draw down.

    Returns None when the provider has no endpoint, or has one but not for the
    credential kind on hand. Never raises.
    """
    routes = _FETCHERS.get(provider)
    if routes is None:
        return None
    oauth_kind, api_kind = routes
    if access_token and oauth_kind is not None:
        return await _run_fetcher(client, oauth_kind, access_token, account_id)
    if api_key and api_kind is not None:
        return await _run_fetcher(client, api_kind, api_key, account_id)
    return None


async def _run_fetcher(
    client: httpx.AsyncClient,
    kind: FetcherKind,
    secret: str,
    account_id: str | None,
) -> UsageReport | None:
    """One fetcher by kind. Split out so the credential choice above stays legible."""
    if kind == "openrouter":
        return await fetch_openrouter(client, secret)
    if kind == "anthropic-oauth":
        return await fetch_anthropic_oauth(client, secret)
    if kind == "openai-oauth":
        return await fetch_openai_oauth(client, secret, account_id)
    if kind == "kimi-oauth":
        return await fetch_kimi_oauth(client, secret)
    if kind == "moonshot-balance":
        return await fetch_moonshot_balance(client, secret)
    if kind == "deepseek-balance":
        return await fetch_deepseek_balance(client, secret)
    if kind == "xai-oauth":
        return await fetch_xai_oauth(client, secret)
    return None


def usage_supported(provider: str) -> bool:
    """Whether ``provider`` has a live quota endpoint at all (either credential)."""
    return provider in USAGE_PROVIDERS


def usage_kinds(provider: str) -> tuple[bool, bool]:
    """``(has_oauth_route, has_api_key_route)`` for ``provider``.

    Exists so a UI can distinguish "this provider has no quota endpoint" from
    "it has one but not for the credential you hold". Those look identical in an
    empty table, and the second is the one the user can act on.
    """
    oauth_kind, api_kind = _FETCHERS.get(provider, (None, None))
    return oauth_kind is not None, api_kind is not None
