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
from typing import Any

import httpx

#: Human labels for the unit a usage amount is measured in.
UNIT_LABELS = {"usd": "USD", "percent": "%", "tokens": "tokens", "requests": "req", "unknown": ""}

#: OpenRouter's who-am-I endpoint doubles as the credit statement. Returns
#: 200 with ``data`` for a valid key, 401 for an invalid one. It is the canonical
#: who-am-I probe; the credits it returns are parsed here into a report.
OPENROUTER_AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"

#: z.AI / GLM token-plan quota — accepts BOTH api_key and oauth credentials.
ZAI_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"

#: Anthropic OAuth usage (subscription plan). OAuth-only; server unreachable
#: with a raw API key.
ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"

#: OpenAI ChatGPT backend usage windows (OAuth device-code / ChatGPT Plus).
OPENAI_WHAM_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

#: Kimi coding-plan usage (OAuth).
KIMI_USAGE_URL = "https://api.kimi.com/coding/v1/usages"

#: xAI Grok subscription usage (OAuth).
XAI_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing"

#: Providers with a live quota endpoint. Kept as a set so the registry and
#: the TUI can answer "does this provider report usage?" without importing
#: every fetcher.
USAGE_PROVIDERS: frozenset[str] = frozenset(
    {"openrouter", "zai", "anthropic", "openai", "openai-device", "kimi", "xai", "xai-oauth"}
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


async def _post_json(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], body: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    try:
        resp = await client.post(url, headers=headers, json=body, timeout=10.0)
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
    """OpenRouter credit statement from ``/api/v1/auth/key``.

    The response carries the key's own usage so it is the closest thing to a
    "how much have I spent / how much is left" without a separate billing
    API. ``is_free_tier`` (unlimited, rate-limited) is surfaced as a note
    rather than a 0/0 budget.
    """
    payload = await _get_json(client, OPENROUTER_AUTH_KEY_URL, _bearer(api_key))
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


async def fetch_zai(client: httpx.AsyncClient, token: str) -> UsageReport | None:
    """z.AI toll plan quota. NOTE: raw token, no ``Bearer`` prefix (matches
    z.AI's own CLI). Accepts both OAuth access tokens and API keys."""
    payload = await _get_json(client, ZAI_QUOTA_URL, {"Authorization": token})
    if payload is None:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    raw_limits = data.get("limits")
    if not isinstance(raw_limits, list):
        return None
    limits: list[UsageLimit] = []
    for item in raw_limits:
        if not isinstance(item, dict):
            continue
        try:
            used = float(item.get("usage") or item.get("currentValue") or 0.0)
            remaining = float(item.get("remaining") or 0.0)
        except (TypeError, ValueError):
            continue
        limit = float(item.get("number") or 0.0) or (used + remaining)
        pct = item.get("percentage")
        used_fraction: float | None = None
        if pct is not None:
            try:
                used_fraction = float(pct) / 100.0
            except (TypeError, ValueError):
                used_fraction = None
        name = str(item.get("type") or item.get("name") or "quota")
        resets = item.get("nextResetTime")
        limits.append(
            UsageLimit(
                id=f"zai:{name}",
                label=name,
                amount=UsageAmount(
                    used=used,
                    limit=limit or None,
                    remaining=remaining or None,
                    used_fraction=used_fraction,
                    unit="unknown",
                ),
                window="window",
                resets_at=str(resets) if resets else None,
            )
        )
    return UsageReport(provider="zai", limits=limits)


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
    return UsageReport(provider="anthropic", limits=limits)


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
    return UsageReport(provider="openai", limits=limits)


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
    return UsageReport(provider="kimi", limits=limits)


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
    weekly = config.get("creditUsagePercent")
    if weekly is not None:
        limits.append(
            UsageLimit(
                id="xai:credits:1w",
                label="Weekly credits",
                amount=UsageAmount(used_fraction=_num(weekly) / 100.0, unit="percent"),
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


#: Provider id (canonical) -> fetcher. ``*`` selects the OAuth access token
#: path; a bare fetcher takes an API key.
_FETCHERS: dict[str, str] = {
    "openrouter": "openrouter",
    "zai": "zai",
    "anthropic": "anthropic-oauth",
    "openai": "openai-oauth",
    "openai-device": "openai-oauth",
    "kimi": "kimi-oauth",
    "xai": "xai-oauth",
    "xai-oauth": "xai-oauth",
}

#: Providers whose usage requires an OAuth access token (not an API key).
OAUTH_USAGE_PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "openai", "openai-device", "kimi", "xai", "xai-oauth"}
)


async def fetch_usage(
    client: httpx.AsyncClient,
    provider: str,
    api_key: str | None = None,
    access_token: str | None = None,
    account_id: str | None = None,
) -> UsageReport | None:
    """Dispatch a provider id to its fetcher.

    ``api_key`` is the resolved API-key cascade value; ``access_token`` is
    the OAuth access token. OAuth-only usage providers require
    ``access_token`` and return None when absent. Unknown providers return
    None (no live quota endpoint).
    """
    kind = _FETCHERS.get(provider)
    if kind is None:
        return None
    if kind == "openrouter":
        if not api_key:
            return None
        return await fetch_openrouter(client, api_key)
    if kind == "zai":
        token = access_token or api_key
        if not token:
            return None
        return await fetch_zai(client, token)
    if kind == "anthropic-oauth":
        if not access_token:
            return None
        return await fetch_anthropic_oauth(client, access_token)
    if kind == "openai-oauth":
        if not access_token:
            return None
        return await fetch_openai_oauth(client, access_token, account_id)
    if kind == "kimi-oauth":
        if not access_token:
            return None
        return await fetch_kimi_oauth(client, access_token)
    if kind == "xai-oauth":
        if not access_token:
            return None
        return await fetch_xai_oauth(client, access_token)
    return None


def usage_supported(provider: str) -> bool:
    """Whether ``provider`` has a live quota endpoint (OAuth or API key)."""
    return provider in USAGE_PROVIDERS
