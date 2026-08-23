"""Provider usage-quota fetchers, normalized to one ``UsageReport`` shape.

A compact per-provider usage layer with a single normalized output shape.
The design is deliberately small because local-operator is mostly a
per-token billing product: most providers here are pay-per-token API keys
with **no** public balance endpoint, so live quota exists only where the
vendor actually publishes one.

What local-operator DOES fetch, grouped by the credential it needs:

- With the plain API key the registry already stores:

  - ``openrouter`` — `GET /api/v1/key` → USD credit usage/limit plus
    free-tier state. The one every local-operator user has.
  - ``kimi`` — Moonshot `GET /v1/users/me/balance`, on whichever host the
    provider is CONFIGURED for, so the key that reaches the chat API is the
    key that reaches the balance. The region also fixes the currency.
  - ``deepseek`` — `GET /user/balance` → one balance per currency.

- With an OAuth access token, when the user logged in via OAuth:
  ``anthropic`` (`/api/oauth/usage`), ``openai``/``openai-device`` (ChatGPT
  backend `/wham/usage`), ``kimi`` (`/coding/v1/usages`), ``xai``/
  ``xai-oauth`` (billing).

``kimi`` is the only provider with both, and its two routes are different
endpoints on different hosts rather than two ways of authenticating one.

Providers deliberately absent, because the credential they need is one
local-operator does not hold: ``google`` (the Gemini quota endpoint is part of
the Gemini CLI's Cloud Code OAuth; local-operator's Google provider is an AI
Studio API key, which that endpoint rejects) and ``alibaba`` (a browser console
session cookie, not the DashScope key). Both would need a new login flow before
a fetcher could reach anything, and advertising a provider whose table can only
ever be empty is worse than being incomplete.

**Vendors report utilization, not spend.** Every subscription endpoint here
quotes a PERCENTAGE — ``utilization`` (Anthropic), ``used_percent``
(OpenAI/Codex), ``creditUsagePercent`` (xAI) — nested inside a per-window
object, and the numeric ones (Kimi) quote strings. Four of the five fetchers
were originally written against invented ``{"used": N, "limit": N}`` shapes that
no vendor sends, which parsed to nothing and returned ``None``: a live 200 with
full data rendered "no usage data", indistinguishable from having no endpoint at
all. The parsers below are pinned to captured live payloads for that reason, and
their tests carry the real bodies rather than hand-written ones.

Every fetcher returns the SAME ``UsageReport`` (or ``None`` when the
provider/credential cannot report) so the TUI renders one shape and is
never coupled to a provider's response schema. A missing endpoint is not an
error — it is a ``None`` report that the caller leaves off the table.

Anything that hits the network uses a caller-provided ``httpx.AsyncClient``
so the TUI can share one connection pool and cancel fetches cleanly; no
code path here owns a client.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from local_operator.providers.registry import get_provider_definition

#: Human labels for the unit a usage amount is measured in. The renderer reads
#: this rather than interpolating the raw key, so an amount prints ``519.86 USD``
#: and ``30%`` instead of ``519.86 usd`` and ``30 percent``.
#:
#: This is the RENDERER's vocabulary, not the vendor's: a currency with no entry
#: here (a CNY balance) is reported ``unknown`` with the currency in the limit's
#: label, which prints a bare number rather than one mislabelled as dollars.
UNIT_LABELS = {
    "usd": "USD",
    "percent": "%",
    "tokens": "tokens",
    "requests": "req",
    "unknown": "",
}

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
#: already stores. Appended to the provider's own ``base_url`` rather than
#: hardcoded: Moonshot runs TWO platforms — mainland ``api.moonshot.cn`` and
#: international ``api.moonshot.ai`` — with separate accounts and separate keys.
#: The registry configures ``kimi`` for ``https://api.moonshot.cn/v1``, so a
#: hardcoded ``.ai`` host 401s on the only key a user can hold, the fetch returns
#: None and the table is empty forever — the exact failure this fetcher exists to
#: fix.
MOONSHOT_BALANCE_PATH = "/users/me/balance"

#: Used only if the registry ever loses its ``kimi`` definition; the mainland host
#: is what the registry, the validation descriptor and the model registry all use.
MOONSHOT_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

#: DeepSeek account balance — plain Bearer with the key already in the registry.
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

#: xAI Grok subscription usage (OAuth).
XAI_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing"

#: QwenCloud Token Plan consumption through the first-party management
#: gateway — the same flat BSS surface the official ``qwencloud-cli`` posts
#: to. The management token comes from the device-flow login
#: (``/login alibaba-token-plan-oauth``); the ``sk-sp-…`` inference key
#: cannot read quota. Verified against production: HTTP 200 with
#: ``code == "200"`` on success and ``ConsoleNeedLogin`` for a dead token.
QWENCLOUD_BSS_URL = "https://cli.qwencloud.com/data/v2/api.json"
QWENCLOUD_BSS_PRODUCT = "BssOpenAPI-V3"
QWENCLOUD_REGION = "ap-southeast-1"
#: Token Plan commodity codes on the international product (qwencloud-cli
#: ``site.ts``): personal = 7-day credit window, teams = monthly seat
#: credits, addon = Credit Packs outside every window. QwenCloud documents
#: NO 5-hour Token Plan window, so none is synthesized.
QWENCLOUD_TOKEN_PLAN_COMMODITIES = {
    "teams": "sfm_tokenplanteams_dp_intl",
    "personal": "sfm_tokenplanpersonal_dp_intl",
    "addon": "sfm_tokenplanteamsaddon_dp_intl",
}

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
    "qwencloud-token-plan",
    "zai-quota",
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
#:
#: ``zai`` was previously absent for exactly this reason: it had a working
#: fetcher and a passing test, but no ``ProviderDefinition`` — so `/login zai`
#: raised, its env var was never read, and no code path could insert the
#: credential row the fetcher needed. It is present now because the registry
#: entry, the credential path and this fetcher landed together; the pairing is
#: the point, and re-adding one without the others would recreate the defect.
_FETCHERS: dict[str, tuple[FetcherKind | None, FetcherKind | None]] = {
    "openrouter": (None, "openrouter"),
    "anthropic": ("anthropic-oauth", None),
    "openai": ("openai-oauth", None),
    "openai-device": ("openai-oauth", None),
    "kimi": ("kimi-oauth", "moonshot-balance"),
    "deepseek": (None, "deepseek-balance"),
    "xai": ("xai-oauth", None),
    "xai-oauth": ("xai-oauth", None),
    "alibaba-token-plan": ("qwencloud-token-plan", None),
    "alibaba-token-plan-oauth": ("qwencloud-token-plan", None),
    # Both slots run the SAME fetcher, because both credential kinds are the
    # same secret: the `zai-oauth` browser sign-in ends by minting an ordinary
    # durable `id.secret` coding-plan key and stores it in `access`, which is
    # exactly what the quota endpoint authenticates with. Leaving the OAuth slot
    # empty made `/provider` advertise Z.AI usage and then report nothing at all
    # for anyone who signed in rather than pasting a key.
    "zai": ("zai-quota", "zai-quota"),
    "zai-oauth": ("zai-quota", "zai-quota"),
}

#: Providers with a live quota endpoint, for callers that only need the question
#: "does this provider report usage?" answered without importing every fetcher.
#:
#: DERIVED from the dispatch table, never hand-listed. This is the set that gates
#: the UI, so a hand-written duplicate that drifted one way would silently unreach
#: a provider with a working fetcher, and the other way would advertise a provider
#: no fetcher serves.
USAGE_PROVIDERS: frozenset[str] = frozenset(_FETCHERS)


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
    #: Epoch ms the window rolls over, parsed from ``resets_at`` when the vendor
    #: sends a timestamp we understand. Kept ALONGSIDE the raw string rather than
    #: replacing it: the renderer wants "resets in 3h 40m", which needs a number,
    #: while a vendor string we could not parse is still worth passing through
    #: rather than dropping.
    resets_at_ms: int | None = None
    #: The model family a per-model cap belongs to (Anthropic's scoped weekly
    #: rows: ``opus``, ``sonnet``, ``fable``). Empty for account-wide windows.
    #: The renderer groups on this so a tier cap is never mistaken for the
    #: umbrella limit that actually gates every request.
    tier: str = ""
    #: True for the account-wide umbrella windows. A tier row hitting 100% stops
    #: one model family; a shared row hitting 100% stops the account, and the two
    #: must not look alike in a list where the user is deciding whether to switch
    #: models or stop working.
    shared: bool = False

    def effective_status(self) -> str:
        return self.status or self.amount.status()

    def resets_in_ms(self, now_ms: float | None = None) -> int | None:
        """Milliseconds until the window rolls over, or None if unknown/past."""
        if self.resets_at_ms is None:
            return None
        now = time.time() * 1000 if now_ms is None else now_ms
        remaining = self.resets_at_ms - now
        return int(remaining) if remaining > 0 else None


@dataclass
class UsageReport:
    """The normalized quota state for one provider at one instant."""

    provider: str
    fetched_at: int = field(default_factory=lambda: int(time.time() * 1000))
    limits: list[UsageLimit] = field(default_factory=list)
    notes: str | None = None
    #: Pass-through identity so the TUI can annotate which account reported.
    identity: str | None = None


@dataclass(frozen=True)
class QuotaHealth:
    """Whether a provider account can accept another message."""

    state: Literal["healthy", "reserve", "depleted", "unknown"]
    remaining_fraction: float | None = None
    reset_after_ms: int | None = None
    scope: Literal["account", "model", "unknown"] = "unknown"
    #: Labels of the windows at or under the reserve threshold (the ones that
    #: set ``remaining_fraction`` and ``reset_after_ms``). The preflight uses
    #: these to tell "everything is spent" from "one secondary window is spent"
    #: without re-deriving the binding set from the raw report.
    binding_labels: tuple[str, ...] = ()
    #: Family slugs of the SCOPED windows in the binding set (Anthropic's
    #: ``7 day (Fable)`` yields ``("fable",)``). This is the dimension a
    #: credential block must be scoped to: a verdict whose only binding
    #: windows are family caps stops THAT family on the account, not every
    #: model the account serves — the shared 5-hour / 7-day windows still
    #: gate the rest. Empty when the binding set is account-wide or the
    #: report carries no family-scoped rows.
    binding_families: tuple[str, ...] = ()


def usage_health(
    report: UsageReport,
    model_id: str,
    *,
    reserve_percent: float = 10.0,
    now_ms: float | None = None,
) -> QuotaHealth:
    """Reduce a normalized report to the limits that gate ``model_id``.

    Shared windows always apply; tier windows apply only when their tier name
    appears in the model id. Anthropic's enabled extra-usage meter supersedes
    the included-plan windows because paid headroom keeps requests runnable
    after the plan reaches 100%.
    """
    lowered_model = model_id.lower()
    extra = [
        limit
        for limit in report.limits
        if limit.id.endswith(":extra") and limit.amount.fraction() is not None
    ]
    if extra:
        relevant = extra
    else:
        relevant = [
            limit
            for limit in report.limits
            if limit.shared
            or (limit.tier and limit.tier.lower() in lowered_model)
            or (not limit.shared and not limit.tier)
        ]
    measured: list[tuple[UsageLimit, float]] = []
    for limit in relevant:
        fraction = limit.amount.fraction()
        if fraction is not None:
            measured.append((limit, max(0.0, min(1.0, 1.0 - fraction))))
    if not measured:
        # Balance-only endpoints (Moonshot/Kimi and DeepSeek) expose an
        # absolute ``remaining`` amount but no total, so a positive balance
        # cannot honestly be converted to a percentage. Exact zero is
        # different: it is definitive exhaustion regardless of the missing
        # denominator. Recognize only that boundary; positive amounts remain
        # unknown/fail-open rather than inventing a quota percentage (review
        # F3).
        balances = [limit for limit in relevant if limit.amount.remaining is not None]
        # DeepSeek returns one row per currency. One empty wallet does not
        # exhaust an account that still has funds in another, so exact-zero is
        # definitive only when EVERY denominator-less balance row is non-
        # positive. A mixed zero/positive report remains unknown rather than
        # being skipped (review F6).
        if not balances or any(
            limit.amount.remaining is not None and limit.amount.remaining > 0 for limit in balances
        ):
            return QuotaHealth("unknown")
        zero_balance = balances
        # This branch is unconditionally a depleted verdict (every balance is
        # zeroed), and a depleted account is usable again the moment ANY spent
        # window reopens -- a request needs only one window with headroom -- so
        # the honest "try again at" is the SOONEST reset among the zeroed
        # windows, not the latest. A two-zeroed-window DeepSeek/Kimi account
        # reopens at the earliest of its resets; keying to the latest would pin
        # a days-long block to the slowest window. The cap in block_credential
        # is the backstop; this is the estimate.
        reset_after = min(
            (
                value
                for value in (limit.resets_in_ms(now_ms) for limit in zero_balance)
                if value is not None
            ),
            default=None,
        )
        scope: Literal["account", "model", "unknown"]
        if all(limit.tier and not limit.shared for limit in zero_balance):
            scope = "model"
        else:
            scope = "account"
        return QuotaHealth(
            "depleted",
            remaining_fraction=0.0,
            reset_after_ms=reset_after,
            scope=scope,
            binding_labels=tuple(limit.label for limit in zero_balance),
            binding_families=tuple(
                dict.fromkeys(limit.tier for limit in zero_balance if limit.tier)
            ),
        )

    remaining = min(value for _limit, value in measured)
    threshold = min(100.0, max(0.0, float(reserve_percent))) / 100.0
    state: Literal["healthy", "reserve", "depleted", "unknown"]
    if remaining <= 0:
        state = "depleted"
    elif remaining <= threshold:
        state = "reserve"
    else:
        state = "healthy"
    binding = [limit for limit, value in measured if value <= threshold]
    # ``reset_after_ms`` answers "when does the condition the STATE names
    # clear", and the horizon must come from the windows that produced that
    # state. A depleted verdict feeds a block that lasts this long, so it may
    # only count fully-spent windows: mixing in a window that is merely in
    # reserve stretched a three-hour 5h-window block out to that window's
    # seven-day reset — a days-long outage written against an account that
    # was usable again the same afternoon.
    if state == "depleted":
        horizon = [limit for limit, value in measured if value <= 0]
    else:
        horizon = binding
    horizon_resets = [limit.resets_in_ms(now_ms) for limit in horizon]
    present = [value for value in horizon_resets if value is not None]
    if state == "depleted":
        # The account is usable again the moment ANY spent window reopens -- a
        # request needs only one window with headroom -- so the honest "try
        # again at" is the SOONEST reset among fully-spent windows, not the
        # latest. Keying to the latest (the old max) pinned a block to the
        # 7-day window when the 5-hour window reopened first. The cap in
        # block_credential is the backstop; this is the estimate. The reserve
        # (else) branch keeps max: for a low-but-usable account the display
        # wants "when does the binding window recover", i.e. the latest.
        reset_after = min(present, default=None)
    else:
        reset_after = max(present, default=None)
    if not binding:
        scope: Literal["account", "model", "unknown"] = "unknown"
    elif all(limit.tier and not limit.shared for limit in binding):
        scope = "model"
    else:
        scope = "account"
    return QuotaHealth(
        state,
        remaining_fraction=remaining,
        reset_after_ms=reset_after,
        scope=scope,
        binding_labels=tuple(limit.label for limit in binding),
        binding_families=tuple(dict.fromkeys(limit.tier for limit in binding if limit.tier)),
    )


def shared_tier_saturation(
    report: UsageReport,
    *,
    reserve_percent: float = 10.0,
) -> tuple[float | None, bool]:
    """How full the shared windows are, and whether they are the binding limit.

    ``usage_health`` reduces a report to the single tightest window, so a
    scoped cap that belongs to a model tier the user never runs (Anthropic's
    ``7 day (Fable)`` against a ``claude-opus`` request) can dominate the
    verdict and hide real headroom in the shared 5-hour / 7-day windows. This
    re-reads the report the other way: the min remaining fraction across the
    SHARED windows alone, and whether at least one scoped (per-tier) window is
    at or under the reserve threshold. "Account exhausted" is only honest when
    the shared windows are binding too; a report whose tight window is scoped
    while the shared windows still hold quota is a tier cap, not an empty
    account, and rotating or failing over on it strands usable reserve.

    The shared fraction is ``None`` when no shared window carries a numeric
    amount — an INDETERMINATE answer, not "full headroom", so a caller cannot
    mistake an unparseable report for a licence to spend.
    """
    threshold = min(100.0, max(0.0, float(reserve_percent))) / 100.0
    shared: list[float] = []
    tier_binding = False
    for limit in report.limits:
        if limit.id.endswith(":extra"):
            continue
        fraction = limit.amount.fraction()
        if fraction is None:
            continue
        remaining = max(0.0, min(1.0, 1.0 - fraction))
        if limit.shared or not limit.tier:
            shared.append(remaining)
        elif remaining <= threshold:
            tier_binding = True
    return (min(shared) if shared else None), tier_binding


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


def moonshot_balance_target(base_url: str | None = None) -> tuple[str, str, str]:
    """``(balance_url, unit, currency_symbol)`` for a Moonshot ``base_url``.

    Moonshot's two platforms are separate products, not two hostnames for one:
    ``api.moonshot.cn`` (mainland) and ``api.moonshot.ai`` (international) have
    separate accounts, separate keys and separate currencies. The balance response
    carries NO currency field, so the HOST is the currency — a mainland balance
    rendered as dollars is off by roughly a factor of seven.

    ``base_url`` defaults to whatever the registry configures for ``kimi``, which
    is the same value the chat client and the key validator use. That is the whole
    point: the endpoint follows the key the user must actually hold.
    """
    if base_url is None:
        definition = get_provider_definition("kimi")
        base_url = (definition.base_url if definition else None) or MOONSHOT_DEFAULT_BASE_URL
    base = base_url.rstrip("/")
    host = (urlsplit(base).hostname or "").lower()
    if host.endswith(".cn"):
        # No UNIT_LABELS entry for CNY, so the currency rides in the label and the
        # amount prints bare rather than wearing a dollar sign it did not earn.
        return base + MOONSHOT_BALANCE_PATH, "unknown", "¥"
    return base + MOONSHOT_BALANCE_PATH, "usd", "$"


async def fetch_moonshot_balance(
    client: httpx.AsyncClient, api_key: str, base_url: str | None = None
) -> UsageReport | None:
    """Moonshot/Kimi account balance from the PLAIN API key.

    The counterpart to :func:`fetch_kimi_oauth`, and the one most Kimi users can
    actually reach: the registry stores ``KIMI_API_KEY`` and the coding-plan
    endpoint only accepts an OAuth token, so an API-key user had a provider that
    claimed to report quota and never did.

    ``available_balance`` is the actionable number — it already nets the voucher
    and cash components the response also breaks out — so it is reported as
    ``remaining`` with no limit. There is no spend figure to pair it with, and
    inventing ``used = limit - remaining`` from a limit we were never given would
    draw a progress bar out of an assumption.
    """
    url, unit, symbol = moonshot_balance_target(base_url)
    currency = UNIT_LABELS.get(unit) or "CNY"
    payload = await _get_json(client, url, _bearer(api_key))
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
        notes = f"voucher {symbol}{voucher:.2f} + cash {symbol}{cash:.2f}"
    return UsageReport(
        provider="kimi",
        limits=[
            UsageLimit(
                id="kimi:balance",
                label=f"Balance ({currency})",
                amount=UsageAmount(remaining=available, unit=unit),
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


#: Z.AI's coding-plan quota endpoint. Lives on the API host but OUTSIDE the
#: `/api/coding/paas/v4` inference prefix, so it is spelled absolutely here
#: rather than derived from the provider's ``base_url``.
ZAI_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"


def _is_zai_chat_model(code: str) -> bool:
    """Whether a ``usageDetails`` code names a GLM chat model.

    A SHAPE test rather than a lookup against the shipped catalogue, and the
    distinction is the whole point: every id in ``glm_models`` already starts
    with ``glm``, so a registry check would answer nothing the prefix does not
    while dragging ``model.registry`` (~300 ms cold) onto a path that runs
    inside a quota parse. What the prefix buys is the ids that do NOT exist
    yet — a model launched after this release must not be mistaken for a
    feature code.

    Case-folded because the codes are vendor strings, not identifiers we mint,
    and ``GLM-6`` naming a model while ``glm-6`` does not would be a
    classification that turns on capitalisation.

    The known feature codes (``search-prime``, ``web-reader``, ``zread``) share
    no prefix with the model family, so the test is unambiguous today; it is
    the tie-break for unknown codes that matters, and it resolves them to
    "feature", which fails toward leaving the account cap alone.
    """
    return code.strip().lower().startswith("glm")


#: The `unit` enum Z.AI uses to describe a quota window's period. Only the
#: values observed on live coding-plan accounts are mapped; anything else falls
#: through to a generic label rather than being guessed at, because mislabelling
#: a monthly cap as hourly would make an exhausted plan look like it resets soon.
#:
#: Only the NAME is carried. An earlier revision paired each unit with a
#: duration in seconds, which nothing read: the reset time arrives as an
#: absolute ``nextResetTime`` from the vendor, so a locally derived window
#: length would be a second, unreconciled answer to a question already
#: answered — and the one that drifts if a window is redefined.
_ZAI_WINDOW_UNITS: dict[int, str] = {
    3: "hour",
    4: "day",
    5: "month",
    6: "week",
}


def _zai_window(item: dict[str, Any]) -> tuple[str, int | None]:
    """``(window_label, resets_at_ms)`` for one Z.AI limit row.

    ``unit`` names the period and ``number`` how many of them, so a 5-hour
    window arrives as ``unit=3, number=5``. ``nextResetTime`` is epoch
    milliseconds on live payloads, but is accepted in seconds too and
    disambiguated by magnitude, matching how the other fetchers here treat
    ambiguous vendor timestamps.
    """
    # `_num` rather than a bare cast on EVERY field, including the ones read
    # only to build a label: a quota fetcher must never raise (the caller drops
    # a None report, but an exception escapes to the UI), and `int()` is the
    # trap — `json.loads` accepts bare NaN/Infinity and an oversized integer,
    # all of which blow up on conversion rather than returning something odd.
    # `_num` rejects both the unconvertible and the non-finite, so anything that
    # reaches an `int()` below is already known-safe. Strings are coerced too,
    # since a vendor that starts quoting `"unit": "3"` would otherwise silently
    # lose the window name while still reporting the numbers.
    count = _num(item.get("number")) or 1
    unit = _num(item.get("unit"))
    mapped = _ZAI_WINDOW_UNITS.get(int(unit)) if unit is not None else None
    label = "quota" if mapped is None else f"{int(count)} {mapped}"
    reset = _num(item.get("nextResetTime"))
    resets_ms: int | None = None
    if reset is not None and reset > 0:
        resets_ms = int(reset if reset > 1_000_000_000_000 else reset * 1000)
    return label, resets_ms


async def fetch_zai_quota(client: httpx.AsyncClient, api_key: str) -> UsageReport | None:
    """Z.AI GLM Coding Plan quota from the monitor endpoint.

    Two row types share one payload and mean different things:

    - ``TOKENS_LIMIT`` is the token allowance for a rolling window. On live
      coding-plan accounts these rows carry ONLY ``percentage`` — no absolute
      used/limit/remaining — so the report is built from the fraction and the
      unit is ``percent``. Inventing token counts from a percentage would put a
      fabricated number in front of the user.
    - ``TIME_LIMIT`` is a REQUEST allowance despite the name. Its ``usage`` field
      is the LIMIT and ``currentValue`` is the amount consumed — inverted
      relative to every other vendor here, which is the trap worth naming.

    A ``TIME_LIMIT`` row whose ``usageDetails`` name NO chat model is the
    separate Zread feature bucket rather than the account-wide request cap, so
    it is tagged as a tier and NOT marked shared: exhausting it stops those
    tools, not the plan. The test is spelled in the negative on purpose — see
    the note at the classification site.
    """
    payload = await _get_json(client, ZAI_QUOTA_URL, _bearer(api_key))
    if payload is None:
        return None
    # The envelope reports business-level failure with a 200, so an unsuccessful
    # body is not a usable report even though the transport succeeded.
    if payload.get("success") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    rows = data.get("limits")
    if not isinstance(rows, list):
        return None

    limits: list[UsageLimit] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        row_type = item.get("type")
        window, resets_ms = _zai_window(item)
        percentage = _num(item.get("percentage"))
        used = _num(item.get("currentValue"))
        limit_value = _num(item.get("usage"))
        remaining = _num(item.get("remaining"))

        if row_type == "TOKENS_LIMIT":
            if percentage is None and used is None:
                continue
            amount = UsageAmount(
                used=used,
                limit=limit_value,
                remaining=remaining,
                used_fraction=(
                    min(max(percentage / 100.0, 0.0), 1.0) if percentage is not None else None
                ),
                # Absolute token counts are absent on coding-plan rows, so the
                # renderer is told to speak in percent unless the vendor gave
                # real numbers to show.
                unit="tokens" if limit_value else "percent",
            )
            limits.append(
                UsageLimit(
                    id=f"zai:tokens:{window.replace(' ', '')}",
                    label=f"Token quota ({window})",
                    amount=amount,
                    window=window,
                    resets_at_ms=resets_ms,
                    shared=True,
                )
            )
        elif row_type == "TIME_LIMIT":
            details = item.get("usageDetails")
            codes = (
                {
                    str(d.get("modelCode"))
                    for d in details
                    if isinstance(d, dict) and d.get("modelCode")
                }
                if isinstance(details, list)
                else set()
            )
            # A row is the feature bucket when its breakdown lists NO chat
            # model — the question is asked in the negative deliberately.
            #
            # Every positive formulation is a maintenance dependency on a vendor
            # enum, and each fails the moment that enum moves. Requiring all
            # known feature codes breaks when Z.AI renames one; requiring the
            # listed codes to be a SUBSET of the known ones breaks on a rename
            # AND on an addition (a new tool code is not in our set, so the row
            # stops looking like a feature bucket). Both fail in the direction
            # that stops work: the row is reclassified as the account-wide cap
            # with ``shared=True``, `usage_health` applies it to EVERY model,
            # and a user with most of their token quota intact is told the
            # account is depleted because the vendor shipped a new tool.
            # Matching ANY known code fails the other way instead, demoting a
            # genuine account cap that merely mentions a feature into a tier.
            #
            # Chat model ids are the stable property: the account-wide request
            # cap is denominated in them, and a feature bucket never lists one.
            # So an unrecognised code is read as a new FEATURE rather than a new
            # model, which keeps a renamed or added tool inside the tier and
            # leaves the account cap alone. An empty breakdown is not a feature
            # bucket.
            is_feature = bool(codes) and not any(_is_zai_chat_model(code) for code in codes)
            if percentage is None and used is None:
                continue
            amount = UsageAmount(
                used=used,
                limit=limit_value,
                remaining=remaining,
                used_fraction=(
                    min(max(percentage / 100.0, 0.0), 1.0) if percentage is not None else None
                ),
                unit="requests" if limit_value else "percent",
            )
            limits.append(
                UsageLimit(
                    id=(
                        f"zai:features:zread:{window.replace(' ', '')}"
                        if is_feature
                        else f"zai:requests:{window.replace(' ', '')}"
                    ),
                    # "Zread quota", not "Zread feature quota": the usage
                    # panel caps its label column at 24 cells (a third of the
                    # panel's own 76-cell maximum), so the longer form truncated
                    # mid-parenthesis at EVERY terminal width and the window was
                    # unreachable — the only row on the panel that could not say
                    # which period it resets over. The word "feature" is the
                    # redundant one: the tier indent and the dimmed label ramp
                    # already say this is not the account-wide cap.
                    #
                    # The bucket covers the non-chat tools together (today:
                    # search-prime, web-reader, zread), so the label names one
                    # member of a set. Kept because "Zread" is the name Z.AI's
                    # own plan page gives this allowance, so it is the word a
                    # user will recognise from their dashboard.
                    label=("Zread quota" if is_feature else "Request quota") + f" ({window})",
                    amount=amount,
                    window=window,
                    resets_at_ms=resets_ms,
                    tier="zread" if is_feature else "",
                    shared=not is_feature,
                )
            )

    # Two rows of the same type and window collide on a generated id, and a
    # duplicate id renders as two identical panel rows that disagree about
    # nothing — the same defensive dedupe `fetch_xai_oauth` applies, for the
    # same reason: the id is derived from vendor fields, so uniqueness is the
    # vendor's promise rather than ours. First occurrence wins.
    seen: set[str] = set()
    limits = [limit for limit in limits if not (limit.id in seen or seen.add(limit.id))]
    if not limits:
        return None
    level = data.get("level")
    notes = f"coding plan: {level}" if isinstance(level, str) and level else None
    return UsageReport(provider="zai", limits=limits, notes=notes)


def _parse_iso_ms(value: Any) -> int | None:
    """An ISO-8601 timestamp as epoch ms, or None when it is not one.

    Anthropic sends ``2026-08-12T03:59:59.196163+00:00``. ``fromisoformat``
    handles that on 3.11+; the ``Z`` suffix other vendors use is rewritten
    because 3.11's parser rejects it outright rather than treating it as UTC.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _anthropic_bucket(value: Any) -> tuple[float, int | None] | None:
    """``(utilization_percent, resets_at_ms)`` from one Anthropic usage bucket.

    The bucket is ``{"utilization": 2.0, "resets_at": "..."}``. It is ``None``
    for every window the account does not have, which is most of them — the
    payload lists a dozen internal codenames (``tangelo``, ``nimbus_quill``,
    ``omelette_promotional``) alongside the two real ones, so "absent" has to be
    ordinary rather than exceptional.

    A bucket with no ``utilization`` is dropped rather than reported as zero:
    Anthropic sends ``resets_at``-only buckets, and rendering those as an empty
    bar says "you have used nothing" when the truth is "nothing was said".
    """
    if not isinstance(value, dict):
        return None
    utilization = _num(value.get("utilization"))
    if utilization is None:
        return None
    return utilization, _parse_iso_ms(value.get("resets_at"))


def _anthropic_limit(
    limit_id: str,
    label: str,
    bucket: tuple[float, int | None] | None,
    *,
    window: str,
    tier: str = "",
    shared: bool = False,
) -> UsageLimit | None:
    """One Anthropic window as a ``UsageLimit``, or None when not reported.

    Utilization arrives as a PERCENT (``100.0`` means the cap is reached), so it
    is carried as ``used`` against a limit of 100 with the fraction stated
    explicitly. Deriving the fraction from used/limit alone would work here, but
    stating it removes any doubt about whether ``2.0`` meant 2% or 200%.
    """
    if bucket is None:
        return None
    utilization, resets_ms = bucket
    clamped = max(0.0, min(100.0, utilization))
    return UsageLimit(
        id=limit_id,
        label=label,
        amount=UsageAmount(
            used=clamped,
            limit=100.0,
            remaining=100.0 - clamped,
            used_fraction=clamped / 100.0,
            unit="percent",
        ),
        window=window,
        resets_at_ms=resets_ms,
        tier=tier,
        shared=shared,
    )


def _anthropic_scoped_weekly(raw_limits: Any) -> list[UsageLimit]:
    """Per-model weekly caps from the generic ``limits[]`` array.

    As of mid-2026 Anthropic returns ``null`` for the legacy ``seven_day_opus`` /
    ``seven_day_sonnet`` buckets and publishes model-scoped caps only here, as
    ``kind: "weekly_scoped"`` entries naming the family in
    ``scope.model.display_name``. Reading only the legacy keys is why every
    per-model row vanished.

    ``is_active`` is deliberately ignored. Live payloads mark only the currently
    BINDING limit active — this account's 100% weekly row is active while its 2%
    session row is not — so filtering on it would hide every window except the
    one already blocking, which is the opposite of what a usage view is for.
    """
    limits: list[UsageLimit] = []
    seen: set[str] = set()
    if not isinstance(raw_limits, list):
        return limits
    for item in raw_limits:
        if not isinstance(item, dict) or item.get("kind") != "weekly_scoped":
            continue
        scope = item.get("scope")
        model = scope.get("model") if isinstance(scope, dict) else None
        display = model.get("display_name") if isinstance(model, dict) else None
        if not isinstance(display, str) or not display.strip():
            continue
        name = display.strip()
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if not slug or slug in seen:
            continue
        seen.add(slug)
        percent = _num(item.get("percent"))
        if percent is None:
            continue
        limit = _anthropic_limit(
            f"anthropic:7d:{slug}",
            f"7 day ({name})",
            (percent, _parse_iso_ms(item.get("resets_at"))),
            window="7 day",
            tier=slug,
        )
        if limit is not None:
            limits.append(limit)
    return limits


def _anthropic_money(value: Any) -> float | None:
    """A ``{amount_minor, exponent, currency}`` object as whole units of USD.

    Minor units with an explicit exponent, so ``20000`` at exponent 2 is
    ``$200.00``. A non-USD currency returns None rather than a number wearing the
    wrong symbol.
    """
    if not isinstance(value, dict):
        return None
    minor = value.get("amount_minor")
    exponent = value.get("exponent")
    if not isinstance(minor, int) or not isinstance(exponent, int):
        return None
    if minor < 0 or exponent < 0:
        return None
    currency = value.get("currency")
    if currency is not None and str(currency).upper() != "USD":
        return None
    return minor / (10**exponent)


def _anthropic_extra_usage(payload: dict[str, Any]) -> UsageLimit | None:
    """The pay-as-you-go credit meter that tops up an exhausted plan.

    Two shapes exist. The newer ``spend`` object uses minor units; the older
    ``extra_usage`` uses ``used_credits`` with ``decimal_places``. ``spend`` wins
    when present because it is what current accounts get.

    Reported ONLY when the account has extra usage enabled. A disabled meter is
    a $0.00/$200.00 row that reads as spare headroom the account cannot actually
    draw on — this account has exactly that, ``enabled: false`` with a
    ``disabled_reason`` — so it becomes a note instead of a bar.
    """
    spend = payload.get("spend")
    if isinstance(spend, dict):
        if spend.get("enabled") is not True:
            return None
        used = _anthropic_money(spend.get("used"))
        if used is None:
            return None
        cap = _anthropic_money(spend.get("limit"))
        return UsageLimit(
            id="anthropic:extra",
            label="Extra usage",
            amount=UsageAmount(
                used=used,
                limit=cap if cap and cap > 0 else None,
                remaining=(cap - used) if cap and cap > 0 else None,
                unit="usd",
            ),
            window="1 month",
        )
    extra = payload.get("extra_usage")
    if not isinstance(extra, dict) or extra.get("is_enabled") is not True:
        return None
    places = extra.get("decimal_places")
    used = _num(extra.get("used_credits"))
    if used is None:
        return None
    monthly = _num(extra.get("monthly_limit"))
    divisor = 10 ** (places if isinstance(places, int) and places >= 0 else 2)
    cap = monthly / divisor if monthly and monthly > 0 else None
    return UsageLimit(
        id="anthropic:extra",
        label="Extra usage",
        amount=UsageAmount(
            used=used,
            limit=cap,
            remaining=(cap - used) if cap else None,
            unit="usd",
        ),
        window="1 month",
    )


def _anthropic_notes(payload: dict[str, Any]) -> str | None:
    """Why the extra-usage meter is off, when it is off and would have shown.

    An account sitting at 100% weekly with no visible credit meter looks like a
    dead end; naming the reason ("extra usage disabled — out of credits") is the
    difference between a user waiting for a reset they cannot avoid and one
    buying credits they did not know existed.
    """
    for key in ("spend", "extra_usage"):
        section = payload.get(key)
        if not isinstance(section, dict):
            continue
        enabled = section.get("enabled") if key == "spend" else section.get("is_enabled")
        if enabled is True or enabled is None:
            continue
        reason = section.get("disabled_reason")
        if isinstance(reason, str) and reason.strip():
            return f"extra usage disabled — {reason.strip().replace('_', ' ')}"
        return "extra usage disabled"
    return None


async def fetch_anthropic_oauth(client: httpx.AsyncClient, access_token: str) -> UsageReport | None:
    """Anthropic subscription usage (5h / 7d windows, model-scoped tiers).

    The response reports each window as a ``utilization`` PERCENT inside a
    ``{"utilization": ..., "resets_at": ...}`` bucket. This fetcher previously
    looked for ``used``/``limit`` keys that the endpoint has never sent, so every
    bucket was skipped, ``limits`` came back empty, and the report was dropped —
    ``/usage`` said "no usage data" for an account whose weekly window was at
    100%. Verified against the live endpoint: a 200 with full data rendered
    nothing.

    The generic ``limits[]`` array is read for the model-scoped weekly caps only.
    Its ``session`` and ``weekly_all`` entries duplicate ``five_hour`` and
    ``seven_day``, so they are a FALLBACK for when the named buckets are absent
    rather than extra rows — reading both unconditionally listed every window
    twice.
    """
    payload = await _get_json(client, ANTHROPIC_USAGE_URL, _bearer(access_token))
    if payload is None:
        return None

    raw_limits = payload.get("limits")
    by_kind: dict[str, tuple[float, int | None]] = {}
    if isinstance(raw_limits, list):
        for item in raw_limits:
            if not isinstance(item, dict):
                continue
            kind = item.get("kind")
            percent = _num(item.get("percent"))
            if not isinstance(kind, str) or kind in by_kind or percent is None:
                continue
            by_kind[kind] = (percent, _parse_iso_ms(item.get("resets_at")))

    five_hour = _anthropic_bucket(payload.get("five_hour")) or by_kind.get("session")
    seven_day = _anthropic_bucket(payload.get("seven_day")) or by_kind.get("weekly_all")

    candidates = [
        _anthropic_limit("anthropic:5h", "5 hour", five_hour, window="5 hour", shared=True),
        _anthropic_limit("anthropic:7d", "7 day", seven_day, window="7 day", shared=True),
        _anthropic_limit(
            "anthropic:7d:opus",
            "7 day (Opus)",
            _anthropic_bucket(payload.get("seven_day_opus")),
            window="7 day",
            tier="opus",
        ),
        _anthropic_limit(
            "anthropic:7d:sonnet",
            "7 day (Sonnet)",
            _anthropic_bucket(payload.get("seven_day_sonnet")),
            window="7 day",
            tier="sonnet",
        ),
    ]
    limits = [limit for limit in candidates if limit is not None]
    seen_tiers = {limit.tier for limit in limits if limit.tier}
    limits.extend(
        limit for limit in _anthropic_scoped_weekly(raw_limits) if limit.tier not in seen_tiers
    )
    extra = _anthropic_extra_usage(payload)
    if extra is not None:
        limits.append(extra)

    if not limits:
        return None
    return UsageReport(
        provider="anthropic",
        limits=limits,
        notes=_anthropic_notes(payload),
    )


def _window_label(seconds: float | None, fallback: str) -> str:
    """``5 hour`` / ``7 day`` from a window length in seconds.

    The endpoint quotes the window as a DURATION (``17940`` seconds is the five
    hour window, a minute short) rather than as a name, so the label is rounded
    from it. Anything under a day reads in hours, at or above in days — the two
    units ChatGPT plans actually use.
    """
    if seconds is None or seconds <= 0:
        return fallback
    if seconds >= 86_400:
        days = round(seconds / 86_400)
        return f"{days} day"
    hours = max(1, round(seconds / 3600))
    return f"{hours} hour"


def _openai_window(raw: Any, now_ms: float) -> tuple[float | None, str, int | None] | None:
    """``(used_percent, window_label, resets_at_ms)`` from one Codex window.

    Two reset encodings exist and both appear in live payloads: an absolute
    ``reset_at`` (seconds OR milliseconds — disambiguated by magnitude, since a
    seconds value large enough to be confused with ms is a year past 33000) and a
    relative ``reset_after_seconds``. A window with neither, and no percentage,
    is not a window.
    """
    if not isinstance(raw, dict):
        return None
    used = _num(raw.get("used_percent"))
    seconds = _num(raw.get("limit_window_seconds"))
    resets_ms: int | None = None
    reset_at = _num(raw.get("reset_at"))
    if reset_at is not None and reset_at > 0:
        resets_ms = int(reset_at if reset_at > 1_000_000_000_000 else reset_at * 1000)
    else:
        after = _num(raw.get("reset_after_seconds"))
        if after is not None and after > 0:
            resets_ms = int(now_ms + after * 1000)
    if used is None and resets_ms is None:
        return None
    return used, _window_label(seconds, "window"), resets_ms


def _openai_limit(
    limit_id: str,
    label: str,
    parsed: tuple[float | None, str, int | None] | None,
    *,
    tier: str = "",
    shared: bool = False,
) -> UsageLimit | None:
    """One Codex window as a ``UsageLimit``, or None when not reported."""
    if parsed is None:
        return None
    used, window, resets_ms = parsed
    if used is None:
        return None
    clamped = max(0.0, min(100.0, used))
    return UsageLimit(
        id=limit_id,
        label=label if label else window,
        amount=UsageAmount(
            used=clamped,
            limit=100.0,
            remaining=100.0 - clamped,
            used_fraction=clamped / 100.0,
            unit="percent",
        ),
        window=window,
        resets_at_ms=resets_ms,
        tier=tier,
        shared=shared,
    )


def _openai_extra_slug(entry: dict[str, Any]) -> tuple[str, str]:
    """``(slug, display_name)`` for one ``additional_rate_limits`` entry.

    ``codex_bengalfox`` is the internal codename for the Spark model; the entry
    also carries the customer-facing ``limit_name``, so the slug is normalised
    to ``spark`` while the label keeps whatever OpenAI called it.
    """
    limit_name = str(entry.get("limit_name") or "").strip()
    metered = str(entry.get("metered_feature") or "").strip()
    probe = f"{limit_name} {metered}".lower()
    if "spark" in probe or "bengalfox" in probe:
        return "spark", limit_name or "Spark"
    source = (metered or limit_name or "extra").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"^codex[-_]", "", source)).strip("-") or "extra"
    return slug, limit_name or slug.replace("-", " ").title()


async def fetch_openai_oauth(
    client: httpx.AsyncClient, access_token: str, account_id: str | None = None
) -> UsageReport | None:
    """OpenAI ChatGPT/Codex plan rate-limit windows.

    The windows are nested under ``rate_limit`` as ``primary_window`` and
    ``secondary_window``. This fetcher read top-level ``primary``/``secondary``
    keys that the endpoint does not send, so it never built a single limit and
    `/usage openai` reported nothing for a logged-in ChatGPT Pro account — the
    same failure as the Anthropic fetcher, one level of nesting deeper.

    ``additional_rate_limits`` carries the per-model caps (Spark). They are
    reported as TIER rows: like Anthropic's model-scoped weekly caps, an
    exhausted Spark window stops one model rather than the account, and merging
    the two is how a usable plan reads as a dead one.
    """
    headers = _bearer(access_token)
    headers["User-Agent"] = "LocalOperator/1.0"
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    payload = await _get_json(client, OPENAI_WHAM_USAGE_URL, headers)
    if payload is None:
        return None
    now_ms = time.time() * 1000

    limits: list[UsageLimit] = []
    rate_limit = payload.get("rate_limit")
    if isinstance(rate_limit, dict):
        for key in ("primary", "secondary"):
            limit = _openai_limit(
                f"openai:{key}",
                "",
                _openai_window(rate_limit.get(f"{key}_window"), now_ms),
                shared=True,
            )
            if limit is not None:
                limits.append(limit)

    extra = payload.get("additional_rate_limits")
    if isinstance(extra, list):
        for entry in extra:
            if not isinstance(entry, dict):
                continue
            nested = entry.get("rate_limit")
            if not isinstance(nested, dict):
                continue
            slug, display = _openai_extra_slug(entry)
            for key in ("primary", "secondary"):
                limit = _openai_limit(
                    f"openai:{slug}:{key}",
                    "",
                    _openai_window(nested.get(f"{key}_window"), now_ms),
                    tier=slug,
                )
                if limit is not None:
                    # The window names itself ("5 hour"); the tier says which
                    # model it counts, and without it two identical rows differ
                    # only by an id the user never sees.
                    limit.label = f"{limit.window} ({display})"
                    limits.append(limit)

    if not limits:
        return None
    plan = payload.get("plan_type")
    notes = f"plan: {plan}" if isinstance(plan, str) and plan.strip() else None
    return UsageReport(provider="openai", limits=limits, notes=notes)


def _kimi_reset_ms(data: dict[str, Any], now_ms: float) -> int | None:
    """The reset instant from whichever key this Kimi object happens to use.

    The payload mixes snake_case and camelCase for the same field across nesting
    levels (``resetTime`` on a detail, ``reset_at`` elsewhere), and expresses it
    as an ISO string, an epoch, or a relative number of seconds. All are read
    here so the caller never has to know which shape it received.
    """
    for key in ("reset_at", "resetAt", "reset_time", "resetTime"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            parsed = _parse_iso_ms(value)
            if parsed is not None:
                return parsed
        elif isinstance(value, (int, float)):
            return int(value if value > 1_000_000_000_000 else value * 1000)
    for key in ("reset_in", "resetIn", "ttl"):
        seconds = _num(data.get(key))
        if seconds is not None and seconds > 0:
            return int(now_ms + seconds * 1000)
    return None


def _kimi_window_label(window: dict[str, Any]) -> str:
    """``5h limit`` from ``{"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"}``.

    Minutes are folded into hours when they divide evenly, because the plan's
    own name for the window is "5h" and ``300m limit`` is the same number said
    in a way no Kimi user would recognise.
    """
    duration = _num(window.get("duration"))
    unit = str(window.get("timeUnit") or window.get("time_unit") or "").upper()
    if duration is None or not unit:
        return ""
    amount = int(duration)
    if "MINUTE" in unit:
        return f"{amount // 60}h limit" if amount >= 60 and amount % 60 == 0 else f"{amount}m limit"
    if "HOUR" in unit:
        return f"{amount}h limit"
    if "DAY" in unit:
        return f"{amount}d limit"
    if "SECOND" in unit:
        return f"{amount}s limit"
    return ""


def _kimi_row(data: dict[str, Any], now_ms: float) -> tuple[UsageAmount, int | None] | None:
    """``(amount, resets_at_ms)`` for one Kimi quota object.

    Kimi quotes ``limit``/``used``/``remaining`` as STRINGS, and often omits
    ``used`` while sending the other two — the five hour window arrives as
    ``{"limit": "100", "remaining": "100"}``. Deriving the missing half is what
    makes that row renderable at all; requiring ``used`` is why it was dropped.
    """
    limit = _num(data.get("limit"))
    used = _num(data.get("used"))
    remaining = _num(data.get("remaining"))
    if used is None and remaining is not None and limit is not None:
        used = limit - remaining
    if used is None and limit is None:
        return None
    amount = UsageAmount(used=used, limit=limit, remaining=remaining, unit="unknown")
    if limit is not None and used is not None and limit > 0:
        amount.used_fraction = max(0.0, min(1.0, used / limit))
    return amount, _kimi_reset_ms(data, now_ms)


async def fetch_kimi_oauth(client: httpx.AsyncClient, access_token: str) -> UsageReport | None:
    """Kimi coding-plan usage windows (OAuth).

    Two things were wrong. The rows live at the TOP level of the response
    (``usage`` for the plan total, ``limits[]`` for each window) rather than
    under a ``data`` envelope, so the fetcher returned None before parsing
    anything. And each ``limits[]`` entry splits its numbers across
    ``detail`` and its window length across ``window``, neither of which the old
    flat read reached.
    """
    payload = await _get_json(client, KIMI_USAGE_URL, _bearer(access_token))
    if payload is None:
        return None
    # An envelope is tolerated but not required: some deployments wrap the body
    # in `data`, and unwrapping it here costs one line and covers both.
    envelope = payload.get("data")
    data: dict[str, Any] = envelope if isinstance(envelope, dict) else payload
    now_ms = time.time() * 1000

    limits: list[UsageLimit] = []
    summary = data.get("usage")
    if isinstance(summary, dict):
        row = _kimi_row(summary, now_ms)
        if row is not None:
            amount, resets = row
            limits.append(
                UsageLimit(
                    id="kimi:total",
                    label="Total quota",
                    amount=amount,
                    window="plan",
                    resets_at_ms=resets,
                    shared=True,
                )
            )

    raw_limits = data.get("limits")
    if isinstance(raw_limits, list):
        for index, item in enumerate(raw_limits):
            if not isinstance(item, dict):
                continue
            raw_detail = item.get("detail")
            detail: dict[str, Any] = raw_detail if isinstance(raw_detail, dict) else item
            raw_window = item.get("window")
            window_data: dict[str, Any] = raw_window if isinstance(raw_window, dict) else {}
            label = (
                str(item.get("name") or item.get("title") or "").strip()
                or _kimi_window_label(window_data)
                or f"Limit {index + 1}"
            )
            row = _kimi_row(detail, now_ms)
            if row is None:
                continue
            amount, detail_reset = row
            # The WINDOW's own reset wins when it has one; Kimi usually leaves it
            # on the detail instead, which is why the fallback exists at all —
            # without it the 5h row renders with no countdown.
            window_reset = _kimi_reset_ms(window_data, now_ms) or detail_reset
            limits.append(
                UsageLimit(
                    id=f"kimi:{index}",
                    label=label,
                    amount=amount,
                    window=_kimi_window_label(window_data) or "window",
                    resets_at_ms=window_reset,
                    shared=True,
                )
            )
    return UsageReport(provider="kimi", limits=limits) if limits else None


def _xai_amount(value: Any) -> float | None:
    """xAI quota amounts, which arrive WRAPPED as ``{"val": 300}``.

    Read as a bare number, the wrapper coerces to None and the monthly rows lost
    both their used and their limit — the report still rendered, with a label and
    no numbers, which is the failure that looks like a working feature.
    """
    if isinstance(value, dict):
        amount = _num(value.get("val"))
    else:
        amount = _num(value)
    return amount if amount is not None and amount >= 0 else None


def _xai_percent_limit(
    limit_id: str, label: str, percent: float, window: str, resets_ms: int | None, *, tier: str = ""
) -> UsageLimit:
    clamped = max(0.0, min(100.0, percent))
    return UsageLimit(
        id=limit_id,
        label=label,
        amount=UsageAmount(
            used=clamped,
            limit=100.0,
            remaining=100.0 - clamped,
            used_fraction=clamped / 100.0,
            unit="percent",
        ),
        window=window,
        resets_at_ms=resets_ms,
        tier=tier,
        shared=not tier,
    )


def _xai_on_demand(config: dict[str, Any]) -> UsageLimit | None:
    """The pay-as-you-go cap that backs an exhausted subscription.

    Same reasoning as Anthropic's extra-usage meter: when an account is out of
    included quota, whether it has on-demand headroom is the difference between
    "wait for the reset" and "keep working".
    """
    cap = _xai_amount(config.get("onDemandCap"))
    used = _xai_amount(config.get("onDemandUsed"))
    if cap is None or cap <= 0 or used is None:
        return None
    return UsageLimit(
        id="xai:on-demand",
        label="On-demand",
        amount=UsageAmount(
            used=used,
            limit=cap,
            remaining=max(0.0, cap - used),
            used_fraction=min(1.0, used / cap),
            unit="unknown",
        ),
        window="1 month",
        shared=True,
    )


def _xai_weekly_limits(config: dict[str, Any]) -> list[UsageLimit]:
    """SuperGrok's legacy weekly credit shape (``?format=credits``)."""
    percent = _num(config.get("creditUsagePercent"))
    if percent is None or not 0 <= percent <= 100:
        return []
    period = config.get("currentPeriod")
    resets_ms = None
    if isinstance(period, dict):
        resets_ms = _parse_iso_ms(period.get("end"))
    limits = [_xai_percent_limit("xai:credits:1w", "Weekly credits", percent, "1 week", resets_ms)]
    products = config.get("productUsage")
    if isinstance(products, list):
        for item in products:
            if not isinstance(item, dict):
                continue
            name = str(item.get("product") or "").strip()
            product_percent = _num(item.get("usagePercent"))
            if not name or product_percent is None:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            if not slug:
                continue
            # Per-PRODUCT rows are tier rows: one being spent does not stop the
            # others, so they must not read as the account-wide credit pool.
            limits.append(
                _xai_percent_limit(
                    f"xai:product:{slug}:1w",
                    f"{name} (weekly)",
                    product_percent,
                    "1 week",
                    resets_ms,
                    tier=slug,
                )
            )
    return limits


def _xai_monthly_limits(config: dict[str, Any]) -> list[UsageLimit]:
    """The unified-billing monthly quota — capped, or bare spend when uncapped.

    Some unified-billing accounts are billed by their subscription plan rather
    than metered against a numeric allowance: the bare billing URL answers
    ``monthlyLimit: {"val": 0}`` while still counting ``used`` up (verified
    against production 2026-08-18 — ``used: {"val": 443}`` on a live SuperGrok
    account). Requiring a positive cap dropped the only number such an account
    reports, every other row was already empty for it, and the whole report
    collapsed to None — so the provider vanished from the usage view,
    indistinguishable from a missing credential.

    A used-only row keeps the account visible and honest: no cap means no
    fraction, so it renders as an unmeasurable row (dotted bar, bare number)
    rather than inventing a denominator, and :func:`usage_health` keeps
    treating the account as "unknown" instead of guessing at depletion.
    """
    limit = _xai_amount(config.get("monthlyLimit"))
    used = _xai_amount(config.get("used"))
    if used is None:
        return []
    resets_ms = _parse_iso_ms(config.get("billingPeriodEnd"))
    if limit is None or limit <= 0:
        return [
            UsageLimit(
                id="xai:usage:1mo",
                label="Monthly usage",
                amount=UsageAmount(used=used, unit="unknown"),
                window="1 month",
                resets_at_ms=resets_ms,
                shared=True,
            )
        ]
    return [
        UsageLimit(
            id="xai:included:1mo",
            label="Monthly included",
            amount=UsageAmount(
                used=used,
                limit=limit,
                remaining=max(0.0, limit - used),
                used_fraction=min(1.0, used / limit),
                unit="unknown",
            ),
            window="1 month",
            resets_at_ms=resets_ms,
            shared=True,
        )
    ]


async def _xai_billing_config(
    client: httpx.AsyncClient, access_token: str, url: str
) -> dict[str, Any] | None:
    """The ``config`` object from one billing URL, or None."""
    headers = _bearer(access_token)
    headers["Accept"] = "application/json"
    headers["X-XAI-Token-Auth"] = "xai-grok-cli"
    payload = await _get_json(client, url, headers)
    if payload is None:
        return None
    config = payload.get("config")
    return config if isinstance(config, dict) else None


async def fetch_xai_oauth(client: httpx.AsyncClient, access_token: str) -> UsageReport | None:
    """xAI Grok subscription usage, across BOTH billing shapes.

    xAI serves two different payloads on one endpoint. ``?format=credits`` is the
    legacy SuperGrok weekly shape (a credit percentage plus a per-product
    breakdown); accounts migrated to unified billing get a monthly included quota
    (``monthlyLimit``/``used``) on the bare URL instead, and report
    ``isUnifiedBillingUser``.

    Only the first was requested before, and its amounts were read as bare
    numbers when xAI wraps them as ``{"val": N}`` — so a unified-billing account
    got an empty report and a weekly one got labels with no numbers. The credits
    URL is still probed first, and the monthly URL only when the account needs
    it, which keeps the common case at one request.
    """
    credits_url = f"{XAI_BILLING_URL}?format=credits"
    config = await _xai_billing_config(client, access_token, credits_url)

    limits: list[UsageLimit] = []
    unified = False
    if config is not None:
        unified = config.get("isUnifiedBillingUser") is True
        limits.extend(_xai_weekly_limits(config))

    if not limits or unified:
        monthly_config = await _xai_billing_config(client, access_token, XAI_BILLING_URL)
        if monthly_config is not None:
            limits.extend(_xai_monthly_limits(monthly_config))
            config = config or monthly_config

    if config is not None:
        on_demand = _xai_on_demand(config)
        if on_demand is not None:
            limits.append(on_demand)

    # Recompute the unified flag from the *effective* config once both URLs have
    # had their chance. The pre-fetch value above only gates whether the monthly
    # URL is probed; when the credits URL 401s but the bare monthly URL answers
    # for an uncapped unified account, that config carries isUnifiedBillingUser
    # too, and without this the explanatory note would be dropped — an
    # unmeasurable bar with no reason reads exactly like the rendering defect
    # this code warns against. Deriving it here keeps both note branches below
    # on the same live signal.
    unified = config is not None and config.get("isUnifiedBillingUser") is True

    # Ids are unique by construction, but both shapes can carry the same
    # on-demand cap when an account reports weekly and monthly at once.
    seen: set[str] = set()
    deduped = [limit for limit in limits if not (limit.id in seen or seen.add(limit.id))]
    if not deduped:
        if config is None:
            # Nothing ANSWERED — transport failure or a dead token. None here
            # lets the caller fall through to other routes and, failing those,
            # report "no usage data", which is the truthful message.
            return None
        # The endpoint answered, but the shape carried nothing renderable
        # (e.g. a unified-billing config with neither ``used`` nor a cap).
        # Returning None here is what made an ANSWERING account disappear from
        # the panel entirely; an empty report keeps the provider block on
        # screen ("no windows reported") with a note saying why, which is the
        # difference between "your login is broken" and "xAI has nothing to
        # meter for this plan".
        note = (
            "unified billing — no metered windows reported"
            if unified
            else "no measurable windows reported"
        )
        return UsageReport(provider="xai", limits=[], notes=note)
    notes = None
    if unified and all(limit.amount.fraction() is None for limit in deduped):
        # Every surviving row is unmeasurable (the uncapped-spend shape): say
        # why the bars are dotted, or the row reads like a rendering defect.
        notes = "unified billing — spend is reported without a metered cap"
    return UsageReport(provider="xai", limits=deduped, notes=notes)


def _num(value: Any, default: float | None = None) -> float | None:
    """Coerce numeric/str fields defensively; None on garbage.

    ``OverflowError`` is caught alongside the obvious two because a JSON body
    may carry an integer too large for a float (``10**400`` parses fine and
    then raises on conversion). Every vendor field in this module reaches a
    fetcher through here, so catching it at the coercion boundary is what keeps
    the "a quota fetcher never raises" contract true for all of them rather
    than for the call sites someone remembered to guard.

    Non-finite values are rejected rather than returned: ``json.loads`` accepts
    bare ``NaN``/``Infinity``, and a float that survives coercion only to blow
    up at the ``int()`` that formats it has merely moved the crash somewhere
    harder to see. ``None`` here means "no usable number", which every caller
    already handles.
    """
    try:
        if value is None:
            return default
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


async def _qwencloud_bss_call(
    client: httpx.AsyncClient, token: str, action: str, params: dict[str, Any]
) -> dict[str, Any] | None:
    """One flat BSS call through the official gateway; None on any failure."""
    flattened = {
        key: (
            value
            if isinstance(value, str)
            else str(value) if isinstance(value, int) else json.dumps(value)
        )
        for key, value in params.items()
    }
    try:
        response = await client.post(
            QWENCLOUD_BSS_URL,
            json={
                "product": QWENCLOUD_BSS_PRODUCT,
                "action": action,
                "region": QWENCLOUD_REGION,
                "params": flattened,
            },
            headers=_bearer(token),
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    payload = response.json() if response.content else {}
    data = payload.get("data")
    if payload.get("code") != "200" or not isinstance(data, dict):
        return None
    return data


def _qwencloud_fr_status(item: dict[str, Any]) -> str:
    """``valid`` from an instance's Status, which is a string or {Code: …}."""
    status = item.get("Status")
    if isinstance(status, dict):
        code = status.get("Code")
        return str(code).lower() if code else ""
    return str(status).lower() if status is not None else ""


def _qwencloud_credits_limit(
    limit_id: str, label: str, window: str, total: float, remaining: float, resets_at_ms: int | None
) -> UsageLimit | None:
    """One subscription window in absolute credits (used falls out of total−remaining)."""
    if not total or total <= 0:
        return None
    return UsageLimit(
        id=limit_id,
        label=label,
        amount=UsageAmount(
            used=max(0.0, total - remaining), limit=total, remaining=remaining, unit="unknown"
        ),
        window=window,
        resets_at_ms=resets_at_ms,
        shared=True,
    )


async def fetch_qwencloud_token_plan(client: httpx.AsyncClient, token: str) -> UsageReport | None:
    """QwenCloud Token Plan credits via the management OAuth token.

    Mirrors the official CLI's data path: accounts migrated to seat-based
    subscriptions (``QuerySubscriptionGray`` → true) answer through
    ``GetSeatSubscriptionSummary`` with a monthly window; everyone else
    reports per-commodity ``DescribeFrInstances``, where the personal
    commodity carries the 7-day credit window (``InitCapacityBaseValue``
    total, ``CurrCapacityBaseValue`` remaining, ``EndTime`` reset) and team
    seats a monthly one. Add-on instances sum into a window-less Credit
    Packs limit — packs draw down outside the subscription window.
    """

    async def instances(commodity: str, page_size: int = 10) -> list[dict[str, Any]]:
        data = await _qwencloud_bss_call(
            client,
            token,
            "DescribeFrInstances",
            {
                "Group": "tokenPlan",
                "CommodityCode": commodity,
                "PageNum": 1,
                "PageSize": page_size,
            },
        )
        rows = data.get("Data") if data else None
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    gray = await _qwencloud_bss_call(client, token, "QuerySubscriptionGray", {})
    if gray is None:
        return None
    packs_rows = await instances(QWENCLOUD_TOKEN_PLAN_COMMODITIES["addon"], 100)
    packs = sum(
        _num(row.get("CurrCapacityBaseValue"), 0.0) or 0.0
        for row in packs_rows
        if _qwencloud_fr_status(row) == "valid"
    )
    limits: list[UsageLimit] = []

    if gray.get("IsGray") is True:
        seat = await _qwencloud_bss_call(
            client,
            token,
            "GetSeatSubscriptionSummary",
            {"productCode": QWENCLOUD_TOKEN_PLAN_COMMODITIES["teams"]},
        )
        if seat is None:
            return None
        # BSS sometimes nests the summary under a second Data envelope.
        nested = seat.get("Data")
        inner: dict[str, Any] = nested if isinstance(nested, dict) else seat
        groups = inner.get("SubscriptionGroupList")
        total = 0.0
        remaining = 0.0
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, dict):
                continue
            equity_list = group.get("EquityList")
            equity = (
                equity_list[0]
                if isinstance(equity_list, list)
                and equity_list
                and isinstance(equity_list[0], dict)
                else {}
            )
            total += _num(equity.get("TotalValue") or group.get("TotalValue"), 0.0) or 0.0
            remaining += _num(equity.get("SurplusValue") or group.get("SurplusValue"), 0.0) or 0.0
        end = _num(inner.get("EndTime"))
        limit = _qwencloud_credits_limit(
            "credits-monthly",
            "Monthly Credits",
            "monthly",
            total,
            remaining,
            int(end) if end and end > 0 else None,
        )
        if limit is not None:
            limits.append(limit)
    else:
        teams_rows = await instances(QWENCLOUD_TOKEN_PLAN_COMMODITIES["teams"])
        personal_rows = await instances(QWENCLOUD_TOKEN_PLAN_COMMODITIES["personal"])
        # Teams first, mirroring the CLI: a team seat outranks a personal
        # subscription on the same account for which window is THE constraint.
        pick = next((row for row in teams_rows if _qwencloud_fr_status(row) == "valid"), None)
        window = "monthly"
        if pick is None:
            pick = next(
                (row for row in personal_rows if _qwencloud_fr_status(row) == "valid"),
                personal_rows[0] if personal_rows else None,
            )
            window = "7d"
        if pick is not None:
            monthly_shift = pick.get("CapacityTypeCode") == "periodMonthlyShift"
            total = _num(pick.get("InitCapacityBaseValue"), 0.0) or 0.0
            remaining = (
                (_num(pick.get("periodCapacityBaseValue")) if monthly_shift else None)
                or _num(pick.get("CurrCapacityBaseValue"), 0.0)
                or 0.0
            )
            end = _num(pick.get("EndTime"))
            if window == "7d" and not monthly_shift:
                limit = _qwencloud_credits_limit(
                    "credits-7d",
                    "7 Day Credits",
                    "7d",
                    total,
                    remaining,
                    int(end) if end and end > 0 else None,
                )
            else:
                limit = _qwencloud_credits_limit(
                    "credits-monthly",
                    "Monthly Credits",
                    "monthly",
                    total,
                    remaining,
                    int(end) if end and end > 0 else None,
                )
            if limit is not None:
                limits.append(limit)

    if packs > 0:
        limits.append(
            UsageLimit(
                id="credits-packs",
                label="Credit Packs",
                amount=UsageAmount(remaining=packs, unit="unknown"),
            )
        )
    return UsageReport(provider="alibaba-token-plan", limits=limits) if limits else None


async def fetch_usage(
    client: httpx.AsyncClient,
    provider: str,
    api_key: str | None = None,
    access_token: str | None = None,
    account_id: str | None = None,
    oauth_creds: dict[str, Any] | None = None,
) -> UsageReport | None:
    """Dispatch a provider id to whichever fetcher its credentials can reach.

    ``api_key`` is the resolved API-key cascade value; ``access_token`` is the
    OAuth access token. The OAuth route is preferred when both are present,
    because for every provider that has both it reports the SUBSCRIPTION the user
    is actually spending (plan windows) while the API-key route reports a
    pay-as-you-go balance that a subscription user does not draw down.
    ``oauth_creds`` is the raw stored OAuth row for split-token providers whose
    wire bearer is not the token the quota endpoint wants.

    Returns None when the provider has no endpoint, or has one but not for the
    credential kind on hand. Never raises.
    """
    routes = _FETCHERS.get(provider)
    if routes is None:
        return None
    oauth_kind, api_kind = routes
    if access_token and oauth_kind is not None:
        return await _run_fetcher(client, oauth_kind, access_token, account_id, creds=oauth_creds)
    if api_key and api_kind is not None:
        return await _run_fetcher(client, api_kind, api_key, account_id)
    return None


async def _run_fetcher(
    client: httpx.AsyncClient,
    kind: FetcherKind,
    secret: str,
    account_id: str | None,
    *,
    creds: dict[str, Any] | None = None,
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
    if kind == "zai-quota":
        return await fetch_zai_quota(client, secret)
    if kind == "xai-oauth":
        return await fetch_xai_oauth(client, secret)
    if kind == "qwencloud-token-plan":
        # Split-token provider: the wire bearer is the sk-sp key, but quota
        # needs the management token from the raw row. ``secret`` is the
        # mapped (wire) key, so only rows carrying ``access`` can report.
        management = (creds or {}).get("access")
        if not management:
            return None
        return await fetch_qwencloud_token_plan(client, management)
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
