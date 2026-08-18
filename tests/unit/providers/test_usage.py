"""Unit tests for the provider usage-quota fetchers and the ProviderController.

The fetchers are tested against canned HTTP responses (httpx MockTransport) so
the parsing contract is pinned without any live network. ``fetch_usage`` and
``usage_supported`` are the public dispatch surface; each provider's parser is
exercised through it.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import httpx
import pytest

from local_operator.providers.usage import (
    UsageAmount,
    UsageLimit,
    UsageReport,
    fetch_usage,
    usage_health,
    usage_supported,
)


def _client_for(payload, status: int = 200) -> httpx.AsyncClient:
    """An httpx async client that returns a canned JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _recording_client(payload, status: int = 200) -> tuple[httpx.AsyncClient, list[str]]:
    """``(client, urls)`` — the same canned body, plus every URL it was asked for.

    The URL is the contract for a region-derived endpoint: the parsed body looks
    identical whichever host answered it.
    """
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(status, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), urls


def test_num_never_yields_a_value_that_crashes_downstream() -> None:
    """``_num`` is the single coercion boundary all nine fetchers share, so its
    contract is what keeps "a quota fetcher never raises" true for fields nobody
    has thought of yet.

    Pinned at this level rather than only through one provider because the
    guarantee is module-wide: a caller is entitled to assume anything `_num`
    returns can be compared, formatted, and passed to ``int()``. ``json.loads``
    accepts bare ``NaN``/``Infinity``, and an integer literal can exceed float
    range (``10**400`` parses, then raises ``OverflowError`` on conversion) —
    all of which must become ``None`` rather than a value that detonates later.
    """
    from local_operator.providers.usage import _num

    # Unusable inputs become None...
    for bad in (
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
        -(10**400),
        "abc",
        "",
        None,
        [],
        {},
        object(),
    ):
        assert _num(bad) is None, bad

    # ...and everything returned survives the operations callers perform on it:
    # `int()` (which is what raised before), formatting, and comparison. NaN
    # would pass an `int()` check by raising, so the finiteness is asserted
    # directly rather than implied by an always-true comparison.
    for good, expected in (
        (0, 0.0),
        (-5, -5.0),
        (3.14, 3.14),
        ("42", 42.0),
        ("3.5", 3.5),
        (1787009983644, 1787009983644.0),
        (10**15, 1e15),
    ):
        value = _num(good)
        assert value == expected
        assert math.isfinite(value)  # type: ignore[arg-type]
        assert isinstance(int(value), int)  # type: ignore[arg-type]

    assert _num(None, 7.0) == 7.0


def test_usage_health_shared_window_reaches_reserve() -> None:
    report = UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:7d",
                label="7 day",
                amount=UsageAmount(used_fraction=0.95),
                shared=True,
                resets_at_ms=20_000,
            )
        ],
    )
    health = usage_health(report, "claude-opus-5", reserve_percent=10, now_ms=10_000)
    assert health.state == "reserve"
    assert health.scope == "account"
    assert health.remaining_fraction == pytest.approx(0.05)
    assert health.reset_after_ms == 10_000


def test_usage_health_ignores_unrelated_model_tier() -> None:
    report = UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:sonnet",
                label="Sonnet",
                amount=UsageAmount(used_fraction=1.0),
                tier="sonnet",
            ),
            UsageLimit(
                id="anthropic:shared",
                label="Shared",
                amount=UsageAmount(used_fraction=0.2),
                shared=True,
            ),
        ],
    )
    assert usage_health(report, "claude-opus-5").state == "healthy"


def test_usage_health_enabled_extra_credit_supersedes_included_plan() -> None:
    report = UsageReport(
        provider="anthropic",
        limits=[
            UsageLimit(
                id="anthropic:7d",
                label="7 day",
                amount=UsageAmount(used_fraction=1.0),
                shared=True,
            ),
            UsageLimit(
                id="anthropic:extra",
                label="Extra usage",
                amount=UsageAmount(used=25, limit=100),
            ),
        ],
    )
    health = usage_health(report, "claude-opus-5")
    assert health.state == "healthy"
    assert health.remaining_fraction == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_openrouter_credits_parse() -> None:
    payload = {"data": {"usage": 12.5, "limit": 100.0, "is_free_tier": False}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="sk-or-test")
    assert report is not None
    assert len(report.limits) == 1
    limit = report.limits[0]
    assert limit.amount.used == pytest.approx(12.5)
    assert limit.amount.limit == pytest.approx(100.0)
    assert limit.amount.fraction() == pytest.approx(0.125)
    assert limit.effective_status() == "ok"


@pytest.mark.asyncio
async def test_openrouter_free_tier_has_no_budget() -> None:
    payload = {"data": {"usage": 0.0, "limit": 0.0, "is_free_tier": True}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="sk-or-test")
    assert report is not None
    assert report.limits == []  # no fabricated 0/0 budget
    assert report.notes is not None and "free" in report.notes


@pytest.mark.asyncio
async def test_openrouter_no_limit_reports_spend_only() -> None:
    payload = {"data": {"usage": 519.7, "limit": 0.0, "is_free_tier": False}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="sk-or-test")
    assert report is not None
    limit = report.limits[0]
    assert limit.amount.used == pytest.approx(519.7)
    assert limit.amount.limit is None
    assert limit.amount.fraction() is None  # no limit -> unmeasurable
    assert limit.effective_status() == "unknown"


@pytest.mark.asyncio
async def test_kimi_api_key_reaches_the_balance_endpoint_the_key_belongs_to() -> None:
    """The widest gap this module had: the registry stores `KIMI_API_KEY` while the
    only Kimi fetcher wanted an OAuth token, so an API-key user was told Kimi
    reports quota and got an empty table forever.

    And the host has to be the one the key is FOR. Moonshot mainland
    (`api.moonshot.cn`) and international (`api.moonshot.ai`) are separate
    platforms with separate accounts and separate keys, so a balance call to the
    other region 401s on the only key the user can hold — reinstating the empty
    table this fetcher exists to remove. The registry configures `kimi` for
    `https://api.moonshot.cn/v1`, so that is where the request must go.
    """
    payload = {
        "data": {"available_balance": 70.0, "voucher_balance": 20.0, "cash_balance": 50.0},
    }
    client, urls = _recording_client(payload)
    async with client:
        report = await fetch_usage(client, "kimi", api_key="sk-moonshot")
    assert urls == ["https://api.moonshot.cn/v1/users/me/balance"]
    assert report is not None
    limit = report.limits[0]
    assert limit.id == "kimi:balance"
    assert limit.amount.remaining == pytest.approx(70.0)
    # A mainland balance is CNY. `usd` here would overstate it ~7x, so the unit is
    # left unlabelled and the currency rides in the label, as DeepSeek does.
    assert limit.amount.unit == "unknown"
    assert limit.label == "Balance (CNY)"
    assert report.notes == "voucher ¥20.00 + cash ¥50.00"


@pytest.mark.asyncio
async def test_the_international_moonshot_region_reports_dollars() -> None:
    """Same payload, same parser, different platform: `api.moonshot.ai` accounts
    are billed in USD and the response carries no currency field, so the host is
    the only thing that says which."""
    from local_operator.providers.usage import fetch_moonshot_balance

    payload = {
        "data": {"available_balance": 12.5, "voucher_balance": 2.5, "cash_balance": 10.0},
    }
    client, urls = _recording_client(payload)
    async with client:
        report = await fetch_moonshot_balance(
            client, "sk-moonshot", base_url="https://api.moonshot.ai/v1"
        )
    assert urls == ["https://api.moonshot.ai/v1/users/me/balance"]
    assert report is not None
    limit = report.limits[0]
    assert limit.amount.unit == "usd"
    assert limit.label == "Balance (USD)"
    assert report.notes == "voucher $2.50 + cash $10.00"


def test_the_balance_endpoint_follows_the_configured_base_url() -> None:
    """Not a hardcoded host: change the provider's base_url and the balance call
    follows it, because the key that reaches the chat API is the key that reaches
    the balance."""
    from local_operator.providers.usage import moonshot_balance_target

    assert moonshot_balance_target("https://api.moonshot.cn/v1") == (
        "https://api.moonshot.cn/v1/users/me/balance",
        "unknown",
        "¥",
    )
    assert moonshot_balance_target("https://api.moonshot.ai/v1/") == (
        "https://api.moonshot.ai/v1/users/me/balance",
        "usd",
        "$",
    )
    # Default = whatever the registry configures, which is the mainland host.
    from local_operator.providers.registry import get_provider_definition

    configured = get_provider_definition("kimi")
    assert configured is not None and configured.base_url is not None
    assert moonshot_balance_target()[0].startswith(configured.base_url)


@pytest.mark.asyncio
async def test_an_oauth_token_still_wins_where_both_routes_exist() -> None:
    """For a provider with both, the OAuth route reports the SUBSCRIPTION the user
    is actually spending; the balance a subscription user never draws down is the
    less useful answer."""
    payload = {"data": {"limits": [{"name": "coding", "used": 1, "limit": 10}]}}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "kimi", api_key="sk-moonshot", access_token="tok")
    assert report is not None
    assert all(lim.id != "kimi:balance" for lim in report.limits), report.limits


@pytest.mark.asyncio
async def test_deepseek_reports_a_balance_per_currency() -> None:
    """A CNY balance rendered as USD would be wrong by roughly a factor of seven,
    so the currency is part of the id and only USD claims the dollar unit."""
    payload = {
        "is_available": True,
        "balance_infos": [
            {"currency": "CNY", "total_balance": "70.00"},
            {"currency": "USD", "total_balance": "9.85"},
        ],
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "deepseek", api_key="sk-ds")
    assert report is not None
    ids = [lim.id for lim in report.limits]
    assert ids == ["deepseek:balance:cny", "deepseek:balance:usd"]
    units = {lim.id: lim.amount.unit for lim in report.limits}
    assert units["deepseek:balance:cny"] == "unknown"
    assert units["deepseek:balance:usd"] == "usd"


#: A REAL Z.AI coding-plan quota body, captured from
#: `GET https://api.z.ai/api/monitor/usage/quota/limit` on 2026-08-17. Pinned
#: verbatim because the shape is the whole contract here: the TOKENS_LIMIT rows
#: carry ONLY a percentage (no absolute counts), and TIME_LIMIT inverts the
#: obvious reading of its fields — `usage` is the limit, `currentValue` the
#: amount consumed. An invented payload would have hidden both.
_ZAI_QUOTA_PAYLOAD: dict[str, Any] = {
    "code": 200,
    "msg": "Operation successful",
    "data": {
        "limits": [
            {
                "type": "TOKENS_LIMIT",
                "unit": 3,
                "number": 5,
                "percentage": 12,
                "nextResetTime": 1787009983644,
            },
            {
                "type": "TOKENS_LIMIT",
                "unit": 6,
                "number": 1,
                "percentage": 42,
                "nextResetTime": 1787542072998,
            },
            {
                "type": "TIME_LIMIT",
                "unit": 5,
                "number": 1,
                "usage": 4000,
                "currentValue": 0,
                "remaining": 4000,
                "percentage": 0,
                "nextResetTime": 1789615672998,
                "usageDetails": [
                    {"modelCode": "search-prime", "usage": 0},
                    {"modelCode": "web-reader", "usage": 0},
                    {"modelCode": "zread", "usage": 0},
                ],
            },
        ],
        "level": "max",
    },
    "success": True,
}


@pytest.mark.asyncio
async def test_zai_reports_token_and_feature_windows() -> None:
    """The coding plan quotes token windows as a bare percentage, so the report is
    built from the fraction rather than inventing token counts to display."""
    client = _client_for(_ZAI_QUOTA_PAYLOAD)
    async with client:
        report = await fetch_usage(client, "zai", api_key="zai-key")
    assert report is not None
    ids = [lim.id for lim in report.limits]
    assert ids == ["zai:tokens:5hour", "zai:tokens:1week", "zai:features:zread:1month"]

    five_hour = report.limits[0]
    assert five_hour.amount.fraction() == pytest.approx(0.12)
    # No absolute counts in the payload, so the renderer is told to speak percent.
    assert five_hour.amount.unit == "percent"
    assert five_hour.amount.used is None
    assert five_hour.shared is True
    assert five_hour.resets_at_ms == 1787009983644
    assert report.notes == "coding plan: max"


@pytest.mark.asyncio
async def test_zai_zread_bucket_is_a_tier_not_the_account_cap() -> None:
    """Exhausting the search/web-reader/zread bucket stops those tools, not the
    plan, so it must not be rendered as the umbrella limit that gates every
    request."""
    client = _client_for(_ZAI_QUOTA_PAYLOAD)
    async with client:
        report = await fetch_usage(client, "zai", api_key="zai-key")
    assert report is not None
    zread = report.limits[-1]
    assert zread.tier == "zread"
    assert zread.shared is False
    # TIME_LIMIT inverts the field names: `usage` is the limit, not the usage.
    assert zread.amount.limit == 4000
    assert zread.amount.used == 0
    assert zread.amount.unit == "requests"


@pytest.mark.asyncio
async def test_zai_hostile_window_fields_do_not_raise() -> None:
    """A quota fetcher must never raise: the caller drops a ``None`` report, but
    an exception escapes to the UI.

    ``json.loads`` accepts bare ``NaN``/``Infinity``, and ``int()`` rejects both
    (ValueError / OverflowError) — so a vendor could crash the usage panel with
    a field that is only ever read to build a LABEL. A quoted ``"unit"`` must
    also still resolve its window name rather than silently degrading.
    """
    # Served as RAW BYTES, not through the JSON-encoding helper: `json.dumps`
    # refuses to emit NaN/Infinity, so round-tripping the payload would destroy
    # the very values under test. This is what the vendor can actually put on
    # the wire — `json.loads` parses both without complaint.
    body = b"""
        {"success": true, "code": 200, "data": {"limits": [
            {"type": "TOKENS_LIMIT", "unit": NaN, "number": 5, "percentage": 10},
            {"type": "TOKENS_LIMIT", "unit": Infinity, "number": NaN, "percentage": 20},
            {"type": "TOKENS_LIMIT", "unit": "3", "number": "5", "percentage": 30}
        ]}}
        """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(client, "zai", api_key="zai-key")
    assert report is not None
    # Unmappable units degrade to the generic label; a quoted unit still maps.
    # Only TWO rows survive: both unmappable rows generate the id
    # `zai:tokens:quota`, and the dedupe keeps the first rather than rendering
    # two identical panel rows.
    labels = [lim.label for lim in report.limits]
    assert labels == ["Token quota (quota)", "Token quota (5 hour)"]
    assert [lim.id for lim in report.limits] == ["zai:tokens:quota", "zai:tokens:5hour"]


@pytest.mark.asyncio
async def test_zai_never_raises_on_any_hostile_numeric_field() -> None:
    """The never-raise contract, enforced over the whole cross-product.

    A previous round fixed the two call sites a review happened to quote and
    left the defect class open: `nextResetTime` still crashed on Infinity, and
    an oversized integer (``10**400`` parses, then raises ``OverflowError`` on
    `float()`) took out every numeric field at once. Testing one field with one
    bad value is what let that through, so this walks every field with every
    hostile value rather than trusting a spot check.
    """
    nasty: list[Any] = [
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
        -(10**400),
        -1,
        "3",
        "abc",
        "",
        None,
        True,
        [],
        {},
    ]
    fields = ["unit", "number", "percentage", "usage", "currentValue", "remaining", "nextResetTime"]
    for row_type in ("TOKENS_LIMIT", "TIME_LIMIT"):
        for field in fields:
            for value in nasty:
                row: dict[str, Any] = {
                    "type": row_type,
                    "unit": 3,
                    "number": 1,
                    "percentage": 10,
                    field: value,
                }
                payload = {"success": True, "data": {"limits": [row]}}
                # `allow_nan=True` + raw bytes: the strict encoder in
                # `_client_for` refuses NaN/Infinity, which would quietly drop
                # the values this test exists to exercise.
                body = json.dumps(payload, allow_nan=True, default=str).encode()

                def handler(request: httpx.Request, _body: bytes = body) -> httpx.Response:
                    return httpx.Response(
                        200, content=_body, headers={"content-type": "application/json"}
                    )

                client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                async with client:
                    # The assertion is that this line returns at all.
                    await fetch_usage(client, "zai", api_key="k")

    # The ENVELOPE shapes, not just the leaf values. The numeric sweep above
    # feeds well-formed containers, so the `isinstance` guards on `data`, on
    # `limits`, and on each row were load-bearing with nothing standing over
    # them: dropping the `data`-is-a-dict or the row-is-a-dict check turns a
    # hostile body into an AttributeError rather than a dropped report, and no
    # test noticed.
    for envelope in (
        {"success": True, "data": "nope"},
        {"success": True, "data": [1, 2]},
        {"success": True, "data": None},
        {"success": True, "data": {"limits": "nope"}},
        {"success": True, "data": {"limits": None}},
        {"success": True, "data": {"limits": ["nope", 3, None]}},
        {"success": True, "data": {}},
    ):
        client = _client_for(envelope)
        async with client:
            assert await fetch_usage(client, "zai", api_key="k") is None, envelope


@pytest.mark.asyncio
async def test_zai_feature_bucket_is_the_breakdown_with_no_chat_model() -> None:
    """The classification must survive Z.AI changing its own feature codes.

    Every positive test is a maintenance dependency on a vendor enum: requiring
    all known codes breaks on a rename, and requiring a SUBSET of them breaks on
    a rename *and* on an addition. Both fail toward ``shared=True``, which
    `usage_health` applies to every model — so a user with most of their token
    quota intact is told the account is depleted because the vendor shipped a
    new tool. The cases below are therefore mostly about codes this code has
    never seen; the live shape is the easy one.
    """

    async def classify(codes: list[str]) -> UsageLimit:
        payload = {
            "success": True,
            "data": {
                "limits": [
                    {
                        "type": "TIME_LIMIT",
                        "unit": 5,
                        "number": 1,
                        "usage": 100,
                        "currentValue": 1,
                        "usageDetails": [{"modelCode": c, "usage": 0} for c in codes],
                    }
                ]
            },
        }
        client = _client_for(payload)
        async with client:
            report = await fetch_usage(client, "zai", api_key="k")
        assert report is not None
        return report.limits[0]

    # The live shape today.
    assert (await classify(["search-prime", "web-reader", "zread"])).tier == "zread"
    # The bucket shrinks to one feature.
    assert (await classify(["zread"])).tier == "zread"
    # Z.AI ADDS a tool code we have never seen. A subset-of-known test fails
    # here and marks the row shared; this must stay a tier.
    added = await classify(["search-prime", "web-reader", "zread", "deep-research"])
    assert added.tier == "zread"
    assert added.shared is False
    # Z.AI RENAMES every feature code. Still no chat model, so still a tier.
    renamed = await classify(["search-prime-v2", "web-reader-v2", "zread-v2"])
    assert renamed.tier == "zread"
    assert renamed.shared is False

    # An account-wide cap whose breakdown mentions a feature alongside real chat
    # models is NOT the feature bucket, and must stay shared or the health check
    # that should gate the account skips it.
    mixed = await classify(["glm-5.3", "zread"])
    assert mixed.tier == ""
    assert mixed.shared is True
    # A cap denominated in a model that does not exist yet: the shape test has
    # to catch it, because the registry cannot.
    future = await classify(["glm-6.0", "zread"])
    assert future.tier == ""
    assert future.shared is True
    # Chat models only, and no breakdown at all: both the plain request cap.
    assert (await classify(["glm-5.3", "glm-4.6"])).shared is True
    # Vendor strings are not identifiers we mint, so a capitalised or padded
    # model id is still a model id — a classification that turned on case would
    # silently demote a real account cap to a tier.
    assert (await classify(["GLM-5.3", "zread"])).shared is True
    assert (await classify([" glm-5.3 ", "zread"])).shared is True
    plain = await classify([])
    assert plain.tier == ""
    assert plain.shared is True


@pytest.mark.asyncio
async def test_zai_labels_fit_the_usage_panel_label_column() -> None:
    """A label the panel cannot render is a window the user can never read.

    The panel caps its label column at a third of its content width, and the
    panel itself is capped at ``PANEL_MAX_WIDTH``, so the ceiling is a constant
    24 cells no matter how wide the terminal is. `Zread feature quota (1 month)`
    needed 31 and truncated mid-parenthesis at EVERY width, leaving the only row
    on the panel whose reset period was unavailable. Pinned here rather than in
    the TUI tests because the label is written in this module.
    """
    from rich.cells import cell_len

    from local_operator.tui.widgets.usage_panel import (
        PANEL_MAX_WIDTH,
        PANEL_PADDING_CELLS,
        TIER_INDENT,
    )

    cap = max(12, (PANEL_MAX_WIDTH - PANEL_PADDING_CELLS) // 3)
    client = _client_for(_ZAI_QUOTA_PAYLOAD)
    async with client:
        report = await fetch_usage(client, "zai", api_key="zai-key")
    assert report is not None
    for limit in report.limits:
        width = cell_len(limit.label) + (len(TIER_INDENT) if limit.tier else 0)
        assert width <= cap, f"{limit.label!r} needs {width} cells, panel caps at {cap}"


@pytest.mark.asyncio
async def test_zai_business_failure_on_a_200_is_not_a_report() -> None:
    """The envelope reports business-level failure with an HTTP 200, so a
    successful transport is not by itself a usable quota reading.

    The payload deliberately carries a FULL, well-formed ``data.limits`` block
    that would parse into real rows if it were trusted. An unsuccessful body
    with no ``data`` key would bail at the later structural guard instead, so
    the test would pass with the envelope check deleted and prove nothing.
    """
    client = _client_for(
        {
            "code": 401,
            "msg": "unauthorized",
            "success": False,
            "data": _ZAI_QUOTA_PAYLOAD["data"],
        }
    )
    async with client:
        report = await fetch_usage(client, "zai", api_key="bad")
    assert report is None


@pytest.mark.asyncio
async def test_an_unavailable_deepseek_account_says_so() -> None:
    """A zero balance and a suspended account look identical in the numbers."""
    payload = {"is_available": False, "balance_infos": [{"currency": "USD", "total_balance": 0.0}]}
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "deepseek", api_key="sk-ds")
    assert report is not None
    assert report.notes == "account not available for requests"


#: A verbatim ``/api/oauth/usage`` body, captured from the live endpoint on
#: 2026-08-08 with a real Claude subscription token (200 OK). Trimmed only of
#: the dozen always-null internal codename buckets it also carries.
#:
#: The shape is the whole point of pinning it. The previous fixture was written
#: by hand against ``{"used": ..., "limit": ...}`` keys that this endpoint has
#: NEVER sent — it reports ``utilization`` as a percent — so the parser and its
#: test agreed with each other and disagreed with Anthropic, and `/usage` said
#: "no usage data" for an account whose weekly window was at 100%.
ANTHROPIC_USAGE_BODY = {
    "five_hour": {"utilization": 2.0, "resets_at": "2026-08-08T20:19:59.196141+00:00"},
    "seven_day": {"utilization": 100.0, "resets_at": "2026-08-12T03:59:59.196163+00:00"},
    "seven_day_opus": None,
    "seven_day_sonnet": None,
    "limits": [
        {
            "kind": "session",
            "percent": 2,
            "resets_at": "2026-08-08T20:19:59.196141+00:00",
            "scope": None,
            "is_active": False,
        },
        {
            "kind": "weekly_all",
            "percent": 100,
            "resets_at": "2026-08-12T03:59:59.196163+00:00",
            "scope": None,
            "is_active": True,
        },
        {
            "kind": "weekly_scoped",
            "percent": 0,
            "resets_at": None,
            "scope": {"model": {"id": None, "display_name": "Fable"}},
            "is_active": False,
        },
    ],
    "extra_usage": {
        "is_enabled": False,
        "monthly_limit": 20000,
        "used_credits": 0.0,
        "currency": "USD",
        "decimal_places": 2,
        "disabled_reason": "out_of_credits",
    },
    "spend": {
        "used": {"amount_minor": 0, "currency": "USD", "exponent": 2},
        "limit": {"amount_minor": 20000, "currency": "USD", "exponent": 2},
        "enabled": False,
        "disabled_reason": "out_of_credits",
    },
}


@pytest.mark.asyncio
async def test_anthropic_oauth_reads_utilization_not_used_limit() -> None:
    """The regression: a 200 with full data rendered an empty table.

    Anthropic reports each window as a ``utilization`` PERCENT. The fetcher
    looked for ``used``/``limit``, found neither, skipped every bucket and
    returned None — indistinguishable, at the `/usage` surface, from a provider
    with no endpoint at all.
    """
    client = _client_for(ANTHROPIC_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["anthropic:5h"].amount.fraction() == pytest.approx(0.02)
    assert by_id["anthropic:7d"].amount.fraction() == pytest.approx(1.0)
    assert by_id["anthropic:7d"].effective_status() == "exhausted"


@pytest.mark.asyncio
async def test_anthropic_reads_model_scoped_weekly_caps_from_limits_array() -> None:
    """As of mid-2026 the legacy per-model buckets are permanently null and the
    scoped caps arrive only as ``weekly_scoped`` entries naming the family in
    ``scope.model.display_name``. Reading the legacy keys alone lost them all."""
    client = _client_for(ANTHROPIC_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    fable = next(limit for limit in report.limits if limit.tier == "fable")
    assert fable.label == "7 day (Fable)"
    assert fable.amount.fraction() == pytest.approx(0.0)
    # A per-model cap is NOT account-wide: a 100% Fable row must not read as an
    # account that can no longer serve any request.
    assert fable.shared is False
    assert next(lim for lim in report.limits if lim.id == "anthropic:7d").shared is True


@pytest.mark.asyncio
async def test_anthropic_does_not_list_each_window_twice() -> None:
    """``limits[]`` repeats ``five_hour``/``seven_day`` as ``session``/
    ``weekly_all``. Reading both unconditionally listed every window twice."""
    client = _client_for(ANTHROPIC_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    ids = [limit.id for limit in report.limits]
    assert len(ids) == len(set(ids)), ids
    assert ids == ["anthropic:5h", "anthropic:7d", "anthropic:7d:fable"]


@pytest.mark.asyncio
async def test_anthropic_falls_back_to_the_generic_limits_array() -> None:
    """The named buckets are the primary source; the generic array is the
    fallback for an account that only gets the newer shape."""
    payload = {
        "five_hour": None,
        "seven_day": None,
        "limits": [
            {"kind": "session", "percent": 40, "resets_at": None},
            {"kind": "weekly_all", "percent": 12, "resets_at": None},
        ],
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["anthropic:5h"].amount.fraction() == pytest.approx(0.4)
    assert by_id["anthropic:7d"].amount.fraction() == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_anthropic_parses_the_reset_timestamp_into_a_countdown() -> None:
    """The panel renders "resets in 3h24m", which needs a number rather than the
    vendor's ISO string."""
    client = _client_for(ANTHROPIC_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    five_hour = next(limit for limit in report.limits if limit.id == "anthropic:5h")
    assert five_hour.resets_at_ms == 1786220399196
    # One hour before the reset, the countdown is an hour — not the raw epoch.
    assert five_hour.resets_in_ms(1786220399196 - 3_600_000) == 3_600_000
    # A window that already rolled over reports nothing rather than a negative.
    assert five_hour.resets_in_ms(1786220399196 + 1) is None


@pytest.mark.asyncio
async def test_anthropic_reports_a_disabled_credit_meter_as_a_note() -> None:
    """A disabled extra-usage meter is a $0.00/$200.00 row that reads as spare
    headroom the account cannot draw on. Naming the reason is what tells a user
    at 100% weekly whether waiting is their only option."""
    client = _client_for(ANTHROPIC_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    assert [limit for limit in report.limits if limit.id == "anthropic:extra"] == []
    assert report.notes == "extra usage disabled — out of credits"


@pytest.mark.asyncio
async def test_anthropic_reports_an_enabled_credit_meter_as_a_limit() -> None:
    """Enabled, it is real spendable headroom and belongs in the table — in
    dollars, decoded from the minor units the endpoint quotes."""
    payload = {
        "five_hour": {"utilization": 10.0, "resets_at": None},
        "spend": {
            "used": {"amount_minor": 1250, "currency": "USD", "exponent": 2},
            "limit": {"amount_minor": 20000, "currency": "USD", "exponent": 2},
            "enabled": True,
        },
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    extra = next(limit for limit in report.limits if limit.id == "anthropic:extra")
    assert extra.amount.used == pytest.approx(12.50)
    assert extra.amount.limit == pytest.approx(200.00)
    assert extra.amount.unit == "usd"
    assert report.notes is None


@pytest.mark.asyncio
async def test_anthropic_drops_a_bucket_that_reported_no_number() -> None:
    """A ``resets_at``-only bucket is not zero usage: rendering it as an empty
    bar claims nothing has been spent when nothing was said."""
    payload = {
        "five_hour": {"resets_at": "2026-08-08T20:19:59+00:00"},
        "seven_day": {"utilization": 5.0, "resets_at": None},
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "anthropic", access_token="tok")
    assert report is not None
    assert [limit.id for limit in report.limits] == ["anthropic:7d"]


#: A real ``/backend-api/wham/usage`` body for a ChatGPT Pro account, matching
#: the live response verified on 2026-08-08 (windows nested under ``rate_limit``,
#: per-model caps under ``additional_rate_limits``).
#:
#: The previous fixture invented top-level ``primary``/``secondary`` keys with a
#: ``window_minutes`` field, none of which this endpoint sends.
OPENAI_USAGE_BODY = {
    "plan_type": "pro",
    "rate_limit": {
        "allowed": True,
        "limit_reached": False,
        "primary_window": {
            "used_percent": 4,
            "limit_window_seconds": 17940,
            "reset_at": 2_000_000_000,
        },
        "secondary_window": {
            "used_percent": 1,
            "limit_window_seconds": 604_740,
            "reset_at": 2_000_500_000,
        },
    },
    "additional_rate_limits": [
        {
            "limit_name": "GPT-5.3-Codex-Spark",
            "metered_feature": "codex_bengalfox",
            "rate_limit": {
                "primary_window": {
                    "used_percent": 17,
                    "limit_window_seconds": 18000,
                    "reset_at": 2_000_001_000,
                }
            },
        }
    ],
}


@pytest.mark.asyncio
async def test_openai_reads_windows_nested_under_rate_limit() -> None:
    """The regression: the windows live under ``rate_limit`` as
    ``primary_window``/``secondary_window``. Reading top-level ``primary`` found
    nothing, so a logged-in ChatGPT Pro account reported no usage at all."""
    client = _client_for(OPENAI_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "openai", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["openai:primary"].amount.fraction() == pytest.approx(0.04)
    assert by_id["openai:secondary"].amount.fraction() == pytest.approx(0.01)
    # The window is quoted as a DURATION in seconds, not a name: 17940s is the
    # five hour window a minute short, and 604740s the seven day one.
    assert by_id["openai:primary"].window == "5 hour"
    assert by_id["openai:secondary"].window == "7 day"
    assert report.notes == "plan: pro"


@pytest.mark.asyncio
async def test_openai_reports_per_model_caps_as_tier_rows() -> None:
    """``additional_rate_limits`` carries the Spark cap. It gates one model, so
    it must not read as an account-wide window the way the plan windows do."""
    client = _client_for(OPENAI_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "openai", access_token="tok")
    assert report is not None
    spark = next(limit for limit in report.limits if limit.tier == "spark")
    assert spark.amount.fraction() == pytest.approx(0.17)
    assert spark.label == "5 hour (GPT-5.3-Codex-Spark)"
    assert spark.shared is False
    assert next(lim for lim in report.limits if lim.id == "openai:primary").shared is True


@pytest.mark.asyncio
async def test_openai_accepts_both_reset_encodings() -> None:
    """``reset_at`` arrives in seconds or milliseconds, and some windows send a
    relative ``reset_after_seconds`` instead. A countdown read off the wrong one
    is out by a factor of a thousand."""
    payload = {
        "rate_limit": {
            "primary_window": {"used_percent": 5, "reset_at": 2_000_000_000},
            "secondary_window": {"used_percent": 5, "reset_after_seconds": 3600},
        }
    }
    client = _client_for(payload)
    async with client:
        report = await fetch_usage(client, "openai", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["openai:primary"].resets_at_ms == 2_000_000_000_000
    relative = by_id["openai:secondary"].resets_at_ms
    assert relative is not None
    # Roughly an hour out, allowing for the clock read inside the fetcher.
    assert 3_500_000 <= relative - time.time() * 1000 <= 3_600_000


#: A real ``/coding/v1/usages`` body. The plan total is at ``usage``; each window
#: splits its numbers across ``detail`` and its length across ``window``, and
#: every number is a STRING.
KIMI_USAGE_BODY = {
    "usage": {
        "limit": "100",
        "used": "28",
        "remaining": "72",
        "resetTime": "2026-07-21T07:43:35.355947Z",
    },
    "limits": [
        {
            "window": {"duration": 300, "timeUnit": "TIME_UNIT_MINUTE"},
            "detail": {
                "limit": "100",
                "remaining": "100",
                "resetTime": "2026-07-18T05:43:35.355947Z",
            },
        }
    ],
}


@pytest.mark.asyncio
async def test_kimi_oauth_reads_the_top_level_usage_and_limits() -> None:
    """The regression: the rows are at the top level, not under a ``data``
    envelope, so the fetcher returned None before parsing anything."""
    client = _client_for(KIMI_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "kimi", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["kimi:total"].label == "Total quota"
    assert by_id["kimi:total"].amount.fraction() == pytest.approx(0.28)
    # 300 minutes is the plan's "5h" window; `300m limit` is the same number
    # said in a way no Kimi user would recognise.
    assert by_id["kimi:0"].label == "5h limit"


@pytest.mark.asyncio
async def test_kimi_derives_used_from_limit_and_remaining() -> None:
    """The 5h window arrives as ``{"limit": "100", "remaining": "100"}`` with no
    ``used``. Requiring ``used`` dropped the row entirely."""
    client = _client_for(KIMI_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "kimi", access_token="tok")
    assert report is not None
    five_hour = next(limit for limit in report.limits if limit.id == "kimi:0")
    assert five_hour.amount.used == pytest.approx(0.0)
    assert five_hour.amount.fraction() == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_kimi_takes_the_window_reset_from_the_limit_detail() -> None:
    """Kimi puts ``resetTime`` on the detail while ``window`` carries only the
    duration, so a window-only read left the 5h row with no countdown."""
    client = _client_for(KIMI_USAGE_BODY)
    async with client:
        report = await fetch_usage(client, "kimi", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["kimi:0"].resets_at_ms == 1_784_353_415_355
    assert by_id["kimi:total"].resets_at_ms == 1_784_619_815_355


#: The legacy SuperGrok weekly billing shape (``?format=credits``). Amounts are
#: WRAPPED objects, which is what the old bare-number read lost.
XAI_WEEKLY_BODY = {
    "config": {
        "currentPeriod": {
            "start": "2026-08-01T00:00:00Z",
            "end": "2026-08-08T00:00:00Z",
            "type": "PERIOD_TYPE_WEEK",
        },
        "creditUsagePercent": 42,
        "productUsage": [{"product": "GrokBuild", "usagePercent": 10}],
        "onDemandCap": {"val": 100},
        "onDemandUsed": {"val": 25},
    }
}

#: The unified-billing monthly shape, served on the BARE billing URL.
XAI_MONTHLY_BODY = {
    "config": {
        "billingPeriodStart": "2026-08-01T00:00:00Z",
        "billingPeriodEnd": "2026-09-01T00:00:00Z",
        "monthlyLimit": {"val": 300},
        "used": {"val": 120},
    }
}


@pytest.mark.asyncio
async def test_xai_weekly_credits_and_per_product_rows() -> None:
    """A per-product row is a tier row: spending Grok Build does not stop the
    rest of the subscription."""
    client = _client_for(XAI_WEEKLY_BODY)
    async with client:
        report = await fetch_usage(client, "xai", access_token="tok")
    assert report is not None
    by_id = {limit.id: limit for limit in report.limits}
    assert by_id["xai:credits:1w"].amount.fraction() == pytest.approx(0.42)
    assert by_id["xai:credits:1w"].shared is True
    product = by_id["xai:product:grokbuild:1w"]
    assert product.amount.fraction() == pytest.approx(0.10)
    assert product.tier == "grokbuild"
    assert product.shared is False
    assert by_id["xai:credits:1w"].resets_at_ms == 1_786_147_200_000


@pytest.mark.asyncio
async def test_xai_unwraps_the_val_object_its_amounts_arrive_in() -> None:
    """The regression: ``monthlyLimit``/``used`` are ``{"val": N}``. Read as bare
    numbers they coerced to None, so a unified-billing account rendered a label
    with no numbers — a row that looks like a working feature and says nothing."""
    client = _client_for(XAI_MONTHLY_BODY)
    async with client:
        report = await fetch_usage(client, "xai", access_token="tok")
    assert report is not None
    monthly = next(limit for limit in report.limits if limit.id == "xai:included:1mo")
    assert monthly.amount.used == pytest.approx(120.0)
    assert monthly.amount.limit == pytest.approx(300.0)
    assert monthly.amount.fraction() == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_xai_reports_the_on_demand_cap_that_backs_an_empty_plan() -> None:
    """Whether on-demand headroom exists is the difference between "wait for the
    reset" and "keep working", so it is a row rather than a footnote."""
    client = _client_for(XAI_WEEKLY_BODY)
    async with client:
        report = await fetch_usage(client, "xai", access_token="tok")
    assert report is not None
    on_demand = next(limit for limit in report.limits if limit.id == "xai:on-demand")
    assert on_demand.amount.fraction() == pytest.approx(0.25)


@pytest.mark.asyncio
async def test_xai_falls_back_to_the_monthly_url_for_a_unified_account() -> None:
    """xAI serves two shapes on one endpoint: the credits URL answers with no
    weekly percentage for a unified account, and the monthly quota is only on the
    bare URL. Probing one alone leaves half the accounts with an empty report."""
    bodies = {"credits": {"config": {"isUnifiedBillingUser": True}}, "bare": XAI_MONTHLY_BODY}
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        key = "credits" if "format=credits" in str(request.url) else "bare"
        return httpx.Response(200, json=bodies[key])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(client, "xai", access_token="tok")
    assert report is not None
    assert any("format=credits" in url for url in urls)
    assert any("format=credits" not in url for url in urls)
    assert [limit.id for limit in report.limits] == ["xai:included:1mo"]


@pytest.mark.asyncio
async def test_xai_asks_once_when_the_weekly_shape_answers() -> None:
    """The common case stays one request: a second probe per `/usage` is a
    round trip spent to learn nothing."""
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        return httpx.Response(200, json=XAI_WEEKLY_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(client, "xai", access_token="tok")
    assert report is not None
    assert len(urls) == 1, urls


@pytest.mark.asyncio
async def test_oauth_provider_without_token_returns_none() -> None:
    # OAuth-only provider with no access token -> None (no report).
    client = _client_for({})
    async with client:
        report = await fetch_usage(client, "anthropic", api_key="sk-key")
    assert report is None


@pytest.mark.asyncio
async def test_unknown_provider_returns_none() -> None:
    client = _client_for({})
    async with client:
        report = await fetch_usage(client, "nonsense", api_key="k")
    assert report is None


def test_usage_supported() -> None:
    assert usage_supported("openrouter") is True
    assert usage_supported("deepseek") is True
    assert usage_supported("nonsense") is False
    # `zai` previously had a working fetcher and no ProviderDefinition, so no code
    # path could ever supply its credential. It is supported now because the
    # registry entry, the credential path and the fetcher landed together.
    assert usage_supported("zai") is True


def test_usage_kinds_distinguishes_no_endpoint_from_no_credential() -> None:
    """Those look identical in an empty table, and only the second is actionable."""
    from local_operator.providers.usage import usage_kinds

    assert usage_kinds("anthropic") == (True, False)  # OAuth only
    assert usage_kinds("openrouter") == (False, True)  # API key only
    assert usage_kinds("kimi") == (True, True)  # both routes
    assert usage_kinds("google") == (False, False)  # no endpoint at all


def test_the_advertised_set_is_the_dispatch_table() -> None:
    """`USAGE_PROVIDERS` is the set that gates the UI, so a hand-maintained copy of
    the dispatch table drifting one way silently unreaches a provider with a
    working fetcher and the other way advertises one no fetcher serves. Derived,
    it cannot drift.

    (`OAUTH_USAGE_PROVIDERS` was deleted rather than derived: the question it
    answered — "OAuth-only?" — is `usage_kinds`, which is what the one caller
    already calls.)
    """
    from local_operator.providers import usage as usage_mod

    assert usage_mod.USAGE_PROVIDERS == frozenset(usage_mod._FETCHERS)
    # 12 since `zai-oauth` joined: the Z.AI browser sign-in is its own provider
    # id, like `xai-oauth`, and needs its own route or a signed-in account
    # reports no usage at all.
    assert usage_mod.USAGE_PROVIDERS is not None and len(usage_mod.USAGE_PROVIDERS) == 12
    assert not hasattr(usage_mod, "OAUTH_USAGE_PROVIDERS")


@pytest.mark.asyncio
async def test_failure_is_not_a_crash() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    async with client:
        report = await fetch_usage(client, "openrouter", api_key="k")
    assert report is None


def test_usage_amount_fraction_precedence() -> None:
    # Explicit used_fraction wins over derived used/limit.
    a = UsageAmount(used=50, limit=100, used_fraction=0.25)
    assert a.fraction() == pytest.approx(0.25)
    # Derive from used/limit when no explicit fraction.
    b = UsageAmount(used=25, limit=100)
    assert b.fraction() == pytest.approx(0.25)
    # Derive from used+remaining when limit absent.
    c = UsageAmount(used=10, remaining=90)
    assert c.fraction() == pytest.approx(0.1)
    # Nothing measurable -> None.
    assert UsageAmount(used=10).fraction() is None


def test_usage_amount_status_thresholds() -> None:
    assert UsageAmount(used_fraction=0.5).status() == "ok"
    assert UsageAmount(used_fraction=0.9).status() == "warning"
    assert UsageAmount(used_fraction=1.0).status() == "exhausted"
    assert UsageAmount().status() == "unknown"


def test_usage_limit_effective_status_defaults_to_amount() -> None:
    limit = UsageLimit(id="x", label="x", amount=UsageAmount(used_fraction=0.95))
    assert limit.effective_status() == "warning"
    explicit = UsageLimit(id="y", label="y", amount=UsageAmount(), status="ok")
    assert explicit.effective_status() == "ok"


# ---------------------------------------------------------------------------
# QwenCloud Token Plan (alibaba-token-plan): management-OAuth BSS gateway.
# ---------------------------------------------------------------------------


def _qwencloud_client(
    responses_by_action: dict[str, dict[str, Any]], requests: list[httpx.Request] | None = None
) -> httpx.AsyncClient:
    """Route official-gateway posts by their ``action``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        body = json.loads(request.content) if request.content else {}
        canned = responses_by_action.get(str(body.get("action")), {"code": "200", "data": {}})
        return httpx.Response(200, json=canned)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _qwencloud_fr_rows(commodity: str) -> dict[str, Any]:
    if commodity == "sfm_tokenplanpersonal_dp_intl":
        return {
            "code": "200",
            "data": {
                "Data": [
                    {
                        "Status": {"Code": "valid"},
                        "InitCapacityBaseValue": 10_000,
                        "CurrCapacityBaseValue": 2_500,
                        "EndTime": 1_800_100_000_000,
                        "TemplateName": "Token Plan Individual Standard",
                    }
                ]
            },
        }
    if commodity == "sfm_tokenplanteamsaddon_dp_intl":
        return {
            "code": "200",
            "data": {"Data": [{"Status": "valid", "CurrCapacityBaseValue": 20_000}]},
        }
    return {"code": "200", "data": {"Data": []}}


@pytest.mark.asyncio
async def test_qwencloud_token_plan_personal_window_and_packs() -> None:
    """The personal commodity is the 7-day credit window; add-on instances are
    Credit Packs outside every window. QwenCloud documents no 5-hour Token Plan
    window, so none may appear even though the console gateway has one."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if body.get("action") == "QuerySubscriptionGray":
            return httpx.Response(200, json={"code": "200", "data": {"IsGray": False}})
        if body.get("action") == "DescribeFrInstances":
            return httpx.Response(200, json=_qwencloud_fr_rows(body["params"]["CommodityCode"]))
        return httpx.Response(200, json={"code": "200", "data": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(
            client,
            "alibaba-token-plan",
            access_token="sk-sp-wire-key",
            oauth_creds={"access": "mgmt-token"},
        )

    assert report is not None
    assert report.provider == "alibaba-token-plan"
    assert [(limit.id, limit.label) for limit in report.limits] == [
        ("credits-7d", "7 Day Credits"),
        ("credits-packs", "Credit Packs"),
    ]
    window = report.limits[0]
    assert window.amount.used == 7_500
    assert window.amount.limit == 10_000
    assert window.amount.remaining == 2_500
    assert window.resets_at_ms == 1_800_100_000_000
    assert window.shared is True
    assert report.limits[1].amount.remaining == 20_000

    # Every gateway call carries the MANAGEMENT token, not the wire key.
    assert len(requests) == 4  # gray + addon + teams + personal
    for request in requests:
        assert request.url == "https://cli.qwencloud.com/data/v2/api.json"
        assert request.headers["Authorization"] == "Bearer mgmt-token"
    gray_body = json.loads(requests[0].content)
    assert gray_body == {
        "product": "BssOpenAPI-V3",
        "action": "QuerySubscriptionGray",
        "region": "ap-southeast-1",
        "params": {},
    }
    personal_bodies = [
        json.loads(request.content)
        for request in requests
        if json.loads(request.content)["action"] == "DescribeFrInstances"
        and json.loads(request.content)["params"]["CommodityCode"]
        == "sfm_tokenplanpersonal_dp_intl"
    ]
    assert personal_bodies[0]["params"] == {
        "Group": "tokenPlan",
        "CommodityCode": "sfm_tokenplanpersonal_dp_intl",
        "PageNum": "1",
        "PageSize": "10",
    }


@pytest.mark.asyncio
async def test_qwencloud_token_plan_team_seat_summary_is_monthly() -> None:
    """Seat-migrated (gray) accounts answer through GetSeatSubscriptionSummary."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("action") == "QuerySubscriptionGray":
            return httpx.Response(200, json={"code": "200", "data": {"IsGray": True}})
        if body.get("action") == "GetSeatSubscriptionSummary":
            return httpx.Response(
                200,
                json={
                    "code": "200",
                    "data": {
                        "Data": {
                            "SubscriptionGroupList": [
                                {"EquityList": [{"TotalValue": "25000", "SurplusValue": "10000"}]}
                            ],
                            "EndTime": 1_800_200_000_000,
                            "PlanName": "Token Plan Team Pro",
                        }
                    },
                },
            )
        return httpx.Response(200, json={"code": "200", "data": {"Data": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(
            client,
            "alibaba-token-plan",
            access_token="sk-sp-wire-key",
            oauth_creds={"access": "mgmt-token"},
        )

    assert report is not None
    assert [(limit.id, limit.amount.used, limit.amount.limit) for limit in report.limits] == [
        ("credits-monthly", 15_000.0, 25_000.0)
    ]
    assert report.limits[0].resets_at_ms == 1_800_200_000_000


@pytest.mark.asyncio
async def test_qwencloud_token_plan_needs_the_management_token_from_the_raw_row() -> None:
    """``access_token`` is the wire-mapped sk-sp key; without the raw row's
    ``access`` there is nothing valid to authenticate with, and no request may
    leave the process."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"code": "200", "data": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(
            client, "alibaba-token-plan", access_token="sk-sp-wire-key", oauth_creds=None
        )
    assert report is None
    assert requests == []


@pytest.mark.asyncio
async def test_qwencloud_token_plan_fails_closed_on_console_need_login() -> None:
    """A dead management token gets code "ConsoleNeedLogin" inside HTTP 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"code": "ConsoleNeedLogin", "message": "You need to log in."}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    async with client:
        report = await fetch_usage(
            client,
            "alibaba-token-plan",
            access_token="sk-sp-wire-key",
            oauth_creds={"access": "expired-mgmt"},
        )
    assert report is None


class TestZaiSignInReportsUsage:
    """`/provider` advertised Z.AI usage and then showed nothing to anyone who
    signed in rather than pasting a key.

    The browser sign-in ends by minting an ordinary durable coding-plan key and
    stores it in `access`, so the OAuth slot wants the SAME fetcher the api-key
    slot uses -- an empty OAuth slot made `fetch_usage` return None.
    """

    def test_both_credential_kinds_route_to_the_quota_fetcher(self) -> None:
        from local_operator.providers.usage import _FETCHERS

        assert _FETCHERS["zai"] == ("zai-quota", "zai-quota")
        # The login flavour resolves too: it is a separate provider id.
        assert _FETCHERS["zai-oauth"] == ("zai-quota", "zai-quota")

    def test_a_signed_in_account_is_reported_as_capable_of_usage(self) -> None:
        from local_operator.providers.usage import usage_kinds

        oauth_kind, api_key_kind = usage_kinds("zai")
        assert oauth_kind is not None, "a signed-in Z.AI account reports no usage"
        assert api_key_kind is not None
