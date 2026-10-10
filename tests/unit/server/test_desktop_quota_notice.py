"""``GET /v1/desktop/quota-notice`` end to end, over an isolated HOME.

WHY THESE EXIST. The route is the contract the desktop renderer will read
(PR3 starts against it), so the tests drive the REAL route through a real
FastAPI app: real AuthStore, real usage cache on disk, real lease path, and
provider usage endpoints stubbed at their fetcher functions — the design's
"small stub for provider usage endpoints". The stubs are the ONLY network
substitution; everything between the HTTP request and the fetcher call is the
production path, which is what makes the cache-hit, refresh-timeout and
401/422 cases real evidence rather than unit assertions.

Isolation follows AGENTS.md: ``HOME`` is redirected (the usage cache derives
from it independently of ``LOCAL_OPERATOR_CONFIG_DIR``), provider env keys
are cleared, and no test reads or writes the operator's stores. The response
never carries an ``identity`` — one case below asserts the absence directly,
because a leak here would be the credential's email on the wire.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import local_operator.providers.usage as usage_module
from local_operator.config import ConfigManager
from local_operator.model import discovery as discovery_module
from local_operator.providers import radient_recovery as rr
from local_operator.providers.controller import ProviderController
from local_operator.providers.quota_notice import report_is_fresh
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import USAGE_REPORT_TTL_MS
from local_operator.server.routes import auth, desktop_quota

TOKEN = "desktop-quota-notice-token"

#: Every provider key that would change a credential decision here. Cleared so
#: the test host's shell cannot decide what these tests measure.
PROVIDER_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "XAI_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "RADIENT_API_KEY",
    "OPENROUTER_API_KEY",
)

pytestmark = pytest.mark.asyncio


class Harness:
    """The app, the client and the handles the cases need."""

    def __init__(
        self,
        client: AsyncClient,
        app: FastAPI,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.client = client
        self.app = app
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch

    @property
    def store(self):
        return self.app.state.desktop_auth.store

    async def notice(
        self, *, provider: str | None = None, model: str | None = None, refresh: bool = False
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if provider is not None:
            params["provider"] = provider
        if model is not None:
            params["model"] = model
        if refresh:
            params["refresh"] = "true"
        response = await self.client.get("/v1/desktop/quota-notice", params=params)
        assert response.status_code == 200, response.text
        return response.json()["result"]


@pytest_asyncio.fixture
async def quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Harness]:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in PROVIDER_ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    rr.reset_recovery_cache()
    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(desktop_quota.router)
    app.state.config_manager = ConfigManager(tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        # One probe request builds the DesktopAuth (and its store) lazily, the
        # same way it is built in production.
        await client.get("/v1/desktop/quota-notice")
        yield Harness(client, app, tmp_path, monkeypatch)
    if getattr(app.state, "desktop_auth", None):
        await app.state.desktop_auth.close()
    rr.reset_recovery_cache()


def _balance_report(provider: str, remaining: float) -> UsageReport:
    return UsageReport(
        provider=provider,
        limits=[
            UsageLimit(
                id=f"{provider}:balance",
                label="Credit balance (USD)",
                amount=UsageAmount(remaining=remaining),
                window="lifetime",
                shared=True,
            )
        ],
    )


def _window_report(
    provider: str, used: float, total: float, *, reset_in_ms: int | None
) -> UsageReport:
    now_ms = int(time.time() * 1000)
    return UsageReport(
        provider=provider,
        limits=[
            UsageLimit(
                id=f"{provider}:window",
                label="5 hour",
                amount=UsageAmount(used=used, limit=total, used_fraction=used / total),
                window="5 hour",
                resets_at_ms=now_ms + reset_in_ms if reset_in_ms is not None else None,
                shared=True,
            )
        ],
    )


def _stub(monkeypatch: pytest.MonkeyPatch, name: str, fn) -> None:
    """Replace one fetcher on the usage module.

    ``_run_fetcher`` resolves its targets as module globals at call time, so
    this is the dispatch point the controller actually uses — not a bypass.
    """
    monkeypatch.setattr(usage_module, name, fn)


def _plant_listing(root: Path, provider: str, rows: list[dict[str, Any]]) -> None:
    """Write a provider's cached listing document the way the app writes it.

    The capture stamp is the module's own (``listing_capture_version``): a
    document from another stamp reads as unusable, so a test that hard-coded
    a number would measure the registry fallback instead of the listing.
    ``root`` is the isolated HOME — the reader resolves its cache under
    ``~/.local-operator/cache``.
    """
    cache = root / ".local-operator" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{provider}.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": discovery_module.listing_capture_version(provider),
                    "models": rows,
                },
            }
        ),
        encoding="utf-8",
    )


def _age_cache_row(root: Path, provider: str, by_ms: int) -> None:
    """Move a cached row back in time — payload and columns on one clock.

    The row's ``expires_at_ms`` (which decides cache freshness) and each
    embedded report's ``fetched_at`` (which the route and the verdict read)
    must move together; moving only one leaves a payload that still claims to
    be seconds old, and every age check in the path would rightly believe it.
    How far back is the caller's choice: past ``REFRESH_FLOOR_MS`` to make a
    forced refresh actually probe, past ``USAGE_REPORT_TTL_MS`` to make the
    row stale. Sleeping instead is not a test.
    """
    conn = sqlite3.connect(root / "usage_cache.db")
    row = conn.execute(
        "SELECT payload FROM usage_reports WHERE provider = ?", (provider,)
    ).fetchone()
    assert row is not None, f"no cache row for {provider} to age"
    payload = json.loads(row[0])
    for report in payload:
        report["fetched_at"] -= by_ms
    conn.execute(
        "UPDATE usage_reports SET payload = ?, fetched_at_ms = fetched_at_ms - ?, "
        "expires_at_ms = expires_at_ms - ?, updated_at_ms = updated_at_ms - ? "
        "WHERE provider = ?",
        (json.dumps(payload), by_ms, by_ms, by_ms, provider),
    )
    conn.commit()
    conn.close()


async def test_depleted_balance_live_then_cached(quota: Harness) -> None:
    """The real path: store credential -> stubbed endpoint -> notice; then a
    cache hit with no second fetch."""
    calls = {"n": 0}

    async def fetch_deepseek(client, api_key):
        calls["n"] += 1
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fetch_deepseek)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})

    first = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert first["state"] == "depleted"
    assert first["kind"] == "balance"
    assert first["source"] == "live"
    assert first["provider"] == "deepseek"
    assert first["title"] == "No balance on DeepSeek"
    assert first["body"] == "No balance on DeepSeek — top up at the DeepSeek platform."
    assert first["actions"] == [
        {
            "id": "open_url",
            "label": "Top up at the DeepSeek platform",
            "url": "https://platform.deepseek.com/top_up",
        },
        {"id": "refresh", "label": "I topped up", "url": None},
    ]
    assert first["age_ms"] is not None and first["age_ms"] >= 0
    assert first["checked_at_ms"] > 0
    assert "identity" not in first
    assert "sk-deepseek-test" not in str(first)
    # The cache row is on disk where the next process will find it.
    assert list(quota.tmp_path.glob("**/usage_cache.db"))
    assert calls["n"] == 1

    second = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert second["state"] == "depleted"
    assert second["source"] == "cached"
    assert calls["n"] == 1, "a fresh cache row must answer with no second fetch"


async def test_fetcher_returning_nothing_is_unknown(quota: Harness) -> None:
    async def fetch_deepseek(client, api_key):
        return None

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fetch_deepseek)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})
    result = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert result["state"] == "unknown"
    assert result["title"] == "" and result["body"] == ""


async def test_free_model_suppresses_the_notice(quota: Harness) -> None:
    """A ``:free`` route suppresses the notice through the REAL catalogue row.

    Round-1 M1: this test used to patch ``ProviderController.entry_for`` —
    which hid that ``entry_for`` alone cannot see a listing-priced zero, because
    aggregators ship no static rows. The free row now comes from a CACHED
    LISTING document (the picker's own source), so the production resolution
    path is what fires.
    """
    _plant_listing(
        quota.tmp_path,
        "openrouter",
        [
            {
                "id": "poolside/laguna-s-2.1:free",
                "context_window": 131_072,
                "input_price": 0.0,
                "output_price": 0.0,
                "free": True,
            }
        ],
    )

    async def fetch_openrouter(client, api_key):
        return _balance_report("openrouter", 0.0)

    _stub(quota.monkeypatch, "fetch_openrouter", fetch_openrouter)
    quota.store.upsert_credential("openrouter", {"key": "sk-or-test", "source": "login"})

    result = await quota.notice(provider="openrouter", model="poolside/laguna-s-2.1:free")
    assert result["state"] == "not_applicable"
    assert result["model_free"] is True
    assert result["body"] == ""

    # Anti-vacuity: with the listing document gone the same pair is unknown to
    # the static registry, so the notice must STAND — which proves the listing
    # row above is what suppressed it, not some other path.
    (quota.tmp_path / ".local-operator" / "cache" / "openrouter.listing.json").unlink()
    unsuppressed = await quota.notice(provider="openrouter", model="poolside/laguna-s-2.1:free")
    assert unsuppressed["state"] == "depleted"
    assert unsuppressed["model_free"] is False


async def test_a_routed_meta_model_is_never_free(quota: Harness) -> None:
    """A zero-priced ROUTED row must keep the notice (round-1 M1's control):
    the user's message goes wherever the router sends it, not to the free
    model, so the spent account is real evidence. Same cached-listing seam as
    the free case — prices exactly 0.0/0.0, ``routed: true`` making the
    difference."""
    _plant_listing(
        quota.tmp_path,
        "openrouter",
        [
            {
                "id": "openrouter/auto",
                "context_window": 2_000_000,
                "input_price": 0.0,
                "output_price": 0.0,
                "free": True,
                "routed": True,
            }
        ],
    )

    async def fetch_openrouter(client, api_key):
        return _balance_report("openrouter", 0.0)

    _stub(quota.monkeypatch, "fetch_openrouter", fetch_openrouter)
    quota.store.upsert_credential("openrouter", {"key": "sk-or-test", "source": "login"})

    result = await quota.notice(provider="openrouter", model="openrouter/auto")
    assert result["state"] == "depleted"
    assert result["model_free"] is False
    assert result["title"] == "No balance on OpenRouter"


async def test_a_healthy_radient_user_never_pays_the_me_probe(quota: Harness) -> None:
    """The /me probe runs only when a DEPLETED Radient notice will render its
    sentence (round-1 m3) — a healthy account must not wait behind it."""
    quota.store.upsert_credential(
        "radient",
        {"type": "oauth", "access": "tok-r", "refresh": "ref-r", "email": "r@example.com"},
    )
    probes = {"n": 0}

    async def probe(token: str):
        probes["n"] += 1
        return rr.VerificationFacts(email_verified=True, signup_grant="claimed", grant_amount=5)

    async def healthy(client, access_token):
        return _balance_report("radient", 9.0)

    quota.monkeypatch.setattr(rr, "_probe_verification_async", probe)
    _stub(quota.monkeypatch, "fetch_radient_balance", healthy)

    result = await quota.notice(provider="radient", model="radient/auto")
    assert result["state"] == "unknown"
    assert probes["n"] == 0, "a healthy account must not pay the /me probe"

    # The depleted case still gets the builder's sentence — recomposed after
    # the verdict, from the probe that only this path pays for.
    async def spent(client, access_token):
        return _balance_report("radient", 0.0)

    _stub(quota.monkeypatch, "fetch_radient_balance", spent)
    _age_cache_row(quota.tmp_path, "radient", 20_000)
    spent_result = await quota.notice(provider="radient", model="radient/auto", refresh=True)
    assert spent_result["state"] == "depleted"
    assert probes["n"] == 1
    assert spent_result["body"].startswith("You're out of credits")


async def test_the_controller_is_closed_on_every_path(quota: Harness) -> None:
    """The controller is closed in ``finally`` — success, error and the
    cancelled-refresh path alike (round-1 m1: replacing the close with ``pass``
    went unnoticed)."""
    import asyncio

    closed: list[str] = []
    real_close = ProviderController.close

    def spy(self):
        closed.append("closed")
        return real_close(self)

    quota.monkeypatch.setattr(ProviderController, "close", spy)

    async def fast(client, api_key):
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fast)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})

    ok = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert ok["state"] == "depleted"
    assert len(closed) == 1, "the success path must close the controller"

    response = await quota.client.get(
        "/v1/desktop/quota-notice", params={"provider": "not-a-provider"}
    )
    assert response.status_code == 422
    assert len(closed) == 2, "the 422 path must close the controller too"

    async def slow(client, api_key):
        await asyncio.sleep(0.5)
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", slow)
    quota.monkeypatch.setattr(desktop_quota, "LIVE_REFRESH_BOUND_S", 0.05)
    _age_cache_row(quota.tmp_path, "deepseek", 20_000)
    timed_out = await quota.notice(provider="deepseek", model="deepseek-chat", refresh=True)
    assert timed_out["source"] == "cached"
    assert len(closed) == 3, "the cancelled-refresh path must close the controller"


async def test_the_refresh_floor_skips_a_just_fetched_row(quota: Harness) -> None:
    """``refresh=true`` inside the floor answers from the cache it just read:
    no second upstream request for a focus loop (round-1 m4)."""
    calls = {"n": 0}

    async def fetch_deepseek(client, api_key):
        calls["n"] += 1
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fetch_deepseek)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})

    first = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert first["source"] == "live" and calls["n"] == 1

    forced = await quota.notice(provider="deepseek", model="deepseek-chat", refresh=True)
    assert forced["source"] == "cached"
    assert calls["n"] == 1, "the floor must cut the second probe short"
    assert forced["age_ms"] is not None
    assert forced["age_ms"] < desktop_quota.REFRESH_FLOOR_MS


async def test_forced_refresh_beyond_the_floor_reaches_live(quota: Harness) -> None:
    """A click past the floor re-probes and reports the NEW numbers — the
    "I topped up" flow the force exists for."""

    async def depleted(client, api_key):
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", depleted)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})
    primed = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert primed["state"] == "depleted"

    async def topped_up(client, api_key):
        return _balance_report("deepseek", 12.5)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", topped_up)
    _age_cache_row(quota.tmp_path, "deepseek", 20_000)
    result = await quota.notice(provider="deepseek", model="deepseek-chat", refresh=True)
    assert result["source"] == "live"
    assert result["state"] == "unknown", "the topped-up balance must be re-read, not the cache"


async def test_a_stale_row_served_from_last_good_stays_cached_sourced(quota: Harness) -> None:
    """A fetch that completes WITHOUT new numbers must not relabel the answer
    ``live`` (round-1 m1: forcing ``source = "live"`` went unnoticed).

    The stub fails fast rather than timing out, so ``fetch_usage`` hands back
    the stale last-good payload — non-empty, but nothing in it was refreshed;
    the verdict then keeps quiet because the numbers are stale."""

    async def fast(client, api_key):
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fast)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})
    primed = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert primed["state"] == "depleted"

    _age_cache_row(quota.tmp_path, "deepseek", 10 * 60_000)

    async def failing(client, api_key):
        return None

    _stub(quota.monkeypatch, "fetch_deepseek_balance", failing)
    result = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert result["source"] == "cached", "last-good is a cached answer"
    assert result["state"] == "unknown", "a stale depleted row must not be shown"


async def test_the_route_mirrors_the_verdict_freshness_rule() -> None:
    """One TTL boundary, two readers (round-1 m5): exactly-TTL is FRESH for
    both the verdict's rule and the route's spend decision — the route used to
    spell it ``>= TTL`` against the verdict's ``> TTL``."""
    fetched = 1_000_000
    report = UsageReport(provider="deepseek", fetched_at=fetched)
    at_boundary = fetched + USAGE_REPORT_TTL_MS
    assert report_is_fresh(report, at_boundary) is True
    assert desktop_quota._needs_live([report], at_boundary) is False
    past_boundary = at_boundary + 1
    assert report_is_fresh(report, past_boundary) is False
    assert desktop_quota._needs_live([report], past_boundary) is True


async def test_multi_account_one_healthy_shows_nothing(quota: Harness) -> None:
    """Two logins, one spent: the provider still works, so no notice — and no
    email may appear in the payload that reports this."""
    quota.store.upsert_credential(
        "anthropic", {"access": "tok-a", "refresh": "ref-a", "email": "a@example.com"}
    )
    quota.store.upsert_credential(
        "anthropic", {"access": "tok-b", "refresh": "ref-b", "email": "b@example.com"}
    )

    async def both_spent(client, token):
        return _window_report("anthropic", 100, 100, reset_in_ms=3_600_000)

    async def one_healthy(client, token):
        if token == "tok-a":
            return _window_report("anthropic", 100, 100, reset_in_ms=3_600_000)
        return _window_report("anthropic", 10, 100, reset_in_ms=3_600_000)

    _stub(quota.monkeypatch, "fetch_anthropic_oauth", both_spent)
    both = await quota.notice(provider="anthropic", model="claude-opus-4")
    assert both["state"] == "limit_reached"
    assert both["kind"] == "subscription"
    assert both["resets_at_ms"] is not None
    assert "resets in " in both["body"]
    assert "a@example.com" not in str(both) and "b@example.com" not in str(both)

    _stub(quota.monkeypatch, "fetch_anthropic_oauth", one_healthy)
    # Take the row past the refresh floor so the forced refresh really probes
    # (inside the floor a force answers from cache by design — see the floor
    # tests below).
    _age_cache_row(quota.tmp_path, "anthropic", 20_000)
    mixed = await quota.notice(provider="anthropic", model="claude-opus-4", refresh=True)
    assert mixed["state"] == "ok"
    assert mixed["body"] == ""


async def test_kimi_mixed_credentials_stay_silent(quota: Harness) -> None:
    """An OAuth login plus a live API key: the skipped route is unproven, so
    even a spent coding plan shows nothing — until the key is gone."""
    quota.store.upsert_credential(
        "kimi", {"access": "tok-k", "refresh": "ref-k", "email": "k@example.com"}
    )

    async def spent_plan(client, token):
        return _window_report("kimi", 100, 100, reset_in_ms=3_600_000)

    _stub(quota.monkeypatch, "fetch_kimi_oauth", spent_plan)
    quota.monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test")

    suppressed = await quota.notice(provider="kimi", model="k3")
    assert suppressed["state"] == "unknown"

    quota.monkeypatch.delenv("KIMI_API_KEY")
    _age_cache_row(quota.tmp_path, "kimi", 20_000)
    warning = await quota.notice(provider="kimi", model="k3", refresh=True)
    assert warning["state"] == "limit_reached"
    # The CODING-PLAN console, not the API-key top-up page: the link is
    # resolved from the kind the report shape derived (round-1 M2), and this
    # assertion is what pins it at the route level too.
    assert warning["actions"][0]["url"] == "https://www.kimi.com/code/console"


async def test_radient_body_comes_from_the_shared_recovery_builder(quota: Harness) -> None:
    """Radient's sentence and its action come from the shared module, and the
    /me probe is the only substitution."""
    quota.store.upsert_credential(
        "radient",
        {"type": "oauth", "access": "tok-r", "refresh": "ref-r", "email": "r@example.com"},
    )

    async def probe(token: str):
        assert token == "tok-r"
        return rr.VerificationFacts(email_verified=False, signup_grant="pending", grant_amount=5)

    async def fetch_radient(client, access_token):
        return _balance_report("radient", 0.0)

    quota.monkeypatch.setattr(rr, "_probe_verification_async", probe)
    _stub(quota.monkeypatch, "fetch_radient_balance", fetch_radient)

    expected_facts = rr.RecoveryFacts(
        signed_in=True,
        verification=rr.VerificationFacts(
            email_verified=False, signup_grant="pending", grant_amount=5
        ),
    )
    result = await quota.notice(provider="radient", model="radient/auto")
    assert result["state"] == "depleted"
    assert result["kind"] == "radient"
    assert result["body"] == rr.recovery_line(expected_facts)
    assert result["actions"][0]["id"] == "open_url"
    assert result["actions"][0]["url"] == "https://console.radienthq.com/dashboard/billing"
    assert result["actions"][-1]["id"] == "refresh"


async def test_forced_refresh_failure_keeps_the_cached_row(quota: Harness) -> None:
    """A slow provider must not hold the banner: the bounded refresh times out
    and the still-fresh cache row answers (a forced refresh that fails on a
    FRESH row falls back to last-good)."""
    import asyncio

    async def fast(client, api_key):
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fast)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})
    primed = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert primed["source"] == "live"

    async def slow(client, api_key):
        await asyncio.sleep(0.5)
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", slow)
    quota.monkeypatch.setattr(desktop_quota, "LIVE_REFRESH_BOUND_S", 0.05)
    # Past the refresh floor but inside the TTL: the forced refresh really
    # attempts and is really cut short.
    _age_cache_row(quota.tmp_path, "deepseek", 20_000)
    started = time.monotonic()
    result = await quota.notice(provider="deepseek", model="deepseek-chat", refresh=True)
    elapsed = time.monotonic() - started
    assert result["state"] == "depleted", "the cached row survives a timed-out refresh"
    assert result["source"] == "cached"
    assert elapsed < 0.4, f"the bound must cut the live fetch short (took {elapsed:.2f}s)"


async def test_stale_cache_failed_refresh_is_silent_and_releases_the_lease(
    quota: Harness,
) -> None:
    """The design's stale-cache rule, plus its open question: a STALE depleted
    cache triggers one bounded live fetch, and a failed fetch shows NOTHING
    (balances can be topped up out of band); the cancelled refresh must not
    strand the usage-cache lease (the design's risk list names exactly this)."""
    import asyncio
    import json
    import sqlite3

    async def fast(client, api_key):
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fast)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})
    primed = await quota.notice(provider="deepseek", model="deepseek-chat")
    assert primed["state"] == "depleted"

    # Age the row past the TTL through the cache DB itself: waiting five
    # minutes is not a test. BOTH clocks move — the row's ``expires_at_ms``
    # (which decides cache freshness) and each embedded report's own
    # ``fetched_at`` (which the route and the verdict read). Moving only the
    # column leaves a payload that still claims to be seconds old, and every
    # age check in the path would rightly believe it. The store lives at
    # ``config_dir()/usage_cache.db`` — the fixture's config dir IS tmp_path.
    cache_db = quota.tmp_path / "usage_cache.db"
    conn = sqlite3.connect(cache_db)
    row = conn.execute("SELECT payload FROM usage_reports WHERE provider = 'deepseek'").fetchone()
    payload = json.loads(row[0])
    for report in payload:
        report["fetched_at"] -= 10 * 60_000
    conn.execute(
        "UPDATE usage_reports SET payload = ?, fetched_at_ms = fetched_at_ms - ?, "
        "expires_at_ms = expires_at_ms - ?, updated_at_ms = updated_at_ms - ? "
        "WHERE provider = 'deepseek'",
        (json.dumps(payload), 10 * 60_000, 10 * 60_000, 10 * 60_000),
    )
    conn.commit()
    conn.close()

    async def slow(client, api_key):
        await asyncio.sleep(1.0)
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", slow)
    quota.monkeypatch.setattr(desktop_quota, "LIVE_REFRESH_BOUND_S", 0.2)
    started = time.monotonic()
    call = asyncio.create_task(quota.notice(provider="deepseek", model="deepseek-chat"))

    # Catch the lease while the slow probe holds it. Without this, a "released"
    # assertion below would be equally green if the refresh never took one.
    held = False
    for _ in range(50):
        await asyncio.sleep(0.01)
        conn = sqlite3.connect(cache_db)
        held = bool(conn.execute("SELECT COUNT(*) FROM usage_fetch_leases").fetchone()[0])
        conn.close()
        if held:
            break
    assert held, "the refresh never took the lease; the release assertion would be vacuous"

    result = await call
    elapsed = time.monotonic() - started
    assert result["state"] == "unknown", "a stale depleted verdict must not be shown"
    assert result["source"] == "cached"
    assert elapsed < 1.5, f"the bound must cut the live fetch short (took {elapsed:.2f}s)"

    # The cancelled refresh held the lease; it must be back on the shelf or
    # every later refresh on this host would serve stale until the lease TTL
    # lapsed.
    conn = sqlite3.connect(cache_db)
    leases = conn.execute("SELECT COUNT(*) FROM usage_fetch_leases").fetchone()[0]
    conn.close()
    assert leases == 0, "the cancelled refresh left its fetch lease held"


async def test_no_bearer_is_401(quota: Harness) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=quota.app), base_url="http://localhost"
    ) as anonymous:
        response = await anonymous.get("/v1/desktop/quota-notice")
    assert response.status_code == 401


async def test_unknown_provider_is_422(quota: Harness) -> None:
    response = await quota.client.get(
        "/v1/desktop/quota-notice", params={"provider": "not-a-provider"}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown provider"


async def test_defaults_come_from_config(quota: Harness) -> None:
    """No query parameters: provider/model come from the config default."""

    async def fetch_deepseek(client, api_key):
        return _balance_report("deepseek", 0.0)

    _stub(quota.monkeypatch, "fetch_deepseek_balance", fetch_deepseek)
    quota.store.upsert_credential("deepseek", {"key": "sk-deepseek-test", "source": "login"})
    quota.app.state.config_manager.update_config(
        {"hosting": "deepseek", "model_name": "deepseek-chat"}, write=False
    )
    result = await quota.notice()
    assert result["provider"] == "deepseek"
    assert result["state"] == "depleted"


async def test_no_selection_at_all_is_not_applicable(quota: Harness) -> None:
    result = await quota.notice()
    assert result["state"] == "not_applicable"
    assert result["provider"] == ""


async def test_a_backend_without_the_route_is_a_clean_404() -> None:
    """The interop contract: the route is a router include, so an older
    backend answers a clean 404 and the renderer shows nothing."""
    bare = FastAPI()
    async with AsyncClient(
        transport=ASGITransport(app=bare), base_url="http://localhost"
    ) as client:
        response = await client.get("/v1/desktop/quota-notice")
    assert response.status_code == 404
