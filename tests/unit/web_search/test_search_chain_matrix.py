"""The web_search chain permutation matrix: FREE -> NATIVE -> RADIENT -> API_KEY.

Operator request (2026-11-24): "if the user is signed in with Radient, we first use
free providers where available, then fail over to native search on each provider,
then fail over to Radient if credits are available, then any API keys."

Each row pins credential state through ``provider_auth_mode`` (the one function
the resolver reads), so nothing touches the network or a real credential store.
Native vendors other than DeepSeek are FAKES registered in ``NATIVE_VENDORS``
(the PR1 seam the native-search family plugs into), so the band code is proved
before those vendors exist.

Where native search through a pay-as-you-go MODEL key lands is
``providers.NATIVE_MODEL_KEY_BAND`` (decided by the operator 2026-11-24: after
Radient). Every expectation is DERIVED from that constant via :func:`_expected`,
and the matrix runs under both values, so flipping it is a one-line change that
keeps this file green.

Module-level imports stay on APIs that exist on ``origin/main`` so rows 2 and 4
can be run against the old code as a reproduction (they fail there).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
import pytest

from local_operator.web_search.models import (
    PROVIDER_IDS,
    SearchProviderId,
    SearchResponse,
    SearchSource,
    WebSearchSettings,
)
from local_operator.web_search.service import (
    WebSearchService,
    reset_round_robin_for_tests,
)

DEFAULT_FREE = ["duckduckgo", "tavily", "exa", "parallel", "perplexity"]


def _cfg(tmp_path):
    return tmp_path / "config"


def _modes(monkeypatch, modes: dict[str, str]) -> None:
    from local_operator.web_search import providers as providers_module

    monkeypatch.setattr(
        providers_module,
        "provider_auth_mode",
        lambda provider_id, _config_dir, _settings: modes.get(provider_id, ""),
    )


FREE_PROVIDER_IDS: tuple[SearchProviderId, ...] = (
    "duckduckgo",
    "tavily",
    "exa",
    "parallel",
    "perplexity",
)
#: Credential vocabulary from the design's §6 fixture table.
FREE_MODES = {
    "duckduckgo": "credential-free",
    "tavily": "keyless",
    "exa": "keyless-mcp",
    "parallel": "keyless-mcp",
    "perplexity": "anonymous",
}
SEARCH_KEYS = {
    "none": {},
    "tavily": {"tavily": "api-key"},
    "tav+brave+exa": {"tavily": "api-key", "brave": "api-key", "exa": "api-key"},
    "brave": {"brave": "api-key"},
}
#: The labels the keyed legs walk under, in API_KEY band (declaration) order.
SEARCH_KEY_LEGS = {
    "none": [],
    "tavily": ["tavily (key)"],
    "tav+brave+exa": ["tavily (key)", "exa (key)", "brave"],
    "brave": ["brave"],
}


def _register_native(monkeypatch, native: str) -> tuple[list[str], list[str]]:
    """Register fake native vendors; return (plan legs, model-key legs) in order."""
    from local_operator.web_search import providers as providers_module
    from local_operator.web_search.providers import NativeClass

    def vendor(vendor_id: str, cls: NativeClass, fee: float):
        def modes(_c, _s) -> list[tuple[str, NativeClass]]:
            return [("plan" if cls == "plan" else "api-key", cls)]

        return providers_module.NativeVendor(vendor_id, modes, fee)

    plan: list[str] = []
    keyed: list[str] = []
    if native in ("plan", "plan+key"):
        # zai registered BEFORE openai but both are plan: plan legs keep
        # registration order (no marginal cash, nothing to sort by).
        monkeypatch.setitem(
            providers_module.NATIVE_VENDORS, "openai", vendor("openai", "plan", 0.0)
        )
        monkeypatch.setitem(providers_module.NATIVE_VENDORS, "zai", vendor("zai", "plan", 0.0))
        plan = ["openai", "zai"]
    if native in ("key", "plan+key"):
        # Anthropic registered first but dearer: paid legs sort cheapest-first.
        monkeypatch.setitem(
            providers_module.NATIVE_VENDORS, "anthropic", vendor("anthropic", "key", 0.01)
        )
        monkeypatch.setitem(providers_module.NATIVE_VENDORS, "google", vendor("google", "key", 0.0))
        keyed = ["google", "anthropic"]
    return plan, keyed


def _expected(
    free: list[str],
    plan: list[str],
    model_key: list[str],
    radient: list[str],
    search_keys: list[str],
) -> list[str]:
    """The walked chain, with model-key native legs placed by the policy constant."""
    from local_operator.web_search.providers import NATIVE_MODEL_KEY_BAND

    if NATIVE_MODEL_KEY_BAND == "native":
        return [*free, *plan, *model_key, *radient, *search_keys]
    return [*free, *plan, *radient, *search_keys, *model_key]


# (row, radient, native, keys, settings kwargs, free legs expected, deepseek?)
MATRIX = [
    (1, "none", "none", "none", {}, DEFAULT_FREE),
    (2, "signed-in", "none", "none", {}, DEFAULT_FREE),
    (3, "402-cached", "none", "none", {}, DEFAULT_FREE),
    (4, "none", "none", "tavily", {}, DEFAULT_FREE),
    (5, "signed-in", "none", "tavily", {}, DEFAULT_FREE),
    (6, "signed-in", "none", "tav+brave+exa", {}, DEFAULT_FREE),
    (7, "none", "plan", "none", {}, DEFAULT_FREE),
    (8, "none", "key", "none", {}, DEFAULT_FREE),
    (9, "signed-in", "plan+key", "tav+brave+exa", {}, DEFAULT_FREE),
    (10, "signed-in", "deepseek", "none", {}, DEFAULT_FREE),
    (11, "signed-in", "key", "brave", {"providers": ["brave", "duckduckgo"]}, DEFAULT_FREE),
    (12, "signed-in", "plan", "none", {"providers": ["radient", "openai"]}, DEFAULT_FREE),
    (13, "signed-in", "key", "tavily", {"excluded_providers": ["radient"]}, DEFAULT_FREE),
    (
        14,
        "signed-in",
        "none",
        "tavily",
        {"excluded_providers": ["tavily"]},
        ["duckduckgo", "exa", "parallel", "perplexity"],
    ),
    (
        15,
        "none",
        "none",
        "none",
        {"excluded_providers": ["duckduckgo", "tavily", "exa", "parallel", "perplexity"]},
        [],
    ),
]


def _setup_row(monkeypatch, tmp_path, radient: str, native: str, keys: str, kwargs: dict[str, Any]):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    modes = dict(FREE_MODES)
    modes.update(SEARCH_KEYS[keys])
    if radient != "none":
        modes["radient"] = "radient-credits"
    if radient == "402-cached":
        radient_module._remember_credit_refusal()
    if native == "deepseek":
        modes["deepseek"] = "login"
    _modes(monkeypatch, modes)
    plan, model_key = _register_native(monkeypatch, native)
    if native == "deepseek":
        model_key = ["deepseek"]
    # `model_construct`: rows 11-13 list/exclude a fake vendor id that is not in
    # this release's `SearchProviderId` (PR2 adds the real ids).
    settings = WebSearchSettings.model_construct(
        **{**WebSearchSettings().model_dump(), "strategy": "ordered", **kwargs}
    )
    return settings, plan, model_key


@pytest.fixture(params=["metered", "native"], ids=["model-key-after-radient", "model-key-native"])
def model_key_band(request, monkeypatch):
    """Run every row under BOTH placements of model-key native search."""
    from local_operator.web_search import providers as providers_module

    monkeypatch.setattr(providers_module, "NATIVE_MODEL_KEY_BAND", request.param)
    return request.param


@pytest.mark.parametrize("row", MATRIX, ids=[f"row{r[0]:02d}" for r in MATRIX])
def test_matrix_ordered(row, model_key_band, tmp_path, monkeypatch) -> None:
    number, radient, native, keys, kwargs, free = row
    settings, plan, model_key = _setup_row(monkeypatch, tmp_path, radient, native, keys, kwargs)
    excluded = set(kwargs.get("excluded_providers", ()))
    search_keys = [leg for leg in SEARCH_KEY_LEGS[keys] if leg.split(" ")[0] not in excluded]
    if number == 11:
        # The listed brave heads the API_KEY band, not the chain.
        search_keys = ["brave"]
    radient_legs = ["radient"] if radient != "none" and "radient" not in excluded else []
    expected = _expected(free, plan, model_key, radient_legs, search_keys)

    service = WebSearchService(settings, _cfg(tmp_path))
    if not expected:
        assert service.candidate_legs() == []
        return
    assert [leg.label for leg in service.candidate_legs()] == expected


@pytest.mark.parametrize("row", MATRIX[:14], ids=[f"row{r[0]:02d}" for r in MATRIX[:14]])
def test_matrix_round_robin_rotates_only_the_free_pool(row, model_key_band, tmp_path, monkeypatch):
    """round_robin: the free pool is a rotation; everything after it is `ordered`."""
    number, radient, native, keys, kwargs, _free = row
    settings, _plan, _key = _setup_row(monkeypatch, tmp_path, radient, native, keys, kwargs)
    ordered = [leg.label for leg in WebSearchService(settings, _cfg(tmp_path)).candidate_legs()]
    from local_operator.web_search.providers import resolve_provider_bands

    bands = resolve_provider_bands(settings, _cfg(tmp_path))
    pool = [leg.label for leg in (*bands.prefix, *bands.rotate)]
    rr = WebSearchService(settings.model_copy(update={"strategy": "round_robin"}), _cfg(tmp_path))
    reset_round_robin_for_tests()
    seen_heads = set()
    for _ in range(len(pool)):
        chain = [leg.label for leg in rr.candidate_legs()]
        head = chain[: len(pool)]
        assert sorted(head) == sorted(pool)
        offset = pool.index(head[0])
        assert head == pool[offset:] + pool[:offset]
        assert chain[len(pool) :] == ordered[len(pool) :]
        seen_heads.add(head[0])
    assert seen_heads == set(pool)


def test_row02_radient_joins_after_free_legs(tmp_path, monkeypatch) -> None:
    """Row 2 on main-compatible API: signed in to Radient, Radient follows the free legs."""
    modes = dict(FREE_MODES)
    modes["radient"] = "radient-credits"
    _modes(monkeypatch, modes)
    settings = WebSearchSettings(strategy="ordered")
    chain = WebSearchService(settings, _cfg(tmp_path)).candidates()
    assert chain == [*DEFAULT_FREE, "radient"]


def test_row04_a_stored_tavily_key_keeps_the_keyless_leg(tmp_path, monkeypatch) -> None:
    """Row 4 on main-compatible API: the keyless leg stays; the key is tried last.

    On origin/main a stored key moved Tavily wholesale into the paid band, so its
    free keyless tier was never tried: the chain was ddg exa parallel perplexity
    tavily.
    """
    modes = dict(FREE_MODES)
    modes["tavily"] = "api-key"
    _modes(monkeypatch, modes)
    settings = WebSearchSettings(strategy="ordered")
    chain = WebSearchService(settings, _cfg(tmp_path)).candidates()
    assert chain == [*DEFAULT_FREE, "tavily"]


def test_row16_17_18_forced_provider(tmp_path, monkeypatch) -> None:
    modes = dict(FREE_MODES)
    modes["radient"] = "radient-credits"
    modes["tavily"] = "api-key"
    _modes(monkeypatch, modes)
    service = WebSearchService(WebSearchSettings(strategy="ordered"), _cfg(tmp_path))
    # Row 16: forced radient walks radient only.
    assert [leg.label for leg in service.candidate_legs("radient")] == ["radient"]
    # Row 18: forced tavily walks BOTH its legs, free first.
    assert [leg.label for leg in service.candidate_legs("tavily")] == ["tavily", "tavily (key)"]
    # Row 17: not signed in -> refused, naming the sign-in command.
    _modes(monkeypatch, FREE_MODES)
    with pytest.raises(RuntimeError, match=r"local-operator login radient"):
        WebSearchService(WebSearchSettings(), _cfg(tmp_path)).candidate_legs("radient")


def test_row19_tainted_session_keeps_index_only_legs(tmp_path, monkeypatch) -> None:
    from local_operator.harness.bridge_egress import INDEX_ONLY_PROVIDERS

    modes = dict(FREE_MODES)
    modes.update({"radient": "radient-credits", "tavily": "api-key", "brave": "api-key"})
    modes["deepseek"] = "login"
    _modes(monkeypatch, modes)
    service = WebSearchService(
        WebSearchSettings(strategy="ordered"), _cfg(tmp_path), restrict_to=INDEX_ONLY_PROVIDERS
    )
    assert [leg.label for leg in service.candidate_legs()] == ["duckduckgo", "brave"]


# -- service-level rows: the Radient leg on the wire ------------------------------


def _failing(provider_id):
    async def search(_client, _config_dir, _settings, _query, _limit) -> SearchResponse:
        raise RuntimeError(f"{provider_id} down")

    return search


def _serving(provider_id):
    async def search(_client, _config_dir, _settings, _query, _limit) -> SearchResponse:
        return SearchResponse(
            provider=provider_id,
            auth_mode="api-key",
            sources=[SearchSource(title="b", url="https://example.com/b")],
        )

    return search


def _radient_service(tmp_path, monkeypatch, handler, *, keys: dict[str, str] | None = None):
    """Every free leg fails; Radient is signed in and answers through ``handler``."""
    from local_operator.web_search import providers as providers_module
    from local_operator.web_search import radient as radient_module

    modes = dict(FREE_MODES)
    modes["radient"] = "radient-credits"
    modes.update(keys or {})
    _modes(monkeypatch, modes)
    for provider_id in FREE_PROVIDER_IDS:
        monkeypatch.setitem(
            providers_module.PROVIDERS,
            provider_id,
            replace(providers_module.PROVIDERS[provider_id], search=_failing(provider_id)),
        )
    monkeypatch.setitem(
        providers_module.PROVIDERS,
        "brave",
        replace(providers_module.PROVIDERS["brave"], search=_serving("brave")),
    )
    monkeypatch.setattr(radient_module, "_radient_base_url", lambda _c: "https://radient.test/v1")

    async def token(_config_dir, _base):
        return "test-token"

    monkeypatch.setattr(radient_module, "_radient_token", token)
    calls: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    service = WebSearchService(
        WebSearchSettings(strategy="ordered"),
        _cfg(tmp_path),
        transport=httpx.MockTransport(recording),
    )
    return service, calls


BILLED = {
    "query": "q",
    "provider": "tavily",
    "total_found": 1,
    "results": [{"title": "T", "url": "https://example.com/t", "content": "c", "score": 0.9}],
    "cost": {
        "v": 1,
        "amount_usd": 0.01,
        "basis": "billed",
        "units": 1,
        "unit": "search",
        "unit_price_usd": 0.01,
        "provider": "tavily",
        "tool_type": "web_search",
        "usage_record_id": "rec-1",
    },
}


@pytest.mark.asyncio
async def test_row20_402_is_cached_and_the_chain_moves_to_api_keys(tmp_path, monkeypatch):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    service, calls = _radient_service(
        tmp_path,
        monkeypatch,
        lambda _r: httpx.Response(402, json={"error": "insufficient credits"}),
        keys={"brave": "api-key"},
    )
    first = await service.search("q")
    assert first.provider == "brave"
    assert any(f.startswith("radient: ") and "credit" in f for f in first.failures)
    assert len(calls) == 1
    second = await service.search("q")
    assert second.provider == "brave"
    # The remembered 402 skips the round trip.
    assert len(calls) == 1
    radient_module.reset_radient_credit_cache_for_tests()


@pytest.mark.asyncio
async def test_row21_401_says_sign_in_again_and_is_not_cached(tmp_path, monkeypatch):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    service, calls = _radient_service(
        tmp_path,
        monkeypatch,
        lambda _r: httpx.Response(401, json={"error": "invalid API key format"}),
        keys={"brave": "api-key"},
    )
    response = await service.search("q")
    assert response.provider == "brave"
    assert any("local-operator login radient" in f for f in response.failures)
    await service.search("q")
    assert len(calls) == 2
    assert radient_module.radient_credit_refusal_remaining() is None


@pytest.mark.asyncio
async def test_row22_200_with_error_fails_with_the_server_text(tmp_path, monkeypatch):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    service, _calls = _radient_service(
        tmp_path,
        monkeypatch,
        lambda _r: httpx.Response(
            200, json={"query": "q", "results": [], "error": "upstream boom"}
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        await service.search("q")
    # The digest crops each leg to its per-leg cap; the server text leads it.
    assert "radient: Radient search failed: upstre" in str(caught.value)


@pytest.mark.asyncio
async def test_row23_success_is_billed_and_recorded_as_radient(tmp_path, monkeypatch):
    from local_operator.web_search import radient as radient_module

    radient_module.reset_radient_credit_cache_for_tests()
    service, calls = _radient_service(
        tmp_path, monkeypatch, lambda _r: httpx.Response(200, json=BILLED)
    )
    response = await service.search("q", limit=3)
    assert response.provider == "radient"
    assert response.auth_mode == "radient-credits"
    assert response.cost is not None and response.cost.usd == 0.01
    assert response.cost.basis == "billed by Radient (tavily)"
    assert response.cost.priced_from_usage is True
    assert response.request_id == "rec-1"
    assert [s.url for s in response.sources] == ["https://example.com/t"]
    request = calls[0]
    assert request.url.path == "/v1/tools/search"
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.url.params["provider"] == "tavily"
    assert request.url.params["max_results"] == "3"


def test_row24_env_only_radient_key_is_not_signed_in(tmp_path, monkeypatch) -> None:
    from local_operator.providers import radient_credentials
    from local_operator.web_search import radient as radient_module

    monkeypatch.setenv("RADIENT_API_KEY", "env-only")
    monkeypatch.setattr(
        radient_module, "_radient_base_url", lambda _c: "https://api.radienthq.com/v1"
    )

    class Store:
        def __init__(self, rows):
            self.rows = rows

        def list_credentials(self, provider):
            assert provider == "radient"
            return self.rows

        def close(self) -> None:
            closed.append(True)

    # The probe reads the token resolver's OWN store (config_dir-scoped and
    # mesh-aware), never the ambient one (review round 1, R1-5).
    seen: list[object] = []
    closed: list[bool] = []

    def store_for(rows):
        def build(config_dir):
            seen.append(config_dir)
            return Store(rows)

        return build

    monkeypatch.setattr(radient_credentials, "_radient_auth_store", store_for([]))
    assert radient_module.radient_login_present(_cfg(tmp_path)) is False
    monkeypatch.setattr(radient_credentials, "_radient_auth_store", store_for([object()]))
    assert radient_module.radient_login_present(_cfg(tmp_path)) is True
    assert seen == [_cfg(tmp_path), _cfg(tmp_path)]
    assert closed == [True, True], "the probe must close the store it opened"


def test_row25_the_402_verdict_lapses_after_its_ttl(monkeypatch) -> None:
    from local_operator.web_search import radient as radient_module

    now = [1000.0]
    monkeypatch.setattr(radient_module.time, "monotonic", lambda: now[0])
    radient_module.reset_radient_credit_cache_for_tests()
    radient_module._remember_credit_refusal()
    assert radient_module.radient_credit_refusal_remaining() == pytest.approx(300.0)
    now[0] += radient_module.RADIENT_CREDIT_TTL_SECONDS - 1
    assert radient_module.radient_credit_refusal_remaining() == pytest.approx(1.0)
    now[0] += 2
    assert radient_module.radient_credit_refusal_remaining() is None
    radient_module.reset_radient_credit_cache_for_tests()


@pytest.mark.asyncio
async def test_row27_a_thirteen_leg_failure_names_every_leg(tmp_path, monkeypatch):
    from rich.cells import cell_len

    from local_operator.tui.widgets.tool_card import REASON_MAX_CELLS
    from local_operator.web_search import providers as providers_module
    from local_operator.web_search import radient as radient_module
    from local_operator.web_search import service as service_module

    assert service_module.REASON_MAX_CELLS == REASON_MAX_CELLS
    radient_module.reset_radient_credit_cache_for_tests()
    _register_native(monkeypatch, "plan+key")
    for vendor_id in ("openai", "zai", "google", "anthropic"):
        monkeypatch.setitem(
            providers_module.PROVIDERS,
            vendor_id,
            replace(providers_module.PROVIDERS["brave"], id=vendor_id, search=_failing(vendor_id)),
        )
    keys = {"tavily": "api-key", "brave": "api-key", "exa": "api-key"}
    service, _calls = _radient_service(
        tmp_path,
        monkeypatch,
        lambda _r: httpx.Response(503, text="All connection attempts failed"),
        keys=keys,
    )
    monkeypatch.setitem(
        providers_module.PROVIDERS,
        "brave",
        replace(providers_module.PROVIDERS["brave"], search=_failing("brave")),
    )
    legs = [leg.label for leg in service.candidate_legs()]
    assert len(legs) == 13
    with pytest.raises(RuntimeError) as caught:
        await service.search("q")
    message = str(caught.value)
    # 13 legs but 11 PROVIDERS: tavily and exa are each walked twice (free leg,
    # then key), and the count names providers (review round 1, R1-7).
    assert len({label.split(" ")[0] for label in legs}) == 11
    assert message.startswith("11/11 providers failed: ")
    for label in legs:
        assert f"{label}: " in message
    assert cell_len(message) <= REASON_MAX_CELLS


def test_provider_ids_match_the_literal() -> None:
    """The import-cheap id list and the typed Literal are one vocabulary."""
    from typing import get_args

    from local_operator.web_search.models import SearchProviderId

    assert set(PROVIDER_IDS) == set(get_args(SearchProviderId))
    assert len(PROVIDER_IDS) == len(set(PROVIDER_IDS))


def test_settings_members_are_the_provider_ids() -> None:
    from local_operator.settings_io import SETTINGS

    rows = {row.key: row for row in SETTINGS if row.key.startswith("web_search.")}
    assert tuple(rows["web_search.providers"].members) == PROVIDER_IDS
    assert tuple(rows["web_search.excluded_providers"].members) == PROVIDER_IDS


def test_listed_rows_name_their_paid_leg_and_the_refused_radient_state(
    tmp_path, monkeypatch
) -> None:
    """Design round 1, D1/D2/D5 (PR #4553).

    D1: a LISTED keyed Tavily (it is in the default order) has a paid leg in the
    API-key band, and its row must say so, not plain `enabled`.
    D2: a listed Radient says `(credits)` like its chain marker, not `(paid)`.
    D5: a remembered 402 marks the Radient row AND its chain leg as refused.
    """
    from local_operator.web_search import radient as radient_module
    from local_operator.web_search.providers import (
        chain_label,
        provider_state_code,
        provider_state_label,
        provider_statuses,
        state_legend,
    )

    radient_module.reset_radient_credit_cache_for_tests()
    _modes(monkeypatch, {**FREE_MODES, "tavily": "api-key", "radient": "radient-credits"})
    monkeypatch.setattr(
        "local_operator.web_search.providers._provider_detail",
        lambda provider_id, _c, _s: "detail",
    )
    settings = WebSearchSettings(providers=["duckduckgo", "tavily", "radient"], strategy="ordered")
    rows = {status.id: status for status in provider_statuses(settings, _cfg(tmp_path))}
    assert provider_state_code(rows["tavily"]) == "enabled, then key"
    assert provider_state_label(rows["tavily"]) == "enabled, then key"
    assert provider_state_code(rows["radient"]) == "enabled (credits)"
    legend = state_legend(list(rows.values()))
    assert "enabled, then key = in your priority order on the free tier" in legend
    assert "enabled (credits) = tried after the free legs" in legend
    assert "Radient (credits)" in chain_label(list(rows.values()))

    radient_module._remember_credit_refusal("https://radient.test/v1", "fp")
    try:
        rows = {status.id: status for status in provider_statuses(settings, _cfg(tmp_path))}
        assert provider_state_code(rows["radient"]) == "radient refused"
        chain = chain_label(list(rows.values()))
        assert "Radient (no credits)" in chain and "Radient (credits)" not in chain
    finally:
        radient_module.reset_radient_credit_cache_for_tests()


def test_the_radient_detail_drops_the_sign_in_clause_once_signed_in(tmp_path, monkeypatch) -> None:
    """Design round 1, D4: a signed-in row must not tell the user to sign in."""
    from local_operator.web_search.providers import provider_statuses

    _modes(monkeypatch, dict(FREE_MODES))
    out = {s.id: s for s in provider_statuses(WebSearchSettings(), _cfg(tmp_path))}
    assert "login radient" in out["radient"].detail
    _modes(monkeypatch, {**FREE_MODES, "radient": "radient-credits"})
    signed = {s.id: s for s in provider_statuses(WebSearchSettings(), _cfg(tmp_path))}
    assert "login radient" not in signed["radient"].detail
    assert "$0.01 per search" in signed["radient"].detail


@pytest.mark.asyncio
async def test_a_listed_signed_out_radient_names_the_sign_in(tmp_path, monkeypatch) -> None:
    """Review round 1, R1-8: the generic `not configured` named nothing to run."""
    from local_operator.web_search import providers as providers_module

    _modes(monkeypatch, {"duckduckgo": "credential-free"})
    monkeypatch.setitem(
        providers_module.PROVIDERS,
        "duckduckgo",
        replace(providers_module.PROVIDERS["duckduckgo"], search=_failing("duckduckgo")),
    )
    settings = WebSearchSettings(
        providers=["radient", "duckduckgo"],
        strategy="ordered",
        excluded_providers=["tavily", "exa", "parallel", "perplexity"],
    )
    with pytest.raises(RuntimeError) as caught:
        await WebSearchService(settings, _cfg(tmp_path)).search("q")
    message = str(caught.value)
    # The sign-in sentence LEADS the digest (it is no longer the skipped
    # `not configured` placeholder) and the tried list names the leg.
    assert message.startswith(
        "2/2 providers failed: You are not signed in to Radient. "
        "Run `local-operator login radient`."
    )
    assert "radient: You are not signed in" in message
    assert "not configured" not in message
