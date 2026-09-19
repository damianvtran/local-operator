from __future__ import annotations

from dataclasses import replace

import pytest

from local_operator.credentials import CredentialManager
from local_operator.web_search.models import (
    PROVIDER_IDS,
    SearchProviderId,
    SearchResponse,
    SearchSource,
    WebSearchSettings,
)
from local_operator.web_search.providers import PROVIDERS
from local_operator.web_search.service import (
    WebSearchService,
    coerce_search_settings,
    reset_round_robin_for_tests,
)


def _credentials(tmp_path) -> CredentialManager:
    return CredentialManager(tmp_path / "config")


def _chain(*ids: SearchProviderId) -> WebSearchSettings:
    """Settings whose chain is EXACTLY ``ids``: every other provider excluded.

    Required rather than cosmetic. ``web_search.providers`` is a priority prefix
    now, so a bare ``WebSearchSettings(providers=[...])`` auto-joins every other
    usable provider -- and on a developer's machine that means real keyless
    requests to Exa and Parallel inside the unit suite. Pinning the chain with
    exclusions is the same mechanism a user has, so a test cannot pass here and
    deploy a chain that behaves differently.
    """
    return WebSearchSettings(
        providers=list(ids),
        excluded_providers=[value for value in PROVIDER_IDS if value not in ids],
    )


def _modes(monkeypatch, modes: dict[str, str]) -> None:
    """Pin each provider's EFFECTIVE auth mode, so no credential or network is read.

    Patched on ``provider_auth_mode`` rather than on ``provider_available``
    because the resolver's tier decision reads the mode too: pinning both would
    let a test describe a provider as available while the resolver classified it
    from a different (real) credential state.
    """
    from local_operator.web_search import providers as providers_module

    monkeypatch.setattr(
        providers_module,
        "provider_auth_mode",
        lambda provider_id, _credentials, _settings: modes.get(provider_id, ""),
    )


def _response(provider: SearchProviderId) -> SearchResponse:
    return SearchResponse(
        provider=provider,
        auth_mode="test",
        sources=[SearchSource(title=provider, url=f"https://{provider}.example")],
    )


@pytest.fixture(autouse=True)
def _reset_rotation() -> None:
    reset_round_robin_for_tests()


@pytest.mark.asyncio
async def test_free_pool_rotates_across_the_prefix_and_the_auto_free_band(
    tmp_path, monkeypatch
) -> None:
    """`round_robin` must spread the first attempt, not just the tail.

    The strategy's whole purpose is that a rate-limited free tier is not the same
    one on every search, and a chain whose prefix is tried first on every call
    withdrew that for exactly the shape most installs run (round-1 M1/Q1/D2/U1).
    """

    async def duck(*_args):
        raise RuntimeError("challenged")

    async def exa(*_args):
        raise RuntimeError("challenged")

    async def parallel(*_args):
        raise RuntimeError("challenged")

    async def tavily(*_args):
        raise RuntimeError("challenged")

    _modes(
        monkeypatch,
        {
            "duckduckgo": "credential-free",
            "tavily": "keyless",
            "exa": "keyless-mcp",
            "parallel": "keyless-mcp",
        },
    )
    monkeypatch.setitem(PROVIDERS, "duckduckgo", replace(PROVIDERS["duckduckgo"], search=duck))
    monkeypatch.setitem(PROVIDERS, "tavily", replace(PROVIDERS["tavily"], search=tavily))
    monkeypatch.setitem(PROVIDERS, "exa", replace(PROVIDERS["exa"], search=exa))
    monkeypatch.setitem(PROVIDERS, "parallel", replace(PROVIDERS["parallel"], search=parallel))
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo", "tavily"]),
        _credentials(tmp_path),
    )

    tried: list[list[SearchProviderId]] = []
    for _ in range(6):
        chain = service.candidates()
        assert chain == [*chain[:4], *chain[4:]]
        tried.append(chain[:4])

    # The pool is the listed free legs in listed order followed by the auto-joined
    # free band, and the cycle walks it one step per call -- both halves asserted,
    # because "the set of first attempts" alone would pass on a pool that merely
    # shuffled.
    assert tried == [
        ["duckduckgo", "tavily", "exa", "parallel"],
        ["tavily", "exa", "parallel", "duckduckgo"],
        ["exa", "parallel", "duckduckgo", "tavily"],
        ["parallel", "duckduckgo", "tavily", "exa"],
        ["duckduckgo", "tavily", "exa", "parallel"],
        ["tavily", "exa", "parallel", "duckduckgo"],
    ]

    # The two original aims of the test survive as the same assertion: `duckduckgo`
    # is no longer the first attempt on every call, and `tavily` -- the second
    # listed leg -- now takes it too.
    first_attempts = {chain[0] for chain in tried}
    assert first_attempts == {"duckduckgo", "tavily", "exa", "parallel"}


@pytest.mark.asyncio
async def test_tavily_oauth_delegate_precedes_direct_transport(tmp_path, monkeypatch) -> None:
    direct_called = False

    async def direct(*_args):
        nonlocal direct_called
        direct_called = True
        return _response("tavily")

    async def oauth(_query: str, _limit: int) -> SearchResponse:
        response = _response("tavily")
        response.auth_mode = "oauth-mcp"
        return response

    monkeypatch.setitem(PROVIDERS, "tavily", replace(PROVIDERS["tavily"], search=direct))
    service = WebSearchService(
        _chain("tavily"),
        _credentials(tmp_path),
        tavily_oauth_search=oauth,
    )

    response = await service.search("query")

    assert response.auth_mode == "oauth-mcp"
    assert direct_called is False


@pytest.mark.asyncio
async def test_failure_falls_through_and_is_reported(tmp_path, monkeypatch) -> None:
    async def duck(*_args):
        raise RuntimeError("challenged")

    async def tavily(*_args):
        return _response("tavily")

    monkeypatch.setitem(PROVIDERS, "duckduckgo", replace(PROVIDERS["duckduckgo"], search=duck))
    monkeypatch.setitem(PROVIDERS, "tavily", replace(PROVIDERS["tavily"], search=tavily))
    service = WebSearchService(
        _chain("duckduckgo", "tavily"),
        _credentials(tmp_path),
    )

    response = await service.search("fallback")

    assert response.provider == "tavily"
    assert response.failures == ["duckduckgo: challenged"]


@pytest.mark.asyncio
async def test_unconfigured_key_provider_is_skipped(tmp_path, monkeypatch) -> None:
    async def duck(*_args):
        return _response("duckduckgo")

    monkeypatch.setitem(PROVIDERS, "duckduckgo", replace(PROVIDERS["duckduckgo"], search=duck))
    service = WebSearchService(
        _chain("brave", "duckduckgo"),
        _credentials(tmp_path),
    )

    response = await service.search("fallback")

    assert response.provider == "duckduckgo"
    assert response.failures == ["brave: not configured"]


@pytest.mark.asyncio
async def test_forced_provider_must_be_in_the_chain(tmp_path, monkeypatch) -> None:
    """The refusal names the command that can actually fix THIS reason.

    `enable` no longer appends to the chain, so the old one-sentence advice sent a
    user with no credential round a loop that could not end (round-1 U3).
    """
    _modes(monkeypatch, {"duckduckgo": "credential-free"})
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo"]),
        _credentials(tmp_path),
    )

    with pytest.raises(
        RuntimeError,
        match=r"cannot serve yet on this install.*search setup brave.*BRAVE_API_KEY",
    ):
        await service.search("query", forced_provider="brave")


@pytest.mark.asyncio
async def test_master_switch_blocks_execution(tmp_path) -> None:
    service = WebSearchService(
        WebSearchSettings(enabled=False),
        _credentials(tmp_path),
    )

    with pytest.raises(RuntimeError, match="Web search is disabled"):
        await service.search("query")


def test_malformed_config_preserves_valid_provider_subset() -> None:
    settings = coerce_search_settings(
        {
            "strategy": "round_robin",
            "providers": ["tavily", "unknown", "tavily", "duckduckgo"],
            "timeout_seconds": 500,
        }
    )

    assert settings.providers == ["tavily", "duckduckgo"]
    assert settings.timeout_seconds == 120


@pytest.mark.asyncio
async def test_provider_reported_failures_survive_a_successful_search(
    tmp_path, monkeypatch
) -> None:
    """A provider's own failure note must reach the caller, not be overwritten.

    The chain's failure list describes providers that were TRIED AND SKIPPED; a
    provider's own list describes something that degraded INSIDE a successful
    call (the DeepSeek evidence pass). Replacing the former with the latter lost
    the degradation exactly when it needed explaining.
    """

    async def duck(*_args):
        raise RuntimeError("upstream 429")

    async def tavily(*_args):
        response = _response("tavily")
        response.failures = ["tavily evidence pass: truncated"]
        return response

    monkeypatch.setitem(PROVIDERS, "duckduckgo", replace(PROVIDERS["duckduckgo"], search=duck))
    monkeypatch.setitem(PROVIDERS, "tavily", replace(PROVIDERS["tavily"], search=tavily))
    service = WebSearchService(
        _chain("duckduckgo", "tavily"),
        _credentials(tmp_path),
    )

    response = await service.search("query", limit=3)

    assert response.provider == "tavily"
    # Both the chain's skip and the provider's own degradation are reported.
    assert any("duckduckgo" in note for note in response.failures)
    assert any("evidence pass" in note for note in response.failures)


# ---------------------------------------------------------------------------
# The resolution model: priority prefix ++ rotating band ++ best-effort ++ paid
# ---------------------------------------------------------------------------


def test_available_unlisted_provider_joins_the_chain(tmp_path, monkeypatch) -> None:
    """The reported bug: an available provider absent from the list was unreachable.

    ``provider_available`` was consulted only INSIDE the candidate loop, so it
    could filter a provider that was already a candidate and never add one.
    """
    _modes(monkeypatch, {"duckduckgo": "credential-free", "deepseek": "login"})
    service = WebSearchService(WebSearchSettings(providers=["duckduckgo"]), _credentials(tmp_path))

    chain = service.resolve()

    assert chain == ["duckduckgo", "deepseek"]
    # The metered leg is a BACKSTOP: reachable, but never first.
    assert service.candidates()[0] == "duckduckgo"


def test_excluded_provider_is_never_a_candidate(tmp_path, monkeypatch) -> None:
    _modes(monkeypatch, {"duckduckgo": "credential-free", "deepseek": "login"})
    settings = WebSearchSettings(providers=["duckduckgo", "deepseek"])
    settings.excluded_providers = [value for value in PROVIDER_IDS if value != "duckduckgo"]
    service = WebSearchService(settings, _credentials(tmp_path))

    assert service.resolve() == ["duckduckgo"]


def test_a_listed_provider_is_excluded_even_when_the_prefix_names_it(tmp_path, monkeypatch) -> None:
    """Exclusion WINS over listing, so `disable` need not rewrite the priority list."""
    _modes(monkeypatch, {"duckduckgo": "credential-free"})
    settings = WebSearchSettings(providers=["duckduckgo"], excluded_providers=["duckduckgo"])

    service = WebSearchService(settings, _credentials(tmp_path))

    assert service.resolve() == []


def test_metered_tail_is_never_rotated(tmp_path, monkeypatch) -> None:
    """THE BAND INVARIANT: rotation happens inside the rotating band only.

    A paid leg rotated to the front of a search would spend money or a model turn
    on a request a free provider could have served, which is the one thing the
    banded chain exists to prevent.
    """
    _modes(
        monkeypatch,
        {
            "duckduckgo": "credential-free",
            "tavily": "keyless",
            "exa": "keyless-mcp",
            "parallel": "keyless-mcp",
            "perplexity": "anonymous",
            "deepseek": "login",
        },
    )
    service = WebSearchService(WebSearchSettings(providers=["duckduckgo"]), _credentials(tmp_path))
    free_legs = 1 + 4  # the listed prefix leg + the four free rotating legs

    for _ in range(8):
        chain = service.candidates()
        assert chain[-1] == "deepseek"
        assert chain.index("deepseek") >= free_legs
        # ...and the best-effort leg is never a first attempt either.
        assert chain.index("perplexity") > chain.index("exa")


def test_a_listed_metered_provider_is_tried_after_every_free_leg(tmp_path, monkeypatch) -> None:
    """The operator's own shape, which is what round-1 M2/Q2/D1/U2 measured.

    ``providers: [duckduckgo, tavily, perplexity, deepseek]`` with a live DeepSeek
    login used to resolve to a chain whose 4th leg was a MODEL TURN, with the two
    free keyless legs behind it -- so the failure path paid while free providers
    were reachable. The strengthened rule holds the listed paid leg back, and the
    free legs still come first in the order they were listed.
    """
    _modes(
        monkeypatch,
        {
            "duckduckgo": "credential-free",
            "tavily": "keyless",
            "perplexity": "anonymous",
            "exa": "keyless-mcp",
            "parallel": "keyless-mcp",
            "deepseek": "login",
        },
    )
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo", "tavily", "perplexity", "deepseek"]),
        _credentials(tmp_path),
    )

    assert service.resolve() == [
        "duckduckgo",
        "tavily",
        "perplexity",
        "exa",
        "parallel",
        "deepseek",
    ]
    for _ in range(6):
        chain = service.candidates()
        assert chain[-1] == "deepseek"
        assert set(chain[:-1]) == {"duckduckgo", "tavily", "perplexity", "exa", "parallel"}


def test_ordered_strategy_walks_the_bands_in_declaration_order(tmp_path, monkeypatch) -> None:
    _modes(
        monkeypatch,
        {
            "tavily": "keyless",
            "exa": "keyless-mcp",
            "parallel": "keyless-mcp",
            "perplexity": "anonymous",
            "deepseek": "login",
        },
    )
    settings = WebSearchSettings(providers=["tavily"], strategy="ordered")
    service = WebSearchService(settings, _credentials(tmp_path))

    assert service.candidates() == ["tavily", "exa", "parallel", "perplexity", "deepseek"]


def test_best_effort_band_is_never_rotated(tmp_path, monkeypatch) -> None:
    _modes(
        monkeypatch,
        {
            "duckduckgo": "credential-free",
            "exa": "keyless-mcp",
            "perplexity": "anonymous",
        },
    )
    service = WebSearchService(WebSearchSettings(providers=["duckduckgo"]), _credentials(tmp_path))

    positions = {service.candidates().index("perplexity") for _ in range(5)}

    # The best-effort leg sits after the prefix and after the free band, always.
    assert positions == {2}


def test_a_listed_keyed_provider_is_held_back_to_the_paid_band(tmp_path, monkeypatch) -> None:
    """A keyed transport is a spend decision, so listing it cannot put it first.

    Listing stays authoritative for ORDER, but only within a band: a listed paid
    leg leads the paid band instead of jumping the free ones (round-1 M2/Q2/D1/U2).
    """
    _modes(monkeypatch, {"duckduckgo": "credential-free", "tavily": "keyless", "exa": "api-key"})

    automatic = WebSearchService(
        WebSearchSettings(providers=["duckduckgo"]), _credentials(tmp_path)
    )
    assert automatic.resolve() == ["duckduckgo", "tavily", "exa"]

    listed = WebSearchService(
        WebSearchSettings(providers=["exa", "duckduckgo"]), _credentials(tmp_path)
    )
    # `exa` is listed FIRST and still resolves last: the keyed leg is held back to
    # the paid band, behind `tavily` which joined automatically and is free.
    assert listed.resolve() == ["duckduckgo", "tavily", "exa"]

    # With free legs behind it, the paid leg still goes last -- and it keeps its
    # listing order relative to the OTHER paid legs, which is what "listing is
    # authoritative within a band" means.
    _modes(
        monkeypatch,
        {
            "duckduckgo": "credential-free",
            "tavily": "keyless",
            "exa": "api-key",
            "serpapi": "api-key",
        },
    )
    mixed = WebSearchService(
        WebSearchSettings(providers=["serpapi", "exa", "tavily", "duckduckgo"]),
        _credentials(tmp_path),
    )
    assert mixed.resolve() == ["tavily", "duckduckgo", "serpapi", "exa"]
    assert mixed.candidates() == ["tavily", "duckduckgo", "serpapi", "exa"]


def test_forced_provider_is_refused_when_excluded(tmp_path, monkeypatch) -> None:
    """A pin must not override an explicit "never" (deviation from omp, §8.5)."""
    _modes(monkeypatch, {"duckduckgo": "credential-free", "deepseek": "login"})
    settings = WebSearchSettings(
        providers=["duckduckgo", "deepseek"], excluded_providers=["deepseek"]
    )
    service = WebSearchService(settings, _credentials(tmp_path))

    with pytest.raises(
        RuntimeError,
        match=r"is excluded, so it will not be used by any search.*search enable deepseek",
    ):
        service.candidates(forced_provider="deepseek")


def test_forced_provider_is_allowed_when_it_only_auto_joined(tmp_path, monkeypatch) -> None:
    _modes(monkeypatch, {"duckduckgo": "credential-free", "deepseek": "login"})
    service = WebSearchService(WebSearchSettings(providers=["duckduckgo"]), _credentials(tmp_path))

    assert service.candidates(forced_provider="deepseek") == ["deepseek"]


@pytest.mark.asyncio
async def test_empty_chain_message_names_enable_and_list(tmp_path, monkeypatch) -> None:
    _modes(monkeypatch, {})
    settings = WebSearchSettings(providers=["duckduckgo"])
    settings.excluded_providers = list(PROVIDER_IDS)
    service = WebSearchService(settings, _credentials(tmp_path))

    with pytest.raises(RuntimeError) as raised:
        await service.search("query")

    message = str(raised.value)
    assert "search enable" in message and "search list" in message


def test_coerce_drops_unknown_exclusions_and_malformed_values_read_as_none() -> None:
    """A typo must not disable the chain, and a hand-edited string is not a crash."""
    coerced = coerce_search_settings(
        {"providers": ["duckduckgo"], "excluded_providers": ["tavily", "nope", "tavily"]}
    )
    assert coerced.excluded_providers == ["tavily"]

    for malformed in ("tavily", None, {"tavily": True}, 7):
        assert (
            coerce_search_settings(
                {"providers": ["duckduckgo"], "excluded_providers": malformed}
            ).excluded_providers
            == []
        )


def test_an_absent_excluded_key_is_an_empty_exclusion_list() -> None:
    """No migration: absence IS the new default (an old config keeps working)."""
    assert coerce_search_settings({}).excluded_providers == []
    assert "excluded_providers" not in {"providers": ["duckduckgo"]}
