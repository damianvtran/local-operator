from __future__ import annotations

from dataclasses import replace

import pytest

from local_operator.credentials import CredentialManager
from local_operator.web_search.models import (
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
async def test_round_robin_rotates_first_success_across_enabled_providers(
    tmp_path, monkeypatch
) -> None:
    async def duck(*_args):
        return _response("duckduckgo")

    async def tavily(*_args):
        return _response("tavily")

    monkeypatch.setitem(PROVIDERS, "duckduckgo", replace(PROVIDERS["duckduckgo"], search=duck))
    monkeypatch.setitem(PROVIDERS, "tavily", replace(PROVIDERS["tavily"], search=tavily))
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo", "tavily"], strategy="round_robin"),
        _credentials(tmp_path),
    )

    first = await service.search("first")
    second = await service.search("second")

    assert first.provider == "duckduckgo"
    assert second.provider == "tavily"


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
        WebSearchSettings(providers=["tavily"]),
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
        WebSearchSettings(providers=["duckduckgo", "tavily"], strategy="ordered"),
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
        WebSearchSettings(providers=["brave", "duckduckgo"], strategy="ordered"),
        _credentials(tmp_path),
    )

    response = await service.search("fallback")

    assert response.provider == "duckduckgo"
    assert response.failures == ["brave: not configured"]


@pytest.mark.asyncio
async def test_forced_provider_must_be_enabled(tmp_path) -> None:
    service = WebSearchService(
        WebSearchSettings(providers=["duckduckgo"]),
        _credentials(tmp_path),
    )

    with pytest.raises(RuntimeError, match="is disabled"):
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
