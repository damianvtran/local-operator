"""``providers.key_check``: one bounded request decides valid / invalid / unknown.

Driven through ``httpx.MockTransport`` so no test reaches a provider. The
classification table is the contract the desktop's key-save route acts on:
``False`` refuses the save, ``None`` saves unverified, ``True`` saves verified.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from local_operator.providers import key_check
from local_operator.providers.registry import get_provider_definition

SECRET = "sk-key-check-secret-never-echoed"


def _run(provider: str, handler, **kwargs) -> tuple[key_check.KeyCheck, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    async def go() -> key_check.KeyCheck:
        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            return await key_check.check_api_key(provider, SECRET, client=client, **kwargs)

    return asyncio.run(go()), seen


@pytest.mark.parametrize(
    ("status", "body", "valid"),
    [
        (200, '{"data": []}', True),
        (401, '{"error": "Authentication Fails, Your api key: sk-... is invalid"}', False),
        # xAI and Google answer a malformed key with 400, measured 2026-09-24.
        (400, '{"error": "Incorrect API key provided."}', False),
        (403, '{"detail": "authentication_error"}', False),
        # Not about the key: a region block, a rate limit, an outage.
        (403, '{"error": "This region is not supported"}', None),
        (429, '{"error": "rate limited"}', None),
        (503, "upstream unavailable", None),
        (400, '{"error": "bad pageSize"}', None),
    ],
)
def test_the_status_table(status: int, body: str, valid: bool | None) -> None:
    result, _ = _run("deepseek", lambda _r: httpx.Response(status, text=body))
    assert result.valid is valid
    if valid is True:
        assert result.reason is None
    else:
        # Every non-valid verdict explains itself in the user's terms, and never
        # carries the provider's body -- DeepSeek echoes the submitted key there.
        assert result.reason and "DeepSeek" in result.reason
        assert "sk-" not in result.reason


def test_a_network_failure_is_unknown_not_invalid() -> None:
    """An offline laptop is not evidence the key is wrong, so it must not refuse."""

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    result, _ = _run("mistral", fail)
    assert result.valid is None
    assert result.reason and "Could not reach Mistral AI" in result.reason
    assert SECRET not in result.reason


@pytest.mark.parametrize(
    ("provider", "url", "header"),
    [
        ("deepseek", "https://api.deepseek.com/v1/models", "authorization"),
        ("anthropic", "https://api.anthropic.com/v1/models", "x-api-key"),
        (
            "google",
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
            "x-goog-api-key",
        ),
        # OpenRouter's /models is PUBLIC (200 for any key), so it cannot judge one.
        ("openrouter", "https://openrouter.ai/api/v1/key", "authorization"),
        # A login flavour checks against the provider it stores under.
        ("xai-oauth", "https://api.x.ai/v1/models", "authorization"),
    ],
)
def test_the_key_goes_to_the_registry_host_in_a_header(
    provider: str, url: str, header: str
) -> None:
    result, seen = _run(provider, lambda _r: httpx.Response(200, json={}))
    assert result.valid is True
    assert len(seen) == 1, "exactly one request per check"
    request = seen[0]
    assert str(request.url) == url
    assert request.method == "GET"
    # In a header, never the query string (which access logs keep).
    assert SECRET in request.headers[header]
    assert SECRET not in str(request.url)


def test_a_provider_that_needs_no_key_is_not_checked() -> None:
    result, seen = _run("ollama", lambda _r: httpx.Response(500))
    assert result == key_check.KeyCheck(None, None)
    assert seen == []


def test_the_check_is_bounded() -> None:
    """The timeout reaches the request, so a hung provider cannot hold a save."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(request.extensions.get("timeout", {}))
        return httpx.Response(200, json={})

    _run("deepseek", handler, timeout=2.5)
    assert captured and all(value == 2.5 for value in captured.values()), captured
    assert key_check.KEY_CHECK_TIMEOUT_S <= 10


def test_every_key_accepting_cloud_provider_has_a_check() -> None:
    """A new provider must not silently fall into "unsupported" (always unverified)."""
    from local_operator.providers.registry import (
        PROVIDER_REGISTRY,
        credential_provider_id,
    )

    for provider in PROVIDER_REGISTRY:
        storage = get_provider_definition(credential_provider_id(provider.id))
        if storage is None or storage.env_keys is None or storage.allows_missing_api_key:
            continue
        assert key_check._request(storage, SECRET) is not None, storage.id
