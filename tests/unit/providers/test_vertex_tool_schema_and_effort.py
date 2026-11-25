"""Regression tests for the two google-vertex defects shipped in v0.108.0.

1. Every tool-carrying Vertex turn 400'd: ``function_declarations[].parameters``
   is Vertex's strict OpenAPI-subset ``Schema`` and rejects the JSON-Schema
   keywords our tool schemas use (``$defs``, ``$ref``, ``exclusiveMinimum``).
   Vertex now gets ``parametersJsonSchema``; AI Studio is left byte-identical.
2. Gemini offered no effort levels. Gemini 3+ takes
   ``generationConfig.thinkingConfig.thinkingLevel`` on ``generateContent``; the
   ladder now comes live from models.dev's ``reasoning_options`` with an
   offline family fallback, and a refused rung is shed by one retry.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
    ToolResult,
)
from local_operator.model import catalogue as catalogue_mod
from local_operator.model import configure as configure_mod
from local_operator.model import discovery, prices
from local_operator.model.effort import default_effort, supported_efforts
from local_operator.providers import clients as clients_mod
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import (
    GOOGLE_API_URL,
    VERTEX_API_URL,
    GoogleClient,
    ProviderError,
)

#: A schema holding every keyword the live Vertex 400 named.
_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "$defs": {"Item": {"type": "object", "properties": {"name": {"type": "string"}}}},
    "properties": {
        "timeout": {"type": "number", "exclusiveMinimum": 0},
        "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
    },
}


async def _unused_execute(*_args: Any, **_kwargs: Any) -> ToolResult:
    raise AssertionError("wire serialization must not execute tools")


def _tool() -> AgentTool:
    return AgentTool(
        name="probe",
        description="A probe tool",
        parameters=_JSON_SCHEMA,
        execute=_unused_execute,
    )


def _request(
    provider: str = "google-vertex",
    model_id: str = "gemini-4-neon",
    *,
    efforts: tuple[str, ...] = (),
    effort: str | None = None,
    tools: bool = True,
) -> ChatRequest:
    return ChatRequest(
        model=ModelSpec(
            provider=provider,
            model_id=model_id,
            context_window=1_000_000,
            max_output_tokens=8192,
            reasoning_efforts=efforts,
            reasoning_effort=effort,
        ),
        messages=[Message(role="user", content=[TextContent(text="hello")])],
        tools=[_tool()] if tools else [],
    )


def _sse_ok() -> httpx.Response:
    payload = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1},
    }
    return httpx.Response(
        200,
        content=f"data: {json.dumps(payload)}\n\n".encode(),
        headers={"content-type": "text/event-stream"},
    )


_OAUTH = OAuthAccess(
    access_token="ya29.synthetic",
    credential_id=1,
    email="user@example.com",
    kind="oauth",
    raw={"project_id": "synthetic-project", "location": "global", "auth_mode": "oauth"},
)


async def _bodies(
    client_factory: Any,
    request: ChatRequest,
    *,
    api_key: str | None = None,
    oauth: OAuthAccess | None = None,
    responses: list[httpx.Response] | None = None,
) -> list[dict[str, Any]]:
    """Every request body the client sent, answered from ``responses`` in order."""
    seen: list[dict[str, Any]] = []
    queued = list(responses or [])

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(http_request.content))
        return queued.pop(0) if queued else _sse_ok()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = client_factory(http)
        _ = [e async for e in client.stream(request, api_key=api_key, oauth_access=oauth)]
    return seen


def _vertex(http: httpx.AsyncClient) -> GoogleClient:
    return GoogleClient(base_url=VERTEX_API_URL, vertex=True, http_client=http)


def _studio(http: httpx.AsyncClient) -> GoogleClient:
    return GoogleClient(base_url=GOOGLE_API_URL, http_client=http)


# -- bug 1: tool schemas ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_key", "oauth"), [("vx-key", None), (None, _OAUTH)], ids=["api-key", "oauth"]
)
async def test_vertex_sends_tool_schema_as_parameters_json_schema(
    api_key: str | None, oauth: OAuthAccess | None
) -> None:
    """Both Vertex endpoints carry the JSON Schema verbatim under the key that
    accepts it; ``parameters`` (the OpenAPI subset that 400'd) is absent."""
    [body] = await _bodies(_vertex, _request(), api_key=api_key, oauth=oauth)
    [declaration] = body["tools"][0]["function_declarations"]
    assert "parameters" not in declaration
    assert declaration["parametersJsonSchema"] == _JSON_SCHEMA


@pytest.mark.asyncio
async def test_ai_studio_tool_schema_key_is_unchanged() -> None:
    """AI Studio is a working path: it keeps ``parameters``, byte-identical."""
    [body] = await _bodies(_studio, _request(provider="google"), api_key="ai-key")
    [declaration] = body["tools"][0]["function_declarations"]
    assert declaration == {
        "name": "probe",
        "description": "A probe tool",
        "parameters": _JSON_SCHEMA,
    }
    assert "parametersJsonSchema" not in declaration


# -- bug 2: the wire ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
@pytest.mark.parametrize("provider", ["google", "google-vertex"])
async def test_offline_neon_effort_rides_as_thinking_level(
    provider: str, level: str, disk_cache: Path
) -> None:
    spec = configure_mod.build_model_spec(provider, "gemini-4-neon")
    assert spec.reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert spec.reasoning_effort is None  # leave the model's default implicit
    request = _request(provider=provider, tools=False).model_copy(
        update={"model": spec.model_copy(update={"reasoning_effort": level})}
    )
    factory = _vertex if provider == "google-vertex" else _studio
    [body] = await _bodies(factory, request, api_key="k")
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": level}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("efforts", "effort"),
    [(("low", "medium", "high"), None), (("low", "medium", "high"), "xhigh"), ((), "high")],
    ids=["unset", "off-ladder", "no-ladder"],
)
async def test_no_thinking_config_without_an_on_ladder_effort(
    efforts: tuple[str, ...], effort: str | None
) -> None:
    """Unset (or a level the ladder does not carry) sends no ``thinkingConfig``,
    so the model's own default depth applies."""
    [body] = await _bodies(_vertex, _request(efforts=efforts, effort=effort), api_key="k")
    assert "thinkingConfig" not in body["generationConfig"]


@pytest.mark.asyncio
async def test_unsupported_thinking_level_is_retried_once_without_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale ladder costs depth, never the turn: the 400 Google answers for a
    refused rung (wording measured live) triggers ONE identical retry without
    ``thinkingConfig``, and a warning names the model and level."""
    refused = httpx.Response(
        400,
        json={
            "error": {
                "code": 400,
                "message": "Thinking level is unsupported: THINKING_LEVEL_XHIGH",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    request = _request(model_id="gemini-3.8-flash", efforts=("low", "xhigh"), effort="xhigh")
    with caplog.at_level(logging.WARNING, logger="local_operator.providers.clients"):
        bodies = await _bodies(_vertex, request, api_key="k", responses=[refused])
    assert len(bodies) == 2
    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "xhigh"}
    assert "thinkingConfig" not in bodies[1]["generationConfig"]
    # Everything else is the same request.
    first = dict(bodies[0], generationConfig={})
    second = dict(bodies[1], generationConfig={})
    assert first == second
    assert any(
        "gemini-3.8-flash" in record.getMessage() and "'xhigh'" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_other_400s_are_not_retried() -> None:
    """An ordinary request defect (here an invalid enum, worded differently by
    Google) surfaces unchanged after ONE request."""
    invalid = httpx.Response(
        400,
        json={"error": {"code": 400, "message": "Invalid value at 'thinking_level' (bogus)"}},
    )
    request = _request(efforts=("low", "high"), effort="high")
    with pytest.raises(ProviderError):
        await _bodies(_vertex, request, api_key="k", responses=[invalid])


@pytest.mark.asyncio
async def test_a_refusal_without_thinking_config_is_not_retried() -> None:
    """The phrase alone is not enough: with no ``thinkingConfig`` in the body
    there is nothing to shed, so a retry would only repeat the failure."""
    refused = httpx.Response(400, json={"error": {"message": "Thinking level is unsupported"}})
    seen: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.append(http_request)
        return refused

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ProviderError):
            _ = [e async for e in _vertex(http).stream(_request(), api_key="k")]
    assert len(seen) == 1


# -- bug 2: the ladder --------------------------------------------------------

_MODELS_DEV_BODY: dict[str, Any] = {
    provider: {
        "models": {
            "gemini-4-neon": {
                "name": "Gemini 4 Neon",
                "reasoning_options": [
                    {"type": "effort", "values": ["low", "medium", "high", "xhigh"]}
                ],
            },
            "gemini-3.8-flash": {
                "name": "Gemini 3.8 Flash",
                "reasoning_options": [{"type": "effort", "values": ["high", "low", "medium"]}],
            },
            # A budget is a different wire parameter: never a ladder.
            "gemini-2.5-pro": {
                "name": "Gemini 2.5 Pro",
                "reasoning_options": [{"type": "budget_tokens", "min": 128, "max": 32768}],
            },
            # Unknown words are dropped by the one ingest rule.
            "gemini-9-odd": {
                "reasoning_options": [{"type": "effort", "values": ["turbo", "low"]}],
            },
        }
    }
    for provider in ("google", "google-vertex", "anthropic")
}


@pytest.fixture
def models_dev(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """The projection as ``models_dev_providers`` would read it from disk."""
    projected = prices.project(_MODELS_DEV_BODY, None)["providers"]
    monkeypatch.setattr(prices, "models_dev_providers", lambda **_kwargs: projected)
    configure_mod._catalogue_effort_memo.clear()
    yield projected
    configure_mod._catalogue_effort_memo.clear()


def test_capture_version_bumped_for_effort_levels() -> None:
    """A v4 document carries no ladder; the bump forces one refetch."""
    assert prices.PRICE_CATALOGUE_CAPTURE == 5
    assert prices.project(_MODELS_DEV_BODY, None)["capture"] == 5


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
def test_models_dev_projects_the_live_ladder(models_dev: dict[str, Any], provider: str) -> None:
    ladder = prices.catalogue_effort_ladder
    assert ladder(provider, "gemini-4-neon") == ("low", "medium", "high", "xhigh")
    assert ladder(provider, "gemini-3.8-flash") == ("low", "medium", "high")
    assert ladder(provider, "gemini-2.5-pro") is None
    assert ladder(provider, "gemini-9-odd") == ("low",)
    assert ladder(provider, "not-listed") is None


def test_catalogue_ladder_is_gated_to_providers_with_a_wire_mapping(
    models_dev: dict[str, Any],
) -> None:
    """The projection is generic but only Google consumes it: other providers
    state their own ladders or have verified table rows."""
    assert prices.catalogue_effort_ladder("anthropic", "gemini-4-neon") is None


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
def test_build_model_spec_prefers_the_catalogue_over_the_table(
    models_dev: dict[str, Any], provider: str
) -> None:
    # A projected ladder narrower than the family fallback proves precedence.
    spec = configure_mod.build_model_spec(provider, "gemini-9-odd")
    assert spec.reasoning_efforts == ("low",)
    # Nothing seeded: the model's own default applies until the user picks.
    assert spec.reasoning_effort is None


def test_build_model_spec_falls_back_to_the_table_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No document at all (offline, first run): the conservative family ladder."""
    monkeypatch.setattr(prices, "models_dev_providers", lambda **_kwargs: None)
    configure_mod._catalogue_effort_memo.clear()
    try:
        spec = configure_mod.build_model_spec("google-vertex", "gemini-5-ultra")
    finally:
        configure_mod._catalogue_effort_memo.clear()
    assert spec.reasoning_efforts == ("low", "medium", "high")
    assert spec.reasoning_effort is None


@pytest.mark.parametrize(
    "model_id", ["gemini-3.8-flash", "gemini-5-x", "models/gemini-3-pro"]
)
def test_offline_table_covers_gemini_3_and_later(model_id: str) -> None:
    assert supported_efforts(model_id) == ("low", "medium", "high")
    assert default_effort(model_id) is None


@pytest.mark.parametrize("model_id", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"])
def test_offline_table_leaves_budget_era_gemini_ladderless(model_id: str) -> None:
    assert supported_efforts(model_id) == ()


def test_discovered_vertex_rows_carry_the_catalogue_ladder(
    models_dev: dict[str, Any], tmp_path: Any
) -> None:
    """The listing every picker reads (``available_models`` and its disk-only
    twin) carries the ladder, so the desktop/TUI effort column shows it."""
    # No credential: the static (offline registry) rows, which must carry the
    # last-known catalogue ladder too.
    rows, _status = discovery.available_models("google-vertex", api_key=None, cache_dir=tmp_path)
    by_id = {row.id: row for row in rows}
    assert by_id["gemini-3.8-flash"].reasoning_efforts == ("low", "medium", "high")
    cached, _ = discovery.cached_available_models("google-vertex", cache_dir=tmp_path)
    assert {row.id: row.reasoning_efforts for row in cached}["gemini-3.8-flash"] == (
        "low",
        "medium",
        "high",
    )
    # A publisher-listing row with no ladder picks up the catalogue's ladder.
    live = discovery.DiscoveredModel(id="gemini-4-neon")
    (filled,) = discovery._with_catalogue_efforts("google-vertex", [live], tmp_path)
    assert filled.reasoning_efforts == ("low", "medium", "high", "xhigh")
    assert filled.reasoning_default_effort is None


def test_a_stated_listing_ladder_is_never_overwritten(models_dev: dict[str, Any]) -> None:
    row = discovery.DiscoveredModel(id="gemini-4-neon", reasoning_efforts=("high",))
    assert prices.with_catalogue_efforts("google-vertex", [row]) == [row]


# -- round-1 remediation (PR #4323) -------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no remembered thinking-level refusal."""
    monkeypatch.setattr(clients_mod, "_REFUSED_THINKING_LEVELS", set())


@pytest.fixture
def disk_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the default catalogue cache at ``tmp_path`` with every memo cold.

    The REAL disk path (``peek_listing`` -> ``_cache_path``), not a patched
    ``models_dev_providers``, because the defect under test is a memo that
    outlived the document it was derived from.
    """
    monkeypatch.setattr(catalogue_mod, "default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(prices, "_providers_memo", None)
    # A stale capture would schedule a real off-loop refetch; the repair is
    # not under test here, only that its pending state is not memoised.
    monkeypatch.setattr(prices, "_repair_unusable_capture", lambda _cache_dir: None)
    configure_mod._catalogue_effort_memo.clear()
    return tmp_path


def _write_models_dev(cache: Path, *, capture: int | None = None) -> None:
    payload = prices.project(_MODELS_DEV_BODY, None)
    if capture is not None:
        payload["capture"] = capture
    catalogue_mod._write_cache(cache / f"{prices.PRICE_CATALOGUE_KEY}.json", payload)


def test_catalogue_ladder_goes_live_once_the_document_lands(disk_cache: Path) -> None:
    """Review #1: a miss caused by NO usable document must not be memoised.

    Cold (no document) and stale-capture (the state every user is in right
    after a capture bump) both answer the offline table ladder; the first spec
    built after the real document lands, in the same process, is live.
    """
    try:
        # Neon now agrees with the projection offline. Use a listing-only id
        # whose projected ladder differs to keep each transition observable.
        model_id = "gemini-9-odd"
        cold = configure_mod.build_model_spec("google-vertex", model_id)
        assert cold.reasoning_efforts == ("low", "medium", "high")

        _write_models_dev(disk_cache, capture=prices.PRICE_CATALOGUE_CAPTURE - 1)
        stale = configure_mod.build_model_spec("google-vertex", model_id)
        assert stale.reasoning_efforts == ("low", "medium", "high")

        _write_models_dev(disk_cache)
        warm = configure_mod.build_model_spec("google-vertex", model_id)
        assert warm.reasoning_efforts == ("low",)
    finally:
        configure_mod._catalogue_effort_memo.clear()


def test_models_dev_providers_parses_once_per_document(
    disk_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review #4: the paint path reuses the parse until the document changes."""
    _write_models_dev(disk_cache)
    reads: list[str] = []
    real_peek = prices.peek_listing

    def counting_peek(key: str, **kwargs: Any) -> Any:
        reads.append(key)
        return real_peek(key, **kwargs)

    monkeypatch.setattr(prices, "peek_listing", counting_peek)
    first = prices.models_dev_providers()
    assert first is not None
    assert prices.models_dev_providers() is first
    assert len(reads) == 1
    # A rewrite (refetch, capture repair, 304 restamp) is picked up at once.
    _write_models_dev(disk_cache)
    assert prices.models_dev_providers() is not None
    assert len(reads) == 2


@pytest.mark.parametrize("provider", ["google", "google-vertex"])
def test_offline_gemini_ladder_on_google_hostings(disk_cache: Path, provider: str) -> None:
    """Review #2: the verified hostings keep the offline family ladder."""
    spec = configure_mod.build_model_spec(provider, "gemini-5-x")
    assert spec.reasoning_efforts == ("low", "medium", "high")


@pytest.mark.parametrize(
    ("provider", "model_id"),
    [("github-copilot", "gemini-3-pro"), ("radient", "google/gemini-3-pro")],
)
def test_offline_gemini_ladder_is_not_offered_on_other_routes(
    disk_cache: Path, provider: str, model_id: str
) -> None:
    """Review #2: ``thinkingLevel`` was verified on Google's own hostings only;
    a reseller route keeps its previous behaviour (no ladder from the table)."""
    assert supported_efforts(model_id, provider) == ()
    spec = configure_mod.build_model_spec(provider, model_id)
    assert spec.reasoning_efforts == ()
    # The provider-agnostic table answer is unchanged for other callers.
    assert supported_efforts(model_id) == ("low", "medium", "high")


@pytest.mark.asyncio
async def test_a_refused_thinking_level_is_remembered(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Review #3 / QA Q1: with a stale ladder, only the FIRST call pays the 400
    and logs; the next request omits the refused level up front."""
    refused = httpx.Response(
        400,
        json={"error": {"code": 400, "message": "Thinking level is unsupported: X"}},
    )
    request = _request(model_id="gemini-3.8-flash", efforts=("low", "xhigh"), effort="xhigh")
    statuses: list[int] = []
    bodies: list[dict[str, Any]] = []
    queued = [refused]

    def handler(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(http_request.content))
        response = queued.pop(0) if queued else _sse_ok()
        statuses.append(response.status_code)
        return response

    with caplog.at_level(logging.WARNING, logger="local_operator.providers.clients"):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = _vertex(http)
            for _ in range(2):
                _ = [e async for e in client.stream(request, api_key="k")]
    assert statuses == [400, 200, 200]
    assert "thinkingConfig" not in bodies[2]["generationConfig"]
    warnings = [r for r in caplog.records if "refused thinking level" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.asyncio
async def test_a_refusal_on_vertex_does_not_strip_ai_studio() -> None:
    """Vertex and AI Studio are separate backends: a refusal is per route."""
    refused = httpx.Response(
        400, json={"error": {"code": 400, "message": "Thinking level is unsupported: X"}}
    )
    vertex_request = _request(model_id="gemini-3.8-flash", efforts=("xhigh",), effort="xhigh")
    await _bodies(_vertex, vertex_request, api_key="k", responses=[refused])
    studio_request = _request(
        "google", model_id="gemini-3.8-flash", efforts=("xhigh",), effort="xhigh"
    )
    (body,) = await _bodies(_studio, studio_request, api_key="k")
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "xhigh"}
