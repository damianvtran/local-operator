"""Live model discovery: layered over the registry, never underneath it.

The registry only knows the models that existed when it was last edited, so a
newly released model is unreachable until discovery surfaces it. The danger is
the other direction: every provider listing is poorer than the bundled data --
Anthropic's returns nothing but an id and a display name -- so a naive
"live overwrites static" merge produces zero prices, missing output caps and a
``context_window`` of ``-1`` whose compaction threshold never fires.

These tests pin both halves: each transport parses a realistic captured payload,
and every merge rule is asserted on its specific field so that reverting the rule
fails a test rather than silently degrading a session.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.model import discovery
from local_operator.model.discovery import (
    GEMINI_MAX_PAGES,
    LYING_MAX_TOKENS,
    DiscoveredModel,
    available_models,
    fetch_models,
    merge_models,
)
from local_operator.model.registry import ModelInfo, static_models
from local_operator.providers.registry import PROVIDER_REGISTRY


class _Response:
    """The slice of ``httpx.Response`` the transports actually touch."""

    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> object:
        # A body that is an exception models a 200 whose payload will not decode,
        # which is a real provider failure mode (an HTML error page served with a
        # 200 by a proxy) and must land on the same ``None`` as a 500.
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _StubClient:
    """A stand-in for ``httpx.Client`` that never touches the network.

    Responses are consumed in order; an exception in the queue is raised from
    ``get`` to model a transport failure. Exhausting the queue is an assertion
    failure rather than a silent extra ``None``, so a test that expects N
    requests fails loudly if the code issues N+1.
    """

    def __init__(self, responses: list[object], *, sticky: bool = False) -> None:
        self._responses = list(responses)
        self._sticky = sticky
        self.calls: list[tuple[str, dict, dict, float]] = []

    def get(self, url: str, *, headers: dict, params: dict, timeout: float) -> _Response:
        self.calls.append((url, dict(headers), dict(params), timeout))
        if self._sticky:
            item = self._responses[0]
        else:
            assert self._responses, f"unexpected extra request to {url}"
            item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, _Response)
        return item


def _info(model_id: str = "m", **overrides: object) -> ModelInfo:
    """A registry row with only the fields a test cares about spelled out."""
    fields: dict[str, object] = {
        "id": model_id,
        "name": f"Static {model_id}",
        "description": "bundled",
    }
    fields.update(overrides)
    return ModelInfo(**fields)  # type: ignore[arg-type]


# -- OpenAI-compatible transport ---------------------------------------------


_OPENROUTER_BODY = {
    "data": [
        {
            "id": "anthropic/claude-sonnet-4",
            "name": "Anthropic: Claude Sonnet 4",
            "context_length": 200_000,
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015",
                "input_cache_read": "0.0000003",
            },
            "top_provider": {"context_length": 200_000, "max_completion_tokens": 64_000},
            "architecture": {"input_modalities": ["text", "image"], "modality": "text+image->text"},
        },
        {
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek V3",
            "context_length": 64_000,
            "pricing": {"prompt": "0.00000027", "completion": "0.0000011"},
            "architecture": {"input_modalities": ["text"]},
        },
    ]
}


def test_openai_compat_parses_a_captured_openrouter_payload() -> None:
    client = _StubClient([_Response(200, _OPENROUTER_BODY)])
    rows = fetch_models("openrouter", api_key="k", client=client)

    assert rows is not None
    assert [row.id for row in rows] == ["anthropic/claude-sonnet-4", "deepseek/deepseek-chat"]
    sonnet = rows[0]
    assert sonnet.name == "Anthropic: Claude Sonnet 4"
    assert sonnet.context_window == 200_000
    assert sonnet.max_tokens == 64_000
    # Per-token strings are scaled to the per-million unit the rest of the app
    # uses; failing to scale understates cost by a factor of a million.
    assert sonnet.input_price == pytest.approx(3.0)
    assert sonnet.output_price == pytest.approx(15.0)
    assert sonnet.cache_read_price == pytest.approx(0.3)
    assert sonnet.supports_images is True
    assert sonnet.supports_prompt_cache is True
    assert rows[1].supports_images is False
    assert rows[1].supports_prompt_cache is False


def test_openai_compat_reads_the_legacy_modality_string() -> None:
    body = {
        "data": [
            {"id": "vision", "architecture": {"modality": "text+image->text"}},
            {"id": "painter", "architecture": {"modality": "text->image"}},
        ]
    }
    rows = fetch_models("openrouter", api_key="k", client=_StubClient([_Response(200, body)]))

    assert rows is not None
    # Only the left of the arrow is input: a model that generates images is not
    # one you can send an image to.
    assert [row.supports_images for row in rows] == [True, False]


def test_openai_compat_accepts_a_models_envelope() -> None:
    body = {"models": [{"id": "local-model", "context_window": 32_768}]}
    rows = fetch_models("ollama", api_key=None, client=_StubClient([_Response(200, body)]))

    assert rows is not None
    assert [(row.id, row.context_window) for row in rows] == [("local-model", 32_768)]


def test_openai_compat_accepts_a_bare_list() -> None:
    body = [{"id": "bare", "max_context_length": 8_192}]
    rows = fetch_models("deepseek", api_key="k", client=_StubClient([_Response(200, body)]))

    assert rows is not None
    assert [(row.id, row.context_window) for row in rows] == [("bare", 8_192)]


def test_openai_compat_sends_a_bearer_key_and_omits_it_when_keyless() -> None:
    keyed = _StubClient([_Response(200, {"data": []})])
    fetch_models("deepseek", api_key="sk-abc", client=keyed)
    assert keyed.calls[0][1]["Authorization"] == "Bearer sk-abc"

    keyless = _StubClient([_Response(200, {"data": []})])
    fetch_models("ollama", api_key=None, client=keyless)
    # A local Ollama rejects a bearer header it never issued, so a keyless
    # provider must send no Authorization at all rather than "Bearer None".
    assert "Authorization" not in keyless.calls[0][1]
    assert keyless.calls[0][0] == "http://localhost:11434/v1/models"


# -- Anthropic transport ------------------------------------------------------


_ANTHROPIC_BODY = {
    "data": [
        {
            "id": "claude-opus-5-20260601",
            "display_name": "Claude Opus 5",
            "created_at": "2026-06-01T00:00:00Z",
            "type": "model",
        },
        {
            "id": "claude-sonnet-4-20250514",
            "display_name": "Claude Sonnet 4",
            "created_at": "2025-05-14T00:00:00Z",
            "type": "model",
        },
    ],
    "has_more": False,
}


def test_anthropic_returns_ids_and_display_names_with_zeroed_numbers() -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    rows = fetch_models("anthropic", api_key="sk-ant", client=client)

    assert rows is not None
    assert [row.id for row in rows] == ["claude-opus-5-20260601", "claude-sonnet-4-20250514"]
    assert rows[0].name == "Claude Opus 5"
    # The listing carries no limits and no prices, so the transport must report
    # "unknown" as zero and leave the numbers to the merge. Any invented default
    # here is what produced a session running at context_window = -1.
    assert rows[0].context_window == 0
    assert rows[0].max_tokens == 0
    assert rows[0].input_price == 0.0
    assert rows[0].output_price == 0.0
    assert client.calls[0][0] == "https://api.anthropic.com/v1/models"
    assert client.calls[0][2]["limit"] == 1000


def test_anthropic_authenticates_an_api_key_with_x_api_key() -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    fetch_models("anthropic", api_key="sk-ant", client=client)

    headers = client.calls[0][1]
    assert headers["x-api-key"] == "sk-ant"
    assert "Authorization" not in headers
    assert headers["anthropic-version"] == discovery.ANTHROPIC_VERSION


def test_anthropic_authenticates_an_oauth_token_with_a_bearer_and_the_beta() -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    fetch_models("anthropic", api_key="oauth-access", is_oauth=True, client=client)

    headers = client.calls[0][1]
    # An OAuth access token sent as x-api-key is 401'd, and the beta opt-in is
    # required for the token to be accepted at all.
    assert headers["Authorization"] == "Bearer oauth-access"
    assert headers["anthropic-beta"] == discovery.ANTHROPIC_OAUTH_BETA
    assert "x-api-key" not in headers


def test_anthropic_base_url_override_does_not_double_the_version_segment() -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    fetch_models("anthropic", api_key="k", base_url="https://proxy.internal/v1", client=client)

    assert client.calls[0][0] == "https://proxy.internal/v1/models"


# -- Gemini transport ---------------------------------------------------------


def _gemini_page(*entries: dict, next_token: str | None = None) -> _Response:
    page: dict[str, object] = {"models": list(entries)}
    if next_token:
        page["nextPageToken"] = next_token
    return _Response(200, page)


def _gemini_entry(model_id: str, **overrides: object) -> dict:
    entry: dict[str, object] = {
        "name": f"models/{model_id}",
        "displayName": model_id.upper(),
        "inputTokenLimit": 1_048_576,
        "outputTokenLimit": 65_536,
        "supportedGenerationMethods": ["generateContent", "countTokens"],
    }
    entry.update(overrides)
    return entry


def test_gemini_strips_the_models_prefix_and_follows_pagination() -> None:
    client = _StubClient(
        [
            _gemini_page(_gemini_entry("gemini-2.5-pro"), next_token="page-2"),
            _gemini_page(_gemini_entry("gemini-2.5-flash")),
        ]
    )
    rows = fetch_models("google", api_key="AIza", client=client)

    assert rows is not None
    # The resource path prefix has to go: every other part of the system
    # addresses the bare id.
    assert [row.id for row in rows] == ["gemini-2.5-pro", "gemini-2.5-flash"]
    assert rows[0].context_window == 1_048_576
    assert rows[0].max_tokens == 65_536
    assert len(client.calls) == 2
    assert client.calls[0][2]["key"] == "AIza"
    assert "pageToken" not in client.calls[0][2]
    assert client.calls[1][2]["pageToken"] == "page-2"


def test_gemini_drops_entries_that_cannot_generate_content() -> None:
    client = _StubClient(
        [
            _gemini_page(
                _gemini_entry("gemini-2.5-pro"),
                _gemini_entry("text-embedding-004", supportedGenerationMethods=["embedContent"]),
                _gemini_entry("mystery-model", supportedGenerationMethods=None),
            )
        ]
    )
    rows = fetch_models("google", api_key="AIza", client=client)

    assert rows is not None
    # An embedding model in the picker 400s on every message, which is worse
    # than not offering it.
    assert [row.id for row in rows] == ["gemini-2.5-pro"]


def test_gemini_stops_at_the_page_cap() -> None:
    pages = [
        _gemini_page(_gemini_entry(f"gemini-{index}"), next_token=f"page-{index}")
        for index in range(GEMINI_MAX_PAGES + 5)
    ]
    client = _StubClient(pages)
    rows = fetch_models("google", api_key="AIza", client=client)

    assert rows is not None
    # Without the cap, a provider that always returns a fresh token makes opening
    # the picker an unbounded run of requests.
    assert len(client.calls) == GEMINI_MAX_PAGES
    assert len(rows) == GEMINI_MAX_PAGES


def test_gemini_stops_when_the_page_token_repeats() -> None:
    client = _StubClient(
        [_gemini_page(_gemini_entry("gemini-2.5-pro"), next_token="stuck")], sticky=True
    )
    rows = fetch_models("google", api_key="AIza", client=client)

    assert rows is not None
    # The first token advances (from none to "stuck"), so page two is fetched;
    # seeing the SAME token again is what proves the server is not paginating.
    # Without the guard this would burn all 25 pages re-reading one page.
    assert len(client.calls) == 2
    assert [row.id for row in rows] == ["gemini-2.5-pro"]


def test_gemini_fails_the_whole_listing_when_a_later_page_fails() -> None:
    client = _StubClient(
        [
            _gemini_page(_gemini_entry("gemini-2.5-pro"), next_token="page-2"),
            _Response(500, {}),
        ]
    )
    # A truncated catalogue presented as authoritative is indistinguishable from
    # a provider that dropped a model, so partial pagination is a failure.
    assert fetch_models("google", api_key="AIza", client=client) is None


def test_gemini_makes_no_request_without_a_key() -> None:
    client = _StubClient([])
    assert fetch_models("google", api_key=None, client=client) is None
    # The key is a query parameter, so a keyless request is a guaranteed 403.
    assert client.calls == []


# -- failure is not emptiness -------------------------------------------------


def test_a_server_error_is_a_failure_not_an_empty_listing() -> None:
    client = _StubClient([_Response(500, {"error": "boom"})])
    assert fetch_models("deepseek", api_key="k", client=client) is None


def test_an_unauthorized_response_is_a_failure_even_with_a_valid_body() -> None:
    # The body is a perfectly well-formed listing: only the STATUS says this is
    # not an answer. Gateways do serve a shaped envelope alongside a 401 or a
    # 429, and parsing it would let a rate-limited request replace the whole
    # catalogue with whatever placeholder the error page carries.
    client = _StubClient([_Response(401, {"data": [{"id": "leaked-placeholder"}]})])
    assert fetch_models("deepseek", api_key="k", client=client) is None


def test_a_timeout_is_a_failure() -> None:
    client = _StubClient([httpx.TimeoutException("timed out")])
    assert fetch_models("deepseek", api_key="k", client=client) is None


def test_an_undecodable_body_is_a_failure() -> None:
    client = _StubClient([_Response(200, ValueError("not json"))])
    assert fetch_models("deepseek", api_key="k", client=client) is None


def test_an_unrecognised_envelope_is_a_failure() -> None:
    client = _StubClient([_Response(200, {"object": "list", "results": []})])
    # An envelope we cannot read must degrade to the registry, not claim the
    # provider has no models.
    assert fetch_models("deepseek", api_key="k", client=client) is None


def test_a_successful_empty_listing_is_an_empty_list() -> None:
    client = _StubClient([_Response(200, {"data": []})])
    rows = fetch_models("deepseek", api_key="k", client=client)

    # Distinct from None: the provider answered, and the answer was "nothing".
    assert rows == []
    assert rows is not None


def test_an_entry_without_an_id_is_dropped_not_fatal() -> None:
    body = {"data": [{"name": "nameless"}, {"id": "usable"}]}
    rows = fetch_models("deepseek", api_key="k", client=_StubClient([_Response(200, body)]))

    assert rows is not None
    assert [row.id for row in rows] == ["usable"]


def test_a_provider_without_a_listing_endpoint_makes_no_request() -> None:
    client = _StubClient([])
    # A base_url is supplied on purpose: "no listing endpoint" is a property of
    # the provider, not merely an absent URL, so an override must not be able to
    # talk the mock wire into a request that has no server behind it.
    assert (
        fetch_models("test", api_key="k", base_url="https://example.invalid", client=client) is None
    )
    assert client.calls == []
    assert "test" in discovery.NO_LISTING_PROVIDERS
    assert "anthropic" not in discovery.NO_LISTING_PROVIDERS


def test_no_listing_providers_is_derived_from_the_provider_registry() -> None:
    expected = {
        definition.id
        for definition in PROVIDER_REGISTRY
        if definition.wire == "mock" or not definition.base_url
    }

    # Derived, not hardcoded: with a literal id list, the provider added next is
    # always the one that silently gets no listing.
    assert discovery.NO_LISTING_PROVIDERS == expected
    assert expected, "the registry should still contain at least one unlistable provider"


def test_an_unknown_provider_makes_no_request() -> None:
    client = _StubClient([])
    assert fetch_models("not-a-provider", api_key="k", client=client) is None
    assert client.calls == []


def test_a_credential_alias_reaches_the_same_listing() -> None:
    client = _StubClient([_Response(200, {"data": [{"id": "grok-4"}]})])
    rows = fetch_models("xai-oauth", api_key="tok", is_oauth=True, client=client)

    assert rows is not None
    assert [row.id for row in rows] == ["grok-4"]
    assert client.calls[0][0] == "https://api.x.ai/v1/models"


def test_a_credential_alias_merges_the_vendor_registry_rows(tmp_path) -> None:
    client = _StubClient([_Response(200, {"data": [{"id": "grok-4"}]})])
    models, status = available_models(
        "xai-oauth", api_key="tok", is_oauth=True, client=client, cache_dir=tmp_path
    )

    assert status == "ok"
    # xai-oauth is a login flavour of xai, not a separate catalogue. Without
    # following store_credentials_as it comes back with no static rows, so every
    # bundled Grok price and window is lost the moment the user logs in by OAuth.
    by_id = {row.id: row for row in models}
    assert set(static_models("xai")) <= set(by_id)
    priced = next(row for name, row in by_id.items() if name in static_models("xai"))
    assert priced.input_price > 0


# -- merge rules --------------------------------------------------------------


def test_merge_keeps_a_static_only_id_the_listing_omits() -> None:
    static = {"kept": _info("kept", context_window=200_000), "listed": _info("listed")}
    merged = merge_models(static, [DiscoveredModel(id="listed")])

    by_id = {row.id: row for row in merged}
    # Pruning static ids makes a model the user runs today unreachable, because
    # gateways filter listings by entitlement.
    assert "kept" in by_id
    assert by_id["kept"].context_window == 200_000


def test_merge_adds_a_live_only_id() -> None:
    merged = merge_models({"old": _info("old")}, [DiscoveredModel(id="brand-new")])

    assert {row.id for row in merged} == {"brand-new", "old"}


def test_merge_returns_the_whole_registry_when_the_fetch_failed() -> None:
    static = {"a": _info("a"), "b": _info("b")}
    merged = merge_models(static, None)

    # None means "keep what we had", so a failed listing must not shrink the list.
    assert {row.id for row in merged} == {"a", "b"}


def test_merge_puts_live_rows_before_registry_only_rows() -> None:
    static = {"old": _info("old"), "listed": _info("listed")}
    merged = merge_models(static, [DiscoveredModel(id="listed"), DiscoveredModel(id="fresh")])

    # Providers list newest-first, and a model released this week is the reason
    # discovery exists; burying it under the registry defeats the feature.
    assert [row.id for row in merged] == ["listed", "fresh", "old"]


def test_merge_drops_a_duplicated_live_id() -> None:
    merged = merge_models({}, [DiscoveredModel(id="dup"), DiscoveredModel(id="dup")])

    assert [row.id for row in merged] == ["dup"]


def test_merge_prefers_a_live_display_name() -> None:
    static = {"m": _info("m", name="Old Label")}
    merged = merge_models(static, [DiscoveredModel(id="m", name="New Label")])

    assert merged[0].name == "New Label"


def test_merge_keeps_the_static_name_when_the_live_name_is_blank() -> None:
    static = {"m": _info("m", name="Claude Sonnet 4")}
    merged = merge_models(static, [DiscoveredModel(id="m", name="   ")])

    assert merged[0].name == "Claude Sonnet 4"


def test_merge_keeps_the_static_name_when_the_live_name_echoes_the_id() -> None:
    static = {"claude-sonnet-4": _info("claude-sonnet-4", name="Claude Sonnet 4")}
    merged = merge_models(static, [DiscoveredModel(id="claude-sonnet-4", name="claude-sonnet-4")])

    # Endpoints that echo the id into `name` must not replace a real label with
    # the raw id.
    assert merged[0].name == "Claude Sonnet 4"


def test_merge_uses_the_echoed_name_when_the_registry_has_nothing_better() -> None:
    merged = merge_models({}, [DiscoveredModel(id="mystery", name="mystery")])

    # The echo rule is about protecting a better name, not about rejecting ids:
    # a live-only model still needs something to display.
    assert merged[0].name == "mystery"


def test_merge_prefers_a_live_context_window() -> None:
    static = {"m": _info("m", context_window=200_000)}
    merged = merge_models(static, [DiscoveredModel(id="m", context_window=1_000_000)])

    assert merged[0].context_window == 1_000_000


def test_merge_keeps_the_static_context_window_when_live_reports_zero() -> None:
    static = {"m": _info("m", context_window=200_000)}
    merged = merge_models(static, [DiscoveredModel(id="m", context_window=0)])

    # This is the compaction bug: a zeroed window yields a threshold that never
    # trips, and the session runs until the provider rejects the request.
    assert merged[0].context_window == 200_000


def test_merge_rejects_a_live_max_tokens_of_exactly_4096() -> None:
    static = {"m": _info("m", max_tokens=32_000)}
    merged = merge_models(static, [DiscoveredModel(id="m", max_tokens=LYING_MAX_TOKENS)])

    # 4096 is an OpenAI-compat listing default, not a limit; believing it caps a
    # 32k-output model at 4k.
    assert merged[0].max_tokens == 32_000


def test_merge_accepts_4096_when_the_registry_knows_less() -> None:
    static = {"m": _info("m", max_tokens=2_048)}
    merged = merge_models(static, [DiscoveredModel(id="m", max_tokens=LYING_MAX_TOKENS)])

    # The 4096 rule is scoped to "a larger static value exists"; otherwise the
    # live number is the best information available.
    assert merged[0].max_tokens == LYING_MAX_TOKENS


def test_merge_accepts_a_live_max_tokens_that_is_not_the_lying_default() -> None:
    static = {"m": _info("m", max_tokens=32_000)}
    merged = merge_models(static, [DiscoveredModel(id="m", max_tokens=64_000)])

    assert merged[0].max_tokens == 64_000


def test_merge_keeps_the_static_max_tokens_when_live_reports_zero() -> None:
    static = {"m": _info("m", max_tokens=32_000)}
    merged = merge_models(static, [DiscoveredModel(id="m", max_tokens=0)])

    assert merged[0].max_tokens == 32_000


def test_merge_keeps_known_prices_when_the_listing_reports_zero() -> None:
    static = {"m": _info("m", input_price=15.0, output_price=75.0, cache_reads_price=1.5)}
    merged = merge_models(static, [DiscoveredModel(id="m")])

    # A zero price is "unknown", never "free": the UI prints the literal word
    # `free` when both legs are zero, so this would advertise Opus as free.
    assert merged[0].input_price == pytest.approx(15.0)
    assert merged[0].output_price == pytest.approx(75.0)
    assert merged[0].cache_read_price == pytest.approx(1.5)


def test_merge_prefers_live_prices_when_they_are_present() -> None:
    static = {"m": _info("m", input_price=15.0, output_price=75.0, cache_reads_price=1.5)}
    live = [
        DiscoveredModel(id="m", input_price=3.0, output_price=15.0, cache_read_price=0.3),
    ]
    merged = merge_models(static, live)

    assert merged[0].input_price == pytest.approx(3.0)
    assert merged[0].output_price == pytest.approx(15.0)
    assert merged[0].cache_read_price == pytest.approx(0.3)


def test_merge_ors_capability_flags_so_a_terse_listing_cannot_downgrade() -> None:
    static = {"m": _info("m", supports_images=True, supports_prompt_cache=True)}
    merged = merge_models(static, [DiscoveredModel(id="m")])

    assert merged[0].supports_images is True
    assert merged[0].supports_prompt_cache is True


def test_merge_ors_capability_flags_so_a_listing_can_upgrade() -> None:
    static = {"m": _info("m", supports_images=False, supports_prompt_cache=False)}
    live = [DiscoveredModel(id="m", supports_images=True, supports_prompt_cache=True)]
    merged = merge_models(static, live)

    assert merged[0].supports_images is True
    assert merged[0].supports_prompt_cache is True


def test_merge_normalises_the_registry_unknown_sentinels_to_zero() -> None:
    static = {"m": _info("m", context_window=-1, max_tokens=-1)}
    merged = merge_models(static, None)

    # -1 survives arithmetic and yields a plausible-looking threshold, which is
    # exactly how the compaction bug hid; 0 is falsy and cannot.
    assert merged[0].context_window == 0
    assert merged[0].max_tokens == 0


def test_anthropic_merge_keeps_static_numbers_and_surfaces_a_new_id() -> None:
    static = static_models("anthropic")
    known_id = "claude-sonnet-4-20250514"
    assert known_id in static, "registry no longer ships the id this test pins"
    known_static = static[known_id]

    live = [
        DiscoveredModel(id="claude-opus-5-20260601", name="Claude Opus 5"),
        DiscoveredModel(id=known_id, name="Claude Sonnet 4"),
    ]
    by_id = {row.id: row for row in merge_models(static, live)}

    # The known model keeps everything the id-and-name-only listing lacks.
    assert by_id[known_id].context_window == known_static.context_window
    assert by_id[known_id].input_price == pytest.approx(known_static.input_price)
    assert by_id[known_id].output_price == pytest.approx(known_static.output_price)
    assert by_id[known_id].max_tokens == known_static.max_tokens
    # The new model is reachable, and honestly reports what the listing did not
    # say rather than inventing a window.
    assert by_id["claude-opus-5-20260601"].name == "Claude Opus 5"
    assert by_id["claude-opus-5-20260601"].context_window == 0


# -- available_models ---------------------------------------------------------


def test_available_models_reports_ok_and_merges_the_registry(tmp_path) -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path
    )

    assert status == "ok"
    by_id = {row.id: row for row in models}
    assert "claude-opus-5-20260601" in by_id
    assert by_id["claude-sonnet-4-20250514"].context_window > 0


def test_available_models_serves_a_fresh_cache_without_a_request(tmp_path) -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    first_models, first_status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path
    )
    second_models, second_status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path
    )

    assert first_status == "ok"
    # A picker that re-listed on every keystroke would block on the network; the
    # cache is what makes the second open instant.
    assert second_status == "cached"
    assert len(client.calls) == 1
    assert [row.id for row in second_models] == [row.id for row in first_models]


def test_available_models_reports_cached_when_a_stale_refetch_fails(tmp_path) -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY), httpx.ConnectError("offline")])
    available_models("anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path)
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, ttl_s=-1
    )

    assert status == "cached"
    # Stale beats absent: the discovered ids are days old at worst, and losing
    # them because the network blipped is the regression this guards.
    assert "claude-opus-5-20260601" in {row.id for row in models}


def test_available_models_reports_static_when_a_cold_fetch_fails(tmp_path) -> None:
    client = _StubClient([httpx.ConnectError("offline")])
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path
    )

    assert status == "static"
    # Still usable offline: the registry alone is a working picker.
    assert {row.id for row in models} == set(static_models("anthropic"))


def test_available_models_reports_empty_on_a_zero_model_listing(tmp_path) -> None:
    client = _StubClient([_Response(200, {"data": []})])
    models, status = available_models("openrouter", api_key="k", client=client, cache_dir=tmp_path)

    # Distinct from "static": the provider answered, so the UI can say so rather
    # than implying the listing was never reached.
    assert status == "empty"
    assert models == []


def test_available_models_reports_static_without_a_listing_endpoint(tmp_path) -> None:
    client = _StubClient([])
    models, status = available_models("test", api_key="k", client=client, cache_dir=tmp_path)

    assert status == "static"
    assert client.calls == []
    assert models == []


def test_available_models_reports_unauthenticated_without_a_key(tmp_path) -> None:
    client = _StubClient([])
    models, status = available_models("anthropic", api_key=None, client=client, cache_dir=tmp_path)

    # The one status the user can act on, so it must not be folded into "static".
    assert status == "unauthenticated"
    assert client.calls == []
    assert {row.id for row in models} == set(static_models("anthropic"))


def test_available_models_lists_a_keyless_local_provider(tmp_path) -> None:
    body = {"data": [{"id": "qwen3:8b"}]}
    client = _StubClient([_Response(200, body)])
    models, status = available_models("ollama", api_key=None, client=client, cache_dir=tmp_path)

    # Ollama declares allows_missing_api_key, so a missing key is normal rather
    # than unauthenticated; treating it as unauthenticated makes every locally
    # pulled model invisible.
    assert status == "ok"
    assert [row.id for row in models] == ["qwen3:8b"]


def test_available_models_uses_a_cache_key_that_cannot_clobber_the_catalogue(tmp_path) -> None:
    catalogue_document = tmp_path / "openrouter.models.json"
    catalogue_document.write_text('{"fetched_at": 0, "payload": {"data": []}}', encoding="utf-8")
    body = {"data": [{"id": "vendor/model", "context_length": 1_000}]}
    client = _StubClient([_Response(200, body)])

    _, status = available_models("openrouter", api_key="k", client=client, cache_dir=tmp_path)

    assert status == "ok"
    # The aggregator catalogue owns the bare provider name; overwriting it with a
    # differently-shaped payload would be served as a fresh hit for a full day.
    assert catalogue_document.read_text(encoding="utf-8") == (
        '{"fetched_at": 0, "payload": {"data": []}}'
    )
    assert (tmp_path / "openrouter.models.models.json").exists()


def test_available_models_survives_a_broken_cache_layer(tmp_path, monkeypatch) -> None:
    def explode(*args: object, **kwargs: object) -> dict:
        raise OSError("cache is on fire")

    monkeypatch.setattr(discovery, "cached_listing", explode)
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=_StubClient([]), cache_dir=tmp_path
    )

    # A model picker must open even when the layer beneath it is broken; a
    # traceback here would take the whole session down.
    assert status == "static"
    assert {row.id for row in models} == set(static_models("anthropic"))
