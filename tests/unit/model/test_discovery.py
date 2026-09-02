"""Live model discovery: layered over the registry, never underneath it.

The registry only knows the models that existed when it was last edited, so a
newly released model is unreachable until discovery surfaces it, and its real
limits are unknowable until a listing states them -- Anthropic's ``/v1/models``
is the only place ``claude-opus-5`` says 1M rather than the shipped family floor.
The danger runs both ways: a listing may also be POORER than the bundled data (a
lean OpenAI-compatible gateway sends an id and nothing else), so a naive
"live overwrites static" merge produces zero prices, missing output caps and a
``context_window`` of ``-1`` whose compaction threshold never fires.

These tests pin both halves: each transport parses a realistic captured payload,
and every merge rule is asserted on its specific field so that reverting the rule
fails a test rather than silently degrading a session.
"""

from __future__ import annotations

import json
import time
import types
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from local_operator.model import discovery
from local_operator.model.discovery import (
    DEFAULT_TIMEOUT_S,
    GEMINI_MAX_PAGES,
    LYING_MAX_TOKENS,
    DiscoveredModel,
    available_models,
    fetch_models,
    merge_models,
)
from local_operator.model.registry import ModelInfo, static_models
from local_operator.providers.registry import PROVIDER_REGISTRY


class _Response(httpx.Response):
    """The slice of ``httpx.Response`` the transports actually touch."""

    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body

    def json(self, **kwargs: Any) -> object:
        # A body that is an exception models a 200 whose payload will not decode,
        # which is a real provider failure mode (an HTML error page served with a
        # 200 by a proxy) and must land on the same ``None`` as a 500.
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _StubClient(httpx.Client):
    """A stand-in for ``httpx.Client`` that never touches the network.

    Responses are consumed in order; an exception in the queue is raised from
    ``get`` to model a transport failure. Exhausting the queue is an assertion
    failure rather than a silent extra ``None``, so a test that expects N
    requests fails loudly if the code issues N+1.
    """

    def __init__(self, responses: Sequence[object], *, sticky: bool = False) -> None:
        self._responses = list(responses)
        self._sticky = sticky
        self.calls: list[tuple[str, dict[str, object], dict[str, object], float]] = []

    def get(
        self,
        url: Any,
        *,
        params: Any = None,
        headers: Any = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> _Response:
        self.calls.append((str(url), dict(headers or {}), dict(params or {}), timeout))
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


#: Captured from ChatGPT's account-scoped Codex catalogue on 2026-08-09,
#: trimmed to one selectable row and one internal hidden row.
_OPENAI_CODEX_BODY = {
    "models": [
        {
            "slug": "gpt-5.6-sol",
            "display_name": "GPT-5.6-Sol",
            "visibility": "list",
            "context_window": 272_000,
            "max_context_window": 272_000,
            "input_modalities": ["text", "image"],
        },
        {
            "slug": "codex-auto-review",
            "display_name": "Codex Auto Review",
            "visibility": "hide",
            "context_window": 272_000,
            "input_modalities": ["text", "image"],
        },
    ]
}

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
                "input_cache_write": "0.00000375",
                "input_cache_write_1h": "0.000006",
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
    # The five-minute write rate, scaled the same way; the `_1h` tier is ignored
    # because every write is billed at the 5m rate unless a caller asks otherwise.
    assert sonnet.cache_write_price == pytest.approx(3.75)
    assert sonnet.supports_images is True
    assert sonnet.supports_prompt_cache is True
    assert rows[1].supports_images is False
    assert rows[1].supports_prompt_cache is False
    assert rows[1].cache_write_price == 0.0


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


def test_openai_oauth_uses_the_account_scoped_codex_catalogue() -> None:
    """ChatGPT subscription tokens receive 403 from ``api.openai.com/v1/models``.

    The Codex catalogue is the endpoint OpenAI's own client uses for those
    tokens. It requires the account header and returns ``slug`` rows rather than
    the public API's ``id`` rows.
    """
    client = _StubClient([_Response(200, _OPENAI_CODEX_BODY)])
    rows = fetch_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=client,
    )

    assert rows is not None
    assert [(row.id, row.name) for row in rows] == [("gpt-5.6-sol", "GPT-5.6-Sol")]
    assert rows[0].context_window == 272_000
    assert rows[0].supports_images is True
    url, headers, params, _timeout = client.calls[0]
    assert url == discovery.OPENAI_CHATGPT_MODELS_URL
    assert headers["Authorization"] == "Bearer chatgpt-token"
    assert headers["ChatGPT-Account-ID"] == "acct-42"
    assert params == {"client_version": discovery.OPENAI_MODELS_CLIENT_VERSION}


def test_openai_oauth_without_an_account_scope_falls_back_without_a_request() -> None:
    client = _StubClient([])

    assert (
        fetch_models(
            "openai",
            api_key="chatgpt-token",
            is_oauth=True,
            account_id=None,
            client=client,
        )
        is None
    )
    assert client.calls == []


def test_an_unrecognised_visibility_value_does_not_hide_every_model() -> None:
    """This listing may PRUNE the registry, so the visibility test is a denylist.

    Testing ``!= "list"`` dropped every row the moment the endpoint renamed or
    added a value, and a listing that parses to nothing then emptied the whole
    OpenAI picker. Showing one internal helper is the far smaller harm.
    """
    body = {
        "models": [
            {"slug": "gpt-5.6-sol", "display_name": "GPT-5.6-Sol", "visibility": "visible"},
            {"slug": "codex-auto-review", "visibility": "HIDE"},
        ]
    }
    rows = fetch_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=_StubClient([_Response(200, body)]),
    )

    assert rows is not None
    assert [row.id for row in rows] == ["gpt-5.6-sol"], "unknown value hid a real model"


def test_a_codex_row_may_spell_its_id_the_public_way() -> None:
    """``slug`` with no fallback made one renamed key cost every row."""
    body = {"models": [{"id": "gpt-5.6-luna", "display_name": "GPT-5.6-Luna"}]}
    rows = fetch_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=_StubClient([_Response(200, body)]),
    )

    assert rows is not None
    assert [row.id for row in rows] == ["gpt-5.6-luna"]


# -- Anthropic transport ------------------------------------------------------


#: Captured from ``GET https://api.anthropic.com/v1/models?limit=50`` on
#: 2026-08-07, trimmed to two entries and to the capability keys this transport
#: reads. The two are chosen to disagree: the 5 generation serves 1M with 128k of
#: output while Opus 4.5 serves 200k with 64k, so a transport that hardcoded
#: either — or that kept reporting zeros and left the family floor to answer —
#: fails on one of them.
_ANTHROPIC_BODY = {
    "data": [
        {
            "id": "claude-opus-5",
            "display_name": "Claude Opus 5",
            "created_at": "2026-06-01T00:00:00Z",
            "type": "model",
            "max_input_tokens": 1_000_000,
            "max_tokens": 128_000,
            "capabilities": {
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "thinking": {"supported": True, "types": {"adaptive": {"supported": True}}},
            },
        },
        {
            "id": "claude-opus-4-5-20251101",
            "display_name": "Claude Opus 4.5",
            "created_at": "2025-11-01T00:00:00Z",
            "type": "model",
            "max_input_tokens": 200_000,
            "max_tokens": 64_000,
            "capabilities": {"image_input": {"supported": True}},
        },
    ],
    "has_more": False,
}

#: The same endpoint as it answered before ``max_input_tokens`` existed, and as a
#: proxy pinned to an older API version still answers. Kept because the transport
#: must degrade to the registry here rather than zero the numbers a session runs on.
_ANTHROPIC_TERSE_BODY = {
    "data": [
        {
            "id": "claude-opus-5",
            "display_name": "Claude Opus 5",
            "created_at": "2026-06-01T00:00:00Z",
            "type": "model",
        }
    ],
    "has_more": False,
}


def test_anthropic_maps_the_window_output_cap_and_image_support() -> None:
    """The reported defect: a session on `claude-opus-5` showed `1.8%/200k`.

    The listing had the truth all along — 1,000,000 `max_input_tokens` — and the
    transport threw it away, so the 200k family floor answered instead and the
    compaction threshold came out at 160k on a model with 1M of room.
    """
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    rows = fetch_models("anthropic", api_key="sk-ant", client=client)

    assert rows is not None
    assert [row.id for row in rows] == ["claude-opus-5", "claude-opus-4-5-20251101"]
    assert rows[0].name == "Claude Opus 5"
    assert rows[0].context_window == 1_000_000
    assert rows[0].max_tokens == 128_000
    assert rows[0].supports_images is True
    # The generations disagree, so the second entry proves the numbers are read
    # per model rather than taken from the first row or a constant.
    assert rows[1].context_window == 200_000
    assert rows[1].max_tokens == 64_000
    # Still no prices on this wire, and a made-up one renders in the status band
    # as fact. Prompt caching is likewise not a listing field.
    assert rows[0].input_price == 0.0
    assert rows[0].output_price == 0.0
    assert rows[0].supports_prompt_cache is False
    assert client.calls[0][0] == "https://api.anthropic.com/v1/models"
    assert client.calls[0][2]["limit"] == 1000


def test_anthropic_reports_unknown_for_everything_the_listing_omits() -> None:
    """An older API version, or a proxy that strips fields, must cost the session
    nothing: the numbers come back as the zero that means "unknown" and the
    capability as ``None`` — "not stated", which the merge fills from the registry.
    Inventing a default here is what produced a session at `context_window = -1`,
    and reading the silence as ``False`` would deny image input no one denied."""
    client = _StubClient([_Response(200, _ANTHROPIC_TERSE_BODY)])
    rows = fetch_models("anthropic", api_key="sk-ant", client=client)

    assert rows is not None
    assert rows[0].context_window == 0
    assert rows[0].max_tokens == 0
    assert rows[0].supports_images is None


@pytest.mark.parametrize(
    "capabilities, stated, merged",
    [
        # Stated denial: the provider's answer, and it must survive a registry row
        # that says otherwise (review finding C-07).
        ({"image_input": {"supported": False}}, False, False),
        # Stated support.
        ({"image_input": {"supported": True}}, True, True),
        # The key exists but says nothing, which is not a denial.
        ({"image_input": {}}, None, True),
        # No capabilities object at all — an older API version, a stripping proxy.
        (None, None, True),
    ],
)
def test_anthropic_capability_states_survive_the_merge(
    tmp_path, capabilities, stated, merged
) -> None:
    """All three states, from the wire to the row a session resolves from.

    Capabilities are objects, not flags: `{"image_input": {"supported": false}}` is
    a text-only model advertising the key, and reading the object's truthiness
    would report every listed capability as supported. That precision only means
    something if it reaches the merge, which is what this pins end to end — the
    shipped `claude-opus-5` row carries `supports_images=True`, so an OR made the
    stated `false` unreachable.
    """
    entry: dict[str, Any] = {"id": "claude-opus-5", "display_name": "Claude Opus 5"}
    if capabilities is not None:
        entry["capabilities"] = capabilities
    body = {"data": [entry]}

    rows = fetch_models("anthropic", api_key="k", client=_StubClient([_Response(200, body)]))
    assert rows is not None
    assert rows[0].supports_images is stated

    resolved, status = available_models(
        "anthropic",
        api_key="k",
        client=_StubClient([_Response(200, body)]),
        cache_dir=tmp_path,
    )
    assert status == "ok"
    by_id = {row.id: row for row in resolved}
    assert static_models("anthropic")["claude-opus-5"].supports_images is True, "fixture drifted"
    assert by_id["claude-opus-5"].supports_images is merged


def test_a_cache_round_trip_does_not_turn_unstated_into_denied(tmp_path) -> None:
    """`dataclasses.asdict` stores the unstated capability as `null`, and reading
    that back as False would make the same model resolve differently from disk than
    live — the second open of a picker disagreeing with the first."""
    body = {"data": [{"id": "claude-opus-5", "display_name": "Claude Opus 5"}]}
    client = _StubClient([_Response(200, body)])

    live, live_status = available_models(
        "anthropic", api_key="k", client=client, cache_dir=tmp_path
    )
    cached, cached_status = available_models(
        "anthropic", api_key="k", client=client, cache_dir=tmp_path
    )

    assert (live_status, cached_status) == ("ok", "cached")
    assert len(client.calls) == 1
    stored = json.loads((tmp_path / "anthropic.listing.json").read_text())
    assert stored["payload"]["models"][0]["supports_images"] is None
    for rows in (live, cached):
        assert {row.id: row.supports_images for row in rows}["claude-opus-5"] is True


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


def _gemini_page(*entries: dict[str, Any], next_token: str | None = None) -> _Response:
    page: dict[str, object] = {"models": list(entries)}
    if next_token:
        page["nextPageToken"] = next_token
    return _Response(200, page)


def _gemini_entry(model_id: str, **overrides: object) -> dict[str, object]:
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


class _SlowPaginator(httpx.Client):
    """A Gemini endpoint that is slow but ALIVE: every page answers inside the
    budget it was granted and hands back a fresh ``nextPageToken`` forever.

    Advances the clock ``discovery`` reads instead of sleeping, so the test can
    measure the elapsed time of a run that really takes minutes. ``share`` is the
    fraction of each granted ceiling the page consumes; 1.0 spends the lot.
    """

    def __init__(self, clock: list[float], *, share: float) -> None:
        self._clock = clock
        self._share = share
        self.calls: list[tuple[str, dict[str, object], dict[str, object], float]] = []

    def get(
        self,
        url: Any,
        *,
        params: Any = None,
        headers: Any = None,
        timeout: Any = None,
        **kwargs: Any,
    ) -> _Response:
        self.calls.append((str(url), dict(headers or {}), dict(params or {}), timeout))
        assert timeout > 0, "a request was issued with no budget left to spend"
        self._clock[0] += timeout * self._share
        page = len(self.calls)
        return _gemini_page(_gemini_entry(f"gemini-{page}"), next_token=f"page-{page}")


def _fake_clock(monkeypatch, clock: list[float]) -> None:
    # Only discovery's own reference is replaced: patching ``time.monotonic``
    # itself would reach every library the test session happens to touch.
    monkeypatch.setattr(discovery, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))


def test_gemini_bounds_the_whole_pagination_run_by_one_timeout(monkeypatch) -> None:
    """``DEFAULT_TIMEOUT_S`` promises an unreachable -- or merely slow -- host
    fails in seconds, but it used to bound ONE request while this transport issues
    up to ``GEMINI_MAX_PAGES``. Measured before the fix with this stub: 25
    requests and 250 s for a single ``resolve_model_info()``, on the synchronous
    session-start path and the TUI's ``_cost_for``.
    """
    clock = [0.0]
    _fake_clock(monkeypatch, clock)
    client = _SlowPaginator(clock, share=0.9)

    fetch_models("google", api_key="AIza", client=client, timeout=DEFAULT_TIMEOUT_S)

    # The ceiling covers the RUN, not each hop: 1e-9 absorbs the accumulation
    # error of adding fractions of a float, not any real slack.
    assert clock[0] <= DEFAULT_TIMEOUT_S + 1e-9
    assert len(client.calls) <= GEMINI_MAX_PAGES
    # Every hop gets what is LEFT of the budget rather than a fresh ceiling.
    assert client.calls[1][3] < client.calls[0][3]


def test_gemini_fails_the_listing_when_the_deadline_cuts_pagination_short(monkeypatch) -> None:
    clock = [0.0]
    _fake_clock(monkeypatch, clock)
    client = _SlowPaginator(clock, share=1.0)

    rows = fetch_models("google", api_key="AIza", client=client, timeout=DEFAULT_TIMEOUT_S)

    # The same choice the failed-page path makes, for the same reason: the pages
    # that did arrive are not the catalogue, and offering them as one silently
    # deletes every model on the pages that did not.
    assert rows is None
    assert len(client.calls) == 1
    assert clock[0] == DEFAULT_TIMEOUT_S


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


def test_merge_prefers_a_live_cache_write_price() -> None:
    static = {"m": _info("m", cache_reads_price=1.5, cache_writes_price=18.75)}
    live = [DiscoveredModel(id="m", cache_read_price=0.3, cache_write_price=3.75)]
    merged = merge_models(static, live)

    assert merged[0].cache_write_price == pytest.approx(3.75)


def test_merge_keeps_the_static_cache_write_price_when_live_is_zero() -> None:
    """Same rule as the read price: a zero is "not quoted", never "free"."""
    static = {"m": _info("m", cache_reads_price=1.5, cache_writes_price=18.75)}
    merged = merge_models(static, [DiscoveredModel(id="m", cache_read_price=0.3)])

    assert merged[0].cache_write_price == pytest.approx(18.75)


def test_a_cache_write_price_survives_the_disk_round_trip(tmp_path) -> None:
    """The document is `asdict` of the row; the reader must pick the field back up."""
    body = {
        "data": [
            {
                "id": "anthropic/claude-fable-5.1",
                "pricing": {
                    "prompt": "0.00001",
                    "completion": "0.00005",
                    "input_cache_read": "0.00000025",
                    "input_cache_write": "0.0000125",
                },
            }
        ]
    }
    client = _StubClient([_Response(200, body)])
    available_models("openrouter", api_key="k", client=client, cache_dir=tmp_path)
    models, status = available_models("openrouter", api_key="k", client=client, cache_dir=tmp_path)

    assert status == "cached"
    assert models[0].cache_write_price == pytest.approx(12.5)


def test_an_openrouter_document_from_capture_one_is_refetched_once(tmp_path) -> None:
    """A version-1 OpenRouter document has no write price; it must not be served for a day.

    Mirrors the Anthropic capture-two guard: the shape is fine and every field
    maps, so nothing but the stamp can notice the writer left a field at zero.
    """
    stale = tmp_path / "openrouter.listing.json"
    stale.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": 1,
                    "models": [{"id": "anthropic/claude-fable-5.1", "input_price": 10.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    body = {
        "data": [
            {
                "id": "anthropic/claude-fable-5.1",
                "pricing": {"prompt": "0.00001", "input_cache_write": "0.0000125"},
            }
        ]
    }
    client = _StubClient([_Response(200, body)])

    models, status = available_models("openrouter", api_key="k", client=client, cache_dir=tmp_path)

    assert status == "ok"
    assert len(client.calls) == 1
    assert models[0].cache_write_price == pytest.approx(12.5)
    assert json.loads(stale.read_text())["payload"]["capture"] == 2


def test_an_unstated_capability_defers_to_the_registry() -> None:
    """Silence is not a denial. Every lean OpenAI-compatible gateway sends an id
    and nothing else, and reading that as "no images, no caching" would downgrade
    every bundled vision model and drop `cache_control` on the priciest ones."""
    static = {"m": _info("m", supports_images=True, supports_prompt_cache=True)}
    live = [DiscoveredModel(id="m")]
    assert live[0].supports_images is None, "the unstated default must be None, not False"
    merged = merge_models(static, live)

    assert merged[0].supports_images is True
    assert merged[0].supports_prompt_cache is True


def test_a_stated_capability_can_upgrade_the_registry() -> None:
    static = {"m": _info("m", supports_images=False, supports_prompt_cache=False)}
    live = [DiscoveredModel(id="m", supports_images=True, supports_prompt_cache=True)]
    merged = merge_models(static, live)

    assert merged[0].supports_images is True
    assert merged[0].supports_prompt_cache is True


def test_an_explicit_false_from_the_provider_beats_a_true_registry_row() -> None:
    """Review finding C-07. An OR made this state unreachable: every shipped
    Anthropic row carries `supports_images=True`, so a live
    `image_input.supported: false` merged straight back to True and a text-only
    model went on advertising vision — the provider's own denial, overruled by a
    hand-transcribed row it was meant to correct.

    Only `supports_images` gets the third state, and only because a wire states it.
    `supports_prompt_cache` has no such field anywhere, so its silence still ORs.
    """
    static = {"m": _info("m", supports_images=True, supports_prompt_cache=True)}
    live = [DiscoveredModel(id="m", supports_images=False)]
    merged = merge_models(static, live)

    assert merged[0].supports_images is False
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
    new_id = "claude-opus-5-20260601"
    assert new_id not in static, "registry now ships the id this test uses as unknown"

    live = [
        DiscoveredModel(id=new_id, name="Claude Opus 5"),
        DiscoveredModel(id=known_id, name="Claude Sonnet 4"),
    ]
    by_id = {row.id: row for row in merge_models(static, live)}

    # A listing that omitted the limits (an older API version, a stripping proxy)
    # must leave the known model with everything the registry knows.
    assert by_id[known_id].context_window == known_static.context_window
    assert by_id[known_id].input_price == pytest.approx(known_static.input_price)
    assert by_id[known_id].output_price == pytest.approx(known_static.output_price)
    assert by_id[known_id].max_tokens == known_static.max_tokens
    # The new model is reachable, and honestly reports what the listing did not
    # say rather than inventing a window. Inheriting its family's window is a
    # RESOLUTION concern (`configure._registry_fallback`), not a merge one: the
    # picker must not claim a number the provider never sent for this id.
    assert by_id[new_id].name == "Claude Opus 5"
    assert by_id[new_id].context_window == 0


# -- available_models ---------------------------------------------------------


def test_available_models_reports_ok_and_merges_the_registry(tmp_path) -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path
    )

    assert status == "ok"
    by_id = {row.id: row for row in models}
    # Live: in the payload, with the payload's window.
    assert by_id["claude-opus-5"].context_window == 1_000_000
    # Registry-only: the listing did not mention it, and a model the user can run
    # today must not vanish from the picker because of that.
    assert by_id["claude-3-5-sonnet-20241022"].context_window == 200_000


def test_openai_oauth_catalogue_is_authoritative_when_available(tmp_path) -> None:
    """A successful account-scoped response replaces the stale static id list."""
    client = _StubClient([_Response(200, _OPENAI_CODEX_BODY)])

    models, status = available_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=client,
        cache_dir=tmp_path,
    )

    assert status == "ok"
    assert [model.id for model in models] == ["gpt-5.6-sol"]
    assert "gpt-4o" in static_models("openai"), "fixture no longer proves static ids were pruned"


def test_openai_oauth_falls_back_to_static_when_the_endpoint_is_unavailable(tmp_path) -> None:
    client = _StubClient([_Response(403, {"detail": "forbidden"})])

    models, status = available_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=client,
        cache_dir=tmp_path,
    )

    assert status == "static"
    assert {model.id for model in models} == set(static_models("openai"))


def test_openai_api_key_and_oauth_catalogues_do_not_share_a_cache(tmp_path) -> None:
    """An API-key cache must not mask an account's newer OAuth-only models."""
    client = _StubClient(
        [
            _Response(200, {"data": [{"id": "api-only"}]}),
            _Response(200, _OPENAI_CODEX_BODY),
        ]
    )

    available_models(
        "openai",
        api_key="sk-api",
        client=client,
        cache_dir=tmp_path,
    )
    oauth_models, oauth_status = available_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=client,
        cache_dir=tmp_path,
    )

    assert oauth_status == "ok"
    assert [model.id for model in oauth_models] == ["gpt-5.6-sol"]
    assert [call[0] for call in client.calls] == [
        "https://api.openai.com/v1/models",
        discovery.OPENAI_CHATGPT_MODELS_URL,
    ]


def test_an_unscoped_oauth_run_never_prunes_from_the_api_key_document(tmp_path) -> None:
    """The pruning decision must require the SAME account scope the cache key
    and the transport require.

    Without the account id ``_cache_key`` degrades to the shared unscoped
    document, so an OAuth run read the listing an API KEY wrote and then used
    it to prune the registry -- 11 bundled ids down to 1, with no HTTP request
    issued and a reassuring ``cached`` status. The same path accepted a
    document written before the account-scoped catalogue existed, so an upgrade
    hit it too.
    """
    client = _StubClient([_Response(200, {"data": [{"id": "api-only"}]})])

    available_models("openai", api_key="sk-api", client=client, cache_dir=tmp_path)
    models, status = available_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id=None,
        client=client,
        cache_dir=tmp_path,
    )

    assert status == "cached"
    listed = {model.id for model in models}
    assert set(static_models("openai")) <= listed, "an unscoped run pruned the registry"
    assert len(client.calls) == 1, "the second run must not issue a request"


def test_a_codex_listing_that_parses_to_nothing_keeps_the_bundled_models(tmp_path) -> None:
    """Upstream schema drift must cost "no new models", never "no models".

    A 200 whose rows all drop (a renamed key, an unknown visibility value, a
    catalogue filtered by the pinned client version) pruned the registry to
    EMPTY and cached the emptiness for the 24h TTL, so the picker offered no
    OpenAI model at all and issued no request that could recover it.
    """
    client = _StubClient([_Response(200, {"models": []})])

    models, status = available_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=client,
        cache_dir=tmp_path,
    )

    assert status == "empty"
    assert {model.id for model in models} == set(static_models("openai"))


def test_a_subscription_token_is_never_spent_on_the_generic_endpoint() -> None:
    """One predicate drives the endpoint choice, the cache scope AND the prune.

    They were spelled three separate ways and disagreed: the prune test omitted
    the account id and keyed off the credential-storage id rather than the
    provider actually fetched. Denying the predicate must therefore stop the
    request too, not fall through to ``api.openai.com/v1/models`` -- that wire
    rejects a subscription credential, so trying it would burn a request and
    report a listing outage for a token that was simply asked the wrong thing.
    """
    client = _StubClient([_Response(200, {"data": [{"id": "gpt-4.1"}]})])

    rows = fetch_models(
        "openai",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id=None,
        client=client,
    )

    assert rows is None
    assert client.calls == [], "an unscoped OAuth listing must issue no request"


def test_listing_authority_requires_every_condition_the_request_needs() -> None:
    """The prune predicate is the only destructive one, so it may not be
    weaker than the predicate that decides the request is issuable.

    ``openai-device`` shares OpenAI's credential row and IS on the
    account-scoped route. A sibling that shared the row WITHOUT being on that
    route would receive a generic ``/v1/models`` snapshot -- explicitly a
    partial entitlement list -- and, under the old ``storage_id == "openai"``
    test, prune the bundled catalogue from it.
    """
    scoped = discovery._serves_account_scoped_catalogue
    kw = {"is_oauth": True, "api_key": "tok", "account_id": "acct-42"}

    assert scoped("openai", **kw) is True
    assert scoped("openai-device", **kw) is True
    # Not on the account-scoped route, whatever credential row it shares.
    assert scoped("openai-enterprise", **kw) is False
    assert scoped("openrouter", **kw) is False
    # Every condition the request itself needs is required here too.
    assert scoped("openai", is_oauth=False, api_key="tok", account_id="acct-42") is False
    assert scoped("openai", is_oauth=True, api_key=None, account_id="acct-42") is False
    assert scoped("openai", is_oauth=True, api_key="tok", account_id=None) is False


def test_openai_device_shares_the_account_scoped_document(tmp_path) -> None:
    """``store_credentials_as`` is what puts ``openai-device`` on the same
    account-scoped route and the same cache document as ``openai``.

    It has no bundled models of its own, so reading it off the provider id
    instead would list a device-flow login against an empty registry and cache
    the answer under a second name for the same account.
    """
    client = _StubClient([_Response(200, _OPENAI_CODEX_BODY)])

    models, status = available_models(
        "openai-device",
        api_key="chatgpt-token",
        is_oauth=True,
        account_id="acct-42",
        client=client,
        cache_dir=tmp_path,
    )

    assert status == "ok"
    assert [model.id for model in models] == ["gpt-5.6-sol"]
    assert client.calls[0][0] == discovery.OPENAI_CHATGPT_MODELS_URL
    written = sorted(path.name for path in tmp_path.glob("*.json"))
    assert written == ["openai.oauth.931c7c164da4.listing.json"], written


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


def test_available_models_reports_stale_when_a_refetch_fails(tmp_path) -> None:
    client = _StubClient([_Response(200, _ANTHROPIC_BODY), httpx.ConnectError("offline")])
    available_models("anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path)
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, ttl_s=-1
    )

    # "stale", not "cached": a fetch was ATTEMPTED and failed, which is the one
    # thing a user hunting for this morning's model needs the footer to say.
    # The two were one status before, so the picker could not tell "fresh
    # enough" from "offline".
    assert status == "stale"
    # Stale beats absent: the numbers are days old at worst, and losing the
    # window because the network blipped is the regression this guards.
    by_id = {row.id: row for row in models}
    assert by_id["claude-opus-5"].context_window == 1_000_000


def _plant_anthropic_document(tmp_path, *, age_s: float, ids: Sequence[str]) -> None:
    """A capture-2 Anthropic document of the given age, listing exactly ``ids``."""
    (tmp_path / "anthropic.listing.json").write_text(
        json.dumps(
            {
                "fetched_at": time.time() - age_s,
                "payload": {
                    "capture": discovery.listing_capture_version("anthropic"),
                    "models": [{"id": model_id, "context_window": 1_000_000} for model_id in ids],
                },
            }
        ),
        encoding="utf-8",
    )


def test_a_missing_want_id_refetches_a_document_old_enough_to_be_wrong(tmp_path) -> None:
    """The incident: a 22h document (inside the hard TTL) without this morning's model.

    Without this trigger the id the user asked for stays unresolvable until the
    document expires, and the memo pins the answer for the day. One refetch,
    inside the caller's budget, for exactly the id being resolved.
    """
    _plant_anthropic_document(tmp_path, age_s=22 * 3600, ids=["claude-opus-5"])
    fresh = {
        "data": [
            {"id": "claude-fable-5-1", "display_name": "Claude Fable 5.1", "type": "model"},
            {"id": "claude-opus-5", "display_name": "Claude Opus 5", "type": "model"},
        ],
        "has_more": False,
    }
    client = _StubClient([_Response(200, fresh)])

    models, status = available_models(
        "anthropic",
        api_key="sk-ant",
        client=client,
        cache_dir=tmp_path,
        want_id="claude-fable-5-1",
    )

    assert status == "ok"
    assert len(client.calls) == 1
    assert "claude-fable-5-1" in {row.id for row in models}


def test_a_missing_want_id_is_believed_when_the_document_is_young(tmp_path) -> None:
    """A typo must not refetch on every resolution: a minute-old miss is a miss."""
    _plant_anthropic_document(tmp_path, age_s=60, ids=["claude-opus-5"])
    client = _StubClient([])

    models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, want_id="claude-typo"
    )

    assert status == "cached"
    assert client.calls == []


def test_a_want_id_that_is_present_makes_no_request(tmp_path) -> None:
    # Past the miss floor (so a miss WOULD refetch) but inside the soft TTL, so
    # the only request this could make is the miss-triggered one.
    _plant_anthropic_document(tmp_path, age_s=20 * 60, ids=["claude-opus-5"])
    client = _StubClient([])

    _models, status = available_models(
        "anthropic",
        api_key="sk-ant",
        client=client,
        cache_dir=tmp_path,
        # Google's `models/` spelling and case count as present too — the same
        # normalisation the lookup applies, so a hit there is not a miss here.
        want_id="models/Claude-Opus-5",
    )

    assert status == "cached"
    assert client.calls == []


@pytest.mark.parametrize(
    ("listed", "wanted"),
    [
        # The R1-1 case: Anthropic's listing carries the dated snapshot only, the
        # user configures the undated alias its API accepts. Before, this was a
        # miss on EVERY process start older than the miss floor: a blocking
        # round trip on boot, forever, for a document the refetch could not fix.
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("claude-opus-4-5-20251101", "claude-opus-4-5"),
        ("claude-haiku-4-5-20251001", "claude-haiku-4-5"),
        # The reverse: an alias-only listing (the aggregators' habit) asked for
        # a snapshot, and the dotted spelling either way.
        ("claude-opus-4-5", "claude-opus-4-5-20251101"),
        ("claude-opus-4.5", "claude-opus-4-5-20251101"),
        ("gpt-5.4", "gpt-5-4"),
    ],
)
def test_a_want_id_listed_under_another_spelling_makes_no_request(
    tmp_path, listed: str, wanted: str
) -> None:
    # Past the miss floor and inside the soft TTL: the only request possible
    # is the miss-triggered one, and there must not be one.
    _plant_anthropic_document(tmp_path, age_s=20 * 60, ids=[listed])
    client = _StubClient([])

    _models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, want_id=wanted
    )

    assert status == "cached"
    assert client.calls == []


def test_a_genuinely_unlisted_id_still_refetches_after_the_alias_widening(tmp_path) -> None:
    """The widening must not swallow the trigger it sits beside: a NEW family
    id that no spelling of any listed row matches is still a miss."""
    _plant_anthropic_document(tmp_path, age_s=20 * 60, ids=["claude-sonnet-4-5-20250929"])
    fresh = {
        "data": [{"id": "claude-fable-5-1", "display_name": "Claude Fable 5.1", "type": "model"}],
        "has_more": False,
    }
    client = _StubClient([_Response(200, fresh)])

    _models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, want_id="claude-fable-5-1"
    )

    assert status == "ok"
    assert len(client.calls) == 1


def test_a_background_revalidation_gets_the_full_default_timeout(tmp_path) -> None:
    """R1-2: the caller's on-path budget (2 s from a repaint) bounds the SYNC
    fetch only. The background refresh is off-path and must get the provider's
    full default, or on any link slower than the caller's budget it fails every
    time and the document only ever advances via the 24 h sync path."""
    from local_operator.model import catalogue

    with catalogue._revalidate_lock:
        catalogue._revalidating.clear()
        catalogue._last_attempt.clear()
        catalogue._threads.clear()
    _plant_anthropic_document(tmp_path, age_s=2 * 3600, ids=["claude-opus-5"])
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])

    _models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, timeout=2.0
    )
    assert status == "cached"
    for thread in catalogue._revalidation_threads():
        thread.join(timeout=5.0)

    assert len(client.calls) == 1
    _url, _headers, _params, timeout = client.calls[0]
    assert timeout == discovery.DEFAULT_TIMEOUT_S

    # And the SYNC path keeps the caller's budget: it is the one waiting.
    _plant_anthropic_document(tmp_path, age_s=25 * 3600, ids=["claude-opus-5"])
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])
    _models, status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path, timeout=2.0
    )
    assert status == "ok"
    assert client.calls[0][3] == 2.0


def test_a_failed_want_id_refetch_keeps_the_document_and_reports_stale(tmp_path) -> None:
    _plant_anthropic_document(tmp_path, age_s=22 * 3600, ids=["claude-opus-5"])
    client = _StubClient([httpx.ConnectError("offline")])

    models, status = available_models(
        "anthropic",
        api_key="sk-ant",
        client=client,
        cache_dir=tmp_path,
        want_id="claude-fable-5-1",
    )

    assert status == "stale"
    assert "claude-opus-5" in {row.id for row in models}


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


def test_available_models_never_reads_a_document_written_by_the_old_layout(tmp_path) -> None:
    """An earlier release cached the provider client's RAW ``list_models()``
    payload under the bare provider id. This reader cannot interpret that shape,
    and the payload is written before anything maps it, so reusing the name would
    serve the failure as a fresh cache hit for a full day. The old documents are
    unreachable by name and swept off disk instead of left to rot there."""
    legacy_document = tmp_path / "openrouter.models.json"
    legacy_document.write_text('{"fetched_at": 0, "payload": {"data": []}}', encoding="utf-8")
    body = {"data": [{"id": "vendor/model", "context_length": 1_000}]}
    client = _StubClient([_Response(200, body)])

    models, status = available_models("openrouter", api_key="k", client=client, cache_dir=tmp_path)

    assert status == "ok"
    assert "vendor/model" in {row.id for row in models}
    assert not legacy_document.exists()
    # One suffix, applied once: the key and the filename builder both adding
    # ``.models`` is what produced ``openrouter.models.models.json``.
    assert [path.name for path in tmp_path.iterdir()] == ["openrouter.listing.json"]


def test_available_models_survives_a_broken_cache_layer(tmp_path, monkeypatch) -> None:
    def explode(*args: object, **kwargs: object) -> dict[str, Any]:
        raise OSError("cache is on fire")

    monkeypatch.setattr(discovery, "read_listing", explode)
    models, status = available_models(
        "anthropic", api_key="sk-ant", client=_StubClient([]), cache_dir=tmp_path
    )

    # A model picker must open even when the layer beneath it is broken; a
    # traceback here would take the whole session down.
    assert status == "static"
    assert {row.id for row in models} == set(static_models("anthropic"))


def test_a_cached_document_that_cannot_be_mapped_is_dropped_so_the_next_call_recovers(
    tmp_path,
) -> None:
    """The payload is cached BEFORE anything interprets it, so a document that
    passes the cache's shape checks and still yields no rows -- here ``models``
    arriving as an object -- would otherwise be served as a FRESH hit on every
    start for the full 24h TTL. Measured before the fix: three consecutive calls
    all returned ``static`` with zero fetches and the document stayed on disk.

    Dropping it was not enough either. The call that drops it has already been
    served from a fresh hit, so the fetch never ran, and the answer fell through
    to the registry's static rows -- of which an aggregator has NONE. So the same
    call re-enters the cache once and recovers immediately.
    """
    poisoned = tmp_path / "openrouter.listing.json"
    poisoned.write_text(
        json.dumps({"fetched_at": time.time(), "payload": {"models": {"not": "a list"}}}),
        encoding="utf-8",
    )
    body = {"data": [{"id": "vendor/model", "context_length": 1_000}]}
    client = _StubClient([_Response(200, body)])

    models, first_status = available_models(
        "openrouter", api_key="k", client=client, cache_dir=tmp_path
    )

    # The FIRST call recovers: the unusable document is dropped and refetched in
    # one pass, because "the next call will fix it" is an empty model list for
    # every provider whose registry has no static rows.
    assert first_status == "ok"
    assert len(client.calls) == 1
    assert "vendor/model" in {row.id for row in models}


def test_a_cached_document_from_an_older_capture_is_refetched_not_served(tmp_path) -> None:
    """The upgrade that fixed the numbers must not be invisible for a day.

    Version 1 of the Anthropic transport wrote every window as zero because it did
    not read ``max_input_tokens``. That document is perfectly well SHAPED, so
    without the capture stamp it is served as a fresh cache hit for the rest of its
    24h TTL: the install that reported ``1.8%/200k`` would have gone on reporting it
    after the fix shipped, which is indistinguishable from the fix not working.
    """
    stale = tmp_path / "anthropic.listing.json"
    stale.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {"models": [{"id": "claude-opus-5", "name": "Claude Opus 5"}]},
            }
        ),
        encoding="utf-8",
    )
    client = _StubClient([_Response(200, _ANTHROPIC_BODY)])

    models, first_status = available_models(
        "anthropic", api_key="sk-ant", client=client, cache_dir=tmp_path
    )

    # One pass: the stale document is dropped AND replaced, because a provider
    # with no static rows would otherwise be left with an empty catalogue until
    # something else happened to ask again.
    assert first_status == "ok"
    assert len(client.calls) == 1
    by_id = {row.id: row for row in models}
    assert by_id["claude-opus-5"].context_window == 1_000_000
    assert json.loads(stale.read_text())["payload"]["capture"] == discovery.listing_capture_version(
        "anthropic"
    )


def test_only_the_transport_that_changed_invalidates_its_cache(tmp_path) -> None:
    """A capture bump is per TRANSPORT, not global.

    A single global number invalidated every provider's cache on upgrade, and for
    an aggregator — whose registry has no static rows to fall back on — the
    replacement answer was an empty model list. Only the Anthropic reader started
    needing a field its writer had not recorded, so only Anthropic's stamp moved.
    """
    assert discovery.listing_capture_version("anthropic") == 2
    assert discovery.listing_capture_version("openrouter") == 2
    # Ollama's reader never changed: an OpenAI-compatible document with no
    # pricing object has nothing new to capture, so its stamp stays at 1.
    assert discovery.listing_capture_version("ollama") == discovery.LISTING_CAPTURE_DEFAULT

    cached = tmp_path / "ollama.listing.json"
    cached.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                "payload": {
                    "capture": discovery.LISTING_CAPTURE_DEFAULT,
                    "models": [{"id": "qwen3:8b", "context_window": 1_000}],
                },
            }
        ),
        encoding="utf-8",
    )
    client = _StubClient([])

    models, status = available_models("ollama", api_key=None, client=client, cache_dir=tmp_path)
    # Served from the cache, no fetch, rows intact — the upgrade did not touch it.
    assert status == "cached"
    assert not client.calls
    assert "qwen3:8b" in {row.id for row in models}


def test_an_unstamped_document_is_the_original_shape_not_a_stale_one(tmp_path) -> None:
    """Pre-upgrade caches carry no `capture` key at all.

    Reading that absence as version 0 rejected every document written before the
    stamp existed — for every transport, including the aggregators the
    per-transport map exists to spare, whose registry has no static rows to answer
    with. Measured before the fix: `static` with 0 models on any failed fetch.
    """
    # Ollama is a transport whose stamp is still version 1, so an unstamped
    # document is exactly its current shape.
    cached = tmp_path / "ollama.listing.json"
    cached.write_text(
        json.dumps(
            {
                "fetched_at": time.time(),
                # Exactly what the pre-stamp writer produced: no `capture`.
                "payload": {"models": [{"id": "qwen3:8b", "context_window": 1_000}]},
            }
        ),
        encoding="utf-8",
    )
    client = _StubClient([])

    models, status = available_models("ollama", api_key=None, client=client, cache_dir=tmp_path)

    assert status == "cached"
    assert not client.calls
    assert "qwen3:8b" in {row.id for row in models}
