"""The keyless price chain: models.dev first, OpenRouter's public listing second.

The defect this module exists for: Anthropic's ``/v1/models`` quotes no prices,
so a model the registry had not been taught was priced from the OpenRouter
listing under a per-provider namespace — and from nowhere else. The day
``claude-fable-5-1`` shipped, that document was six hours old and predated the
row, and a user signed in only to Anthropic ran the whole day at ``$0.00``.
models.dev carried ``10/50/0.25/12.5`` on release day. It is the primary now;
OpenRouter is the independent secondary so that a gap in ONE community source
is not a gap in every direct provider's price.

No live network anywhere here: ``httpx.get`` is patched with a canned response,
the OpenRouter leg is stubbed by an autouse fixture (``openrouter``) that
answers with whatever rows a test hands it, and every document lands in
``tmp_path``.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from local_operator.model import catalogue, prices
from local_operator.model.discovery import DiscoveredModel
from local_operator.model.prices import (
    PRICE_CATALOGUE_CAPTURE,
    PRICE_CATALOGUE_KEY,
    price_catalogue_row,
    price_row,
    project,
)

#: Captured from https://models.dev/api.json on 2026-09-01 (ETag
#: ``"2afcb862acafd97c7717b83be8fa940b"``), trimmed to the providers and
#: fields the tests need. ``description``, ``modalities`` and the rest of a real
#: entry are kept on one row so the projection can be shown to drop them.
_MODELS_DEV_BODY: dict[str, Any] = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "claude-fable-5-1": {
                "id": "claude-fable-5-1",
                "name": "Claude Fable 5.1",
                "description": "Claude model for demanding reasoning and long-horizon agentic work",
                "family": "claude-fable",
                "attachment": True,
                "reasoning": True,
                "tool_call": True,
                "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
                "release_date": "2026-09-01",
                "limit": {"context": 1_000_000, "output": 128_000},
                "cost": {"input": 10, "output": 50, "cache_read": 0.25, "cache_write": 12.5},
            },
            "claude-opus-4.5": {
                "id": "claude-opus-4.5",
                "name": "Claude Opus 4.5",
                "limit": {"context": 200_000, "output": 64_000},
                "cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25},
            },
        },
    },
    "openai": {
        "id": "openai",
        "models": {
            "gpt-5.4": {
                "id": "gpt-5.4",
                "name": "GPT-5.4",
                "limit": {"context": 400_000, "output": 128_000},
                "cost": {"input": 2.5, "output": 15, "cache_read": 0.25},
            }
        },
    },
    "google": {
        "id": "google",
        "models": {
            "gemini-2.5-pro": {
                "id": "gemini-2.5-pro",
                "name": "Gemini 2.5 Pro",
                "limit": {"context": 1_048_576, "output": 65_536},
                "cost": {"input": 1.25, "output": 10, "cache_read": 0.31},
            }
        },
    },
    "moonshotai": {
        "id": "moonshotai",
        "models": {
            "kimi-k2.5": {
                "id": "kimi-k2.5",
                "name": "Kimi K2.5",
                "limit": {"context": 262_144, "output": 262_144},
                "cost": {"input": 0.6, "output": 2.5, "cache_read": 0.1},
            }
        },
    },
    "kimi-for-coding": {
        "id": "kimi-for-coding",
        "models": {
            "k3": {
                "id": "k3",
                "name": "Kimi K3",
                "limit": {"context": 262_144, "output": 32_768},
                # The plan bills credits, not USD per token — models.dev states
                # that as explicit zeros, the same shape as ``zai/glm-4.7-flash``.
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            }
        },
    },
    "zai": {
        "id": "zai",
        "models": {
            "glm-4.7-flash": {
                "id": "glm-4.7-flash",
                "name": "GLM-4.7-Flash",
                "limit": {"context": 200_000, "output": 128_000},
                # Z.AI lists this model at $0 on its own pricing page. The
                # OpenRouter sibling (``z-ai/glm-4.7-flash``) is what a
                # THIRD-PARTY host charges for the open weights.
                "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            }
        },
    },
    "openrouter": {
        "id": "openrouter",
        "models": {
            "anthropic/claude-fable-5.1": {
                "id": "anthropic/claude-fable-5.1",
                "name": "Anthropic: Claude Fable 5.1",
                "limit": {"context": 1_000_000, "output": 128_000},
                "cost": {"input": 10, "output": 50, "cache_read": 0.25, "cache_write": 12.5},
            }
        },
    },
    # A provider this tree does not map: must not reach disk.
    "amazon-bedrock": {
        "id": "amazon-bedrock",
        "models": {"anthropic.claude-fable-5-1": {"cost": {"input": 10, "output": 50}}},
    },
}

_ETAG = '"2afcb862acafd97c7717b83be8fa940b"'


class _Canned:
    """``httpx.get`` stand-in: one response per call, in order, recording headers."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        assert self.responses, f"unexpected request to {url}"
        return self.responses.pop(0)


def _ok(body: Any = None, etag: str = _ETAG) -> httpx.Response:
    return httpx.Response(
        200, json=_MODELS_DEV_BODY if body is None else body, headers={"etag": etag}
    )


def _not_modified() -> httpx.Response:
    return httpx.Response(304, headers={"etag": _ETAG})


@pytest.fixture
def canned():
    """Patch the transport; yield the recorder so tests can assert on requests."""
    recorder = _Canned([_ok()])
    with patch("httpx.get", recorder):
        yield recorder


class _OpenRouterStub:
    """The secondary leg, offline: answers ``rows`` and records every call.

    Autouse (below) because the chain reaches OpenRouter on every models.dev
    miss, and that leg goes through discovery's ``httpx.Client`` rather than
    ``httpx.get`` — without this stub a models.dev-miss test would hit the real
    endpoint. Tests that want a secondary answer append to ``rows``.
    """

    def __init__(self) -> None:
        self.rows: list[DiscoveredModel] = []
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> list[DiscoveredModel]:
        self.calls.append(kwargs)
        return list(self.rows)


@pytest.fixture(autouse=True)
def openrouter():
    stub = _OpenRouterStub()
    with patch("local_operator.model.prices.openrouter_rows", stub):
        yield stub


_OPENROUTER_FABLE = DiscoveredModel(
    id="anthropic/claude-fable-5.1",
    name="Anthropic: Claude Fable 5.1",
    context_window=1_000_000,
    max_tokens=128_000,
    input_price=10.0,
    output_price=50.0,
    cache_read_price=0.25,
    cache_write_price=12.5,
    supports_images=True,
    supports_prompt_cache=True,
)


def _plant(tmp_path, document: dict[str, Any], *, age_s: float = 0.0) -> None:
    (tmp_path / f"{PRICE_CATALOGUE_KEY}.json").write_text(
        json.dumps({"fetched_at": time.time() - age_s, "payload": document}), encoding="utf-8"
    )


def _stored(tmp_path) -> dict[str, Any]:
    return json.loads((tmp_path / f"{PRICE_CATALOGUE_KEY}.json").read_text(encoding="utf-8"))


# -- projection ---------------------------------------------------------------


def test_the_projection_keeps_only_mapped_providers_and_the_five_fields() -> None:
    document = project(_MODELS_DEV_BODY, _ETAG)

    assert document["capture"] == PRICE_CATALOGUE_CAPTURE
    assert document["etag"] == _ETAG
    assert "amazon-bedrock" not in document["providers"]
    assert set(document["providers"]) == {
        "anthropic",
        "openai",
        "google",
        "moonshotai",
        "kimi-for-coding",
        "zai",
        "openrouter",
    }
    fable = document["providers"]["anthropic"]["claude-fable-5-1"]
    # Structural, not numeric: the point of the projection is that the bulk of
    # a real entry never reaches disk.
    assert set(fable) == {"name", "cost", "limit", "attachment", "release_date"}
    assert "description" not in json.dumps(document)
    assert fable["cost"] == {"input": 10, "output": 50, "cache_read": 0.25, "cache_write": 12.5}


def test_the_projection_skips_malformed_entries_rather_than_raising() -> None:
    body = {
        "anthropic": {"models": {"ok": {"cost": {"input": 1}}, "bad": "not a mapping", 3: {}}},
        "openai": "not a provider",
        "google": {"models": ["not", "a", "mapping"]},
    }
    document = project(body, None)
    assert set(document["providers"]) == {"anthropic"}
    assert set(document["providers"]["anthropic"]) == {"ok"}
    assert document["etag"] is None


# -- the document on disk -----------------------------------------------------


def test_a_cold_lookup_fetches_and_writes_the_projection(tmp_path, canned) -> None:
    row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)

    assert row is not None
    assert (row.input_price, row.output_price) == (10.0, 50.0)
    assert (row.cache_read_price, row.cache_write_price) == (0.25, 12.5)
    assert (row.context_window, row.max_tokens) == (1_000_000, 128_000)
    assert row.supports_prompt_cache is True
    assert row.name == "Claude Fable 5.1"
    # The FIRST fetch carries no validator; there is nothing to validate against.
    assert "If-None-Match" not in canned.calls[0]["headers"]
    stored = _stored(tmp_path)["payload"]
    assert stored["etag"] == _ETAG
    assert "amazon-bedrock" not in stored["providers"]


def test_a_matching_etag_returns_the_previous_payload_and_restamps_it(tmp_path) -> None:
    """Hourly freshness costs one header round trip, not a 4.4 MB body."""
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=25 * 3600)
    before = _stored(tmp_path)["fetched_at"]
    recorder = _Canned([_not_modified()])

    with patch("httpx.get", recorder):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)

    assert row is not None and row.input_price == 10.0
    assert recorder.calls[0]["headers"] == {"If-None-Match": _ETAG}
    stored = _stored(tmp_path)
    assert stored["payload"]["etag"] == _ETAG
    assert stored["fetched_at"] > before, "a 304 is 'validated just now'"


def test_a_fresh_document_is_served_without_a_request(tmp_path) -> None:
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=60)
    recorder = _Canned([])
    with patch("httpx.get", recorder):
        row = price_catalogue_row("openai", "gpt-5.4", cache_dir=tmp_path)
    assert row is not None and row.output_price == 15.0
    assert recorder.calls == []


def test_a_failed_fetch_keeps_the_stale_document(tmp_path) -> None:
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=25 * 3600)
    with patch("httpx.get", side_effect=httpx.ConnectError("offline")):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)
    assert row is not None and row.input_price == 10.0


def test_a_non_200_never_overwrites_the_document(tmp_path) -> None:
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=25 * 3600)
    with patch("httpx.get", _Canned([httpx.Response(503, text="down")])):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)
    assert row is not None and row.input_price == 10.0
    assert _stored(tmp_path)["payload"]["providers"]["anthropic"]


def test_a_failed_cold_fetch_returns_none_and_schedules_a_background_retry(tmp_path) -> None:
    """The 4.4 MB cold download timing out inside a 3s leg budget is evidence the
    budget was too small, not that the network is down: retry off-path with the
    full timeout so the NEXT resolution finds the document."""
    with catalogue._revalidate_lock:
        catalogue._revalidating.clear()
        catalogue._last_attempt.clear()
    gate = __import__("threading").Event()
    recorder = _Canned([_ok()])

    def first_times_out_then_answers(url, **kwargs):
        if not recorder.calls:
            recorder.calls.append({"url": url, **kwargs})
            raise httpx.ReadTimeout("slow link")
        gate.wait(timeout=5.0)
        return recorder(url, **kwargs)

    with patch("httpx.get", first_times_out_then_answers):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", timeout=0.5, cache_dir=tmp_path)
        assert row is None
        threads = catalogue._revalidation_threads()
        assert len(threads) == 1, "one background retry expected"
        gate.set()
        for thread in threads:
            thread.join(timeout=5.0)

    assert recorder.calls[0]["timeout"] == 0.5
    assert recorder.calls[1]["timeout"] == prices.DEFAULT_TIMEOUT_S
    assert _stored(tmp_path)["payload"]["providers"]["anthropic"]["claude-fable-5-1"]


def test_a_background_revalidation_gets_the_full_default_timeout(tmp_path) -> None:
    """R1-2, this document: the leg budget (3 s at most) bounds only the sync
    fetch; a soft-window refresh runs off-path with the full default."""
    with catalogue._revalidate_lock:
        catalogue._revalidating.clear()
        catalogue._last_attempt.clear()
        catalogue._threads.clear()
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=2 * 3600)
    recorder = _Canned([_ok()])
    with patch("httpx.get", recorder):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", timeout=0.5, cache_dir=tmp_path)
        assert row is not None
        threads = catalogue._revalidation_threads()
        assert len(threads) == 1
        for thread in threads:
            thread.join(timeout=5.0)
    assert recorder.calls[0]["timeout"] == prices.DEFAULT_TIMEOUT_S


def test_a_document_from_an_older_capture_is_refetched_in_the_same_call(tmp_path, canned) -> None:
    _plant(tmp_path, {"capture": 0, "etag": '"old"', "providers": {}}, age_s=60)
    row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)
    assert row is not None and row.input_price == 10.0
    assert _stored(tmp_path)["payload"]["capture"] == PRICE_CATALOGUE_CAPTURE


def test_a_missing_id_in_an_old_document_is_refetched_once(tmp_path) -> None:
    """The `want_id` rule on this document: a release that landed since the last
    fetch is found on the first resolution, usually via a 0-byte 304 otherwise."""
    without = json.loads(json.dumps(_MODELS_DEV_BODY))
    del without["anthropic"]["models"]["claude-fable-5-1"]
    _plant(tmp_path, project(without, '"before"'), age_s=20 * 60)
    recorder = _Canned([_ok()])

    with patch("httpx.get", recorder):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)

    assert row is not None and row.cache_write_price == 12.5
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["headers"] == {"If-None-Match": '"before"'}


def test_a_missing_id_in_a_young_document_is_believed(tmp_path) -> None:
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=60)
    recorder = _Canned([])
    with patch("httpx.get", recorder):
        assert price_catalogue_row("anthropic", "claude-typo", cache_dir=tmp_path) is None
    assert recorder.calls == []


# -- lookup -------------------------------------------------------------------


def _lookup(provider: str, model_id: str, tmp_path):
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=60)
    with patch("httpx.get", _Canned([])):
        return price_catalogue_row(provider, model_id, cache_dir=tmp_path)


def test_lookup_matches_the_exact_id(tmp_path) -> None:
    row = _lookup("openai", "gpt-5.4", tmp_path)
    assert row is not None and row.id == "gpt-5.4"


def test_lookup_matches_the_normalised_id(tmp_path) -> None:
    """Google's own docs spell the id `models/gemini-2.5-pro`; case is folded too."""
    row = _lookup("google", "models/Gemini-2.5-Pro", tmp_path)
    assert row is not None and row.id == "gemini-2.5-pro"


def test_lookup_matches_a_dotted_version_spelling(tmp_path) -> None:
    """`claude-opus-4-5-20251101` (Anthropic's spelling) finds `claude-opus-4.5`."""
    row = _lookup("anthropic", "claude-opus-4-5-20251101", tmp_path)
    assert row is not None and row.id == "claude-opus-4.5"
    assert row.input_price == 5.0
    dashed = _lookup("openai", "gpt-5-4", tmp_path)
    assert dashed is not None and dashed.id == "gpt-5.4"


def test_kimi_tries_moonshotai_then_the_coding_plan(tmp_path) -> None:
    priced = _lookup("kimi", "kimi-k2.5", tmp_path)
    assert priced is not None and priced.input_price == 0.6
    # `k3` is only in the coding-plan catalogue, which quotes limits and a
    # stated 0/0 cost — the plan bills credits, so zero is the ANSWER, not a
    # miss. The chain must not let OpenRouter price it.
    plan_only = _lookup("kimi", "k3", tmp_path)
    assert plan_only is not None and plan_only.context_window == 262_144
    assert plan_only.input_price == 0.0 and plan_only.output_price == 0.0


def test_openrouter_ids_are_looked_up_under_their_own_namespace(tmp_path) -> None:
    row = _lookup("openrouter", "anthropic/claude-fable-5.1", tmp_path)
    assert row is not None and row.cache_write_price == 12.5


def test_an_unmapped_provider_returns_none_without_a_request(tmp_path) -> None:
    recorder = _Canned([])
    with patch("httpx.get", recorder):
        assert price_catalogue_row("radient", "some/model", cache_dir=tmp_path) is None
        assert price_catalogue_row("ollama", "qwen3:8b", cache_dir=tmp_path) is None
    assert recorder.calls == []
    assert not (tmp_path / f"{PRICE_CATALOGUE_KEY}.json").exists()


def test_supports_images_is_never_taken_from_the_price_catalogue(tmp_path) -> None:
    """`attachment: true` is projected but NOT read as a capability: a stated
    `false` is the provider's denial, and a second-hand source cannot issue one."""
    row = _lookup("anthropic", "claude-fable-5-1", tmp_path)
    assert row is not None
    assert row.supports_images is None


# -- the ranked chain: models.dev, then OpenRouter, then nothing ----------------


def _providers(body: dict[str, Any] | None = None) -> dict[str, Any]:
    return project(_MODELS_DEV_BODY if body is None else body, _ETAG)["providers"]


def _without(model_id: str, provider: str = "anthropic") -> dict[str, Any]:
    body = json.loads(json.dumps(_MODELS_DEV_BODY))
    body[provider]["models"].pop(model_id)
    return body


def test_a_models_dev_miss_is_priced_from_openrouter() -> None:
    """The day-0 gap: the row is not in models.dev yet, OpenRouter has it under
    the vendor namespace with the dotted spelling. The chain answers 10/50 and
    carries the write price, instead of leaving the id unpriced."""
    row = price_row(
        "anthropic",
        "claude-fable-5-1",
        models_dev=_providers(_without("claude-fable-5-1")),
        openrouter=[_OPENROUTER_FABLE],
    )
    assert row is not None
    assert (row.input_price, row.output_price, row.cache_write_price) == (10.0, 50.0, 12.5)


def test_both_sources_missing_is_none_so_the_registry_decides() -> None:
    row = price_row(
        "anthropic", "claude-nova-9", models_dev=_providers(), openrouter=[_OPENROUTER_FABLE]
    )
    assert row is None


def test_a_models_dev_price_is_never_overridden_by_openrouter() -> None:
    """Primary wins outright, even when the secondary quotes something else."""
    cheaper = DiscoveredModel(
        id="anthropic/claude-fable-5.1", input_price=1.0, output_price=2.0, context_window=42
    )
    row = price_row("anthropic", "claude-fable-5-1", models_dev=_providers(), openrouter=[cheaper])
    assert row is not None
    assert (row.input_price, row.output_price) == (10.0, 50.0)
    assert row.context_window == 1_000_000, "and its native limits ride along"


def test_a_disagreement_over_five_percent_is_logged_and_not_acted_on(caplog) -> None:
    caplog.set_level("DEBUG", logger="local_operator.model.prices")
    cheaper = DiscoveredModel(id="anthropic/claude-fable-5.1", input_price=9.0, output_price=50.0)
    row = price_row("anthropic", "claude-fable-5-1", models_dev=_providers(), openrouter=[cheaper])
    assert row is not None and row.input_price == 10.0
    assert any("disagree" in record.message for record in caplog.records)
    caplog.clear()
    within = DiscoveredModel(id="anthropic/claude-fable-5.1", input_price=9.7, output_price=50.0)
    price_row("anthropic", "claude-fable-5-1", models_dev=_providers(), openrouter=[within])
    assert not any("disagree" in record.message for record in caplog.records)


def test_a_models_dev_stated_zero_is_an_answer_not_a_miss(tmp_path) -> None:
    """``zai/glm-4.7-flash`` is $0 on Z.AI's own pricing page; models.dev states
    that as ``cost: {input: 0, output: 0, ...}``. That stated zero is the
    ANSWER — the OpenRouter row for the same weights is a third-party host's
    rate, and the chain must not print a number the user is not paying.
    Reproduced from the reviewer's repro on the real 2026-09-02 documents:
    main resolved 0.0/0.0, the pre-fix branch resolved 0.06/0.40."""
    # An OpenRouter sibling EXISTS for this model — that is the trap.
    openrouter_sibling = DiscoveredModel(
        id="z-ai/glm-4.7-flash", input_price=0.06, output_price=0.40
    )
    row = price_row(
        "zai", "glm-4.7-flash", models_dev=_providers(), openrouter=[openrouter_sibling]
    )
    assert row is not None
    assert (row.input_price, row.output_price) == (
        0.0,
        0.0,
    ), "a stated zero from models.dev is the answer, not a miss"
    assert row.context_window == 200_000
    # The struct never carries a negative price out of the chain.
    assert row.input_price >= 0 and row.output_price >= 0


def test_a_stated_zero_never_reaches_the_openrouter_leg(tmp_path, openrouter) -> None:
    """The resolver path: a models.dev 0/0 costs NO OpenRouter read, the same
    short-circuit a priced row gets."""
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG))
    openrouter.rows.append(
        DiscoveredModel(id="z-ai/glm-4.7-flash", input_price=0.06, output_price=0.40)
    )
    row = price_catalogue_row("zai", "glm-4.7-flash", cache_dir=tmp_path)
    assert row is not None and (row.input_price, row.output_price) == (0.0, 0.0)
    assert openrouter.calls == [], "a stated zero answers the chain before the secondary"


def test_an_absent_cost_is_a_miss_and_falls_through(tmp_path) -> None:
    """``google/gemma-4-31b-it``: models.dev has the row with limits but NO
    ``cost`` mapping — genuinely unanswered, so the secondary may fill it.
    The distinction is the presence of numeric ``input``/``output``, not the
    numbers' value."""
    secondary = DiscoveredModel(id="google/gemma-4-31b-it", input_price=0.09, output_price=0.34)
    row = price_row(
        "google",
        "gemma-4-31b-it",
        models_dev=_providers(),
        openrouter=[secondary],
    )
    # gemma-4-31b-it is absent from the fixture body entirely: a miss, so the
    # secondary answers — same rule as an empty ``cost`` (``gemma`` rows ship
    # ``cost: {}`` upstream, which projects identically to absent).
    assert row is not None
    assert (row.input_price, row.output_price) == (0.09, 0.34)

    # A ``cost`` mapping with NO numeric input/output is equally unanswered.
    body = json.loads(json.dumps(_MODELS_DEV_BODY))
    body["google"]["models"]["gemma-4-31b-it"] = {
        "id": "gemma-4-31b-it",
        "limit": {"context": 262_144, "output": 32_768},
        "cost": {},
    }
    row = price_row("google", "gemma-4-31b-it", models_dev=_providers(body), openrouter=[secondary])
    assert row is not None
    assert (row.input_price, row.output_price) == (0.09, 0.34)
    assert row.context_window == 262_144, "the stub's native limits ride along"


def test_a_models_dev_stub_with_limits_takes_openrouter_money_and_keeps_its_limits() -> None:
    """A models.dev row whose ``cost`` quotes no numeric ``input``/``output``
    has not answered the money question: the secondary fills the money and the
    native window stays — every field is first-source-that-has-it. This is the
    EMPTY-``cost`` shape (``google/gemma-4-31b-it``), distinct from a stated
    ``0/0`` which stops the chain (see the stated-zero tests above)."""
    body = json.loads(json.dumps(_MODELS_DEV_BODY))
    body["kimi-for-coding"]["models"]["k3"]["cost"] = {}
    secondary = DiscoveredModel(
        id="moonshotai/k3", input_price=0.6, output_price=2.5, context_window=131_072
    )
    row = price_row("kimi", "k3", models_dev=_providers(body), openrouter=[secondary])
    assert row is not None
    assert (row.input_price, row.output_price) == (0.6, 2.5)
    assert row.context_window == 262_144


def test_openrouter_ids_never_enter_the_secondary_leg() -> None:
    """An ``openrouter/*`` id is that listing's own business (leg 1 / leg 3 of the
    resolver); the chain has no namespace for it and does not invent one."""
    assert prices.openrouter_lookup([_OPENROUTER_FABLE], "openrouter", "some/model") is None
    row = price_row(
        "openrouter", "anthropic/claude-fable-5.1", models_dev=_providers(), openrouter=[]
    )
    assert row is not None and row.input_price == 10.0, "models.dev's openrouter key still answers"


def test_a_plan_provider_never_borrows_the_pay_per_token_rate() -> None:
    """``alibaba-token-plan`` bills credits; models.dev's 0/0 is the intended answer
    and the ``qwen/`` USD rate must not be printed for it."""
    assert "alibaba-token-plan" not in prices.OPENROUTER_NAMESPACE
    priced = DiscoveredModel(id="qwen/qwen3.7-max", input_price=1.6, output_price=6.4)
    assert prices.openrouter_lookup([priced], "alibaba-token-plan", "qwen3.7-max") is None


def test_an_unpriced_openrouter_row_is_a_routing_stub_not_a_hit() -> None:
    stub = DiscoveredModel(id="anthropic/claude-fable-5.1", context_window=1_000_000)
    assert prices.openrouter_lookup([stub], "anthropic", "claude-fable-5-1") is None


def test_no_documents_at_all_is_none_without_raising() -> None:
    assert price_row("anthropic", "claude-fable-5-1", models_dev=None, openrouter=None) is None
    assert price_row("anthropic", "claude-fable-5-1", models_dev=None, openrouter=[]) is None


def test_the_resolver_leg_reaches_openrouter_only_on_a_models_dev_miss(
    tmp_path, openrouter
) -> None:
    """A hit on the primary costs NO OpenRouter read: the second document is
    never parsed on a path the first already answered."""
    _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG))
    row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)
    assert row is not None and row.input_price == 10.0
    assert openrouter.calls == []

    _plant(tmp_path, project(_without("claude-fable-5-1"), _ETAG))
    openrouter.rows.append(_OPENROUTER_FABLE)
    with patch("httpx.get", _Canned([_ok(_without("claude-fable-5-1"))])):
        row = price_catalogue_row("anthropic", "claude-fable-5-1", cache_dir=tmp_path)
    assert row is not None and (row.input_price, row.output_price) == (10.0, 50.0)
    assert len(openrouter.calls) == 1
    assert (
        openrouter.calls[0]["want_id"] == "anthropic/claude-fable-5-1"
    ), "the miss-refetch rule is armed for the namespaced row"
    assert openrouter.calls[0]["cache_dir"] == tmp_path


def test_the_openrouter_leg_gets_what_the_primary_left_of_the_budget(tmp_path, openrouter) -> None:
    _plant(
        tmp_path,
        project(_without("claude-fable-5-1"), _ETAG),
        age_s=catalogue.MISS_REFETCH_MIN_AGE_S + 1,
    )
    clock = [100.0]

    def slow_primary(url, **kwargs):
        clock[0] += 2.0  # the models.dev refetch burns two of the three seconds
        return _ok(_without("claude-fable-5-1"))

    with (
        patch("httpx.get", slow_primary),
        patch("local_operator.model.prices.time.monotonic", side_effect=lambda: clock[0]),
    ):
        price_catalogue_row("anthropic", "claude-fable-5-1", timeout=3.0, cache_dir=tmp_path)
    assert len(openrouter.calls) == 1
    assert openrouter.calls[0]["timeout"] == pytest.approx(1.0)


def test_models_dev_providers_reads_disk_only_and_never_fetches(tmp_path) -> None:
    """The picker's bulk read: whatever is on disk, at any age, no request."""
    recorder = _Canned([])
    with patch("httpx.get", recorder):
        assert prices.models_dev_providers(cache_dir=tmp_path) is None
        _plant(tmp_path, project(_MODELS_DEV_BODY, _ETAG), age_s=10 * 24 * 3600)
        providers = prices.models_dev_providers(cache_dir=tmp_path)
    assert providers is not None and "claude-fable-5-1" in providers["anthropic"]
    assert recorder.calls == []
    # An older capture reads as absent rather than as a half-usable document.
    _plant(tmp_path, {"capture": 0, "providers": {}})
    assert prices.models_dev_providers(cache_dir=tmp_path) is None
