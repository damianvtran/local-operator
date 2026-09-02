"""The models.dev price catalogue: provider-neutral prices with no OpenRouter coupling.

The defect this module exists for: Anthropic's ``/v1/models`` quotes no prices,
so a model the registry had not been taught was priced from the OpenRouter
listing under a per-provider namespace. The day ``claude-fable-5-1`` shipped,
that document was six hours old and predated the row, and a user signed in only
to Anthropic ran the whole day at ``$0.00``. models.dev carried
``10/50/0.25/12.5`` on release day.

No live network anywhere here: ``httpx.get`` is patched with a canned response,
and every document lands in ``tmp_path``.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from local_operator.model import catalogue, prices
from local_operator.model.prices import (
    PRICE_CATALOGUE_CAPTURE,
    PRICE_CATALOGUE_KEY,
    price_catalogue_row,
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
                "cost": {},
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
    # `k3` is only in the coding-plan catalogue, which quotes limits and no cost.
    plan_only = _lookup("kimi", "k3", tmp_path)
    assert plan_only is not None and plan_only.context_window == 262_144
    assert plan_only.input_price == 0.0


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
