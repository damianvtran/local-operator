"""Receipt authority must not depend on metadata availability or bool coercion."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import replace
from typing import Any

import httpx
import pytest

from local_operator.analytics.model import CallSnapshot, price_snapshot
from local_operator.analytics.recorder import AnalyticsRecorder
from local_operator.analytics.store import AnalyticsStore
from local_operator.harness.types import (
    ChatRequest,
    Message,
    ModelSpec,
    StreamUsageEvent,
)
from local_operator.model import configure
from local_operator.model.registry import get_model_info, unknown_model_info
from local_operator.providers.clients import OpenAICompatClient, _usd_cost
from local_operator.session.spend import price_call


def _snapshot(receipt: Any) -> CallSnapshot:
    return CallSnapshot(
        ts_ms=time.time_ns() // 1_000_000,
        session_id="receipt-test",
        provider="openrouter",
        model_id="unknown/receipt-sku",
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=1000,
        usd_cost=receipt,
        billing_basis="billed",
        cost_source="provider_reported",
    )


@pytest.mark.parametrize("receipt", [0.0075, "0.0075", 0, 0.0, -0.0, "0", "-0"])
def test_valid_receipt_prices_both_paths_without_metadata(receipt, monkeypatch):
    attempted = []

    def unavailable(*args, **kwargs):
        attempted.append(args)
        raise OSError("metadata unavailable")

    monkeypatch.setattr(configure, "resolve_model_info", unavailable)
    snapshot = _snapshot(receipt)
    expected = (round(float(receipt) * 1_000_000), True)
    assert price_snapshot(snapshot) == expected
    assert price_call(snapshot.provider, snapshot.model_id, snapshot) == expected
    assert attempted == []


def test_priced_override_still_wins_over_receipt_and_metadata(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("a final price must not be resolved again")

    monkeypatch.setattr(configure, "resolve_model_info", unavailable)
    snapshot = replace(_snapshot(0.0075), priced=True, cost_micro=23, cost_known=False)
    assert price_snapshot(snapshot) == (23, False)


@pytest.mark.parametrize("receipt", [None, -1, float("nan"), float("inf"), "bad", True, False])
def test_invalid_receipt_retains_catalogue_estimate_parity(receipt, monkeypatch):
    # Published rates, not a fabricated estimate: both pricers must still use
    # the shared tariff calculation when the raw receipt is not money.
    info = get_model_info("anthropic", "claude-sonnet-4-20250514")
    attempted = []

    def catalogue(*args, **kwargs):
        attempted.append(args)
        return info

    monkeypatch.setattr(configure, "resolve_model_info", catalogue)
    snapshot = _snapshot(receipt)
    clean = replace(snapshot, usd_cost=None)
    expected = round(configure.cost_for_usage(snapshot.provider, info, clean) * 1_000_000)
    assert expected not in (0, 1_000_000)
    assert price_snapshot(snapshot) == (expected, True)
    assert price_call(snapshot.provider, snapshot.model_id, snapshot) == (expected, True)
    assert len(attempted) == 2
    assert configure._usage_cost(snapshot) is None
    assert _usd_cost({"cost": receipt}) is None


@pytest.mark.parametrize("raises", [False, True])
def test_missing_receipt_remains_unknown_when_metadata_cannot_price(raises, monkeypatch):
    def unavailable(*args, **kwargs):
        if raises:
            raise OSError("metadata unavailable")
        return unknown_model_info

    monkeypatch.setattr(configure, "resolve_model_info", unavailable)
    snapshot = _snapshot(None)
    assert price_snapshot(snapshot) == (0, False)
    assert price_call(snapshot.provider, snapshot.model_id, snapshot) == (0, False)


def test_queued_receipt_settles_without_entering_cold_transport(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def held_metadata(self, request, *args, **kwargs):
        entered.set()
        # A controlled unavailable catalogue, never a manufactured rate or a
        # widened recorder deadline. Pre-fix, the unchanged five-second barrier
        # fails while this real transport boundary holds the writer.
        if not release.wait(10):
            raise RuntimeError("receipt test controller failed to release")
        return httpx.Response(503, request=request)

    monkeypatch.setattr(httpx.Client, "send", held_metadata)
    configure.invalidate_model_info_cache()
    snapshot = replace(
        _snapshot(0.0075),
        parent_session_id="synthetic-parent",
        purpose="child",
        component_chars={"conversation": 30, "tool_inventory": 10},
    )
    store = AnalyticsStore(tmp_path / "receipt.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.record(snapshot)
        rec.flush_for_test()
        assert not entered.is_set()
        assert rec._queue.unfinished_tasks == 0 and rec.dropped == 0
        assert rec._write_failures == {}
        with sqlite3.connect(tmp_path / "receipt.db") as conn:
            assert conn.execute(
                "SELECT provider, model_id, cost_micro, cost_known, billing_basis, "
                "cost_source, parent_session_id, purpose, c_conversation, c_tool_inventory "
                "FROM calls"
            ).fetchall() == [
                (
                    snapshot.provider,
                    snapshot.model_id,
                    7500,
                    1,
                    "billed",
                    "provider_reported",
                    "synthetic-parent",
                    "child",
                    750,
                    250,
                )
            ]
    finally:
        release.set()
        rec.close()


def test_missing_receipt_still_enters_the_real_metadata_transport(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def held_metadata(self, request, *args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError("fallback test controller failed to release")
        return httpx.Response(503, request=request)

    monkeypatch.setattr(httpx.Client, "send", held_metadata)
    configure.invalidate_model_info_cache()
    store = AnalyticsStore(tmp_path / "fallback.db")
    rec = AnalyticsRecorder(store=store)
    try:
        rec.record(_snapshot(None))
        assert entered.wait(5), "missing receipt incorrectly bypassed the catalogue"
        assert rec._thread is not None and rec._thread.is_alive()
        assert rec._queue.unfinished_tasks == 1
        with sqlite3.connect(tmp_path / "fallback.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM calls").fetchone() == (0,)
        release.set()
        rec.flush_for_test()
        assert rec._write_failures == {} and rec._queue.unfinished_tasks == 0
        with sqlite3.connect(tmp_path / "fallback.db") as conn:
            assert conn.execute("SELECT cost_micro, cost_known FROM calls").fetchone() == (
                0,
                0,
            )
    finally:
        release.set()
        rec.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [True, False])
async def test_raw_wire_bool_is_absent_before_usage_float_coercion(receipt):
    # Drive the actual SSE parser: fixing only the duck-typed helper would let
    # Usage's float field convert a raw bool back into a seemingly valid bill.
    chunks = [
        {"choices": [{"delta": {"content": "ok"}, "index": 0}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": receipt}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )
    )
    async with httpx.AsyncClient(transport=transport) as wire:
        client = OpenAICompatClient("https://receipt.test/v1", http_client=wire)
        request = ChatRequest(
            model=ModelSpec(provider="openrouter", model_id="unknown/receipt-sku"),
            messages=[Message.user("hi")],
        )
        usages = [
            event.usage
            async for event in client.stream(request, "synthetic-test-key")
            if isinstance(event, StreamUsageEvent)
        ]
    assert usages and all(usage.usd_cost is None for usage in usages)
