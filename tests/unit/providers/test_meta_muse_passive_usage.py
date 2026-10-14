"""A real serving stream updates only its account; stale snapshots never mean zero."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from local_operator.harness.types import ChatRequest, StreamEndEvent
from local_operator.model.configure import build_model_spec
from local_operator.providers.auth_store import AuthStore, OAuthAccess
from local_operator.providers.clients import client_for_spec
from local_operator.providers.controller import ProviderController
from local_operator.providers.failover import stream_with_failover
from local_operator.providers.usage import (
    META_MUSE_ALLOWANCE_NOTE,
    META_MUSE_OBSERVATION_TTL_MS,
    META_MUSE_PASSIVE_NOTE,
    META_MUSE_STALE_NOTE,
    meta_muse_usage_report,
)


def observation(age=0, **overrides):
    return {
        "observedAtMs": int(time.time() * 1000) - age,
        "source": "responses",
        "window": {"usedPercent": 47, "windowDurationMins": 300},
        "weekly": {"usedPercent": 72},
        **overrides,
    }


@pytest.mark.parametrize("zero", [True, False])
def test_sign_in_zero_or_nonzero_never_claims_current_usage(zero):
    raw = observation(86_400_000)
    raw.pop("source")
    if zero:
        raw["window"]["usedPercent"] = raw["weekly"]["usedPercent"] = 0
    report = meta_muse_usage_report({"subscription_observation": raw})
    assert not report.limits
    assert report.fetched_at == raw["observedAtMs"]
    assert report.notes == META_MUSE_ALLOWANCE_NOTE


@pytest.mark.parametrize("expired_reset", [True, False])
def test_expired_response_keeps_age_but_hides_meters(expired_reset):
    raw = observation(0 if expired_reset else META_MUSE_OBSERVATION_TTL_MS + 1)
    if expired_reset:
        raw["weekly"]["resetsAtMs"] = int(time.time() * 1000) - 1
    report = meta_muse_usage_report({"subscription_observation": raw})
    assert report.limits == []
    assert report.fetched_at == raw["observedAtMs"]
    assert report.notes == META_MUSE_STALE_NOTE


def test_verified_fresh_zero_is_not_an_unknown_default():
    raw = observation(window={"usedPercent": 0}, weekly={"usedPercent": 0})
    report = meta_muse_usage_report({"subscription_observation": raw})
    assert [limit.amount.fraction() for limit in report.limits] == [0, 0]
    assert report.notes == META_MUSE_PASSIVE_NOTE


def test_an_open_pane_expires_verified_zero_without_mutating_shared_report():
    from local_operator.tui.widgets.usage_panel import build_usage_body

    raw = observation(window={"usedPercent": 0}, weekly={"usedPercent": 0})
    report = meta_muse_usage_report({"subscription_observation": raw})
    body = build_usage_body(
        [report], width=100, now_ms=raw["observedAtMs"] + META_MUSE_OBSERVATION_TTL_MS + 1
    )
    text = "\n".join(line.plain for line in body.lines)
    assert "current usage unknown" in text
    assert "0%" not in text
    assert len(report.limits) == 2
    assert report.notes == META_MUSE_PASSIVE_NOTE


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_cache", [False, True])
async def test_serving_dispatch_persists_sanitized_receipt_and_refresh_reads_it(
    tmp_path, monkeypatch, broken_cache, caplog
):
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    controller = ProviderController(store, tmp_path)
    row = store.upsert_credential(
        "meta-muse",
        {"type": "oauth", "access": "LLM|1234|fixture_subscription_key", "email": "a@example.test"},
    )
    other = store.upsert_credential(
        "meta-muse",
        {"type": "oauth", "access": "LLM|5678|other_subscription_key", "email": "b@example.test"},
    )
    wire = {
        "window": {"used_percent": 47, "window_duration_mins": 300, "resets_at": 1_800_000_000},
        "weekly": {"used_percent": 72, "resets_at": 1_800_100_000},
        "opaque_secret": "must-not-be-saved",
    }
    writer_threads = []
    write_receipt = store.record_muse_usage

    def observe_writer(*args):
        writer_threads.append(threading.get_ident())
        return write_receipt(*args)

    monkeypatch.setattr(store, "record_muse_usage", observe_writer)
    if broken_cache:

        def broken_write(*_):
            raise RuntimeError("secret-bearing failure must not be logged")

        monkeypatch.setattr(store, "record_muse_usage", broken_write)
    frames = [
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
        {"type": "response.subscription_usage", "subscription": wire},
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, content="".join("data: " + json.dumps(f) + "\n\n" for f in frames)
            )
        )
    ) as http:

        async def client_for(spec):
            return client_for_spec(spec, http_client=http)

        request = ChatRequest(model=build_model_spec("meta-muse", "muse-wonder-1.0"))
        events = [e async for e in stream_with_failover(request, store, {}, client_for)]
        assert any(isinstance(e, StreamEndEvent) for e in events)
    try:
        retained = store.get_credential(row.id)
        assert retained is not None
        data = retained.data
        if broken_cache:
            assert "subscription_observation" not in data
            assert "secret-bearing" not in caplog.text
            assert "Muse allowance receipt could not be retained" in caplog.text
            return
        assert writer_threads and all(t != threading.get_ident() for t in writer_threads)
        assert data["subscription_observation"]["source"] == "responses"
        assert "must-not-be-saved" not in json.dumps(data)
        untouched = store.get_credential(other.id)
        assert untouched is not None
        assert "subscription_observation" not in untouched.data
        reports = await controller.fetch_usage(force_refresh=True)
        served = next(r for r in reports if r.identity == "a@example.test")
        assert [x.amount.fraction() for x in served.limits] == [0.47, 0.72]
        assert [x.label for x in served.limits] == ["5 hour", "7 day"]
        assert [x.resets_at_ms for x in served.limits] == [1_800_000_000_000, 1_800_100_000_000]
    finally:
        store.close()


@pytest.mark.parametrize("fresh", [True, False])
def test_read_only_snapshot_projects_receipts_even_without_usage_cache(tmp_path, fresh):
    from local_operator.providers.usage_cache import UsageCacheStore

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    row = store.upsert_credential(
        "meta-muse", {"type": "oauth", "access": "LLM|1234|fixture_subscription_key"}
    )
    if fresh:
        access = OAuthAccess(access_token="fixture", credential_id=row.id, raw=row.data)
        assert store.record_muse_usage(access, observation())
    store.close()
    metadata = AuthStore.open_read_only(tmp_path / "auth.db")
    try:
        controller = ProviderController(
            metadata, tmp_path, usage_cache=UsageCacheStore(tmp_path / "absent.db", read_only=True)
        )
        snapshot = controller.cached_persisted_usage_reports_read_only("meta-muse")
        assert not snapshot.unavailable
        assert len(snapshot.reports) == 1
        assert bool(snapshot.reports[0].limits) is fresh
        assert snapshot.reports[0].identity is None
        assert snapshot.reports[0].notes is not None
        if fresh:
            assert snapshot.reports[0].notes == META_MUSE_PASSIVE_NOTE
        else:
            assert "refresh does not query Muse" in snapshot.reports[0].notes
        assert not (tmp_path / "absent.db").exists()
    finally:
        metadata.close()


def test_receipt_rejects_wrong_key_and_older_stream_without_resealing(tmp_path):
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = store.upsert_credential(
            "meta-muse", {"type": "oauth", "access": "LLM|1234|fixture_subscription_key"}
        )
        access = OAuthAccess(access_token="fixture", credential_id=row.id, raw=row.data)
        assert store.record_muse_usage(access, observation())
        retained = store.get_credential(row.id)
        assert retained is not None
        current = retained.data
        assert current["access_secret_ref"] == row.data["access_secret_ref"]
        assert not store.record_muse_usage(access, observation(1000))
        wrong = OAuthAccess(
            access_token="fixture", credential_id=row.id, raw={"access_secret_ref": "wrong"}
        )
        assert not store.record_muse_usage(wrong, observation())
        unchanged = store.get_credential(row.id)
        assert unchanged is not None
        assert unchanged.data == current
    finally:
        store.close()
