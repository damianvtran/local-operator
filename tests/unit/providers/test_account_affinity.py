"""Local affinity expiry/restart contract with wall and monotonic controls."""

import json

import pytest

from local_operator.providers import account_affinity as affinity

NOW = 1_800_000_000_000


@pytest.fixture
def clock(monkeypatch):
    current = {"wall": NOW, "mono": 1000.0}
    monkeypatch.setattr(affinity.time, "time", lambda: current["wall"] / 1000)
    monkeypatch.setattr(affinity.time, "monotonic", lambda: current["mono"])
    return current


def hint(ttl=300_000, signature="compatible"):
    return affinity.CacheAffinityHint("anthropic:messages", signature, ttl, "explicit-wire")


def record(store, **overrides):
    values = dict(
        session_id="synthetic-session",
        provider="anthropic",
        account_id=2,
        account_created_at=42,
        model_id="claude-sonnet-5",
        hint=hint(),
        sent_at_ms=NOW,
        sent_monotonic=1000.0,
    )
    values.update(overrides)
    return store.record(**values)


def get(store, **overrides):
    values = dict(
        session_id="synthetic-session",
        provider="anthropic",
        model_id="claude-sonnet-5",
        compatibility="compatible",
    )
    values.update(overrides)
    return store.get(**values)


def test_reading_never_creates_or_renews(clock, tmp_path):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    assert get(store) is None
    assert not store.root.exists()
    assert record(store)
    original = store.path_for("synthetic-session").read_bytes()
    for elapsed in (1, 59_999, 299_999):
        clock.update(wall=NOW + elapsed, mono=1000 + elapsed / 1000)
        receipt = get(store)
        assert receipt.account_id == 2
        assert receipt.expires_at_ms == NOW + 300_000
        assert store.path_for("synthetic-session").read_bytes() == original
    clock.update(wall=NOW + 300_000, mono=1300)
    assert get(store) is None


def test_resume_preserves_original_expiry_and_namespace(clock, tmp_path):
    path = tmp_path / "auth.db"
    first = affinity.AccountAffinityStore(path)
    assert record(first)
    clock.update(wall=NOW + 200_000, mono=1200)
    resumed = affinity.AccountAffinityStore(path)
    assert get(resumed).expires_at_ms == NOW + 300_000
    assert get(resumed, session_id="copied-session") is None
    other = affinity.AccountAffinityStore(tmp_path / "different.db")
    assert get(other) is None
    assert record(other, account_id=1, sent_at_ms=NOW + 100_000, sent_monotonic=1100)
    assert get(other).account_id == 1
    assert get(resumed).account_id == 2
    assert first.path_for("synthetic-session") != other.path_for("synthetic-session")
    clock.update(wall=NOW + 300_000, mono=1300)
    assert get(resumed) is None
    assert get(affinity.AccountAffinityStore(path)) is None


def test_clock_rollback_and_monotonic_expiry(clock, tmp_path):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    assert record(store)
    clock.update(wall=NOW - 1, mono=1001)
    assert get(store) is None
    assert get(affinity.AccountAffinityStore(tmp_path / "auth.db")) is None
    clock.update(wall=NOW + 10_000, mono=1300)
    assert get(store) is None  # wall clock alone must not extend an active lease


def test_model_signature_and_long_request_invalidate(clock, tmp_path):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    assert record(store)
    assert get(store, model_id="claude-opus-4") is None
    assert get(store, compatibility="compacted-or-new-settings") is None
    assert not record(store, completed_at_ms=NOW + 300_000)
    assert not record(store, hint=hint(0))
    assert not record(store, hint=hint(86_400_001))
    assert not record(store, account_created_at=0)
    assert not record(store, account_id=-1)


def test_late_completion_never_overwrites_newer_success(clock, tmp_path):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    assert record(store)
    clock.update(wall=NOW + 5000, mono=1005)
    assert record(store, sent_at_ms=NOW + 4000, sent_monotonic=1004)
    assert not record(store, account_id=1, completed_at_ms=NOW + 5000)
    assert get(store).account_id == 2
    assert get(store).expires_at_ms == NOW + 304_000


@pytest.mark.parametrize(
    "corruption",
    ["json", "schema", "namespace", "future", "account", "oversize", "ttl", "endpoint", "model"],
)
def test_corrupt_persisted_receipts_fail_open_without_rewrite(clock, tmp_path, corruption):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    assert record(store)
    path = store.path_for("synthetic-session")
    payload = json.loads(path.read_text())
    if corruption == "schema":
        payload["schema"] = 999
    elif corruption == "namespace":
        payload["providers"]["anthropic"]["namespace"] = "other-computer"
    elif corruption == "future":
        payload["providers"]["anthropic"]["completed_at_ms"] = NOW + 1
    elif corruption == "account":
        payload["providers"]["anthropic"]["account_id"] = True
    elif corruption == "ttl":
        payload["providers"]["anthropic"]["expires_at_ms"] += 1
    elif corruption == "endpoint":
        payload["providers"]["anthropic"]["endpoint_mode"] = "unknown:gateway"
    elif corruption == "model":
        payload["providers"]["anthropic"]["model_id"] = ""
    raw = (
        "{"
        if corruption == "json"
        else "x" * 65_537 if corruption == "oversize" else json.dumps(payload)
    )
    path.write_text(raw)
    assert get(affinity.AccountAffinityStore(tmp_path / "auth.db")) is None
    assert path.read_text() == raw


def test_failed_persistence_degrades_to_memory_without_replay(clock, tmp_path, monkeypatch):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    monkeypatch.setattr(
        affinity.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("fixture"))
    )
    assert record(store)
    assert get(store).account_id == 2
    assert get(affinity.AccountAffinityStore(tmp_path / "auth.db")) is None
    assert not list(store.root.glob(".affinity-*"))


def test_maximum_same_identity_expiry_survives_shorter_and_late_completions(clock, tmp_path):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    clock.update(wall=NOW + 5000, mono=1005)
    assert record(store, sent_at_ms=NOW + 4000, sent_monotonic=1004)
    assert record(store, hint=hint(3_600_000))
    assert get(store).expires_at_ms == NOW + 3_600_000
    assert not record(store, sent_at_ms=NOW + 5000, sent_monotonic=1005)
    assert get(store).expires_at_ms == NOW + 3_600_000


def test_own_failed_attempt_can_invalidate_receipt(clock, tmp_path):
    store = affinity.AccountAffinityStore(tmp_path / "auth.db")
    assert record(store)
    store.invalidate("synthetic-session", "anthropic")
    assert get(store) is None
    assert get(affinity.AccountAffinityStore(tmp_path / "auth.db")) is None
