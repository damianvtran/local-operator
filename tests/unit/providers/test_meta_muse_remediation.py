"""Mixed-panel vintage, legacy-cache and CAS/idle-pane regression surfaces."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest
from rich.text import Text
from textual.content import Content

from local_operator.providers.auth_store import AuthStore, OAuthAccess
from local_operator.providers.controller import ProviderController
from local_operator.providers.usage import (
    META_MUSE_ALLOWANCE_NOTE,
    META_MUSE_PASSIVE_NOTE,
    META_MUSE_STALE_NOTE,
    UsageReport,
    meta_muse_usage_report,
)
from local_operator.providers.usage_cache import UsageCacheStore
from local_operator.tui.widgets.usage_panel import UsagePanel, build_usage_body
from tests.unit.providers.test_meta_muse_passive_usage import observation


@pytest.mark.parametrize("width", [100, 40])
@pytest.mark.parametrize("note", [META_MUSE_ALLOWANCE_NOTE, META_MUSE_STALE_NOTE])
def test_unknown_muse_age_survives_fresh_sibling_and_compaction(width, note):
    now = int(time.time() * 1000)
    muse = UsageReport(provider="meta-muse", fetched_at=now - 86_400_000, notes=note)
    fresh = UsageReport(provider="deepseek", fetched_at=now)
    body = build_usage_body([muse, fresh], width, now, header_ms=now)
    text = "\n".join(line.plain for line in body.lines)
    muse_text = text.split("deepseek")[0]
    assert "1d ago" in muse_text
    assert "r won't query Muse" in muse_text
    assert "no windows reported" not in muse_text
    assert "last known\n" not in muse_text
    panel = UsagePanel()
    rows, _ = panel._window_rows(body, 3)
    compact = "\n".join(row.plain for row in rows)
    assert "1d ago" in compact
    assert "r won't query Muse" in compact


@pytest.mark.parametrize("width", [100, 40])
def test_passive_age_is_visible_before_five_minute_expiry(width):
    raw = observation(240_000)
    muse = meta_muse_usage_report({"subscription_observation": raw})
    now = raw["observedAtMs"] + 240_000
    body = build_usage_body(
        [muse, UsageReport(provider="deepseek", fetched_at=now)], width, now, now
    )
    text = "\n".join(line.plain for line in body.lines)
    assert "4m ago" in text
    assert "47%" in text
    assert "r won't query Muse" in text


@pytest.mark.asyncio
async def test_old_zero_cache_never_shadows_current_metadata_on_any_projection(tmp_path):
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    cache = UsageCacheStore(tmp_path / "usage.db")
    controller = ProviderController(store, tmp_path, usage_cache=cache)
    row = store.upsert_credential(
        "meta-muse", {"type": "oauth", "access": "LLM|1234|synthetic_key"}
    )
    old = meta_muse_usage_report(
        {
            "subscription_observation": observation(
                window={"usedPercent": 0}, weekly={"usedPercent": 0}
            )
        }
    )
    old.fetched_at -= 86_400_000
    old.notes = "Last sign-in allowance; refresh does not query Muse."
    cache.set(
        controller._usage_cache_key("meta-muse"),
        "meta-muse",
        [old],
        expires_at_ms=int(time.time() * 1000) + 600_000,
    )
    # Make the seeded legacy schema/row visible to the intentionally immutable
    # metadata-only reader; an uncheckpointed foreign schema is unavailable.
    connection = cache._conn
    assert connection is not None
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert store.record_muse_usage(
        OAuthAccess(access_token="fixture", credential_id=row.id, raw=row.data), observation()
    )
    try:
        for reports in (
            controller.cached_usage_reports("meta-muse"),
            await controller.fetch_usage(["meta-muse"]),
            controller.cached_persisted_usage_reports_read_only("meta-muse").reports,
        ):
            assert len(reports) == 1
            assert [x.amount.fraction() for x in reports[0].limits] == [0.47, 0.72]
            assert reports[0].notes == META_MUSE_PASSIVE_NOTE
        assert controller.usage_cache_age_ms("meta-muse") == 0
    finally:
        controller.close()
        store.close()


@pytest.mark.parametrize("provider", [None, "meta-muse"])
def test_absent_metadata_and_usage_cache_keep_null_cached_at_without_writes(tmp_path, provider):
    auth = tmp_path / "absent-auth.db"
    usage = tmp_path / "absent-usage.db"
    controller = ProviderController.read_only_for_persisted_usage(auth, usage)
    try:
        snapshot = controller.cached_persisted_usage_reports_read_only(provider)
        assert snapshot.reports == []
        assert snapshot.cached_at is None
        assert not snapshot.unavailable
        assert not auth.exists() and not usage.exists()
    finally:
        controller.close()
        # The read-only factory's protocol omits close, but its concrete store
        # owns the metadata handle. Narrow rather than suppress that contract.
        assert isinstance(controller.auth_store, AuthStore)
        controller.auth_store.close()


@pytest.mark.parametrize("race_before", ["SELECT data FROM", "UPDATE auth_credentials SET data"])
def test_cas_lost_race_preserves_concurrent_metadata(tmp_path, monkeypatch, race_before):
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    row = store.upsert_credential(
        "meta-muse", {"type": "oauth", "access": "LLM|1234|synthetic_key"}
    )
    original = store._conn
    concurrent = {**row.data, "source": "concurrent-login"}

    class Race:
        def execute(self, sql, args=()):
            if sql.startswith(race_before):
                # A second writer lands either between parsed/raw reads or
                # between the final read and CAS. Both guards must discriminate.
                with sqlite3.connect(tmp_path / "auth.db") as other:
                    other.execute(
                        "UPDATE auth_credentials SET data = ? WHERE id = ?",
                        (json.dumps(concurrent), row.id),
                    )
            return original.execute(sql, args)

        def __getattr__(self, name):
            return getattr(original, name)

    monkeypatch.setattr(store, "_conn", Race())
    try:
        assert not store.record_muse_usage(
            OAuthAccess(access_token="fixture", credential_id=row.id, raw=row.data), observation()
        )
        retained = store.get_credential(row.id)
        assert retained is not None
        assert retained.data == concurrent
    finally:
        store.close()


@pytest.mark.asyncio
async def test_real_usage_command_refresh_and_idle_expiry(tmp_path, monkeypatch):
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    row = store.upsert_credential(
        "meta-muse", {"type": "oauth", "access": "LLM|1234|synthetic_key"}
    )
    controller = ProviderController(store, tmp_path)
    app = OperatorApp(lambda: _factory(FakeSession()))
    try:
        async with app.run_test(size=(120, 38)) as pilot:
            await pilot.pause()
            app._providers = controller
            app._cmd_usage("meta-muse", app._system_notice)
            await app.workers.wait_for_complete()
            panel = app.query_one(UsagePanel)
            assert panel.display
            assert panel._reports[0].limits == []
            raw = observation()
            assert store.record_muse_usage(
                OAuthAccess(access_token="fixture", credential_id=row.id, raw=row.data), raw
            )
            await pilot.press("r")
            await app.workers.wait_for_complete()
            assert [x.amount.fraction() for x in panel._reports[0].limits] == [0.47, 0.72]
            # Wait for the actual mounted timer, not a direct pure-render call.
            repainted = asyncio.Event()
            repaint = panel._repaint

            def observe_repaint():
                repaint()
                repainted.set()

            monkeypatch.setattr(panel, "_repaint", observe_repaint)
            panel.set_clock(raw["observedAtMs"] + 300_001)
            await asyncio.wait_for(repainted.wait(), 3)
            rendered = panel.render()
            assert isinstance(rendered, (Text, Content))
            text = rendered.plain
            assert "unknown" in text.lower()
            assert "47%" not in text
            assert "r won't query Muse" in text
    finally:
        controller.close()
        store.close()
