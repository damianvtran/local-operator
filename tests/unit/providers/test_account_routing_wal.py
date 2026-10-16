"""Routing must read committed WAL facts, not the phone's immutable snapshot."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import UsageCacheStore, account_preflight_key


def quota(identity: str, fraction: float) -> UsageReport:
    now = int(time.time() * 1000)
    return UsageReport(
        "anthropic",
        fetched_at=now,
        identity=identity,
        limits=[
            UsageLimit(
                id="anthropic:5h",
                label="5 hour",
                window="5 hour",
                shared=True,
                amount=UsageAmount(used_fraction=fraction),
                resets_at_ms=now + 3_600_000,
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_root", [False, True], ids=["default-root", "explicit-root"])
async def test_public_selection_observes_committed_depletion_with_writer_held_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_root: bool
) -> None:
    ambient = tmp_path / "ambient"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    root = tmp_path / "explicit" if explicit_root else ambient
    auth = AuthStore(db_path=root / "auth.db") if explicit_root else AuthStore()
    rows = [
        auth.upsert_credential(
            "anthropic",
            {
                "type": "oauth",
                "access": f"synthetic-access-{index}",
                "refresh": "synthetic-refresh",
                "expires": None,
                "email": f"synthetic-{index}@example.invalid",
            },
        )
        for index in range(2)
    ]
    keys = [account_preflight_key("anthropic", row.data["email"]) for row in rows]
    writer = UsageCacheStore(auth.db_path.parent / "usage_cache.db")
    display = UsageCacheStore(writer.db_path, read_only=True)
    fresh = None
    supplied = None
    probes: list[bool] = []

    async def forbidden_quota_probe(*args, **kwargs):
        probes.append(True)
        raise AssertionError("cached account ranking must not fetch quota")

    monkeypatch.setattr("local_operator.providers.usage.fetch_usage", forbidden_quota_probe)
    try:
        expires = int(time.time() * 1000) + 600_000
        for key, row, fraction in zip(keys, rows, (0.1, 0.3), strict=True):
            writer.set(
                key, "anthropic", [quota(row.data["email"], fraction)], expires_at_ms=expires
            )
        connection = writer._connect()
        assert connection is not None
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.set(keys[0], "anthropic", [quota(rows[0].data["email"], 1.0)], expires_at_ms=expires)
        # This control must stay stale: the paired-phone display deliberately
        # avoids coordination sidecars. Routing must NOT borrow its snapshot.
        shown = display.get(keys[0])
        assert shown is not None and shown[0].limits[0].amount.fraction() == 0.1
        current = writer.get(keys[0])
        assert current is not None and current[0].limits[0].amount.fraction() == 1.0
        auth.configure_account_selection_policy("headroom")
        access = await auth.get_oauth_access(
            "anthropic", "synthetic-wal", model_id="claude-sonnet-5"
        )
        assert access is not None and access.credential_id == rows[1].id
        # A separately constructed store must see the same committed fact, not
        # a pointer or connection accidentally retained by the first selector.
        fresh = AuthStore(db_path=auth.db_path)
        fresh.configure_account_selection_policy("headroom")
        access = await fresh.get_oauth_access(
            "anthropic", "synthetic-wal-fresh", model_id="claude-sonnet-5"
        )
        assert access is not None and access.credential_id == rows[1].id
        supplied = AuthStore(db_path=auth.db_path, usage_cache=writer)
        supplied.configure_account_selection_policy("headroom")
        access = await supplied.get_oauth_access(
            "anthropic", "synthetic-wal-supplied", model_id="claude-sonnet-5"
        )
        assert access is not None and access.credential_id == rows[1].id
        assert probes == []
        if explicit_root:
            assert not (ambient / "usage_cache.db").exists()
    finally:
        if fresh is not None:
            fresh.close()
        if supplied is not None:
            supplied.close()
        display.close()
        auth.close()
        writer.close()
