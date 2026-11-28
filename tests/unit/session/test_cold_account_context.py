"""A cold open reads current account metadata without owning the account's lifetime."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from local_operator.model import configure
from local_operator.model.registry import ModelInfo
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.registry import store_provider_key
from local_operator.session import cold_model
from local_operator.session.frontend_state import FrontendModelSpec

pytestmark = pytest.mark.asyncio

_WINDOWS = {"synthetic-a": 150_000, "synthetic-b": 250_000, "synthetic-rotated": 350_000}


@pytest.fixture(autouse=True)
def _offline_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "OPENAI_OAUTH_TOKEN", "LOCAL_OPERATOR_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(configure, "resolve_model_info", _listing)


def _listing(provider: str, model_id: str, *, credential: Any, **kwargs: Any) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=model_id,
        description="",
        context_window=_WINDOWS[credential[0]],
        max_tokens=8192,
    )


async def _resolve(
    root: Path, key: str = "same-session", model_id: str = "gpt-5.1"
) -> FrontendModelSpec:
    return await cold_model.resolve_context_metadata(
        root, FrontendModelSpec(provider="openai", model_id=model_id), stickiness_key=key
    )


def _seed(root: Path, *, bearer: str = "synthetic-a", expired: bool = False) -> None:
    with closing(AuthStore(root / "auth.db")) as store:
        store.upsert_credential(
            "openai",
            {
                "type": "oauth",
                "access": bearer,
                "refresh": "synthetic-refresh",
                "expires": 1 if expired else int(time.time() * 1000) + 3_600_000,
                "account_id": "synthetic-account",
                "org_id": "synthetic-org",
            },
        )


def _rotation() -> dict[str, Any]:
    return {
        "access": "synthetic-rotated",
        "refresh": "synthetic-rotated-refresh",
        "expires": int(time.time() * 1000) + 3_600_000,
    }


def _read_pair(root: Path) -> tuple[str, str]:
    with closing(AuthStore(root / "auth.db")) as store:
        row = store.list_credentials("openai")[0]
        return row.data["access"], row.data["refresh"]


async def _entered(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5), "worker did not reach the synthetic barrier"


async def test_current_schema_cold_opens_do_not_wait_for_a_sqlite_writer(tmp_path: Path) -> None:
    _seed(tmp_path)
    path = tmp_path / "auth.db"
    before = path.read_bytes()
    with closing(sqlite3.connect(path)) as writer:
        # Keep the real writer reserved until BOTH distinct sessions return. A
        # request that tries to reserve it itself fails SQLite's busy timeout;
        # there is no wall-clock threshold or chance to run after its release.
        writer.execute("BEGIN IMMEDIATE")
        for session in ("first-session", "second-session"):
            answer = await _resolve(tmp_path, session)
            assert answer.context_window == 150_000 and answer.context_metadata_resolved
            assert writer.in_transaction
        writer.rollback()
    assert path.read_bytes() == before
    with closing(AuthStore(path)) as store:
        assert all(
            store.session_credential_id("openai", session) is None
            for session in ("first-session", "second-session")
        )


async def test_same_session_reads_ambient_changes_then_supported_provider_store_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.secrets import client as secrets_client

    # Match the desktop host's config root for both auth and the provider store.
    # Use the real keyfile fallback without a broker or host keychain.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(secrets_client, "PEER_AUTHENTICATION_SUPPORTED", False)
    assert (await _resolve(tmp_path)).context_window == 128_000
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-a")
    assert (await _resolve(tmp_path)).context_window == 150_000
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-b")
    assert (await _resolve(tmp_path)).context_window == 250_000
    monkeypatch.delenv("OPENAI_API_KEY")
    assert (await _resolve(tmp_path)).context_window == 128_000

    store_provider_key("OPENAI_API_KEY", "synthetic-b", base=tmp_path)
    configure.invalidate_model_info_cache()  # same invalidation as the credentials route
    assert (tmp_path / "secrets" / "store.db").is_file()
    assert (await _resolve(tmp_path)).context_window == 250_000


async def test_same_session_reads_committed_login_changes_and_replaced_database(
    tmp_path: Path,
) -> None:
    _seed(tmp_path)
    assert (await _resolve(tmp_path)).context_window == 150_000
    _seed(tmp_path, bearer="synthetic-b")  # same account, new persisted access
    assert (await _resolve(tmp_path)).context_window == 250_000

    replacement = tmp_path / "replacement"
    _seed(replacement, bearer="synthetic-rotated")
    # Every store is already closed, so no old WAL is attached to this path.
    path = tmp_path / "auth.db"
    old_inode = path.stat().st_ino
    (replacement / "auth.db").replace(path)
    assert path.stat().st_ino != old_inode
    assert (await _resolve(tmp_path)).context_window == 350_000


async def test_next_same_session_lookup_recovers_from_failover_soft_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    real = AuthStore.get_oauth_access
    calls = 0

    async def transient_busy(self: AuthStore, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("synthetic transient database failure")
        return await real(self, *args, **kwargs)

    # The real failover bridge catches this and produces a valid fallback spec.
    # The next cold resolution must ask again, not replay that fallback.
    monkeypatch.setattr(AuthStore, "get_oauth_access", transient_busy)
    assert (await _resolve(tmp_path)).context_window == 128_000
    assert (await _resolve(tmp_path)).context_window == 150_000
    assert calls == 2


async def test_slow_listing_does_not_serialize_another_cold_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocked_listing(provider: str, model_id: str, **kwargs: Any) -> ModelInfo:
        if model_id == "gpt-5.2":
            entered.set()
            assert release.wait(10), "test did not release listing"
        return _listing(provider, model_id, **kwargs)

    monkeypatch.setattr(configure, "resolve_model_info", blocked_listing)
    slow = asyncio.create_task(_resolve(tmp_path, "slow", "gpt-5.2"))
    try:
        await _entered(entered)
        fast = await asyncio.wait_for(_resolve(tmp_path, "fast"), 5)
        assert fast.context_window == 150_000 and not slow.done()
    finally:
        release.set()
        await slow


@pytest.mark.skipif(sys.platform == "win32", reason="config directory symlink requires privileges")
async def test_retargeted_config_does_not_close_an_old_db_with_refresh_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, new, live = (tmp_path / name for name in ("old", "new", "live"))
    _seed(old, expired=True)
    _seed(new, bearer="synthetic-b")
    live.symlink_to(old, target_is_directory=True)
    entered, release = threading.Event(), threading.Event()

    async def refresh(creds: dict[str, Any]) -> dict[str, Any]:
        entered.set()
        assert await asyncio.to_thread(release.wait, 10), "test did not release refresh"
        return _rotation()

    monkeypatch.setattr(AuthStore, "_refresh_fn", lambda self, provider: refresh)
    ongoing = asyncio.create_task(_resolve(live))
    try:
        await _entered(entered)
        switch = tmp_path / "switch"
        switch.symlink_to(new, target_is_directory=True)
        os.replace(switch, live)
        replacement = await asyncio.wait_for(_resolve(live, "new-session"), 5)
        assert replacement.context_window == 250_000 and not ongoing.done()
    finally:
        release.set()
        earlier = await ongoing
    assert earlier.context_window == 350_000
    assert _read_pair(old) == ("synthetic-rotated", "synthetic-rotated-refresh")
    assert _read_pair(new) == ("synthetic-b", "synthetic-refresh")
    assert (await _resolve(live)).context_window == 250_000  # original session key


async def test_normal_loop_shutdown_drains_refresh_after_viewer_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path, expired=True)
    stopping, drained, release = threading.Event(), threading.Event(), threading.Event()

    async def viewer() -> None:
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        shutdown = loop.shutdown_default_executor

        async def refresh(creds: dict[str, Any]) -> dict[str, Any]:
            loop.call_soon_threadsafe(entered.set)
            assert await asyncio.to_thread(release.wait, 10), "test did not release refresh"
            return _rotation()

        async def at_shutdown(*args: Any, **kwargs: Any) -> None:
            # Run the signal on the next loop turn. With no executor job to
            # drain, real shutdown returns immediately, *before* that signal.
            loop.call_soon(stopping.set)
            await shutdown(*args, **kwargs)
            drained.set()

        monkeypatch.setattr(AuthStore, "_refresh_fn", lambda self, provider: refresh)
        monkeypatch.setattr(loop, "shutdown_default_executor", at_shutdown)
        ongoing = asyncio.create_task(_resolve(tmp_path))
        await asyncio.wait_for(entered.wait(), 5)
        ongoing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ongoing

    def run() -> tuple[str, str]:
        asyncio.run(viewer())
        return _read_pair(tmp_path)

    owner = asyncio.create_task(asyncio.to_thread(run))
    try:
        await _entered(stopping)
        assert not drained.is_set(), "normal shutdown escaped the already-sent refresh"
    finally:
        release.set()
        persisted = await owner
    assert persisted == ("synthetic-rotated", "synthetic-rotated-refresh")
