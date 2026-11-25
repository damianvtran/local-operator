"""Cold-open account context: one store per process, memoised, single-loop.

Regression cover for the v0.113.3 open trace, where the ``account_context``
phase of two openai cold opens took 701 ms and 1851 ms because every open
bootstrapped a NEW writable ``AuthStore`` (cross-process file lock, WAL pragma,
schema script, ``BEGIN IMMEDIATE``) on the auth.db ~25 lop processes share.
See ``cold_model._AccountContextWorker`` for the design these tests pin.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.session import cold_model
from local_operator.session.frontend_state import FrontendModelSpec

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Any:
    for var in ("OPENAI_API_KEY", "OPENAI_OAUTH_TOKEN", "LOCAL_OPERATOR_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cold_model._reset_account_context_cache()
    yield
    cold_model._reset_account_context_cache()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    # A config root as a running process has it: config.yml and auth.db exist
    # before the first cold open, so their fingerprints are stable.
    ConfigManager(config_dir=tmp_path).get_config()
    AuthStore(tmp_path / "auth.db").close()
    return tmp_path


@pytest.fixture
def connects(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count writable-store bootstraps (``AuthStore._connect``)."""
    calls: list[int] = []
    real = AuthStore._connect

    def _counting(self: AuthStore) -> Any:
        calls.append(1)
        return real(self)

    monkeypatch.setattr(AuthStore, "_connect", _counting)
    return calls


@pytest.fixture
def no_listing(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Keep ``resolve_model_info`` offline and record which credential it saw."""
    from local_operator.model import configure
    from local_operator.model.registry import ModelInfo

    seen: list[tuple[Any, ...]] = []

    def _info(provider: str, model_id: str, *, credential: Any = None, base: Any = None) -> Any:
        seen.append(credential)
        token = str(credential[0]) if credential else ""
        # A window derived from the bearer, so two credentials give two answers.
        window = 100_000 + int(hashlib.sha1(token.encode()).hexdigest()[:4], 16)
        return ModelInfo(
            id=model_id, name=model_id, description="", context_window=window, max_tokens=8192
        )

    monkeypatch.setattr(configure, "resolve_model_info", _info)
    return seen


def _spec() -> FrontendModelSpec:
    return FrontendModelSpec(provider="openai", model_id="gpt-5.1")


def _seed(root: Path, provider: str, credential: dict[str, Any]) -> None:
    store = AuthStore(root / "auth.db")
    try:
        store.upsert_credential(provider, credential)
    finally:
        store.close()


def _db_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in root.iterdir()
        if p.name.startswith("auth.db") and not p.name.endswith("-shm")
    }


async def test_two_cold_opens_build_at_most_one_writable_store(
    root: Path, connects: list[int], no_listing: list[Any]
) -> None:
    _seed(root, "openai", {"type": "api_key", "key": "sk-one", "source": "login"})
    connects.clear()
    a = await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    b = await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-b")
    assert a.context_metadata_resolved and b.context_metadata_resolved
    assert len(connects) <= 1, f"{len(connects)} writable AuthStore bootstraps"


async def test_concurrent_resolutions_are_single_flight(
    root: Path, monkeypatch: pytest.MonkeyPatch, no_listing: list[Any]
) -> None:
    _seed(root, "openai", {"type": "api_key", "key": "sk-one", "source": "login"})
    from local_operator.providers import failover

    real = failover._resolve_access_for_provider
    calls: list[str] = []

    async def _slow(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[2])
        await asyncio.sleep(0.05)
        return await real(*args, **kwargs)

    monkeypatch.setattr(failover, "_resolve_access_for_provider", _slow)
    results = await asyncio.gather(
        *(
            cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
            for _ in range(5)
        )
    )
    assert calls == ["sess-a"]
    assert len({r.context_window for r in results}) == 1
    # Each caller owns its copy: mutating one cannot leak into the memo.
    results[0].context_window = 1
    again = await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert again.context_window == results[1].context_window
    assert calls == ["sess-a"]


async def test_expired_token_refresh_runs_on_one_loop_without_cross_loop_error(
    root: Path, monkeypatch: pytest.MonkeyPatch, no_listing: list[Any]
) -> None:
    """``read_only`` does NOT stop an expired bearer refreshing, and the store's
    refresh single-flight is ``asyncio.Lock``s. Three sessions resolving the same
    expired login concurrently contend on that lock; on per-call loops the lock
    binds to one and the next waiter raises "bound to a different event loop"."""
    expired = {
        "type": "oauth",
        "refresh": "r1",
        "access": "access-1",
        "expires": int(time.time() * 1000) + 1000,  # inside the refresh skew
        "account_id": "acct-1",
        "org_id": "org-1",
    }
    _seed(root, "openai", expired)
    loops: set[int] = set()
    refreshes: list[str] = []

    async def _refresh(creds: dict[str, Any]) -> dict[str, Any]:
        loops.add(id(asyncio.get_running_loop()))
        refreshes.append(creds["refresh"])
        await asyncio.sleep(0.05)
        return {"access": "access-2", "expires": int(time.time() * 1000) + 3600_000}

    monkeypatch.setattr(AuthStore, "_refresh_fn", lambda self, provider: _refresh)
    for round_ in range(2):
        results = await asyncio.gather(
            *(
                cold_model.resolve_context_metadata(
                    root, _spec(), stickiness_key=f"sess-{round_}-{i}"
                )
                for i in range(3)
            )
        )
        assert all(r.context_metadata_resolved for r in results)
        # Expire it again so the second round contends on the same lock anew.
        store = AuthStore(root / "auth.db")
        store.upsert_credential("openai", {**expired, "access": f"access-r{round_}"})
        store.close()
    assert refreshes, "the expired login was never refreshed — the test lost its subject"
    assert len(loops) == 1, "refreshes ran on more than one event loop"
    assert all(c is not None and c[0] == "access-2" for c in no_listing)


async def test_read_only_resolution_writes_nothing_to_auth_db(
    root: Path, no_listing: list[Any]
) -> None:
    store = AuthStore(root / "auth.db")
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-one", "source": "login"})
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-two", "source": "login"})
    store.close()
    before = _db_bytes(root)
    for key in ("sess-a", "sess-b", "sess-c"):
        await cold_model.resolve_context_metadata(root, _spec(), stickiness_key=key)
    cold_model._reset_account_context_cache()  # close the worker's store first
    assert _db_bytes(root) == before


async def test_memo_keys_on_session_and_credential_and_expires(
    root: Path, monkeypatch: pytest.MonkeyPatch, no_listing: list[Any]
) -> None:
    store = AuthStore(root / "auth.db")
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-one", "source": "login"})
    first = await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert [c[0] for c in no_listing] == ["sk-one"]

    # A different conversation never reuses another's answer: it may be pinned
    # to a different account.
    await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-b")
    assert len(no_listing) == 2

    # Same key, unchanged state: served from the memo.
    await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert len(no_listing) == 2

    # A credential change (here: the login replaced) misses at once.
    for row in store.list_credentials("openai"):
        store.delete_credential(row.id)
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-new", "source": "login"})
    store.close()
    changed = await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert no_listing[-1][0] == "sk-new"
    assert changed.context_window != first.context_window

    # And a cached entry expires after the TTL.
    seen = len(no_listing)
    clock = [time.monotonic()]
    monkeypatch.setattr(cold_model.time, "monotonic", lambda: clock[0])
    await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert len(no_listing) == seen
    clock[0] += cold_model._ACCOUNT_CONTEXT_TTL_S + 1
    await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert len(no_listing) == seen + 1


async def test_failure_is_not_cached(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.providers import failover

    calls: list[int] = []

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(failover, "_resolve_access_for_provider", _boom)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cold_model.resolve_context_metadata(root, _spec(), stickiness_key="sess-a")
    assert len(calls) == 2

