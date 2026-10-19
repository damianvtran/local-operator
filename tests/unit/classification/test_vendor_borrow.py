"""The provider-credential borrow rung the classification seam was missing.

Measured on ``cloud-node-1``: the Radient login lived on the head and was a
placement holder on the node, but ``classification/vendors.py`` read only
``providers.auth_store.shared_auth_store`` — a plain ``AuthStore`` with no mesh
rung — so the leg reported Radient absent and the cascade silently fell through
to OpenRouter. These tests pin the rung that closes that gap: after the local
store says nothing, ONE borrow is tried, and its absence, value and failures all
degrade to the same "this tier said nothing" contract the module documents.
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = pytest.mark.asyncio


async def test_a_login_lent_by_an_owner_reaches_the_classification_leg(
    holder_root, monkeypatch
) -> None:
    """The local store is empty; the placement holder row makes the borrow the answer."""
    from local_operator.classification import vendors
    from local_operator.network import credentials as credentials_pkg

    calls: list[tuple[str, object]] = []

    async def fake_borrow(provider: str, config_dir=None) -> str:
        calls.append((provider, config_dir))
        return "lent-key"

    monkeypatch.setattr(credentials_pkg, "borrowed_provider_key", fake_borrow)

    assert await vendors.auth_store_api_key(holder_root, "radient") == "lent-key"
    assert calls == [("radient", holder_root)]


async def test_a_zero_peer_device_returns_none_without_constructing_a_mesh_store(
    tmp_path, monkeypatch
) -> None:
    """Absence must cost nothing: no placement means no store, exactly as before."""
    from local_operator.classification import vendors
    from local_operator.network.credentials import store as mesh_store

    root = tmp_path / "bare"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    def explode(*args, **kwargs):
        raise AssertionError("a zero-peer device must not construct a mesh store")

    monkeypatch.setattr(mesh_store, "build_auth_store", explode)

    assert await vendors.auth_store_api_key(root, "radient") is None


async def test_a_raising_borrow_degrades_to_no_credential(holder_root, monkeypatch) -> None:
    """A provider leg that cannot resolve is not a failed turn — even mid-borrow."""
    from local_operator.classification import vendors
    from local_operator.network import credentials as credentials_pkg

    async def explode(*args, **kwargs):
        raise RuntimeError("relay is down")

    monkeypatch.setattr(credentials_pkg, "borrowed_provider_key", explode)

    assert await vendors.auth_store_api_key(holder_root, "radient") is None


class _FakeStore:
    """The mesh-aware store the helper builds, without a relay behind it."""

    def __init__(self, key: str | None, *, delay_s: float = 0.0) -> None:
        self._key = key
        self._delay_s = delay_s
        self.closed = False

    async def get_api_key(self, provider: str, session_id=None, *, read_only=False, **kwargs):
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._key

    def close(self) -> None:
        self.closed = True


async def test_a_slow_borrow_is_bounded_and_always_closed(holder_root, monkeypatch) -> None:
    """F1: the borrow has its own ceiling, well below the classification budget."""
    from local_operator.network.credentials import borrowed_provider_key
    from local_operator.network.credentials import store as mesh_store

    fake = _FakeStore("too-late", delay_s=2.0)
    monkeypatch.setattr(mesh_store, "build_auth_store", lambda *args, **kwargs: fake)

    started = time.monotonic()
    key = await borrowed_provider_key("radient", holder_root, timeout_s=0.05)
    elapsed = time.monotonic() - started

    assert key is None
    assert elapsed < 1.0
    assert fake.closed, "the store is closed even when the borrow times out"


async def test_a_fast_borrow_resolves_the_preferred_leg(holder_root, monkeypatch) -> None:
    """The positive half of F1: a live borrow still wins the first leg."""
    from local_operator.classification.cascade import resolve_vendor_with_trace
    from local_operator.network.credentials import store as mesh_store

    monkeypatch.setattr(
        mesh_store, "build_auth_store", lambda *args, **kwargs: _FakeStore("lent-key")
    )

    resolution = await resolve_vendor_with_trace(holder_root)

    assert resolution.vendor is not None and resolution.vendor.name == "radient"
    assert resolution.skipped == ()


async def test_a_local_row_short_circuits_the_borrow(holder_root, monkeypatch) -> None:
    """F4(a): local-first still wins — a local login never pays for a borrow."""
    from local_operator.classification import vendors
    from local_operator.network import credentials as credentials_pkg
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=holder_root / "auth.db", config_dir=holder_root)
    store.upsert_credential("radient", {"key": "local-key", "source": "login", "type": "api_key"})
    store.close()

    async def explode(*args, **kwargs):
        raise AssertionError("a local credential must short-circuit the borrow")

    monkeypatch.setattr(credentials_pkg, "borrowed_provider_key", explode)

    assert await vendors.auth_store_api_key(holder_root, "radient") == "local-key"


async def test_the_borrow_reads_the_explicit_roots_store(holder_root, tmp_path, monkeypatch):
    """F2: ``config_dir`` does not select the DB; the explicit root must be read."""
    from local_operator.network.credentials import borrowed_provider_key
    from local_operator.providers.auth_store import AuthStore

    ambient = tmp_path / "ambient"
    ambient.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    ambient_store = AuthStore(db_path=ambient / "auth.db", config_dir=ambient)
    ambient_store.upsert_credential(
        "radient", {"key": "ambient-key", "source": "login", "type": "api_key"}
    )
    ambient_store.close()

    explicit_store = AuthStore(db_path=holder_root / "auth.db", config_dir=holder_root)
    explicit_store.upsert_credential(
        "radient", {"key": "explicit-key", "source": "login", "type": "api_key"}
    )
    explicit_store.close()

    assert await borrowed_provider_key("radient", holder_root) == "explicit-key"


async def test_the_borrow_never_reads_the_ambient_store(holder_root, tmp_path, monkeypatch):
    """F2 regression: a root with no local row must not surface the ambient profile's."""
    from local_operator.network.credentials import borrowed_provider_key
    from local_operator.providers.auth_store import AuthStore

    ambient = tmp_path / "ambient"
    ambient.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(ambient))
    ambient_store = AuthStore(db_path=ambient / "auth.db", config_dir=ambient)
    ambient_store.upsert_credential(
        "radient", {"key": "ambient-key", "source": "login", "type": "api_key"}
    )
    ambient_store.close()

    # The explicit root has placement but no local row; its owner is not reachable
    # from this test, so the only correct answer is "no credential" — never ambient.
    assert await borrowed_provider_key("radient", holder_root) is None
