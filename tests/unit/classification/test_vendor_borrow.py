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
