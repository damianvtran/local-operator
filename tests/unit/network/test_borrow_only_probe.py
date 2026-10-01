"""A mesh BORROW-ONLY device reads UNAVAILABLE from the persisted-credential probe.

The STT cascade lane asked for this to be stated and pinned rather than left to
fall out of ``__getattr__``. ``MeshAwareAuthStore`` forwards
``has_persisted_credential`` to its LOCAL store, and a brokered grant is not a
persisted row there, so a device whose only credential is borrowed does not
advertise the rung -- while the CALL-time path (``get_api_key``) still serves the
borrowed bearer. That is the persisted-only rule as written; whether brokered
logins should advertise is a decision for the mesh owners, and this test is what
changes if they decide it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.network.credentials.types import CredentialRef, Grant
from local_operator.providers.auth_store import AuthStore

OWNER = "d_00000000000000000000000000000022"
SELF = "d_00000000000000000000000000000011"


class _BorrowMesh:
    """The borrower seam: always willing to borrow, always granted."""

    self_device: str = SELF
    grants: Any = None
    placement: Any = None

    def should_borrow(self, key: str) -> bool:
        return True

    def owner_of(self, key: str) -> str:
        return OWNER

    def owner_label(self, key: str) -> str:
        return "owner-laptop"

    def owner_last_seen_s(self, device: str) -> float | None:
        return 1.0

    async def grant_async(self, key: str, **_kwargs: Any) -> Grant:
        return Grant(
            access_token="borrowed-bearer",
            kind="bearer",
            token_expires_at_ms=0,
            grant_expires_at_ms=int(time.time() * 1000) + 60_000,
            credential_ref=CredentialRef(
                owner_device=OWNER,
                owner_device_name="owner-laptop",
                provider="radient",
                kind="oauth",
                credential_id=7,
            ),
            served_by=OWNER,
        )

    def report_sync(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_a_borrow_only_device_reads_unavailable_but_still_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    local = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    mesh: Any = _BorrowMesh()
    store = MeshAwareAuthStore(local, mesh=mesh, config_dir=tmp_path)
    try:
        # Call time: the borrow works, exactly as before.
        assert await store.get_api_key("radient", "s1", read_only=True) == "borrowed-bearer"
        # Availability: not a persisted row on THIS device.
        assert await store.has_persisted_credential("radient", "s1") is False
        # A device with its OWN login advertises, borrow or not.
        local.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "refresh": "r",
                "access": "a",
                "expires": int(time.time() * 1000) + 3_600_000,
            },
        )
        assert await store.has_persisted_credential("radient", "s1") is True
    finally:
        local.close()
